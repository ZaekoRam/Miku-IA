"""
Modulo 1: Oidos de Miku (STT)

Escucha el microfono en segundo plano, detecta la palabra clave "Oye
Miku" y transcribe el comando que sigue usando faster-whisper sobre GPU
(CUDA).

Uso como script (prueba manual, habla al microfono):
    python -m core.stt_listener

Uso como modulo (para el orquestador / modulo cognitivo, mas adelante):
    from core.stt_listener import MikuListener
    listener = MikuListener()
    listener.escuchar(on_comando=lambda texto: print("Comando:", texto))
"""

from __future__ import annotations

import os
import sys
import sysconfig

# Agregar las DLLs de NVIDIA (cuBLAS/cuDNN) al PATH de Windows ANTES de
# importar faster-whisper. Sin esto, ctranslate2 no encuentra
# cublas64_12.dll y falla (no al cargar el modelo, sino en la primera
# transcripcion real, que es cuando de verdad usa cuBLAS).
#
# sysconfig.get_paths()["purelib"] devuelve el site-packages del Python
# que esta corriendo AHORA -- sirve igual con venv que sin el; derivarlo
# a mano desde sys.executable asume un layout de venv y apunta a la
# carpeta equivocada en una instalacion global de Python.
site_packages = sysconfig.get_paths()["purelib"]
nvidia_paths = [
    os.path.join(site_packages, "nvidia", "cublas", "bin"),
    os.path.join(site_packages, "nvidia", "cudnn", "bin"),
]
for path in nvidia_paths:
    if os.path.exists(path):
        os.add_dll_directory(path)
        os.environ["PATH"] = path + os.path.pathsep + os.environ["PATH"]

import gc
import math
import queue
import re
import threading
import time
from collections import deque
from typing import Callable

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

import config
from core.utils import normalizar_texto


class MikuListener:
    def __init__(
        self,
        palabra_clave: str = config.PALABRA_CLAVE,
        variantes_palabra_clave: tuple[str, ...] = config.VARIANTES_PALABRA_CLAVE,
        modelo: str = config.MODELO_WHISPER,
        dispositivo: str = config.DISPOSITIVO_WHISPER,
        tipo_computo: str = config.TIPO_COMPUTO_WHISPER,
        dispositivo_entrada: int | None = config.DISPOSITIVO_ENTRADA,
    ):
        self.palabra_clave = normalizar_texto(palabra_clave)
        self._variantes_palabra_clave = [self._limpiar_para_deteccion(v) for v in variantes_palabra_clave]
        self.dispositivo_entrada = self._preparar_dispositivo_entrada(dispositivo_entrada)

        # Un solo modelo para deteccion de palabra clave y transcripcion
        # del comando: evita tener dos modelos distintos ocupando VRAM
        # a la vez.
        print(f"[Miku-STT] Cargando modelo Whisper '{modelo}' en {dispositivo}...")
        self._whisper = WhisperModel(modelo, device=dispositivo, compute_type=tipo_computo)

        self._cola_audio: "queue.Queue[np.ndarray]" = queue.Queue()

        # Cola/bandera aparte para el "monitor de interrupcion" (ver
        # activar_monitor_interrupcion): solo se llena mientras Miku esta
        # hablando, para poder detectar "Miku silencio" a MITAD de una
        # respuesta sin competir por los mismos bloques que el bucle
        # principal de escuchar() (que en ese momento esta bloqueado
        # procesando el comando, no leyendo mas audio).
        self._cola_monitor: "queue.Queue[np.ndarray]" = queue.Queue()
        self._monitor_activo = threading.Event()

    def cerrar(self) -> None:
        """
        Libera el modelo Whisper de la GPU. Aunque en teoria el driver
        de NVIDIA recupera toda la VRAM de un proceso al terminar (ni
        siquiera hace falta llamar esto para que Ctrl+C libere memoria),
        se llama de forma explicita al salir (ver main.py,
        _limpiar_al_salir) para no dejarlo como el unico modelo en GPU
        sin una ruta de apagado propia, igual que MikuBrain.descargar_modelo()
        y MikuVoice.cerrar().
        """
        if self._whisper is None:
            return
        self._whisper = None
        gc.collect()
        print("[Miku-STT] Modelo Whisper liberado de la GPU.")

    # ------------------------------------------------------------------ #
    # Dispositivos de entrada (microfonos)
    # ------------------------------------------------------------------ #

    @staticmethod
    def listar_dispositivos_entrada() -> None:
        """Imprime todos los dispositivos de entrada de audio disponibles."""
        try:
            predeterminado = sd.default.device[0]
        except Exception:
            predeterminado = -1

        print("[Miku-STT] Dispositivos de entrada disponibles:")
        for indice, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0:
                marca = " <- predeterminado de Windows" if indice == predeterminado else ""
                print(f"    [{indice}] {info['name']} ({info['max_input_channels']} canales){marca}")

    def _preparar_dispositivo_entrada(self, dispositivo_entrada: int | None) -> int:
        """
        Imprime la lista de microfonos y resuelve cual usar: el indice
        explicito que se pase (o venga de config.DISPOSITIVO_ENTRADA), o
        si no hay ninguno, el microfono predeterminado de Windows.
        """
        self.listar_dispositivos_entrada()

        if dispositivo_entrada is not None:
            info = sd.query_devices(dispositivo_entrada)
            print(f"[Miku-STT] Usando el dispositivo configurado: [{dispositivo_entrada}] {info['name']}")
            return dispositivo_entrada

        predeterminado = sd.default.device[0]
        info = sd.query_devices(predeterminado)
        print(f"[Miku-STT] Usando el microfono predeterminado de Windows: [{predeterminado}] {info['name']}")
        return predeterminado

    # ------------------------------------------------------------------ #
    # Captura de audio
    # ------------------------------------------------------------------ #

    def _callback_audio(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[Miku-STT] Aviso de audio: {status}", file=sys.stderr)
        # Solo copiamos el bloque a la cola; el trabajo pesado (Whisper)
        # nunca debe correr dentro del callback de audio.
        bloque = indata[:, 0].copy()
        self._cola_audio.put(bloque)
        if self._monitor_activo.is_set():
            self._cola_monitor.put(bloque)

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0

    @staticmethod
    def _medidor_nivel(rms: float, ancho: int = 30) -> str:
        """Convierte el RMS de un bloque de audio en una barra visual tipo VU-metro + dB."""
        minimo_db, maximo_db = -60.0, 0.0
        db = 20 * math.log10(rms) if rms > 1e-9 else minimo_db
        db = max(minimo_db, min(maximo_db, db))
        proporcion = (db - minimo_db) / (maximo_db - minimo_db)
        llenado = int(proporcion * ancho)
        barra = "#" * llenado + "-" * (ancho - llenado)
        return f"[{barra}] {db:5.1f} dB"

    @staticmethod
    def _es_transcripcion_confiable(segmentos: list) -> bool:
        """
        faster-whisper da, por cada segmento, avg_logprob (que tan segura
        estuvo la decodificacion -- mas cerca de 0 es mejor, valores como
        -1.0 o mas negativos indican que el modelo "adivino" bastante) y
        no_speech_prob (la probabilidad de que ese segmento sea silencio o
        ruido, no habla real). Ruido ambiente o habla poco clara suele
        transcribirse igual como palabras sueltas con sentido gramatical
        pero sin ningun sentido real (confirmado con ejemplos reales del
        diario) -- eso no se puede filtrar mirando SOLO el texto, pero
        Whisper ya sabe internamente que no estaba seguro. Se usa esto en
        vez de intentar adivinar "esto suena a texto incoherente" con
        reglas de texto, que es mucho menos confiable.
        """
        if not segmentos:
            return False
        return all(
            s.avg_logprob >= config.UMBRAL_LOGPROB_STT and s.no_speech_prob <= config.UMBRAL_NO_SPEECH_STT
            for s in segmentos
        )

    @staticmethod
    def _tiene_voz(audio: np.ndarray) -> bool:
        """
        Filtro previo con el modelo Silero VAD que trae faster-whisper,
        usado SUELTO (get_speech_timestamps) en vez de pasarle
        vad_filter=True a transcribe() -- esa opcion altera la
        segmentacion INTERNA de donde salen los timestamps por palabra
        y rompio la deteccion de "oye miku" en vivo (ver notas en
        _transcribir_con_palabras). Aca solo se usa como una pregunta de
        si/no sobre el audio crudo, sin tocar como despues se transcribe
        -- si no hay ni un tramo de voz, ni vale la pena mandarlo a
        Whisper (evita alucinaciones tipo "gracias por ver el video"
        sobre puro ruido/golpes al microfono, y de paso ahorra computo).
        """
        return bool(get_speech_timestamps(audio, VadOptions(min_speech_duration_ms=100), sampling_rate=config.SAMPLE_RATE))

    def _transcribir(self, audio: np.ndarray, modelo: WhisperModel) -> str:
        # beam_size mas alto (5 en vez de 1) porque esto corre UNA vez
        # por comando, no en un loop continuo -- hay margen de latencia
        # para ganar precision. vad_filter descarta tramos de silencio/
        # ruido antes de transcribir (menos alucinaciones tipo "Meco",
        # "Susururos" con audio que no era voz real). condition_on_
        # previous_text=False evita que Whisper seque un segmento
        # alucinado arrastrando texto del segmento anterior.
        segmentos = list(
            modelo.transcribe(
                audio,
                language=config.IDIOMA,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )[0]
        )
        texto = " ".join(segmento.text.strip() for segmento in segmentos).strip()
        if texto and not self._es_transcripcion_confiable(segmentos):
            print(f"\n[Miku-STT] Transcripcion de baja confianza descartada (probable ruido): {texto!r}")
            return ""
        return texto

    def _transcribir_con_palabras(self, audio: np.ndarray, modelo: WhisperModel) -> tuple[str, list]:
        """
        Como _transcribir, pero tambien pide timestamps por palabra --
        un poco mas lento, asi que solo se usa en la ventana corta (~2.5s)
        de deteccion de la palabra clave, para poder ubicar donde termina
        "miku"/"mico" y no perder lo que se dijo justo despues en la
        misma respiracion (ver _fin_palabra_clave_seg).
        """
        # OJO: a diferencia de _transcribir(), aca NO se descartan
        # transcripciones de baja confianza -- esto se usa para detectar
        # la palabra clave, y un "oye miku" dicho rapido/con poco
        # contexto alrededor ya tiende a salir con confianza mas baja;
        # filtrarlo aca arriesgaba volver a pedir que se hable mas fuerte
        # (ya costo varias rondas bajar ese umbral). El filtro de
        # confianza aplica solo al COMANDO ya capturado (_transcribir).
        # beam_size se queda en 1 aca (a diferencia de _transcribir): esto
        # SI corre en un loop continuo mientras se espera "oye miku", asi
        # que la latencia importa mas que exprimir precision extra.
        #
        # NO se le agrega vad_filter aca (a diferencia de _transcribir):
        # se probo y rompio la deteccion en vivo -- "oye miku" dejaba de
        # reactivar la escucha despues de un par de segundos. Esta
        # ventana ya es muy corta (~2.5s) y _fin_palabra_clave_seg()
        # depende de los timestamps por palabra para saber donde corta
        # "miku" del resto -- el VAD de faster-whisper puede recortar o
        # correr esos timestamps en ventanas tan cortas, y esa funcion es
        # delicada (ver nota arriba, ya costo varias rondas afinarla).
        segmentos = list(
            modelo.transcribe(
                audio,
                language=config.IDIOMA,
                beam_size=1,
                word_timestamps=True,
            )[0]
        )
        texto_partes: list[str] = []
        palabras: list = []
        for segmento in segmentos:
            texto_partes.append(segmento.text.strip())
            if segmento.words:
                palabras.extend(segmento.words)
        return " ".join(texto_partes).strip(), palabras

    # ------------------------------------------------------------------ #
    # Deteccion de la palabra clave
    # ------------------------------------------------------------------ #

    @staticmethod
    def _limpiar_para_deteccion(texto: str) -> str:
        """
        Minusculas, sin acentos (via normalizar_texto) y sin signos de
        puntuacion, para que "oye miku,", "oye. miku!" o "oyemiku" se
        comparen igual que "oye miku".
        """
        texto = normalizar_texto(texto)
        texto = re.sub(r"[^a-z0-9\s]", " ", texto)
        return re.sub(r"\s+", " ", texto).strip()

    @staticmethod
    def _distancia_edicion(a: str, b: str) -> int:
        """
        Distancia de Levenshtein simple (sin dependencias externas -- las
        palabras que compara son cortas, 4-5 letras, asi que el costo no
        importa). Se usa para tolerar transcripciones de Whisper con un
        error de una sola letra (ej. "miku" -> "meco"/"miko"/"niku").
        """
        if a == b:
            return 0
        filas, columnas = len(a) + 1, len(b) + 1
        dp = [[0] * columnas for _ in range(filas)]
        for i in range(filas):
            dp[i][0] = i
        for j in range(columnas):
            dp[0][j] = j
        for i in range(1, filas):
            for j in range(1, columnas):
                costo = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + costo)
        return dp[-1][-1]

    def _es_palabra_clave(self, texto: str) -> bool:
        """
        True si el texto transcrito contiene la palabra clave, alguna
        variante ya conocida (VARIANTES_PALABRA_CLAVE), una palabra
        suelta a distancia de edicion <=1 de "miku"/"mico" -- Whisper a
        veces la transcribe con un error de una letra (ej. "meco",
        "miko", "niku") que no matchea ningun substring exacto, y
        obligaba a repetir "oye miku" varias veces para que activara --,
        o dos palabras SEGUIDAS que juntas (sin el espacio) tambien caen
        a esa misma distancia -- probado en vivo que Whisper a veces
        parte "miku" en dos fragmentos separados por un espacio ("me
        co", "me cu", "me con") en vez de una sola palabra, y el chequeo
        por palabra suelta no los detectaba ninguno de los dos por
        separado.
        """
        limpio = self._limpiar_para_deteccion(texto)
        if not limpio:
            return False

        if any(variante in limpio for variante in self._variantes_palabra_clave):
            return True

        palabras = limpio.split()
        if any(
            self._distancia_edicion(palabra, nucleo) <= 1
            for palabra in palabras
            for nucleo in ("miku", "mico")
        ):
            return True

        return any(
            self._distancia_edicion(palabras[i] + palabras[i + 1], nucleo) <= 1
            for i in range(len(palabras) - 1)
            for nucleo in ("miku", "mico")
        )

    @classmethod
    def _es_palabra_nucleo(cls, palabra: str) -> bool:
        """El mismo criterio que usa _es_palabra_clave para una sola palabra suelta (sin el prefijo "oye")."""
        limpia = cls._limpiar_para_deteccion(palabra)
        if not limpia:
            return False
        if "miku" in limpia or "mico" in limpia:
            return True
        return any(cls._distancia_edicion(limpia, nucleo) <= 1 for nucleo in ("miku", "mico"))

    @classmethod
    def _fin_palabra_clave_seg(cls, palabras: list) -> float | None:
        """
        Busca, entre las palabras transcritas con sus timestamps (ver
        _transcribir_con_palabras), la ULTIMA que suene a "miku"/"mico"
        y devuelve el segundo (relativo al inicio del audio) en que
        termina -- para poder recortar justo ahi y no perder lo que el
        usuario haya dicho a continuacion en el mismo aliento (ej. "oye
        miku, que tal el dia" sin pausa). None si no se encontro ninguna
        (se sigue el camino de siempre: empezar a grabar desde cero).
        """
        fin = None
        for palabra in palabras:
            texto_palabra = getattr(palabra, "word", "")
            if cls._es_palabra_nucleo(texto_palabra):
                fin = getattr(palabra, "end", None)
        return fin

    # ------------------------------------------------------------------ #
    # Captura del comando completo, una vez detectada la palabra clave
    # ------------------------------------------------------------------ #

    def _grabar_comando(self, audio_inicial: np.ndarray | None = None) -> str:
        """
        Si se pasa audio_inicial (el sobrante de audio que ya se habia
        grabado en la misma respiracion que "oye miku", ver
        _fin_palabra_clave_seg), arranca la grabacion con eso en vez de
        empezar de cero -- para no perder el comando si el usuario no
        hizo pausa despues de la palabra clave.

        Usa dos umbrales de silencio distintos, no uno solo:
        - Mientras el usuario TODAVIA NO dijo nada (ej. recien empieza el
          modo conversacion y esta pensando que preguntar), se le da un
          margen mas largo (ESPERA_INICIO_HABLA_SEG) antes de rendirse.
          Sin esto, una pausa de mas de SILENCIO_PARA_CORTAR_SEG para
          pensar cortaba la grabacion vacia y el modo conversacion se
          cerraba solo sin que el usuario dijera nada.
        - Una vez que YA empezo a hablar, se usa el umbral corto de
          siempre (SILENCIO_PARA_CORTAR_SEG) para no demorar de mas el
          corte cuando termina su frase.
        """
        print("[Miku-STT] Escuchando comando...")
        bloques: list[np.ndarray] = [audio_inicial] if audio_inicial is not None and len(audio_inicial) else []
        empezo_a_hablar = bool(audio_inicial is not None and len(audio_inicial))
        bloques_silencio_seguidos = 0
        bloques_para_cortar = max(1, int(config.SILENCIO_PARA_CORTAR_SEG / config.DURACION_BLOQUE_SEG))
        bloques_espera_inicio = max(1, int(config.ESPERA_INICIO_HABLA_SEG / config.DURACION_BLOQUE_SEG))
        tiempo_inicio = time.monotonic()

        while True:
            try:
                bloque = self._cola_audio.get(timeout=config.ESPERA_INICIO_HABLA_SEG + 1)
            except queue.Empty:
                break

            bloques.append(bloque)

            if self._rms(bloque) < config.UMBRAL_SILENCIO_RMS:
                bloques_silencio_seguidos += 1
            else:
                bloques_silencio_seguidos = 0
                empezo_a_hablar = True

            tiempo_transcurrido = time.monotonic() - tiempo_inicio
            tiempo_agotado = tiempo_transcurrido >= config.DURACION_MAX_COMANDO_SEG

            if empezo_a_hablar:
                silencio_suficiente = bloques_silencio_seguidos >= bloques_para_cortar
                # Exigimos al menos un par de bloques grabados para no
                # cortar apenas empieza (silencio inicial mientras la
                # persona respira).
                if (silencio_suficiente and tiempo_transcurrido > config.DURACION_BLOQUE_SEG * 2) or tiempo_agotado:
                    break
            else:
                if bloques_silencio_seguidos >= bloques_espera_inicio or tiempo_agotado:
                    break

        audio_completo = np.concatenate(bloques) if bloques else np.array([], dtype=np.float32)
        return self._transcribir(audio_completo, self._whisper)

    # ------------------------------------------------------------------ #
    # Monitor de interrupcion ("Miku silencio" a mitad de una respuesta)
    # ------------------------------------------------------------------ #

    def activar_monitor_interrupcion(self, frases: tuple[str, ...], on_interrupcion: Callable[[], None]) -> None:
        """
        Arranca un hilo aparte que escucha SOLO mientras Miku esta
        hablando (main.py lo activa justo antes de empezar a reproducir
        una respuesta y lo desactiva apenas termina) -- para poder decir
        "Miku silencio" y cortarla a mitad de una respuesta larga, como
        en Alexa.

        Usa su PROPIA cola de audio (_cola_monitor, ver _callback_audio)
        en vez de _cola_audio: el bucle principal de escuchar() esta
        bloqueado procesando el comando mientras Miku habla, asi que no
        esta leyendo _cola_audio en ese momento -- una cola aparte evita
        que este monitor y ese bucle compitan por los mismos bloques
        cuando el bucle principal se reactive.
        """
        with self._cola_monitor.mutex:
            self._cola_monitor.queue.clear()
        self._monitor_activo.set()
        hilo = threading.Thread(
            target=self._bucle_monitor_interrupcion, args=(frases, on_interrupcion), daemon=True
        )
        hilo.start()

    def desactivar_monitor_interrupcion(self) -> None:
        self._monitor_activo.clear()

    def _bucle_monitor_interrupcion(self, frases: tuple[str, ...], on_interrupcion: Callable[[], None]) -> None:
        """
        Junta ~1.5s de audio a la vez (ventana corta, para reaccionar
        rapido) y los transcribe buscando SOLO las frases de
        interrupcion -- no hace falta nada de la logica de deteccion de
        "oye miku" aca. Se corta solo (return) apenas encuentra una
        coincidencia; main.py se encarga de llamar a
        desactivar_monitor_interrupcion() cuando Miku termina de hablar
        por su cuenta.
        """
        tam_ventana = max(1, int(1.5 / config.DURACION_BLOQUE_SEG))
        ventana: deque = deque(maxlen=tam_ventana)

        while self._monitor_activo.is_set():
            try:
                bloque = self._cola_monitor.get(timeout=0.3)
            except queue.Empty:
                continue

            ventana.append(bloque)
            if len(ventana) < ventana.maxlen:
                continue

            audio = np.concatenate(list(ventana))
            if self._rms(audio) < config.UMBRAL_SILENCIO_RMS:
                continue

            texto = self._transcribir(audio, self._whisper)
            if not texto:
                continue

            limpio = normalizar_texto(texto)
            if any(frase in limpio for frase in frases):
                on_interrupcion()
                return

    # ------------------------------------------------------------------ #
    # Bucle principal
    # ------------------------------------------------------------------ #

    def escuchar(
        self,
        on_comando: Callable[[str], bool],
        on_activacion: Callable[[bool], None] | None = None,
        on_fin_escucha: Callable[[], None] | None = None,
        on_reanudar_escucha: Callable[[], None] | None = None,
        on_silencio: Callable[[], None] | None = None,
    ) -> None:
        """
        Bucle infinito: escucha el microfono, detecta "Oye Miku" y llama
        a on_comando(texto) con cada comando transcrito. Se detiene con
        Ctrl+C.

        on_comando debe devolver True si hay que seguir la conversacion
        (grabar el siguiente turno directo, SIN pedir "Oye Miku" de
        nuevo) o False si hay que cerrarla (ej. una despedida, "Miku
        callate", o el usuario se quedo en silencio). Este "modo
        conversacion" es lo que permite un ida y vuelta natural sin
        repetir la palabra clave en cada turno.

        Si se pasa on_activacion, se llama con UN booleano justo cuando
        se detecta la palabra clave la PRIMERA vez (no en cada turno
        dentro del modo conversacion) -- lo usa main.py para avisarle al
        avatar 3D que salga del "modo reposo" y diga la frase de
        activacion. El booleano es True si "oye miku" se dijo SOLA (sin
        nada pegado en la misma respiracion) y False si ya traia una
        instruccion pegada (ej. "oye miku, que hora es") -- main.py lo
        usa para decidir si vale la pena el gesto/frase de "¿si?" (tiene
        sentido cuando espera instruccion, no cuando ya se la dijeron).

        Si se pasa on_fin_escucha, se llama justo despues de que termina
        de grabarse CADA comando (haya quedado texto o no) -- lo usa
        main.py para apagar el "halo" de escuchando en la ventana antes
        de procesar la respuesta.

        Si se pasa on_silencio, se llama cuando un comando se graba
        VACIO (el usuario activo a Miku pero no dijo nada, o se quedo
        callado) -- a diferencia de on_comando, este SI se llama en ese
        caso. Sin esto, nada le avisaba al avatar que se ocultara si no
        habia una respuesta real de por medio (se quedaba en pantalla
        indefinidamente).

        Si se pasa on_reanudar_escucha, se llama al arrancar cada turno
        DENTRO del modo conversacion (cuando ya no hace falta "Oye
        Miku") -- lo usa main.py para prender el halo de escuchando de
        nuevo sin repetir la animacion de entrada ni la frase "¿Si?".
        """
        tam_bloque = int(config.SAMPLE_RATE * config.DURACION_BLOQUE_SEG)
        ventana = deque(maxlen=max(1, int(config.VENTANA_WAKE_SEG / config.DURACION_BLOQUE_SEG)))

        print(f"[Miku-STT] Escuchando... di '{config.PALABRA_CLAVE}' para activar (Ctrl+C para salir)")

        with sd.InputStream(
            device=self.dispositivo_entrada,
            samplerate=config.SAMPLE_RATE,
            channels=config.CANALES,
            blocksize=tam_bloque,
            dtype="float32",
            callback=self._callback_audio,
        ):
            try:
                while True:
                    bloque = self._cola_audio.get()
                    ventana.append(bloque)

                    rms = self._rms(bloque)
                    # Indicador de nivel en vivo (se reescribe en la misma
                    # linea) para confirmar que el microfono si esta
                    # mandando senal, aunque todavia no diga la palabra clave.
                    print(f"\r[Miku-STT] Nivel de audio: {self._medidor_nivel(rms)}", end="", flush=True)

                    if rms < config.UMBRAL_SILENCIO_RMS:
                        continue  # no vale la pena transcribir silencio

                    audio_ventana = np.concatenate(list(ventana))

                    # Filtro de voz (Silero VAD, ver _tiene_voz): un
                    # golpe al microfono o ruido de fondo pasa el umbral
                    # de RMS de arriba (es "sonido"), pero no es VOZ --
                    # sin este filtro, Whisper alucinaba frases enteras
                    # ("gracias por ver el video") sobre ese ruido.
                    if not self._tiene_voz(audio_ventana):
                        continue

                    texto, palabras = self._transcribir_con_palabras(audio_ventana, self._whisper)

                    if texto:
                        print(f"\n[Miku-STT] (escuchando) '{texto}'")

                    if self._es_palabra_clave(texto):
                        print("[Miku-STT] ¡Wake word detectada!")

                        # Si el usuario siguio hablando en la misma
                        # respiracion (sin pausa despues de "oye miku"),
                        # eso ya quedo grabado en audio_ventana -- se
                        # rescata en vez de tirarlo, para no obligar a
                        # pausar como con un asistente clasico. Esto se
                        # calcula ANTES de llamar a on_activacion (no
                        # despues, como era antes) para poder avisarle si
                        # hay o no un comando pegado -- el gesto de
                        # asentir/"¿si?" solo tiene sentido cuando "oye
                        # miku" se dice SOLA, no cuando ya viene con la
                        # instruccion pegada (ver on_activacion abajo).
                        audio_sobrante = None
                        fin_palabra_clave = self._fin_palabra_clave_seg(palabras)
                        if fin_palabra_clave is not None:
                            corte = int(fin_palabra_clave * config.SAMPLE_RATE)
                            resto = audio_ventana[corte:]
                            # Menos de ~150ms no vale la pena (seria solo
                            # el silencio/respiro justo despues de decir
                            # la palabra clave, no contenido real) -- Y
                            # ADEMAS se confirma con el VAD (_tiene_voz)
                            # que de verdad hay voz ahi, no solo
                            # silencio/respiracion/ruido de cola que dura
                            # mas de 150ms por casualidad. Sin este
                            # segundo chequeo, casi cualquier "oye miku"
                            # SOLA terminaba contando como "trae
                            # instruccion pegada" (por la colita de
                            # silencio de la ventana), y el gesto/frase de
                            # "¿si?" casi nunca se disparaba.
                            if len(resto) > int(0.15 * config.SAMPLE_RATE) and self._tiene_voz(resto):
                                audio_sobrante = resto

                        if on_activacion:
                            on_activacion(audio_sobrante is None)

                        ventana.clear()
                        with self._cola_audio.mutex:
                            self._cola_audio.queue.clear()

                        comando = self._grabar_comando(audio_inicial=audio_sobrante)
                        if on_fin_escucha:
                            on_fin_escucha()
                        if not comando and on_silencio:
                            on_silencio()

                        # Modo conversacion: mientras haya comando Y
                        # on_comando diga que siga (True), se graba el
                        # siguiente turno directo, sin volver a pedir la
                        # palabra clave. Si el usuario se queda callado
                        # (comando vacio), se corta solo -- no hace falta
                        # una despedida explicita.
                        seguir_conversacion = bool(comando) and on_comando(comando)
                        while seguir_conversacion:
                            if on_reanudar_escucha:
                                on_reanudar_escucha()
                            comando = self._grabar_comando()
                            if on_fin_escucha:
                                on_fin_escucha()
                            if not comando and on_silencio:
                                on_silencio()
                            seguir_conversacion = bool(comando) and on_comando(comando)

                        print(f"[Miku-STT] Escuchando... di '{config.PALABRA_CLAVE}' para activar")
            except KeyboardInterrupt:
                print("\n[Miku-STT] Detenido por el usuario.")


if __name__ == "__main__":
    listener = MikuListener()
    listener.escuchar(on_comando=lambda texto: print(f"\n>>> COMANDO RECIBIDO: {texto}\n"))
