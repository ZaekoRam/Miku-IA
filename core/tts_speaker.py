"""
Modulo TTS: la voz de Miku.

Convierte el texto de la respuesta del LLM a audio con Kokoro (TTS local
por GPU via onnxruntime-gpu/CUDA -- reemplaza a Edge-TTS, que necesitaba
red) y lo reproduce con pygame.

Si config.USE_RVC esta activo, el audio de Kokoro pasa despues por RVC
para convertirlo a la voz de personaje de Miku, usando el modelo en
models/rvc/. RVC corre en su propio venv aislado (rvc_env/, Python
3.10) porque sus dependencias (fairseq) no instalan en el Python
principal del proyecto.

La conversion RVC la hace un SERVIDOR PERSISTENTE (core/rvc_servidor.py)
que MikuVoice arranca una sola vez y mantiene vivo como subprocess,
hablandole por JSON sobre stdin/stdout. Antes cada frase lanzaba un
proceso nuevo que recargaba PyTorch + fairseq + HuBERT + RMVPE + el
modelo (~7s solo de carga); con el servidor persistente esa carga pasa
una sola vez al arrancar y cada frase despues solo paga el costo real
de la inferencia.

Si se le pasa un PuenteWebSocket (core/ws_bridge.py), tambien manda las
senales de lip-sync (hablando_inicio/nivel/fin) para que el avatar 3D en
gui_3d/ mueva la boca mientras Miku habla. Como no decodificamos el
audio real, el "nivel" que se manda es una simulacion (paseo aleatorio
acotado) sincronizada con la duracion de la reproduccion -- no es la
amplitud real del audio, pero da un movimiento de boca creible.

Uso como script (prueba manual, deberias escuchar la frase):
    python -m core.tts_speaker

Uso como modulo:
    from core.tts_speaker import MikuVoice
    voz = MikuVoice()
    voz.hablar("Hola, soy Miku.")
    voz.cerrar()
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pygame
import requests

import config
from core.ws_bridge import PuenteWebSocket

# Import diferido y con manejo de error propio (no al tope del archivo):
# kokoro_onnx / onnxruntime cargan CUDA en el import, y necesitan las
# DLLs de NVIDIA ya puestas en el PATH -- main.py lo hace ANTES de
# importar este modulo (mismo patron que ya usaba core/stt_listener.py
# para faster-whisper/ctranslate2), asi que para cuando este import
# corre, el PATH ya deberia estar listo.
from kokoro_onnx import Kokoro

# Debe ser identico al de core/rvc_servidor.py: rvc_python/fairseq/torch
# imprimen cosas sueltas a stdout por su cuenta (logs, prints internos),
# asi que solo las lineas que empiezan con este marcador son parte de
# nuestro protocolo -- el resto se ignora.
_MARCADOR_PROTOCOLO_RVC = "###RVC_MIKU### "


class MikuVoice:
    def __init__(
        self,
        voz: str = config.KOKORO_VOZ,
        velocidad: float = config.KOKORO_VELOCIDAD,
        idioma: str = config.KOKORO_IDIOMA,
        puente_ws: PuenteWebSocket | None = None,
    ):
        self.voz = voz
        self.velocidad = velocidad
        self.idioma = idioma
        self.puente_ws = puente_ws

        try:
            pygame.mixer.init()
        except Exception as error:
            print(f"[Miku-Voz] No se pudo inicializar el audio de salida: {error}")

        self._kokoro = self._cargar_kokoro()

        self._proceso_rvc: subprocess.Popen | None = None
        if config.USE_RVC:
            self._iniciar_servidor_rvc()

        # Evita que la muletilla y la primera oracion real de una
        # respuesta intenten sonar al mismo tiempo por el mismo canal
        # de pygame.mixer.music -- reproducir_audio() lo toma antes de
        # tocar el mixer, asi que si la muletilla ya esta sonando, la
        # oracion real simplemente espera a que termine (ver
        # reproducir_muletilla_aleatoria + main.py).
        self._lock_reproduccion = threading.Lock()

        # "Miku silencio"/"Miku callate" a mitad de una respuesta (ver
        # detener_habla) -- reproducir_audio() lo chequea en cada vuelta
        # de su loop de espera para poder cortar de inmediato en vez de
        # esperar a que termine de sonar solo.
        self._evento_detener_habla = threading.Event()

    # ------------------------------------------------------------------ #
    # Kokoro (TTS local por GPU)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _descargar_si_falta(ruta: Path, url: str, etiqueta: str) -> None:
        if ruta.is_file():
            return
        ruta.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Miku-Voz] Descargando {etiqueta} (no esta en {ruta})...")
        ruta_temporal = ruta.with_suffix(ruta.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(ruta_temporal, "wb") as f:
                for bloque in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(bloque)
        ruta_temporal.rename(ruta)
        print(f"[Miku-Voz] {etiqueta} descargado.")

    def _cargar_kokoro(self) -> Kokoro:
        """
        Carga el modelo de Kokoro (descargandolo primero si hace falta).
        onnxruntime elige el proveedor de ejecucion solo (ver
        kokoro_onnx/session.py: resolve_providers()) segun que
        distribucion de onnxruntime este instalada -- con
        onnxruntime-gpu instalado, usa CUDA automaticamente. Fijamos
        ONNX_PROVIDER explicito para que no intente TensorRT primero
        (no esta instalado, solo ensucia la consola con un warning
        antes de caer a CUDA de todas formas).
        """
        self._descargar_si_falta(config.KOKORO_MODEL_PATH, config.KOKORO_MODEL_URL, "modelo de Kokoro")
        self._descargar_si_falta(config.KOKORO_VOICES_PATH, config.KOKORO_VOICES_URL, "voces de Kokoro")

        os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")

        print("[Miku-Voz] Cargando Kokoro en GPU...")
        kokoro = Kokoro(str(config.KOKORO_MODEL_PATH), str(config.KOKORO_VOICES_PATH))
        # "Calienta" la sesion de CUDA con una frase corta -- la primera
        # inferencia de una sesion de onnxruntime en GPU suele ser mucho
        # mas lenta (compilacion/cache de kernels) que las siguientes; sin
        # esto, esa demora la pagaria la primera frase real de Miku.
        kokoro.create("Hola", voice=self.voz, lang=self.idioma)
        print("[Miku-Voz] Kokoro listo.")
        return kokoro

    # ------------------------------------------------------------------ #
    # Servidor RVC persistente
    # ------------------------------------------------------------------ #

    def _iniciar_servidor_rvc(self) -> None:
        """
        Arranca core/rvc_servidor.py como subprocess de larga vida, UNA
        sola vez, y espera a que confirme que ya cargo el modelo. Si
        algo falta (venv, modelo), sigue sin RVC en vez de trabar el
        arranque de todo el programa.
        """
        if not config.RVC_MODEL_PATH.is_file():
            print(f"[Miku-Voz] RVC activado pero no se encontro el modelo: {config.RVC_MODEL_PATH}")
            return

        if not config.RVC_VENV_PYTHON.is_file():
            print(f"[Miku-Voz] RVC activado pero no existe el venv: {config.RVC_VENV_PYTHON}")
            print("[Miku-Voz] Corre: py -3.10 -m venv rvc_env  (y luego instala rvc-python ahi).")
            return

        comando = [
            str(config.RVC_VENV_PYTHON),
            str(config.RVC_SERVIDOR_PATH),
            "--modelo", str(config.RVC_MODEL_PATH),
            "--indice", str(config.RVC_INDEX_PATH) if config.RVC_INDEX_PATH.is_file() else "",
            "--metodo-pitch", config.RVC_METODO_PITCH,
            "--dispositivo", config.RVC_DISPOSITIVO,
        ]

        print("[Miku-Voz] Arrancando el servidor RVC persistente (carga el modelo una sola vez)...")
        try:
            self._proceso_rvc = subprocess.Popen(
                comando,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except Exception as error:
            print(f"[Miku-Voz] No se pudo arrancar el servidor RVC: {error}")
            self._proceso_rvc = None
            return

        estado = self._leer_respuesta_protocolo()
        if estado is None or not estado.get("listo"):
            detalle = estado.get("error") if estado else "no respondio nada por stdout"
            error_stderr = self._proceso_rvc.stderr.read() if self._proceso_rvc.stderr else ""
            print(f"[Miku-Voz] El servidor RVC fallo al cargar el modelo: {detalle}")
            if error_stderr:
                print(f"[Miku-Voz] Detalle (stderr): {error_stderr[-1500:]}")
            self._proceso_rvc = None
            return

        print("[Miku-Voz] Servidor RVC listo -- las siguientes frases ya no van a recargar el modelo.")

    def _leer_respuesta_protocolo(self, max_lineas: int = 500) -> dict | None:
        """
        Lee lineas de stdout del servidor RVC hasta encontrar una que
        empiece con el marcador del protocolo (ver nota junto a
        _MARCADOR_PROTOCOLO_RVC) y la devuelve ya parseada. Lineas de
        ruido de las librerias (prints/logs sueltos de rvc_python,
        fairseq, torch) se ignoran. Devuelve None si el proceso se cerro
        o si no aparecio ninguna linea del protocolo en max_lineas.
        """
        for _ in range(max_lineas):
            linea = self._proceso_rvc.stdout.readline()
            if not linea:
                return None  # el proceso cerro stdout (se murio)
            if linea.startswith(_MARCADOR_PROTOCOLO_RVC):
                try:
                    return json.loads(linea[len(_MARCADOR_PROTOCOLO_RVC):])
                except json.JSONDecodeError:
                    return None
        return None

    def dormir_rvc(self) -> bool:
        """
        Modo juego: apaga el servidor RVC por completo (no solo
        descarga el modelo adentro) para liberar el 100% de su VRAM.
        despertar_rvc() lo vuelve a lanzar desde cero cuando haga falta.

        Se probo primero mandarle un comando para que descargara el
        modelo sin matar el proceso (mas rapido para despertar despues),
        pero recargarlo con rvc.load_model() sobre la misma instancia
        dispara un subprocess interno de alguna libreria (fairseq/numba)
        que en Windows a veces se lanza con el interprete de Python
        equivocado y se cuelga, trabando todo el servidor. Matar y
        relanzar el proceso completo es mas lento pero usa el mismo
        camino de arranque ya probado (_iniciar_servidor_rvc), sin ese
        riesgo.
        """
        if not self._proceso_rvc:
            return False

        print("[Miku-Voz] Apagando el servidor RVC (modo juego)...")
        self._apagar_proceso_rvc()
        print("[Miku-Voz] RVC apagado, VRAM liberada.")
        return True

    def despertar_rvc(self) -> bool:
        """Modo juego: vuelve a lanzar el servidor RVC desde cero."""
        if self._proceso_rvc:
            return True  # ya esta corriendo, no hay nada que hacer

        self._iniciar_servidor_rvc()
        return self._proceso_rvc is not None

    def _apagar_proceso_rvc(self) -> None:
        """
        Apaga self._proceso_rvc si esta corriendo. Intenta el apagado
        prolijo ("_salir" por stdin) y despues, sin importar si eso
        funciono o no, mata el arbol de procesos completo con
        taskkill /T -- por si el servidor llego a dejar algun proceso
        hijo huerfano (ver nota en dormir_rvc).
        """
        if not self._proceso_rvc:
            return

        proceso = self._proceso_rvc
        if proceso.poll() is None:
            try:
                proceso.stdin.write(json.dumps({"_salir": True}) + "\n")
                proceso.stdin.flush()
                proceso.wait(timeout=5)
            except Exception:
                pass

        if proceso.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proceso.pid)],
                    capture_output=True,
                )
            else:
                proceso.kill()

        self._proceso_rvc = None

    def cerrar(self) -> None:
        """Apaga el servidor RVC persistente. Llamar al salir del programa."""
        self._apagar_proceso_rvc()

    @staticmethod
    def _limpiar_para_voz(texto: str) -> str:
        """
        Quita marcado de Markdown (**negritas**, *, #, etc.) para que no
        se lea en voz alta. Tambien colapsa puntos suspensivos ("...' o
        "…") a un punto simple -- Edge-TTS los lee con una pausa larga
        que sonaba como un silencio muerto a mitad de frase.
        """
        texto = re.sub(r"[*_`#]+", "", texto)
        texto = re.sub(r"\.{3,}|…", ".", texto)
        return re.sub(r"\s+", " ", texto).strip()

    def _generar_audio(self, texto: str, ruta_salida: Path, velocidad: float | None = None) -> None:
        """Sintetiza texto con Kokoro (GPU, local) y lo escribe como WAV mono 16-bit."""
        audio, sample_rate = self._kokoro.create(
            texto, voice=self.voz, speed=velocidad if velocidad is not None else self.velocidad, lang=self.idioma
        )
        # Kokoro devuelve float32 en [-1, 1]; WAV de 16 bits necesita
        # enteros con signo -- el "* 32767" + clip evita overflow si
        # algun pico se pasa por poco de 1.0.
        muestras = np.clip(audio, -1.0, 1.0)
        muestras_int16 = (muestras * 32767).astype(np.int16)
        with wave.open(str(ruta_salida), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(muestras_int16.tobytes())

    def _convertir_con_rvc(self, ruta_audio_original: Path) -> Path | None:
        """
        Le pide al servidor RVC persistente que convierta el audio,
        mandandole la peticion por stdin y esperando la respuesta por
        stdout (protocolo JSON por linea, ver core/rvc_servidor.py).

        Si el servidor no esta corriendo o la conversion falla, devuelve
        None y quien llama sigue con la voz de Edge-TTS sin convertir en
        vez de trabar toda la respuesta.
        """
        if not self._proceso_rvc or self._proceso_rvc.poll() is not None:
            return None

        ruta_salida = ruta_audio_original.with_name(ruta_audio_original.stem + "_rvc.wav")
        peticion = {
            "entrada": str(ruta_audio_original),
            "salida": str(ruta_salida),
            "semitonos": config.RVC_TRANSPOSICION_SEMITONOS,
        }

        try:
            self._proceso_rvc.stdin.write(json.dumps(peticion) + "\n")
            self._proceso_rvc.stdin.flush()
            respuesta = self._leer_respuesta_protocolo()
        except Exception as error:
            print(f"[Miku-Voz] No se pudo hablar con el servidor RVC: {error}")
            return None

        if respuesta is None:
            print("[Miku-Voz] El servidor RVC se cayo, sigo con la voz de Edge-TTS sin convertir.")
            self._proceso_rvc = None
            return None

        if not respuesta.get("ok") or not ruta_salida.is_file():
            print(f"[Miku-Voz] La conversion RVC fallo: {str(respuesta.get('error', '(sin detalle)'))[:500]}")
            return None

        return ruta_salida

    def preparar_audio(self, texto: str, velocidad: float | None = None) -> Path | None:
        """
        Genera el audio de una frase con Kokoro y, si aplica, lo
        convierte con RVC -- pero NO lo reproduce. Separado de
        reproducir_audio() para que quien llama pueda preparar la
        SIGUIENTE frase en un hilo aparte mientras la actual todavia
        esta sonando (ver main.py, _hablar_oraciones_en_cola), en vez
        de dejar un silencio entre frase y frase esperando a Kokoro/RVC.

        velocidad, si se pasa, sobreescribe self.velocidad solo para
        esta frase (ver main.py, _velocidad_para -- frases largas/
        informativas se hablan mas lento que la conversacion casual).

        Devuelve la ruta del audio final (ya convertido si USE_RVC
        aplico), o None si el texto quedo vacio tras limpiarlo o algo
        fallo generandolo.
        """
        texto = self._limpiar_para_voz(texto)
        if not texto:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as archivo_temporal:
            ruta_audio_original = Path(archivo_temporal.name)

        try:
            t0 = time.monotonic()
            self._generar_audio(texto, ruta_audio_original, velocidad)
            t1 = time.monotonic()
            print(f"[Miku-Voz] Kokoro: {t1 - t0:.2f}s")

            ruta_audio_final = ruta_audio_original
            if config.USE_RVC:
                ruta_audio_rvc = self._convertir_con_rvc(ruta_audio_original)
                if ruta_audio_rvc:
                    ruta_audio_final = ruta_audio_rvc
                    ruta_audio_original.unlink(missing_ok=True)
                print(f"[Miku-Voz] RVC: {time.monotonic() - t1:.2f}s")

            return ruta_audio_final
        except Exception as error:
            print(f"[Miku-Voz] No se pudo generar el audio: {error}")
            ruta_audio_original.unlink(missing_ok=True)
            return None

    def reproducir_audio(self, ruta_audio: Path, borrar_al_terminar: bool = True) -> None:
        """
        Reproduce un archivo de audio ya generado (por preparar_audio) y
        manda las senales de lip-sync mientras suena. Bloquea hasta que
        termina, para no pisar una frase con la siguiente.

        borrar_al_terminar=False se usa para las muletillas cacheadas
        (ver reproducir_muletilla_aleatoria), que se reutilizan en
        turnos futuros y no deben borrarse despues de sonar una vez.

        Toma _lock_reproduccion mientras dura la reproduccion: si una
        muletilla y la primera oracion real de la respuesta quedaran
        listas casi al mismo tiempo, esto evita que ambas intenten usar
        pygame.mixer.music a la vez -- la segunda simplemente espera a
        que la primera termine.
        """
        with self._lock_reproduccion:
            try:
                pygame.mixer.music.load(str(ruta_audio))
                pygame.mixer.music.play()

                if self.puente_ws:
                    self.puente_ws.notificar_inicio()

                nivel_boca = 0.0
                while pygame.mixer.music.get_busy():
                    if self._evento_detener_habla.is_set():
                        pygame.mixer.music.stop()
                        break
                    if self.puente_ws:
                        # Paseo aleatorio acotado: no es la amplitud real del
                        # audio (no lo decodificamos), pero da un movimiento
                        # de boca que se ve natural mientras dura el habla.
                        nivel_boca += random.uniform(-0.35, 0.35)
                        nivel_boca = max(0.15, min(1.0, nivel_boca))
                        self.puente_ws.notificar_nivel(nivel_boca)
                    time.sleep(0.05)
            except Exception as error:
                print(f"[Miku-Voz] No se pudo reproducir el audio: {error}")
            finally:
                if self.puente_ws:
                    self.puente_ws.notificar_fin()
                if pygame.mixer.get_init() and hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()
                if borrar_al_terminar:
                    ruta_audio.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Interrupcion ("Miku silencio"/"Miku callate" a mitad de una respuesta)
    # ------------------------------------------------------------------ #

    def detener_habla(self) -> None:
        """
        Corta de inmediato lo que este sonando ahora mismo. Para de
        verdad al toque (no espera al siguiente chequeo del loop de
        reproducir_audio): pygame.mixer.music.stop() hace que
        get_busy() pase a False casi al instante.
        """
        self._evento_detener_habla.set()
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def limpiar_interrupcion(self) -> None:
        """Se llama al arrancar CADA turno nuevo, para que una interrupcion anterior no siga afectando al siguiente."""
        self._evento_detener_habla.clear()

    def hablando_interrumpido(self) -> bool:
        return self._evento_detener_habla.is_set()

    # ------------------------------------------------------------------ #
    # Cache generico de frases cortas (muletillas, activacion) -- se
    # generan una sola vez (Edge-TTS + RVC) y se reproducen despues
    # directo desde disco, casi al instante.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ruta_cache_frase(carpeta: Path, indice: int) -> Path | None:
        coincidencias = list(carpeta.glob(f"frase_{indice}.*"))
        return coincidencias[0] if coincidencias else None

    def _preparar_cache_frases(self, frases: tuple[str, ...], carpeta: Path, etiqueta: str) -> None:
        carpeta.mkdir(parents=True, exist_ok=True)
        faltantes = [i for i in range(len(frases)) if self._ruta_cache_frase(carpeta, i) is None]
        if not faltantes:
            return

        print(f"[Miku-Voz] Generando {len(faltantes)} frase(s) de {etiqueta} para el cache...")
        for indice in faltantes:
            ruta_generada = self.preparar_audio(frases[indice])
            if ruta_generada:
                ruta_cache = carpeta / f"frase_{indice}{ruta_generada.suffix}"
                shutil.copyfile(ruta_generada, ruta_cache)
                ruta_generada.unlink(missing_ok=True)
            else:
                print(f"[Miku-Voz] No se pudo generar la frase de {etiqueta}: {frases[indice]!r}")
        print(f"[Miku-Voz] Frases de {etiqueta} listas.")

    def _reproducir_frase_cacheada_aleatoria(self, frases: tuple[str, ...], carpeta: Path) -> None:
        indices_disponibles = [i for i in range(len(frases)) if self._ruta_cache_frase(carpeta, i) is not None]
        if not indices_disponibles:
            return

        ruta = self._ruta_cache_frase(carpeta, random.choice(indices_disponibles))
        self.reproducir_audio(ruta, borrar_al_terminar=False)

    # ------------------------------------------------------------------ #
    # Muletillas de respuesta instantanea (opcional, ver config.HABILITAR_MULETILLAS)
    # ------------------------------------------------------------------ #

    def preparar_muletillas(self) -> None:
        if not config.HABILITAR_MULETILLAS:
            return
        self._preparar_cache_frases(config.MULETILLAS, config.MULETILLAS_DIR, "muletilla")

    def reproducir_muletilla_aleatoria(self) -> None:
        if not config.HABILITAR_MULETILLAS:
            return
        self._reproducir_frase_cacheada_aleatoria(config.MULETILLAS, config.MULETILLAS_DIR)

    # ------------------------------------------------------------------ #
    # Frase de activacion instantanea ("Oye Miku" -> "¿Que necesitas?")
    # ------------------------------------------------------------------ #

    def preparar_frases_activacion(self) -> None:
        self._preparar_cache_frases(config.FRASES_ACTIVACION, config.FRASES_ACTIVACION_DIR, "activacion")

    def reproducir_frase_activacion_aleatoria(self) -> None:
        """
        Reproduce, casi al instante, una confirmacion corta ("¿Que
        necesitas?", "¿Si?") apenas se detecta la palabra clave -- antes
        de que exista ninguna respuesta real que preparar. Se llama en
        paralelo con la animacion de aparicion del avatar (ver main.py).
        """
        self._reproducir_frase_cacheada_aleatoria(config.FRASES_ACTIVACION, config.FRASES_ACTIVACION_DIR)

    def hablar(self, texto: str) -> None:
        """
        Atajo que genera y reproduce una frase en un solo paso,
        bloqueando hasta que termina. Para frases sueltas (ej. las
        confirmaciones de "Miku duerme"/"despierta") donde no hace
        falta solapar la preparacion de una siguiente frase.
        """
        ruta_audio = self.preparar_audio(texto)
        if ruta_audio:
            self.reproducir_audio(ruta_audio)


if __name__ == "__main__":
    voz = MikuVoice()
    print(f"[Miku-Voz] Probando la voz '{voz.voz}'...")
    voz.hablar("Hola, soy Miku. Esta es una prueba de mi voz.")
    voz.cerrar()
    print("[Miku-Voz] Listo.")
