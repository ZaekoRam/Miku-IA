"""
Modulo Cognitivo: el cerebro de Miku, con un LLM local EN EL MISMO
PROCESO (llama-cpp-python, GPU via CUDA) -- ya no depende de Ollama por
HTTP para el chat en vivo. El modelo (un .gguf, ver config.LLM_GGUF_PATH)
se carga UNA sola vez al arrancar con todas sus capas en GPU
(n_gpu_layers=-1) y se reusa turno a turno: llama.cpp cachea el contexto
ya procesado, asi que si el turno nuevo comparte el mismo prefijo que el
anterior (mismo System Prompt + historial), no hace falta re-procesar
todo desde cero -- eso era lo que hacia que Ollama por HTTP tardara
~2.1-2.4s fijos en cada turno (reprocesaba el System Prompt completo,
~7000 caracteres con Personalidad.md, en cada llamada).

Nota: Ollama sigue usandose para el analisis de memoria en segundo plano
(ver core/obsidian_manager.py, config.MODELO_MEMORIA) -- esa tarea no es
sensible a la latencia, asi que no valia la pena migrarla tambien.

Con la personalidad de Miku como System Prompt (leida de
04_System_Prompts/Personalidad.md) mas el contexto del perfil del
usuario y el diario reciente, leidos UNA SOLA VEZ al arrancar desde su
Vault (MIKU IA/02_Perfil_Usuario/ y 01_Diario/) -- no en cada turno, para
no sumar latencia releyendo archivos que no cambiaron.

Tambien mantiene un historial corto de la conversacion (los ultimos
config.MAX_MENSAJES_HISTORIAL mensajes) para que el modelo no pierda el
hilo entre turnos.

Las respuestas se piden en streaming, por oracion completa (ver
AcumuladorChunks): se probo cortar en chunks mas chicos (3-5 palabras)
para bajar mas la latencia, pero sonaba entrecortado/robotico -- Kokoro
pierde la entonacion natural cuando sintetiza pedazos muy chicos sin el
contexto de la oracion completa.
"""

from __future__ import annotations

import gc
import random
import re
from typing import Callable

import requests
from llama_cpp import Llama

import config


SYSTEM_PROMPT_BASE = (
    "Eres Hatsune Miku, la compañera virtual de Carlo. Habla en español latino "
    "neutro (nunca 'vos'/'decime', usa 'tú'/'dime'). En conversacion normal "
    "responde alegre, amigable y directa en 1-2 oraciones breves. Si Carlo te "
    "pide un saludo o sugerencia, hazlo inmediatamente sin disclaimers de IA. Si "
    "te pide informacion o codigo (ej. 'que es X', 'como hago Y'), responde "
    "completo sin cortar la idea a la mitad, pero con maximo 4 puntos o unas "
    "80-100 palabras (codigo exceptuado) -- nunca repitas la misma idea con "
    "otras palabras solo para alargar. Si Carlo te presenta a alguien nuevo (ej. 'te "
    "presento a Alan') o alguien se te presenta directamente, muestra "
    "curiosidad genuina con UNA sola pregunta corta sobre como la conoce Carlo "
    "o algun gusto de esa persona, sin interrogatorios -- eso SOLO aplica ahi, "
    "en el resto de la conversacion no hagas preguntas de seguimiento por tu "
    "cuenta ni ofrezcas actividades/contenido extra sin que te lo pidan. "
    "JAMAS inventes links, URLs ni nombres de sitios web -- no tienes internet "
    "real, cualquier link que digas es falso."
)

# Frases-disclaimer que el modelo repetia en bucle casi textuales turno
# tras turno (ej. "Lo siento, pero como Miku, no tengo un diario
# personal...", "No tengo acceso a la hora actual. Mi funcion es ser una
# asistente virtual..."). Si una respuesta asi se guarda en el
# historial, el modelo tiende a imitarla (o repetirla textual) en el
# siguiente turno tambien -- ver mas abajo, donde se filtra del
# historial en vez de dejar que se auto-refuerce. Se usa .search() (no
# .match()) porque a veces el disclaimer no arranca la respuesta.
_PATRON_RESPUESTA_EN_BUCLE = re.compile(
    r"lo\s+siento,?\s+pero\s+como\s+miku"
    r"|no\s+tengo\s+(un\s+)?diario\s+personal"
    r"|mi\s+funci[oó]n\s+es\s+ser\s+una\s+asistente\s+virtual"
    r"|no\s+tengo\s+acceso\s+a\s+la\s+hora"
    # El saludo generico "Hola Carlo, como estas / en que puedo ayudarte"
    # -- confirmado en el diario que el modelo lo repetia CASI TEXTUAL
    # turno tras turno, incluso para preguntas que no tenian nada que ver
    # con un saludo, una vez que ya habia quedado guardado en el
    # historial un par de veces. Anclado al INICIO de la respuesta (^)
    # para no descartar una respuesta real que solo mencione "como estas"
    # de pasada mas adelante.
    r"|^\s*¡?hola,?\s+carlo!?.{0,60}"
    r"(c[oó]mo\s+est[aá]s|en\s+qu[eé]\s+puedo\s+ayudarte|hay\s+algo\s+en\s+lo\s+que\s+pueda\s+ayudarte)",
    re.IGNORECASE,
)


# Separadores "duros": siempre cortan un chunk apenas aparecen.
# Probamos cortar tambien por coma/punto y coma/dos puntos (chunks de
# 3-5 palabras) para bajar la latencia al maximo -- funcionaba para el
# PRIMER chunk, pero la voz sonaba entrecortada/robotica el resto de la
# respuesta: Kokoro sintetiza cada chunk por separado, sin el contexto
# de la oracion completa, asi que pierde la entonacion natural cuando
# los pedazos son demasiado chicos. Con el LLM local ya generando toda
# la respuesta en ~0.2-1.5s, ese medio segundo extra de esperar a que
# cierre una oracion completa no se nota, y suena mucho mas natural.
_SEPARADORES_FUERTES = (".", "!", "?", "\n")
# Si una oracion se alarga demasiado sin ningun separador duro, se corta
# igual (en el espacio mas cercano, para no partir una palabra a la
# mitad) -- evita esperar de mas a un punto que tarda en llegar.
_LONGITUD_MAXIMA_SIN_SEPARADOR = 90


class AcumuladorChunks:
    """
    Junta los fragmentos de texto que van llegando en streaming del LLM
    y libera oraciones completas apenas se detecta un separador duro
    (. ! ? salto de linea) o se supera _LONGITUD_MAXIMA_SIN_SEPARADOR.

    Esto es lo que permite mandar el primer chunk a Kokoro/RVC apenas
    esta listo, sin esperar a que el LLM termine de generar toda la
    respuesta ni siquiera una oracion completa.
    """

    def __init__(self):
        self._buffer = ""

    def agregar(self, fragmento: str) -> list[str]:
        """Agrega un pedazo de texto y devuelve los chunks nuevos que quedaron completos (0, 1 o mas)."""
        self._buffer += fragmento
        chunks = []

        while True:
            indice_corte = self._buscar_corte(self._buffer)
            if indice_corte is None:
                break
            chunk = self._buffer[:indice_corte].strip()
            self._buffer = self._buffer[indice_corte:]
            if chunk:
                chunks.append(chunk)

        return chunks

    def vaciar_resto(self) -> str | None:
        """Al terminar el streaming, devuelve lo que haya quedado sin cerrar (si algo)."""
        resto = self._buffer.strip()
        self._buffer = ""
        return resto or None

    @staticmethod
    def _buscar_corte(texto: str) -> int | None:
        for indice, caracter in enumerate(texto):
            if caracter in _SEPARADORES_FUERTES:
                return indice + 1

        if len(texto) > _LONGITUD_MAXIMA_SIN_SEPARADOR:
            corte = texto.rfind(" ", 0, _LONGITUD_MAXIMA_SIN_SEPARADOR)
            if corte == -1:
                corte = _LONGITUD_MAXIMA_SIN_SEPARADOR
            return corte + 1

        return None


def _descargar_gguf_si_falta() -> None:
    """Descarga config.LLM_GGUF_PATH desde Hugging Face si todavia no existe en disco."""
    if config.LLM_GGUF_PATH.is_file():
        return

    config.LLM_MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Miku-Brain] No se encontro el modelo LLM, descargando {config.LLM_GGUF_NOMBRE} (~4.4GB)...")
    ruta_temporal = config.LLM_GGUF_PATH.with_suffix(config.LLM_GGUF_PATH.suffix + ".part")
    with requests.get(config.LLM_GGUF_URL_DESCARGA, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(ruta_temporal, "wb") as f:
            for bloque in resp.iter_content(chunk_size=1024 * 1024):
                f.write(bloque)
    ruta_temporal.rename(config.LLM_GGUF_PATH)
    print("[Miku-Brain] Modelo LLM descargado.")


class MikuBrain:
    def __init__(self, obsidian=None):
        # Se leen UNA sola vez aqui (no en cada preguntar()): ni la
        # personalidad ni el perfil/diario cambian turno a turno,
        # releerlos solo sumaria latencia sin cambiar el contenido. Para
        # editar la personalidad de Miku alcanza con tocar
        # 04_System_Prompts/Personalidad.md en Obsidian y reiniciar.
        self._personalidad = self._cargar_personalidad(obsidian)
        self._contexto_usuario = self._cargar_contexto_inicial(obsidian)
        self._system_prompt = self._construir_system_prompt(self._personalidad, self._contexto_usuario)

        self._historial: list[dict[str, str]] = []
        self._llm: Llama | None = None
        self._cargar_llm()

    # ------------------------------------------------------------------ #
    # Personalidad + contexto del usuario (una sola vez, al arrancar)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cargar_personalidad(obsidian) -> str:
        """
        04_System_Prompts/Personalidad.md manda si existe y tiene
        contenido -- SYSTEM_PROMPT_BASE queda solo como respaldo si el
        archivo no existe, esta vacio, o no hay ObsidianManager (ej.
        pruebas sueltas de este modulo sin Vault).
        """
        if obsidian is None:
            return SYSTEM_PROMPT_BASE
        try:
            personalidad = obsidian.leer_personalidad()
        except Exception as error:
            print(f"[Miku-Brain] No se pudo leer Personalidad.md: {error}")
            return SYSTEM_PROMPT_BASE
        return personalidad or SYSTEM_PROMPT_BASE

    @staticmethod
    def _cargar_contexto_inicial(obsidian) -> str:
        if obsidian is None:
            return ""
        try:
            return obsidian.leer_perfil_y_diario_reciente()
        except Exception as error:
            print(f"[Miku-Brain] No se pudo leer el perfil/diario de Obsidian: {error}")
            return ""

    @staticmethod
    def _construir_system_prompt(personalidad: str, contexto_usuario: str) -> str:
        if not contexto_usuario:
            return personalidad

        return (
            f"{personalidad}\n\n"
            "--- Contexto sobre Carlo (perfil y diario reciente en Obsidian) ---\n"
            f"{contexto_usuario}"
        )

    # ------------------------------------------------------------------ #
    # Carga/descarga del modelo (llama-cpp-python, GPU, en el proceso)
    # ------------------------------------------------------------------ #

    def _cargar_llm(self) -> None:
        _descargar_gguf_si_falta()

        print(
            f"[Miku-Brain] Cargando {config.LLM_GGUF_PATH.name} "
            f"(n_gpu_layers={config.LLM_N_GPU_LAYERS}, n_ctx={config.LLM_N_CTX})..."
        )
        self._llm = Llama(
            model_path=str(config.LLM_GGUF_PATH),
            n_gpu_layers=config.LLM_N_GPU_LAYERS,
            n_ctx=config.LLM_N_CTX,
            verbose=False,
        )

        # "Calienta" el modelo con una peticion minima: la PRIMERISIMA
        # inferencia de una sesion de llama.cpp en GPU tarda varios
        # segundos de mas (compilacion/cache de kernels CUDA la primera
        # vez que se usan) -- medido en pruebas, ~7.5s la primera vez
        # contra ~250ms de ahi en mas. Sin este calentamiento, esa
        # demora la pagaria la primera pregunta real del usuario.
        for _ in self._llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hola"}], max_tokens=1, stream=True
        ):
            pass
        print("[Miku-Brain] Modelo LLM listo.")

    def verificar_conexion(self) -> bool:
        """Compatibilidad con main.py: True si el modelo esta cargado en GPU."""
        return self._llm is not None

    def descargar_modelo(self) -> None:
        """
        Libera el modelo de la GPU (modo juego). llama-cpp-python no
        tiene un "keep_alive" como Ollama -- para soltar la VRAM de
        verdad hay que cerrar y tirar la instancia entera.
        """
        if self._llm is None:
            return
        self._llm.close()
        self._llm = None
        gc.collect()
        print("[Miku-Brain] Modelo LLM liberado de la GPU.")

    def calentar_modelo(self) -> None:
        """Vuelve a cargar el modelo en GPU (al despertar del modo juego)."""
        if self._llm is not None:
            return
        self._cargar_llm()

    # ------------------------------------------------------------------ #
    # Pregunta
    # ------------------------------------------------------------------ #

    def preguntar(self, texto_usuario: str, on_token: Callable[[str], None] | None = None) -> str:
        """
        Manda texto_usuario al LLM (con el system prompt + historial
        reciente ya armados) y devuelve la respuesta completa en texto
        plano.

        La pide en streaming: si se pasa on_token, se llama con cada
        fragmento de texto conforme va llegando (para imprimirlo en vivo
        en la consola y para AcumuladorChunks, en vez de esperar a que
        termine toda la respuesta).

        Nota sobre hora/fecha e identidad ("quien sos"/"quien soy"):
        main.py intercepta esas preguntas ANTES de llegar aca (ver
        FRASES_HORA/FRASES_FECHA/FRASES_IDENTIDAD) -- un modelo tan
        chico no las contesta bien ni siquiera con la info correcta
        puesta justo antes de la pregunta: probado en vivo que
        alucinaba una fecha cualquiera, y que a "¿sabes quien soy?"
        contestaba generico ("eres tu, el usuario") sin usar ni el
        nombre de Carlo ni su propia identidad, a pesar de tener las dos
        cosas en el System Prompt (Personalidad.md + Perfil.md).
        """
        mensajes = [{"role": "system", "content": self._system_prompt}]
        mensajes.extend(self._historial)
        mensajes.append({"role": "user", "content": texto_usuario})

        fragmentos: list[str] = []
        for chunk in self._llm.create_chat_completion(
            messages=mensajes,
            stream=True,
            max_tokens=config.MAX_TOKENS_RESPUESTA,
            temperature=config.TEMPERATURA_RESPUESTA,
            # Sin "seed" explicito, llama-cpp-python parecia reusar el
            # mismo estado de muestreo entre llamadas -- confirmado en
            # vivo: "hola" y "hola miku!" (prompts casi identicos)
            # devolvian la respuesta EXACTAMENTE igual, dos veces. Una
            # semilla nueva por turno fuerza un muestreo independiente
            # cada vez, incluso para prompts muy parecidos.
            seed=random.randint(0, 2**31 - 1),
            # El modelo de 1.5B, en respuestas largas, caia en bucles de
            # repeticion casi textual -- frequency_penalty/presence_penalty
            # fuertes (0.4) lo arreglaban ahi, pero con Qwen2.5-7B (que
            # ya no repite asi de entrada) esos mismos valores altos
            # sobre una respuesta larga terminaban empujando al modelo a
            # "escapar" del espanol cambiando a chino a mitad de
            # respuesta (probado en vivo) -- son modelos bilingues
            # zh/en/es y penalizar de mas el vocabulario ya usado
            # empuja hacia vocabulario nuevo en OTRO idioma. repeat_penalty
            # (ventana corta de tokens recientes, no todo el historial)
            # no causa ese efecto, se deja mas moderado; frequency/presence
            # se bajan a algo casi simbolico.
            repeat_penalty=1.15,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        ):
            contenido = chunk["choices"][0]["delta"].get("content", "")
            if contenido:
                fragmentos.append(contenido)
                if on_token:
                    on_token(contenido)

        respuesta = "".join(fragmentos).strip()

        if _PATRON_RESPUESTA_EN_BUCLE.search(respuesta):
            # No se guarda en el historial: si quedara ahi, el modelo
            # tiende a imitar su propio disclaimer en el siguiente turno
            # y se queda repitiendolo en bucle.
            print("[Miku-Brain] Respuesta tipo disclaimer detectada, no se guarda en el historial.")
        else:
            self._historial.append({"role": "user", "content": texto_usuario})
            self._historial.append({"role": "assistant", "content": respuesta})
            self._historial = self._historial[-config.MAX_MENSAJES_HISTORIAL:]

        return respuesta


if __name__ == "__main__":
    cerebro = MikuBrain()
    if cerebro.verificar_conexion():
        pregunta = "Hola Miku, ¿quien eres?"
        print(f"\n[Tu] {pregunta}")
        print("[Miku] ", end="", flush=True)
        cerebro.preguntar(pregunta, on_token=lambda t: print(t, end="", flush=True))
        print()
