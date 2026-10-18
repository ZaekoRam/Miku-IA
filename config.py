"""
Configuracion central de Miku IA.

Aqui vive la ruta al Vault de Miku y los parametros del modulo de voz
(STT). Si algo se mueve de carpeta, este es el unico archivo que hay
que tocar.

Por diseno (seguridad de datos), el codigo de Miku solo tiene acceso de
lectura/escritura a su propio Vault. El Vault Principal del usuario
("Obsidian Vault", con sus proyectos) queda fuera del alcance del
codigo: el usuario lo documenta manualmente, incluyendo el avance de
este proyecto en Proyectos/Miku IA/.
"""

from pathlib import Path

# --- Vault de Miku ------------------------------------------------------------

# Vault de memoria/diario de Miku (perfil, diario, prompts de personalidad)
VAULT_MIKU_IA = Path(r"C:\Users\ramir\OneDrive\Documents\MIKU IA")

# Subcarpetas dentro del vault de Miku
DIARIO_DIR = VAULT_MIKU_IA / "01_Diario"
PERFIL_USUARIO_DIR = VAULT_MIKU_IA / "02_Perfil_Usuario"
SYSTEM_PROMPTS_DIR = VAULT_MIKU_IA / "04_System_Prompts"


def verificar_rutas_vaults(crear_subcarpetas: bool = True) -> bool:
    """
    Verifica que el Vault de Miku exista y, opcionalmente, crea sus
    subcarpetas de memoria si todavia no existen.

    El Vault en si (VAULT_MIKU_IA) debe existir de antemano: es el Vault
    de Obsidian del usuario y no se crea automaticamente para evitar
    generar un Vault "fantasma" en un lugar equivocado.

    Devuelve True si el Vault raiz existe, False si no.
    """
    if not VAULT_MIKU_IA.is_dir():
        print(f"[ERROR] Vault de Miku no encontrado en: {VAULT_MIKU_IA}")
        return False

    print(f"[OK] Vault de Miku (memoria): {VAULT_MIKU_IA}")

    if crear_subcarpetas:
        for carpeta in (DIARIO_DIR, PERFIL_USUARIO_DIR, SYSTEM_PROMPTS_DIR):
            carpeta.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Estructura de memoria verificada en: {VAULT_MIKU_IA}")

    return True


# --- Modulo de voz (STT) -----------------------------------------------------

# Palabra clave que activa a Miku, para mostrar en los mensajes de la consola.
PALABRA_CLAVE = "oye miku"

# Variantes que activan a Miku al detectarlas. El modelo "tiny" (rapido
# pero menos preciso) suele transcribir mal "miku" -> "mico", o pegar
# "oye"+"miku" sin espacio; cualquiera de estas variantes, encontrada
# como substring del texto ya limpio (minusculas, sin acentos ni signos
# de puntuacion), activa el listener.
VARIANTES_PALABRA_CLAVE = (
    "oye miku",
    "oye mico",
    "oyemiku",
    "oyemico",
    "miku",
    "mico",
)

# Un solo modelo de Whisper para todo (deteccion de palabra clave y
# transcripcion del comando). Antes se usaban dos modelos distintos
# ("tiny" para detectar + "small" para el comando) cargados a la vez en
# VRAM; con uno solo hay mas VRAM libre y menos latencia por turno.
# Subido de "small" a "medium": probado en vivo que "small" seguia
# alucinando frases (ej. "miku" transcrito como "me cago me cago").
# "medium" tarda un poco mas por transcripcion, pero con la GPU sobrada
# de VRAM vale la pena el canje por precision real.
MODELO_WHISPER = "medium"

# "cuda" para usar la GPU (RTX) via CUDA, "cpu" como respaldo
DISPOSITIVO_WHISPER = "cuda"

# float16 aprovecha mejor la GPU; usar "int8" si hay poca VRAM o falla CUDA
TIPO_COMPUTO_WHISPER = "float16"

IDIOMA = "es"

# --- Audio --------------------------------------------------------------------

# Indice del dispositivo de entrada a usar (ver la lista que imprime
# stt_listener.py al arrancar). None = usar el microfono predeterminado
# de Windows. Si el predeterminado resulta ser el equivocado (ej. la
# webcam en vez de la diadema), pon aqui el numero de indice correcto.
DISPOSITIVO_ENTRADA = None

SAMPLE_RATE = 16000          # faster-whisper espera 16 kHz
CANALES = 1
DURACION_BLOQUE_SEG = 0.5    # tamano de cada bloque leido del microfono
VENTANA_WAKE_SEG = 2.5       # ventana de audio analizada para detectar la palabra clave

# Por debajo de esto se considera silencio. Bajado de 0.01 (-40dB) a
# 0.003 (-50dB) y despues a 0.0022 (-53dB) tras probar en vivo: seguia
# haciendo falta hablar fuerte para activar "oye miku". Todavia por
# encima del piso de ruido ambiente (~-60dB) para no disparar con nada.
UMBRAL_SILENCIO_RMS = 0.0022

DURACION_MAX_COMANDO_SEG = 12

# Filtro de confianza del COMANDO ya grabado (ver MikuListener._transcribir
# y _es_transcripcion_confiable): ruido ambiente o habla poco clara se
# transcribia igual como frases con sentido gramatical pero sin sentido
# real (confirmado con ejemplos reales del diario, ej. "¡Suchis! ¡Suchis!"),
# y esas terminaban mandandose igual al LLM. avg_logprob mas cerca de 0 =
# mas seguro; no_speech_prob mas alto = mas probable que sea silencio/ruido,
# no habla real. Si cualquiera de los dos falla, se descarta la
# transcripcion entera (se trata como si el usuario no hubiera dicho nada).
UMBRAL_LOGPROB_STT = -0.9
UMBRAL_NO_SPEECH_STT = 0.6

# Cuanto silencio seguido hace falta para dar por terminado el comando y
# cortar la grabacion, UNA VEZ que el usuario ya empezo a hablar.
# Subido de 0.6 -> 1.0 -> 1.5 -> 2.0s tras varias rondas de prueba en
# vivo: la grabacion seguia cortando a mitad de frase con cualquier
# pausa normal al hablar (thinking pause, respirar, etc.). El limite
# duro de DURACION_MAX_COMANDO_SEG sigue ahi por si alguien se queda
# callado a mitad de una idea.
SILENCIO_PARA_CORTAR_SEG = 2.0

# Cuanto se espera a que el usuario EMPIECE a hablar antes de rendirse
# y devolver un comando vacio. Es un umbral aparte (mas largo) de
# SILENCIO_PARA_CORTAR_SEG: en el "modo conversacion" (ver main.py), el
# usuario suele tomarse un segundo o dos para pensar la siguiente
# pregunta antes de arrancar a hablar -- con un solo umbral corto, esa
# pausa para pensar cortaba la grabacion vacia y cerraba la conversacion
# sola sin que el usuario dijera nada.
ESPERA_INICIO_HABLA_SEG = 6.0

# --- Modulo Cognitivo: LLM local en el mismo proceso (llama-cpp-python) --------

# El chat en vivo ya NO pasa por Ollama (ver core/llm_client.py): corre
# en el mismo proceso con llama-cpp-python, cargando este .gguf directo
# en GPU. Si no existe, se descarga solo la primera vez.
# Subido de 1.5B a 7B (Q4_K_M, ~4.4GB en VRAM): el 1.5B repetia
# parrafos, alucinaba datos y no respetaba limites de longitud del
# System Prompt -- sintomas tipicos de un modelo chico, no bugs de
# codigo. Con 16GB de VRAM y ~10-11GB en uso total (Whisper+Kokoro+RVC+
# este LLM), sigue quedando margen. El .gguf de Qwen (repo oficial)
# viene partido en varios archivos para 7B Q4_K_M -- se uso el quant de
# bartowski (mismos pesos, un solo archivo) para no complicar la
# descarga automatica de un solo .gguf.
LLM_MODELOS_DIR = Path(__file__).resolve().parent / "models" / "llm"
LLM_GGUF_NOMBRE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LLM_GGUF_PATH = LLM_MODELOS_DIR / LLM_GGUF_NOMBRE
LLM_GGUF_URL_DESCARGA = (
    "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/"
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
)

LLM_N_GPU_LAYERS = -1  # -1 = todas las capas en GPU
LLM_N_CTX = 8192       # de sobra para Personalidad.md + Perfil/diario + historial + respuesta

# Ollama sigue instalado y corriendo SOLO para el analisis de memoria en
# segundo plano (extraer_y_guardar_hechos_clave en obsidian_manager.py,
# modelo MODELO_MEMORIA) -- esa tarea no es sensible a la latencia (corre
# despues de responderle al usuario, en un hilo aparte) asi que no vale
# la pena migrarla; el chat en vivo es lo unico que se movio a local.
OLLAMA_HOST = "http://localhost:11434"

# Ruta al ejecutable de Ollama para pararlo por CLI en modo juego (ver
# main.py, _detener_modelo_ollama_cli: "ollama stop <modelo>"). Hace
# falta la ruta completa porque la carpeta de instalacion de Ollama no
# queda en el PATH que hereda este proceso de Python (aunque "ollama"
# si funcione en una terminal interactiva nueva). Si el ejecutable no
# esta aqui, main.py cae de vuelta a confiar en el PATH.
OLLAMA_EXE_PATH = Path(r"C:\Users\ramir\AppData\Local\Programs\Ollama\ollama.exe")

# Modelo usado SOLO para el analisis de memoria en segundo plano
# (extraer_y_guardar_hechos_clave en obsidian_manager.py). Corre
# despues de responderle al usuario y no bloquea nada, asi que no hay
# motivo para sacrificarle precision por velocidad: en pruebas,
# llama3.2:1b (el modelo chico que se usaba antes para el chat en vivo)
# se equivocaba clasificando hechos incluso con ejemplos en el prompt;
# qwen2.5:7b es mas confiable para esta tarea puntual.
MODELO_MEMORIA = "qwen2.5:7b"

# Cuanto tiempo se queda el modelo cargado en VRAM en Ollama despues de
# la ultima peticion, antes de descargarse solo. "10m" evita que la
# siguiente frase pague de nuevo el costo de cargar el modelo si hay
# unos minutos de silencio entre turnos.
OLLAMA_KEEP_ALIVE = "10m"

# temperature mas baja = respuestas mas directas y predecibles, menos
# se desvia divagando. Subida de 0.3 a 0.5 tras notar en pruebas reales
# que con 0.3 el modelo repetia respuestas casi identicas (a veces
# textuales) para preguntas parecidas en turnos distintos -- 0.5 da algo
# mas de variedad sin volverse incoherente.
TEMPERATURA_RESPUESTA = 0.5

# Limite de tokens que el LLM genera por respuesta. Con 150 las
# respuestas conversacionales (1-2 oraciones, que es lo que pide el
# System Prompt) sobraban de margen, pero cualquier pregunta
# informativa/tecnica (ej. "que es el manifiesto agil", codigo) se
# cortaba a la mitad de una idea -- el System Prompt ya le pide a Miku
# que sea breve en conversacion normal y resumida (no relleno) en
# respuestas informativas, asi que este limite ahora es solo un techo
# de seguridad para que una respuesta tecnica larga no se quede
# truncada, no un objetivo de longitud.
MAX_TOKENS_RESPUESTA = 300  # subido de 220: Qwen2.5-7B sigue mejor el limite de "4 puntos/80-100 palabras" del System Prompt que el 1.5B anterior, asi que se le da un poco mas de margen antes de cortar en vez de necesitar un techo tan ajustado

# Cuantos mensajes (usuario + Miku, sumados) se mandan como historial en
# cada llamada a Ollama, para no perder el hilo de la conversacion. 4 =
# ultimos 2 intercambios -- un historial mas chico procesa mas rapido en
# cada turno (menos contexto que Ollama tiene que re-leer).
MAX_MENSAJES_HISTORIAL = 4

# Cuantos archivos recientes de 01_Diario/ se le dan de contexto a Miku
# al arrancar (ademas de todo lo que haya en 02_Perfil_Usuario/).
DIAS_DIARIO_CONTEXTO = 3

# --- Modulo TTS (Kokoro local, GPU) ----------------------------------------------

# Kokoro-onnx corre 100% local en GPU (via onnxruntime-gpu, CUDA) --
# reemplaza a Edge-TTS, que necesitaba red. Si el modelo/voces no
# existen, core/tts_speaker.py los descarga solos la primera vez.
KOKORO_DIR = Path(__file__).resolve().parent / "models" / "kokoro"
KOKORO_MODEL_PATH = KOKORO_DIR / "kokoro-v1.0.fp16-gpu.onnx"
KOKORO_VOICES_PATH = KOKORO_DIR / "voices-v1.0.bin"
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16-gpu.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# "ef_dora" es la unica voz femenina en espanol (acento latino) que trae
# Kokoro v1.0 -- calza con el modelo RVC "Latin-Spanish" que ya usamos.
KOKORO_VOZ = "ef_dora"
KOKORO_IDIOMA = "es"
KOKORO_VELOCIDAD = 1.05  # >1.0 = un poco mas rapido que el default

# Frases largas/densas (respuestas informativas) se hablan mas lento que
# la conversacion casual, para que de tiempo a procesar la informacion
# en vez de que se sienta atropellada. Se decide por oracion (ver
# main.py, _velocidad_para): si tiene mas de UMBRAL_PALABRAS_VELOCIDAD_LENTA
# palabras, se usa KOKORO_VELOCIDAD_LENTA en vez de KOKORO_VELOCIDAD.
KOKORO_VELOCIDAD_LENTA = 0.85
UMBRAL_PALABRAS_VELOCIDAD_LENTA = 14

# --- RVC (conversion de la voz de Edge-TTS a la voz de personaje de Miku) --------

# Si esta en False, MikuVoice se queda con la voz "cruda" de Edge-TTS
# (sirve para desactivar RVC sin tocar codigo si algo sale mal).
USE_RVC = True

# Path (no string) y absolutas, como el resto de las rutas de este
# archivo -- si fueran relativas, se romperian en cuanto algo corra el
# programa desde un directorio de trabajo distinto al del proyecto.
RVC_MODEL_PATH = Path(__file__).resolve().parent / "models" / "rvc" / "hatsune-miku-Latin-Spanish.pth"
RVC_INDEX_PATH = Path(__file__).resolve().parent / "models" / "rvc" / "hatsune-miku-Latin-Spanish.index"

# Extractor de pitch (f0). "rmvpe" es el que pide el usuario: mas preciso
# que el "harvest"/"crepe" clasicos, buen equilibrio precision/velocidad.
RVC_METODO_PITCH = "rmvpe"

# Semitonos que se le suman al pitch dentro de RVC, para acercar la voz
# base de Kokoro al tono agudo tipico de Vocaloid/Miku.
RVC_TRANSPOSICION_SEMITONOS = 10

RVC_DISPOSITIVO = "cuda:0"

# Las dependencias de RVC (fairseq, etc.) no instalan en el Python
# principal del proyecto (3.12) -- viven en su propio venv aislado con
# Python 3.10. MikuVoice arranca RVC_SERVIDOR_PATH como un subprocess
# PERSISTENTE ahi (una sola vez, con los modelos ya cargados en
# memoria) en vez de RVC_SCRIPT_PATH (que recargaba todo -- PyTorch,
# fairseq, HuBERT, RMVPE, el modelo -- en cada frase, ~7s de latencia
# por turno). RVC_SCRIPT_PATH se deja para pruebas manuales sueltas.
RVC_VENV_PYTHON = Path(__file__).resolve().parent / "rvc_env" / "Scripts" / "python.exe"
RVC_SCRIPT_PATH = Path(__file__).resolve().parent / "core" / "rvc_convertir.py"
RVC_SERVIDOR_PATH = Path(__file__).resolve().parent / "core" / "rvc_servidor.py"

# --- Muletillas de respuesta instantanea -------------------------------------------

# Desactivadas: con la animacion de aparicion + la frase de activacion
# instantanea (ver FRASES_ACTIVACION abajo), la muletilla de "Mmm..." /
# "A ver..." quedaba de mas -- Miku ya da feedback inmediato al aparecer.
HABILITAR_MULETILLAS = False

# Frases cortas que Miku diria apenas termina de escuchar al usuario,
# mientras Ollama + Edge-TTS + RVC generan la respuesta real en segundo
# plano. Solo se usan si HABILITAR_MULETILLAS = True.
MULETILLAS = ("Mmm...", "¡Claro!", "A ver...", "Okey...", "Veamos...")

# Carpeta donde se cachean las muletillas ya convertidas (Edge-TTS + RVC)
# la primera vez que arranca el programa. Sin cache, generarlas tardaria
# lo mismo que cualquier otra frase (~1-1.5s) y no serviria de nada como
# respuesta "instantanea"; con el archivo ya listo en disco, reproducirla
# es casi inmediato (<0.3s).
MULETILLAS_DIR = Path(__file__).resolve().parent / "cache" / "muletillas"

# --- Frase de activacion instantanea ("Oye Miku") ------------------------------------

# Apenas se detecta la palabra clave (antes de grabar el comando), Miku
# reproduce una de estas frases desde el cache -- confirmacion audible
# casi inmediata de que "escucho", mientras el avatar hace su animacion
# de aparicion en la ventana.
FRASES_ACTIVACION = ("¿Qué necesitas?", "¿Sí?")

FRASES_ACTIVACION_DIR = Path(__file__).resolve().parent / "cache" / "activacion"

# --- Puente WebSocket (avatar 3D en gui_3d/) -------------------------------------

WEBSOCKET_HOST = "localhost"
WEBSOCKET_PUERTO = 8765

# Carpeta de la app de Electron (ventana 3D del avatar), relativa a este archivo.
RUTA_GUI_3D = Path(__file__).resolve().parent / "gui_3d"


if __name__ == "__main__":
    verificar_rutas_vaults()
