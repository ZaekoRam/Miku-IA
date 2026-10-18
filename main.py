"""
Punto de entrada de Miku IA.

Conecta los modulos construidos hasta ahora:
  1. Oidos    (core.stt_listener.MikuListener)          -> transcribe lo que se dice
  2. Cerebro  (core.llm_client.MikuBrain)                -> LLM local (llama-cpp-python, GPU), por chunks
  3. Voz      (core.tts_speaker.MikuVoice)               -> Kokoro (GPU) + RVC, convierte cada chunk a audio
  4. Avatar   (core.ws_bridge.PuenteWebSocket + gui_3d/) -> ventana 3D que mueve la boca al hablar
  5. Memoria  (core.obsidian_manager.ObsidianManager)    -> diario + extractor de hechos clave

Dos cosas que vale la pena entender antes de tocar este archivo:

- Streaming por oracion completa: en vez de esperar a que el LLM
  termine TODA la respuesta para recien ahi mandarla a TTS/RVC, se manda
  a hablar cada oracion apenas esta completa (ver AcumuladorChunks en
  core/llm_client.py -- se probo cortar en chunks mas chicos de 3-5
  palabras para bajar mas la latencia, pero sonaba entrecortado). Un
  hilo aparte ("hablador") va sacando oraciones de una cola y
  reproduciendolas una por una mientras el LLM sigue generando las
  siguientes, preparando (Kokoro+RVC) la proxima oracion en paralelo
  mientras la actual todavia suena.

- "Miku duerme" / "modo juego" / "Miku despierta": comandos especiales
  que se detectan ANTES de mandarle nada al LLM (deteccion por texto, no
  por el modelo -- mas rapido y mas confiable para un par de frases
  fijas). Dormida libera la VRAM del LLM local y de RVC para dejarsela a
  otras apps (juegos); despierta la recupera. El AFK (5s sin actividad)
  es distinto: solo oculta el avatar en la ventana, nunca toca VRAM ni
  modelos (ver notificar_respuesta_completa -> renderer.js).
"""

import atexit
import os
import queue
import random
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from datetime import datetime

# La consola de Windows usa cp1252 por defecto, que no puede imprimir
# cualquier caracter (ej. Whisper alucinando un caracter chino desde
# ruido). Sin esto, un solo print() con un caracter asi tumba TODO el
# proceso con un UnicodeEncodeError -- se vio en vivo cerrando la
# ventana a mitad de una transcripcion. errors="replace" cambia el
# caracter problematico por "?" en vez de crashear.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Agregar las DLLs de NVIDIA (cuBLAS/cuDNN/cuFFT/cuRAND/nvJitLink/cudart)
# al PATH de Windows ANTES de importar cualquier libreria que las
# necesite: faster-whisper/ctranslate2 (STT), llama-cpp-python (LLM
# local) y onnxruntime-gpu/kokoro-onnx (TTS local) todas cargan CUDA en
# su propio import y fallan si no encuentran estas DLLs -- no al cargar
# el modelo, sino en la primera inferencia real, que es cuando de verdad
# las usan.
#
# sysconfig.get_paths()["purelib"] devuelve el site-packages del Python
# que esta corriendo AHORA -- sirve igual con venv (<venv>\Lib\site-packages)
# que sin el (<PythonXY>\Lib\site-packages); derivarlo a mano desde
# sys.executable con dirname(dirname(...)) asume un layout de venv y
# apunta a la carpeta equivocada en una instalacion global de Python.
site_packages = sysconfig.get_paths()["purelib"]
nvidia_paths = [
    os.path.join(site_packages, "nvidia", "cublas", "bin"),
    os.path.join(site_packages, "nvidia", "cudnn", "bin"),
    os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
    os.path.join(site_packages, "nvidia", "cufft", "bin"),
    os.path.join(site_packages, "nvidia", "curand", "bin"),
    os.path.join(site_packages, "nvidia", "nvjitlink", "bin"),
]
for path in nvidia_paths:
    if os.path.exists(path):
        os.add_dll_directory(path)
        os.environ["PATH"] = path + os.path.pathsep + os.environ["PATH"]

import config
from core.llm_client import AcumuladorChunks, MikuBrain
from core.obsidian_manager import ObsidianManager
from core.stt_listener import MikuListener
from core.tts_speaker import MikuVoice
from core.utils import normalizar_texto
from core.ws_bridge import PuenteWebSocket

# Frases que activan/desactivan el modo juego. Se comparan ya
# normalizadas (sin acentos, minusculas) como substring del mensaje, asi
# que "Oye Miku, modo juego porfa" tambien dispara.
FRASES_DORMIR = ("miku duerme", "duermete", "modo juego")
FRASES_DESPERTAR = ("miku despierta",)

# Comandos explicitos para cortar el "modo conversacion" (ver mas abajo)
# sin decir nada de vuelta -- "callate" no deberia contestarse hablando.
FRASES_SILENCIO = ("miku callate", "miku cállate", "miku silencio")

# Despedidas cortas: solo cuentan si son (casi) todo el mensaje (ver
# _es_despedida_corta), para no cortar la conversacion en medio de una
# pregunta que de casualidad contenga "gracias" u "ok".
FRASES_DESPEDIDA_CORTAS = (
    "gracias", "muchas gracias", "ok", "okay", "adios", "hasta luego",
    "nos vemos", "eso es todo", "nada mas", "ya esta", "esta bien",
)

FRASES_DESPEDIDA_RESPUESTA = ("¡De nada, Carlo!", "¡Un gusto ayudarte! Nos vemos.", "¡Cuando quieras!")


def _es_despedida_corta(texto_normalizado: str) -> bool:
    """
    True si el mensaje es (casi) puramente una despedida corta.

    El chequeo exacto (limpio == frase) se quedaba corto en la practica:
    variantes tan comunes como "no, gracias" o "ya esta, gracias" no
    matcheaban ninguna entrada de FRASES_DESPEDIDA_CORTAS tal cual, y la
    conversacion se quedaba esperando en vez de cerrarse. Para cubrir
    eso sin arriesgar cortar una pregunta larga que de casualidad
    mencione "gracias" a mitad, se permite tambien un mensaje CORTO
    (maximo 4 palabras) que CONTENGA una despedida completa como
    palabra/frase entera (con espacios de por medio, no como pedazo de
    otra palabra).
    """
    limpio = texto_normalizado.strip(" .,!¡¿?")
    if not limpio:
        return False

    if limpio in FRASES_DESPEDIDA_CORTAS:
        return True

    if any(limpio in (f"{frase} miku", f"miku {frase}") for frase in FRASES_DESPEDIDA_CORTAS):
        return True

    palabras = limpio.split()
    if len(palabras) <= 4:
        acolchado = f" {limpio} "
        if any(f" {frase} " in acolchado for frase in FRASES_DESPEDIDA_CORTAS):
            return True

    return False


# Frase de arranque: se dice una sola vez al terminar de cargar todo (ver
# main(), justo antes de listener.escuchar()). El proposito real no es
# el saludo en si -- es forzar una conversion REAL de RVC durante el
# arranque, no solo cargar el modelo. La primerisima inferencia de RVC
# tarda ~15-20s de mas (compilacion de kernels CUDA la primera vez que
# se usan, confirmado varias veces en vivo); sin esto, esa demora la
# pagaba el primer mensaje real del usuario en vez de pagarla mientras
# arranca el programa.
FRASES_LISTA_PARA_SERVIR = (
    "¡Hola, Carlo! Ya estoy lista para servirte.",
    "¡Ya estoy despierta y lista, Carlo!",
    "¡Hola! Todo cargado, decime 'oye miku' cuando quieras.",
)


# Preguntas de hora/fecha se responden DIRECTO desde Python, sin pasarle
# nada a Ollama: un modelo tan chico (llama3.2:1b) no tiene forma real
# de saber la hora, y probado en vivo, ni siquiera usaba la fecha/hora
# real que se le inyectaba en el prompt -- alucinaba una fecha
# cualquiera ("2023-12-29, 02:00") de forma consistente. Interceptarlo
# aca garantiza que la respuesta sea siempre correcta.
FRASES_HORA = ("que hora es", "que hora tienes", "que horas son", "dime la hora", "me dices la hora")
FRASES_FECHA = ("que dia es hoy", "que dia es", "en que fecha estamos", "que fecha es hoy", "que fecha es", "que dia estamos")

_DIAS_SEMANA = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
_MESES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)


def _respuesta_hora() -> str:
    ahora = datetime.now()
    return random.choice((
        f"Son las {ahora:%H:%M}, Carlo.",
        f"¡Ahorita son las {ahora:%H:%M}!",
        f"El reloj marca las {ahora:%H:%M}.",
    ))


def _respuesta_fecha() -> str:
    ahora = datetime.now()
    dia_semana = _DIAS_SEMANA[ahora.weekday()]
    mes = _MESES[ahora.month - 1]
    return random.choice((
        f"Hoy es {dia_semana} {ahora.day} de {mes}.",
        f"Estamos a {ahora.day} de {mes}, un {dia_semana}.",
    ))


# Preguntas de identidad ("sabes quien soy?"/"sabes quien eres?") tambien
# se responden directo desde Python: probado en vivo que el modelo
# contestaba generico ("eres tu, el usuario") sin usar ni el nombre de
# Carlo ni su propia identidad, a pesar de tener las dos cosas en el
# System Prompt (Personalidad.md + Perfil.md) -- la misma limitacion que
# ya vimos con la hora/fecha, un modelo tan chico no siempre usa bien el
# contexto largo para este tipo de pregunta puntual.
FRASES_IDENTIDAD_MIKU = ("quien eres", "quien sos", "como te llamas", "cual es tu nombre")
FRASES_IDENTIDAD_USUARIO = ("quien soy", "como me llamo", "cual es mi nombre")


def _respuesta_identidad_miku() -> str:
    return random.choice((
        "Soy Hatsune Miku, tu compañera virtual.",
        "Soy Miku -- tu asistente y compañera, Carlo.",
    ))


def _respuesta_identidad_usuario() -> str:
    return random.choice((
        "¡Claro que sí! Sos Carlo, mi usuario y creador.",
        "Sos Carlo, obvio.",
    ))


def _detener_modelo_ollama_cli(nombre_modelo: str) -> None:
    """
    Corre "ollama stop <modelo>" para descargarlo de la GPU al instante
    (modo juego). Se usa el CLI en vez de la API HTTP (keep_alive=0)
    porque es el mismo comando que correrias a mano para confirmar que
    ya no esta cargado -- mas directo para este caso puntual.
    """
    ejecutable = str(config.OLLAMA_EXE_PATH) if config.OLLAMA_EXE_PATH.is_file() else "ollama"
    try:
        subprocess.run([ejecutable, "stop", nombre_modelo], capture_output=True, timeout=10)
        print(f"[Main] Modelo '{nombre_modelo}' detenido (ollama stop).")
    except Exception as error:
        print(f"[Main] No se pudo detener '{nombre_modelo}' con 'ollama stop': {error}")


def lanzar_electron() -> subprocess.Popen | None:
    """
    Lanza la ventana 3D de Miku (gui_3d/, Electron) en segundo plano con
    "npm start". Si gui_3d/ todavia no tiene sus dependencias instaladas
    (node_modules), avisa y el programa sigue sin la GUI en vez de
    truncarse por completo.
    """
    if not (config.RUTA_GUI_3D / "node_modules").is_dir():
        print(f"[Main] {config.RUTA_GUI_3D} no tiene node_modules todavia.")
        print("[Main] Corre 'npm install' ahi para poder ver el avatar. Sigo sin la GUI por ahora.")
        return None

    print("[Main] Abriendo la ventana 3D de Miku (Electron)...")
    # shell=True porque en Windows "npm" es un .cmd, no un ejecutable
    # directo; con shell=True cmd.exe lo resuelve por el PATH.
    return subprocess.Popen("npm start", cwd=config.RUTA_GUI_3D, shell=True)


def cerrar_electron(proceso: subprocess.Popen) -> None:
    """
    "npm start" en Windows corre por debajo de cmd.exe -> node -> electron.exe;
    proceso.terminate() solo mataria a cmd.exe y dejaria a electron.exe
    huerfano. taskkill /T mata todo el arbol de procesos.
    """
    print("[Main] Cerrando la ventana de Miku...")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proceso.pid)],
            capture_output=True,
        )
    else:
        proceso.terminate()


def main() -> None:
    if not config.verificar_rutas_vaults():
        print("Corrige las rutas en config.py antes de continuar.")
        return

    # ------------------------------------------------------------------ #
    # Cierre de emergencia: registrado YA, antes de crear nada -- si no,
    # un Ctrl+C a mitad del arranque (ej. mientras carga el servidor RVC,
    # que puede tardar bastante) caia en el manejo por defecto de Python
    # (traceback feo, sin liberar nada) en vez de en nuestra limpieza,
    # porque el registro pasaba recien cerca del final de main().
    #
    # "recursos" es mutable y se va llenando a medida que cada cosa se
    # crea mas abajo -- _limpiar_al_salir() solo limpia lo que SI llego a
    # existir, sin importar en que punto del arranque haya llegado la
    # senal.
    # ------------------------------------------------------------------ #

    recursos = {"cerebro": None, "voz": None, "proceso_electron": None, "listener": None}
    _ya_se_limpio = {"hecho": False}

    def _limpiar_al_salir() -> None:
        """
        Red de seguridad para el cierre del proceso: mata ACTIVAMENTE el
        subproceso de RVC (taskkill /T, ver MikuVoice._apagar_proceso_rvc)
        y libera el LLM local de la GPU (cerebro.descargar_modelo()).
        Registrada con atexit (cubre salidas normales y excepciones no
        atrapadas) y se llama tambien DIRECTO (no solo via atexit) desde
        el manejador de SIGINT/SIGTERM -- levantar una excepcion y
        confiar en que el resto del stack se desenrede bien no
        garantizaba que esto corriera a tiempo; llamarlo aca sincrono,
        antes de salir del proceso, si.

        _ya_se_limpio evita correr esto dos veces si Ctrl+C se presiona
        mas de una vez (la segunda señal podria llegar a mitad de la
        primera limpieza).
        """
        if _ya_se_limpio["hecho"]:
            return
        _ya_se_limpio["hecho"] = True

        print("[Main] Cerrando: liberando VRAM y matando procesos...")
        if recursos["cerebro"] is not None:
            try:
                recursos["cerebro"].descargar_modelo()
            except Exception as error:
                print(f"[Main] Error liberando el LLM: {error}")
        if recursos["voz"] is not None:
            try:
                recursos["voz"].cerrar()
            except Exception as error:
                print(f"[Main] Error cerrando la voz/RVC: {error}")
        if recursos["listener"] is not None:
            try:
                recursos["listener"].cerrar()
            except Exception as error:
                print(f"[Main] Error liberando el STT: {error}")
        proceso_electron_actual = recursos["proceso_electron"]
        if proceso_electron_actual and proceso_electron_actual.poll() is None:
            try:
                cerrar_electron(proceso_electron_actual)
            except Exception as error:
                print(f"[Main] Error cerrando Electron: {error}")
        print("[Main] VRAM liberada.")

    atexit.register(_limpiar_al_salir)

    def _manejar_senal_cierre(signum, _frame) -> None:
        print(f"[Main] Senal de cierre recibida ({signum}), liberando VRAM antes de salir...")
        _limpiar_al_salir()
        # Salida inmediata: la limpieza critica (RVC + LLM) ya corrio a
        # mano arriba, asi que no hace falta esperar a que el resto del
        # programa se desenrede solo -- eso es justo lo que no queriamos
        # depender (y durante el arranque, buena parte de main() ni
        # siquiera corrio todavia).
        os._exit(0)

    signal.signal(signal.SIGINT, _manejar_senal_cierre)
    signal.signal(signal.SIGTERM, _manejar_senal_cierre)

    # ObsidianManager se crea antes que MikuBrain porque el cerebro le
    # lee el perfil + diario reciente UNA sola vez al arrancar (ver
    # MikuBrain.__init__ -> _cargar_contexto_inicial).
    obsidian = ObsidianManager(config.VAULT_MIKU_IA)
    obsidian.limpiar_contaminacion_nombre()

    cerebro = MikuBrain(obsidian=obsidian)
    recursos["cerebro"] = cerebro
    if not cerebro.verificar_conexion():
        print("El modelo LLM local no cargo -- revisa el log de arriba.")
        return

    puente_ws = PuenteWebSocket()
    puente_ws.iniciar()

    proceso_electron = lanzar_electron()
    recursos["proceso_electron"] = proceso_electron

    voz = MikuVoice(puente_ws=puente_ws)
    recursos["voz"] = voz
    voz.preparar_muletillas()
    voz.preparar_frases_activacion()

    # Avatar + saludo real de arranque -- ver FRASES_LISTA_PARA_SERVIR:
    # esto es lo que de verdad calienta RVC (una conversion real, no solo
    # cargar el modelo), para que tu primer mensaje ya no pague los
    # ~15-20s de arranque en frio.
    puente_ws.notificar_activacion()
    voz.hablar(random.choice(FRASES_LISTA_PARA_SERVIR))
    puente_ws.notificar_respuesta_completa()

    listener = MikuListener()
    recursos["listener"] = listener

    # El microfono (hilo principal) y el chat de texto de la ventana
    # (hilo aparte, via PuenteWebSocket) llaman a la misma funcion; el
    # lock evita que dos respuestas se reproduzcan encimadas en pygame.
    bloqueo_respuesta = threading.Lock()

    # Modo juego: True mientras Ollama/RVC estan descargados de la GPU.
    # Es un solo hilo el que lo lee/escribe (siempre dentro de
    # bloqueo_respuesta), asi que no hace falta una primitiva atomica.
    estado = {"dormida": False}

    # ------------------------------------------------------------------ #
    # Modo hibrido voz/texto: responder por el mismo canal por el que
    # llego el comando (ver al_recibir_comando -- via_texto)
    # ------------------------------------------------------------------ #

    def _responder(texto_respuesta: str, via_texto: bool) -> None:
        """
        Contesta siempre en voz alta -- y ADEMAS manda el texto al chat
        de la ventana si el comando llego escrito (via_texto), para que
        quede visible ahi tambien.
        """
        if via_texto:
            puente_ws.notificar_respuesta_texto(texto_respuesta)
        voz.hablar(texto_respuesta)

    def _al_interrumpir_habla() -> None:
        """
        Callback del monitor de interrupcion (ver
        listener.activar_monitor_interrupcion): se llama apenas se
        escucha "Miku silencio"/"Miku callate" MIENTRAS Miku esta
        hablando. Corta el audio de inmediato -- _hablar_oraciones_en_cola
        se entera por voz.hablando_interrumpido() y deja de preparar/decir
        lo que quedaba de la respuesta.
        """
        print("[Main] 'Miku silencio' detectado a mitad de la respuesta -- cortando.")
        voz.detener_habla()

    # ------------------------------------------------------------------ #
    # Modo juego: liberar/recuperar VRAM
    # ------------------------------------------------------------------ #

    def _manejar_modo_dormir(via_texto: bool) -> None:
        print("[Main] Modo juego: Miku se va a dormir (liberando VRAM)...")

        # Se dice/escribe la frase ANTES de descargar nada -- si
        # dormimos RVC primero, ya no tendria con que hablar con su
        # propia voz (por texto no importa, pero se mantiene el mismo orden).
        _responder("¡Entendido! Libero la tarjeta, me avisas cuando termines de jugar.", via_texto)

        cerebro.descargar_modelo()  # libera el LLM local (llama-cpp-python) de la GPU
        _detener_modelo_ollama_cli(config.MODELO_MEMORIA)  # el modelo de memoria si sigue siendo de Ollama

        voz.dormir_rvc()
        puente_ws.notificar_dormir()

        estado["dormida"] = True
        print("[Main] Miku esta dormida. VRAM liberada para juegos.")

    def _manejar_modo_despertar(via_texto: bool) -> None:
        print("[Main] Despertando a Miku...")

        puente_ws.notificar_activacion()  # reanuda el render 3D
        voz.despertar_rvc()  # recarga el modelo de RVC
        cerebro.calentar_modelo()  # recarga Ollama de antemano

        estado["dormida"] = False

        # Ya con RVC recargado, puede responder con su voz de verdad (o
        # por texto, si asi la despertaron).
        _responder("¡Ya estoy lista de nuevo, Carlo!", via_texto)
        puente_ws.notificar_respuesta_completa()
        print("[Main] Miku esta despierta.")

    # ------------------------------------------------------------------ #
    # "Oye Miku" detectada: aparece el avatar + confirmacion instantanea
    # ------------------------------------------------------------------ #

    def _al_detectar_activacion(palabra_clave_sola: bool) -> None:
        # En modo juego, "oye miku" solo debe despertarla si el comando
        # que sigue es "Miku despierta" -- eso ya lo resuelve
        # al_recibir_comando. Pero mostrar el avatar y reanudar el
        # render aqui (antes de saber que se dijo) iria en contra del
        # proposito del modo juego, asi que mientras esta dormida no
        # aparece ni suena nada al detectar la palabra clave.
        if estado["dormida"]:
            return

        # Le avisa al avatar que aparezca (animacion de entrada) apenas
        # se escucho la palabra clave, sin esperar a que termine de
        # grabarse el comando. El gesto de asentir solo tiene sentido
        # cuando "oye miku" se dijo SOLA (esta esperando instruccion) --
        # si ya vino pegada una instruccion en la misma respiracion (ej.
        # "oye miku, que hora es"), asentir se veria/sonaria raro
        # justo antes de responder de una.
        puente_ws.notificar_activacion(asentir=palabra_clave_sola)

        # La frase ("¿Que necesitas?"/"¿Si?") tambien es solo para
        # cuando esta esperando instruccion -- si ya la tiene, sonaria
        # raro preguntar "¿si?" y de inmediato contestar lo que ya le
        # dijeron. Cacheada de antemano (ver voz.preparar_frases_activacion
        # en main()), asi que sonar es casi instantaneo -- pero esta
        # llamada BLOQUEA hasta que termina de sonar, a proposito: si
        # _grabar_comando() arrancara en paralelo mientras Miku todavia
        # esta hablando, el microfono (sin cancelacion de eco) podria
        # captar su propia voz de fondo y mezclarla con el comando real.
        # Grabar recien DESPUES de que termina de decir "¿Si?" evita ese
        # problema -- no aplica cuando se salta la frase, porque ahi no
        # habla nada antes de grabar.
        if palabra_clave_sola:
            voz.reproducir_frase_activacion_aleatoria()

        # Prende el halo de "escuchando" justo cuando arranca a grabar
        # de verdad (stt_listener.py llama a _grabar_comando() apenas
        # on_activacion() retorna) -- se apaga en _al_terminar_escucha()
        # apenas termina de grabarse el comando.
        puente_ws.notificar_escuchando_inicio()

    def _al_terminar_escucha() -> None:
        puente_ws.notificar_escuchando_fin()

    def _al_quedarse_callado() -> None:
        """
        Se activo a Miku (o se seguia en modo conversacion) pero no se
        grabo ningun comando real -- sin esto, nada le avisaba al
        avatar que se ocultara, y se quedaba en pantalla indefinidamente
        despues de una activacion en silencio. Reusa la misma senal de
        "respuesta completa" que ya programa el fade-out normal, aunque
        aqui no hubo ninguna respuesta de verdad.
        """
        puente_ws.notificar_respuesta_completa()

    def _al_reanudar_escucha() -> None:
        """
        Se llama al arrancar cada turno DENTRO del modo conversacion
        (ver al_recibir_comando -- devuelve True para seguir), cuando ya
        no hace falta "Oye Miku" de nuevo: solo prende el halo de
        escuchando, sin repetir la animacion de entrada del avatar ni la
        frase "¿Si?" (esas son solo para la activacion inicial).
        """
        puente_ws.notificar_escuchando_inicio()

    # ------------------------------------------------------------------ #
    # Comando normal: streaming por chunks cortos
    # ------------------------------------------------------------------ #

    def _velocidad_para(oracion: str) -> float:
        """
        Frases largas (tipicamente respuestas informativas, ver el
        limite de 4 puntos/80-100 palabras en Personalidad.md) se
        hablan mas lento que una respuesta casual corta, para que de
        tiempo a procesar la informacion en vez de sonar atropellada.
        """
        if len(oracion.split()) > config.UMBRAL_PALABRAS_VELOCIDAD_LENTA:
            return config.KOKORO_VELOCIDAD_LENTA
        return config.KOKORO_VELOCIDAD

    def _hablar_oraciones_en_cola(
        cola: "queue.Queue[str | None]",
        on_primer_audio: "Callable[[], None] | None" = None,
    ) -> None:
        """
        Corre en un hilo aparte: va sacando chunks de la cola y
        reproduciendolos uno por uno, pero preparando (Kokoro + RVC) el
        SIGUIENTE chunk en un hilo aparte mientras el actual todavia
        esta sonando -- asi la voz no se queda en silencio entre chunk y
        chunk esperando a que termine de convertirse el que sigue.

        Si se pasa on_primer_audio, se llama justo antes de reproducir
        el primer chunk (para el benchmark de latencia en al_recibir_comando).

        Si se pide "Miku silencio"/"Miku callate" a mitad de camino (ver
        voz.detener_habla(), disparado por el monitor de interrupcion en
        core/stt_listener.py), deja de preparar y reproducir lo que
        queda -- pero SIGUE sacando (y marcando task_done) todo lo que
        llegue a la cola hasta el None final, para no dejar colgado el
        cola.join() de al_recibir_comando mientras el LLM sigue
        generando el resto de la respuesta en paralelo.
        """
        interrumpido = voz.hablando_interrumpido()
        oracion = cola.get()
        ruta_lista = (
            voz.preparar_audio(oracion, _velocidad_para(oracion))
            if (oracion is not None and not interrumpido)
            else None
        )
        cola.task_done()

        es_el_primero = True
        while oracion is not None:
            resultado: dict[str, object] = {}

            def _preparar_siguiente() -> None:
                oracion_siguiente = cola.get()
                resultado["oracion"] = oracion_siguiente
                resultado["ruta"] = (
                    voz.preparar_audio(oracion_siguiente, _velocidad_para(oracion_siguiente))
                    if (oracion_siguiente is not None and not interrumpido)
                    else None
                )
                cola.task_done()

            hilo_siguiente = threading.Thread(target=_preparar_siguiente, daemon=True)
            hilo_siguiente.start()

            if ruta_lista and not interrumpido:
                if es_el_primero and on_primer_audio:
                    on_primer_audio()
                voz.reproducir_audio(ruta_lista)
            elif ruta_lista:
                ruta_lista.unlink(missing_ok=True)  # se llego a generar pero no se va a hablar
            es_el_primero = False

            if voz.hablando_interrumpido():
                interrumpido = True

            hilo_siguiente.join()
            oracion = resultado["oracion"]
            ruta_lista = resultado["ruta"]

    def al_recibir_comando(texto_usuario: str, via_texto: bool = False) -> bool:
        """
        Devuelve True si la conversacion debe seguir abierta (stt_listener.py
        sigue grabando el siguiente turno SIN pedir "Oye Miku" de nuevo)
        o False si se cierra (hay que decir la palabra clave otra vez) --
        solo le importa al modo conversacion por voz.

        via_texto=True es el "modo hibrido": el mensaje llego escrito en
        el chat de la ventana en vez de hablado. Miku sigue respondiendo
        EN VOZ ALTA igual (asi lo confirmo el usuario tras probarlo), pero
        ADEMAS manda la respuesta como texto al chat de la ventana -- ver
        _responder(). El resto de la logica (dormir/despertar, despedidas,
        hora/fecha, el LLM) es identica para los dos canales.
        """
        with bloqueo_respuesta:
            print(f"[Tu] {texto_usuario}")
            texto_normalizado = normalizar_texto(texto_usuario)

            if via_texto:
                # "Miku aparece en pantalla" apenas escribis, sin
                # necesitar "Oye Miku" -- a diferencia de la activacion
                # por voz, sin frase hablada ni halo de escuchando (esto
                # no paso por el microfono). asentir=False: un mensaje de
                # texto siempre trae la instruccion "pegada" (nunca
                # escribis solo "oye miku" y esperas), asi que el gesto
                # de "¿si?" no tiene sentido aca -- solo debe aparecer,
                # sin asentir en cada mensaje.
                puente_ws.notificar_activacion(asentir=False)

            if any(frase in texto_normalizado for frase in FRASES_DORMIR):
                if not estado["dormida"]:
                    _manejar_modo_dormir(via_texto)
                else:
                    print("[Main] Ya esta dormida.")
                    _responder("Ya estoy dormida -- decime 'Miku despierta' para reactivarme.", via_texto)
                    puente_ws.notificar_respuesta_completa()
                return False

            if any(frase in texto_normalizado for frase in FRASES_DESPERTAR):
                if estado["dormida"]:
                    _manejar_modo_despertar(via_texto)
                else:
                    print("[Main] Ya esta despierta.")
                    _responder("Ya estoy despierta, Carlo.", via_texto)
                    puente_ws.notificar_respuesta_completa()
                return False

            if estado["dormida"]:
                print("[Main] (Miku esta dormida -- di 'Miku despierta' para reactivarla; ignoro el resto)")
                if via_texto:
                    _responder("Estoy dormida (modo juego) -- decime 'Miku despierta' primero.", via_texto)
                    puente_ws.notificar_respuesta_completa()
                return False

            if any(frase in texto_normalizado for frase in FRASES_SILENCIO):
                print("[Main] Miku silenciada, cierro la conversacion.")
                puente_ws.notificar_respuesta_completa()
                return False

            if _es_despedida_corta(texto_normalizado):
                print("[Main] Despedida detectada, cierro la conversacion.")
                _responder(random.choice(FRASES_DESPEDIDA_RESPUESTA), via_texto)
                puente_ws.notificar_respuesta_completa()
                return False

            if any(frase in texto_normalizado for frase in FRASES_HORA):
                _responder(_respuesta_hora(), via_texto)
                puente_ws.notificar_respuesta_completa()
                return True

            if any(frase in texto_normalizado for frase in FRASES_FECHA):
                _responder(_respuesta_fecha(), via_texto)
                puente_ws.notificar_respuesta_completa()
                return True

            pregunta_por_miku = any(frase in texto_normalizado for frase in FRASES_IDENTIDAD_MIKU)
            pregunta_por_usuario = any(frase in texto_normalizado for frase in FRASES_IDENTIDAD_USUARIO)
            if pregunta_por_miku or pregunta_por_usuario:
                # Si pregunto las dos cosas en el mismo mensaje (ej.
                # "sabes quien soy? sabes quien eres?"), se contestan las
                # dos juntas, en el mismo orden en que las pregunto.
                partes = []
                if pregunta_por_usuario:
                    partes.append(_respuesta_identidad_usuario())
                if pregunta_por_miku:
                    partes.append(_respuesta_identidad_miku())
                _responder(" ".join(partes), via_texto)
                puente_ws.notificar_respuesta_completa()
                return True

            # t0 marca el instante en que el STT ya termino de transcribir
            # la voz (o, si via_texto, el instante en que llego el
            # mensaje) y arranca a procesarse el comando -- de aca sale
            # el benchmark de latencia hasta el primer audio.
            t0 = time.monotonic()

            # Limpia cualquier interrupcion de un turno anterior antes de
            # empezar este, y prende el monitor que escucha SOLO "Miku
            # silencio"/"Miku callate" mientras dure esta respuesta (se
            # apaga mas abajo, apenas termina de hablar).
            voz.limpiar_interrupcion()
            listener.activar_monitor_interrupcion(FRASES_SILENCIO, _al_interrumpir_habla)

            # Muletilla casi instantanea (cacheada de antemano, ver
            # voz.preparar_muletillas): le da feedback audible al
            # usuario mientras el LLM/Kokoro/RVC generan la respuesta
            # real en paralelo, en vez de dejarlo en silencio unos
            # segundos. reproducir_audio() usa un lock interno, asi que
            # si la primera oracion real queda lista antes de que
            # termine la muletilla, simplemente espera su turno.
            threading.Thread(target=voz.reproducir_muletilla_aleatoria, daemon=True).start()

            def _al_sonar_primer_audio() -> None:
                print(f"[Benchmark] Latencia hasta el primer audio: {(time.monotonic() - t0) * 1000:.0f}ms")

            cola_habla: "queue.Queue[str | None]" = queue.Queue()
            hilo_hablador = threading.Thread(
                target=_hablar_oraciones_en_cola, args=(cola_habla, _al_sonar_primer_audio), daemon=True
            )
            hilo_hablador.start()

            acumulador = AcumuladorChunks()
            primera_oracion_lista = False

            def al_recibir_fragmento(fragmento: str) -> None:
                nonlocal primera_oracion_lista
                print(fragmento, end="", flush=True)
                for oracion in acumulador.agregar(fragmento):
                    if not primera_oracion_lista:
                        primera_oracion_lista = True
                        print(f"\n[Main] Primera oracion lista a los {time.monotonic() - t0:.2f}s: {oracion!r}")
                    cola_habla.put(oracion)

            print("[Miku] ", end="", flush=True)
            respuesta = cerebro.preguntar(texto_usuario, on_token=al_recibir_fragmento)
            print()
            print(f"[Main] LLM (stream completo): {time.monotonic() - t0:.2f}s")

            resto = acumulador.vaciar_resto()
            if resto:
                cola_habla.put(resto)

            # Espera a que el hilo hablador termine de decir todo antes
            # de soltar el lock -- si no, un segundo comando podria
            # pisarle el audio a este mientras todavia esta sonando.
            cola_habla.put(None)
            cola_habla.join()
            hilo_hablador.join(timeout=1)
            listener.desactivar_monitor_interrupcion()

            if via_texto:
                # Ademas de hablarla, la respuesta tambien se manda como
                # texto al chat de la ventana (llego escrita).
                puente_ws.notificar_respuesta_texto(respuesta)

            # Termino de decir TODA la respuesta (no cada oracion suelta):
            # el avatar programa su animacion de salida unos segundos
            # despues (ver renderer.js, caso "respuesta_completa").
            puente_ws.notificar_respuesta_completa()

            resumen = f"**Tu:** {texto_usuario}\n\n**Miku:** {respuesta}"
            ruta = obsidian.escribir_diario_miku(resumen)
            print(f"[Main] Interaccion registrada en: {ruta}")

            # Analisis de memoria en segundo plano (hilo aparte, no
            # bloquea): si el mensaje trae un dato permanente (pareja,
            # escuela, etc.), se anexa solo a Perfil.md.
            obsidian.extraer_y_guardar_hechos_clave(texto_usuario)

            # Igual pero para gente nueva que le presentan a Miku (ver
            # curiosidad con personas nuevas en Personalidad.md): si
            # aplica, se anexa a 03_Companeros/Companeros.md.
            obsidian.extraer_y_guardar_companero(texto_usuario)

            # Modo conversacion: sigue escuchando el siguiente turno sin
            # pedir "Oye Miku" de nuevo, hasta una despedida/silencio o
            # hasta que el usuario se quede callado (ver
            # MikuListener.escuchar -> on_reanudar_escucha).
            return True

    def al_recibir_mensaje_texto(texto_usuario: str) -> None:
        """
        Mensajes de texto escritos en la barra de chat de la ventana
        (ver gui_3d/renderer.js) llegan aca -- mismo pipeline que los
        comandos de voz (al_recibir_comando), pero marcados via_texto=True
        para que Miku responda por el chat ademas de hablar.

        FRASES_SILENCIO se chequea ACA, antes de intentar tomar
        bloqueo_respuesta: si Miku ya esta hablando (una respuesta
        anterior sigue en curso, con el lock tomado), escribir "Miku
        silencio" no puede esperar a que ese turno termine solo para
        recien ahi cortarlo -- por eso corta de una, sin pasar por
        al_recibir_comando ni por el lock.
        """
        if any(frase in normalizar_texto(texto_usuario) for frase in FRASES_SILENCIO):
            print("[Main] Miku silenciada desde el chat.")
            voz.detener_habla()
            return
        al_recibir_comando(texto_usuario, via_texto=True)

    puente_ws.al_recibir_mensaje_usuario = al_recibir_mensaje_texto

    try:
        listener.escuchar(
            on_comando=al_recibir_comando,
            on_activacion=_al_detectar_activacion,
            on_fin_escucha=_al_terminar_escucha,
            on_reanudar_escucha=_al_reanudar_escucha,
            on_silencio=_al_quedarse_callado,
        )
    finally:
        voz.cerrar()
        if proceso_electron and proceso_electron.poll() is None:
            cerrar_electron(proceso_electron)


if __name__ == "__main__":
    main()
