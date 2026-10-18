"""
ObsidianManager: acceso de Miku a su propio Vault de Obsidian.

Por diseno, Miku solo lee y escribe dentro de su Vault (MIKU IA):
notas diarias, perfil de usuario y prompts de personalidad. El Vault
Principal del usuario (con sus proyectos) queda fuera del alcance de
esta clase; el usuario lo documenta manualmente ahi.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path

import requests

import config
from core.utils import normalizar_texto

# Rutas de archivo/carpeta de Windows: "C:\...", "C:/...", "\\servidor\recurso\...".
# El contexto que se le manda al LLM no deberia incluir nunca detalles
# del sistema de archivos del usuario -- solo el contenido real de las
# notas -- asi que se quitan antes de armar el bloque de contexto.
_PATRON_RUTA_SISTEMA = re.compile(
    r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n\s]*"
    r"|[A-Za-z]:/(?:[^/:*?\"<>|\r\n]+/)*[^/:*?\"<>|\r\n\s]*"
    r"|\\\\[^\\\s]+(?:\\[^\\\s]+)+"
)


def _sanitizar_texto(texto: str) -> str:
    """Quita rutas de archivo/carpeta de Windows de un texto, dejando solo el contenido real."""
    return _PATRON_RUTA_SISTEMA.sub("[ruta omitida]", texto)


# El LLM alucino en algun momento el nombre "Ramiro" en vez de "Carlo"
# en un par de respuestas que quedaron guardadas tal cual en el diario.
# Como el diario reciente se manda de vuelta como contexto (ver
# leer_perfil_y_diario_reciente), ese error se auto-realimentaba en
# cada arranque. \b requiere limite de palabra para no tocar nombres
# que solo contengan "ramiro" como substring de otra palabra.
_PATRON_NOMBRE_INCORRECTO = re.compile(r"\bRamiro\b", re.IGNORECASE)


class ObsidianManager:
    def __init__(self, vault_miku: Path):
        self.vault_miku = Path(vault_miku)

        self.diario_dir = self.vault_miku / "01_Diario"
        self.perfil_dir = self.vault_miku / "02_Perfil_Usuario"
        self.companeros_dir = self.vault_miku / "03_Companeros"
        self.prompts_dir = self.vault_miku / "04_System_Prompts"

        self._verificar_vault()

    def _verificar_vault(self) -> None:
        if not self.vault_miku.is_dir():
            raise FileNotFoundError(f"No se encontro el Vault de Miku en: {self.vault_miku}")

        # Estas subcarpetas son parte de la estructura que Miku administra,
        # asi que se crean solas si todavia no existen.
        self.diario_dir.mkdir(parents=True, exist_ok=True)
        self.perfil_dir.mkdir(parents=True, exist_ok=True)
        self.companeros_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Limpieza de contaminacion (nombre incorrecto en el diario)
    # ------------------------------------------------------------------ #

    def limpiar_contaminacion_nombre(self) -> None:
        """
        Corrige "Ramiro" -> "Carlo" en todas las notas de 01_Diario/,
        directamente en disco. Pensado para llamarse una vez al
        arrancar, ANTES de que MikuBrain lea el diario como contexto
        (ver main.py) -- si no, el error seguiria realimentandose turno
        a turno aunque el modelo ya no se equivoque de nuevo.
        """
        notas_corregidas = 0
        for nota in self.diario_dir.glob("*.md"):
            contenido_original = nota.read_text(encoding="utf-8")
            contenido_limpio = _PATRON_NOMBRE_INCORRECTO.sub("Carlo", contenido_original)
            if contenido_limpio != contenido_original:
                nota.write_text(contenido_limpio, encoding="utf-8")
                notas_corregidas += 1

        if notas_corregidas:
            print(
                f"[Miku-Memoria] Limpieza de arranque: corregido 'Ramiro' -> 'Carlo' "
                f"en {notas_corregidas} nota(s) del diario."
            )

    # ------------------------------------------------------------------ #
    # Personalidad (System Prompt editable desde Obsidian)
    # ------------------------------------------------------------------ #

    def leer_personalidad(self) -> str:
        """
        Lee 04_System_Prompts/Personalidad.md -- el "quien es Miku" base
        del System Prompt, editable a mano desde Obsidian sin tocar
        codigo. Devuelve "" si el archivo no existe o esta vacio, para
        que MikuBrain pueda caer de vuelta a un prompt por defecto
        embebido en el codigo en vez de romperse.
        """
        archivo_personalidad = self.prompts_dir / "Personalidad.md"
        if not archivo_personalidad.is_file():
            return ""
        return _sanitizar_texto(archivo_personalidad.read_text(encoding="utf-8").strip())

    # ------------------------------------------------------------------ #
    # Diario de Miku
    # ------------------------------------------------------------------ #

    def escribir_diario_miku(self, resumen: str) -> Path:
        """
        Anade una entrada al diario del dia en:
        MIKU IA/01_Diario/YYYY-MM-DD.md

        Si el archivo del dia no existe, lo crea con un encabezado.
        Devuelve la ruta del archivo escrito.
        """
        ahora = datetime.now()
        archivo_diario = self.diario_dir / f"{ahora:%Y-%m-%d}.md"

        if not archivo_diario.exists():
            encabezado = f"# Diario - {ahora:%Y-%m-%d}\n\n"
            archivo_diario.write_text(encabezado, encoding="utf-8")

        entrada = f"## {ahora:%H:%M}\n\n{resumen.strip()}\n\n"

        with archivo_diario.open("a", encoding="utf-8") as f:
            f.write(entrada)

        return archivo_diario

    # ------------------------------------------------------------------ #
    # Memoria: perfil + diario reciente
    # ------------------------------------------------------------------ #

    def leer_perfil_y_diario_reciente(
        self,
        dias_diario: int = config.DIAS_DIARIO_CONTEXTO,
        max_caracteres: int = 4000,
    ) -> str:
        """
        Junta todo lo que hay en 02_Perfil_Usuario/ (ej. Perfil.md) con
        las entradas mas recientes de 01_Diario/ en un solo bloque de
        texto, para dárselo una sola vez al LLM como contexto al
        arrancar -- no en cada turno, porque re-leer archivos por cada
        pregunta solo sumaba latencia sin cambiar el contenido.

        Si el texto junto queda muy largo, se recorta al final (lo mas
        reciente) para no inflar de mas el prompt.

        Solo se manda el TEXTO de las notas -- ni rutas de archivo, ni
        nombres de carpeta del sistema, ni nada por el estilo -- para
        que el LLM no tenga forma de "leer" ni mencionar detalles del
        sistema de archivos del usuario (ver _sanitizar_texto). El
        nombre del archivo (nota.stem) SI se manda como encabezado
        porque es solo una fecha o "Perfil", no una ruta.
        """
        bloques: list[str] = []

        for nota in sorted(self.perfil_dir.glob("*.md")):
            contenido = _sanitizar_texto(nota.read_text(encoding="utf-8").strip())
            if contenido:
                bloques.append(f"## Perfil: {nota.stem}\n{contenido}")

        for nota in sorted(self.companeros_dir.glob("*.md")):
            contenido = _sanitizar_texto(nota.read_text(encoding="utf-8").strip())
            if contenido:
                bloques.append(f"## Companeros de Carlo: {nota.stem}\n{contenido}")

        notas_diario = sorted(self.diario_dir.glob("*.md"))[-dias_diario:]
        for nota in notas_diario:
            contenido = _sanitizar_texto(nota.read_text(encoding="utf-8").strip())
            if contenido:
                bloques.append(f"## Diario del {nota.stem}\n{contenido}")

        texto = "\n\n".join(bloques)
        return texto[-max_caracteres:] if len(texto) > max_caracteres else texto

    # ------------------------------------------------------------------ #
    # Extractor de memoria inteligente (Perfil.md se actualiza solo)
    # ------------------------------------------------------------------ #

    PROMPT_EXTRACCION_HECHOS = """Analiza este mensaje y decide si el usuario esta DECLARANDO un DATO PERMANENTE sobre si mismo que valga la pena recordar para siempre: su escuela/trabajo, su pareja o familiares (con nombre), o un gusto/preferencia duradera.

IGNORA eventos del momento (que comio, como se siente hoy, planes de hoy, el clima, etc).
IGNORA preguntas -- si el usuario esta PREGUNTANDO algo (aunque sea sobre un dato que ya sepas), NO es una declaracion nueva, responde NINGUNO.
IGNORA muletillas y frases de cierre que no aportan informacion nueva, como "es todo", "eso es todo", "nada mas", "ya esta", "solo eso" -- estas NUNCA son un dato en si mismas, aunque aparezcan pegadas al final de una oracion que si declaro algo (en ese caso, el dato ya se extrajo de la parte anterior; la muletilla no agrega nada nuevo que resumir).

Si SI hay un dato permanente siendo declarado, responde con UNA sola linea corta en tercera persona resumiendolo, nada mas.
Si NO hay ningun dato permanente nuevo, responde exactamente: NINGUNO

Ejemplos:
Mensaje: "Tu novia se llama Keyli" -> Su pareja se llama Keyli.
Mensaje: "Estudio en la Universidad de Colima" -> Estudia en la Universidad de Colima.
Mensaje: "Hoy comi tacos" -> NINGUNO
Mensaje: "Tengo sueno" -> NINGUNO
Mensaje: "¿Como se llama mi novia?" -> NINGUNO
Mensaje: "Quiero que sepas que es mi mejor amigo, es todo" -> NINGUNO"""

    def extraer_y_guardar_hechos_clave(self, mensaje_usuario: str) -> None:
        """
        Lanza en un hilo aparte (no bloquea) un analisis del mensaje del
        usuario: le pregunta a Ollama si contiene un dato PERMANENTE
        sobre el (escuela, pareja, familiares, gustos duraderos) y, si
        lo encuentra, lo anexa a 02_Perfil_Usuario/Perfil.md. Si es solo
        algo del momento (ej. "hoy comi tacos"), no toca el archivo.

        Pensado para llamarse justo despues de responderle al usuario --
        corre en paralelo, no retrasa la respuesta que ya escucho/leyo.
        """
        hilo = threading.Thread(
            target=self._analizar_y_guardar_hecho,
            args=(mensaje_usuario,),
            daemon=True,
        )
        hilo.start()

    def _analizar_y_guardar_hecho(self, mensaje_usuario: str) -> None:
        try:
            hecho = self._preguntar_si_hay_hecho_clave(mensaje_usuario)
        except Exception as error:
            print(f"[Miku-Memoria] No se pudo analizar el mensaje en segundo plano: {error}")
            return

        if not hecho:
            return

        if self._hecho_ya_guardado(hecho):
            print(f"[Miku-Memoria] Hecho ya estaba guardado, no lo repito: {hecho}")
            return

        ruta = self._anexar_a_perfil(hecho)
        print(f"[Miku-Memoria] Hecho nuevo guardado en {ruta.name}: {hecho}")

    def _hecho_ya_guardado(self, hecho: str) -> bool:
        """
        Chequeo simple (substring, sin acentos/mayusculas) para no
        anexar el mismo hecho una y otra vez cada vez que se vuelve a
        mencionar o preguntar en la conversacion. No detecta
        reformulaciones distintas del mismo dato -- para eso hairia
        falta comparacion semantica, que no vale la pena para esto.
        """
        archivo_perfil = self.perfil_dir / "Perfil.md"
        if not archivo_perfil.exists():
            return False

        contenido_normalizado = normalizar_texto(archivo_perfil.read_text(encoding="utf-8"))
        return normalizar_texto(hecho) in contenido_normalizado

    def _preguntar_si_hay_hecho_clave(self, mensaje_usuario: str) -> str | None:
        payload = {
            "model": config.MODELO_MEMORIA,
            "messages": [
                {"role": "system", "content": self.PROMPT_EXTRACCION_HECHOS},
                {"role": "user", "content": mensaje_usuario},
            ],
            "stream": False,
            "options": {"num_predict": 60, "temperature": 0.2},
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }

        resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=20)
        resp.raise_for_status()

        texto = resp.json()["message"]["content"].strip()

        # Filtro defensivo: si el modelo (chico, a veces se desvia) no
        # contesto "NINGUNO" ni una frase corta razonable, mejor no
        # guardar nada que guardar basura en el perfil.
        if not texto or texto.upper().startswith("NINGUNO") or len(texto) > 200:
            return None

        return texto

    def _anexar_a_perfil(self, hecho: str) -> Path:
        """Agrega una linea con el hecho nuevo a Perfil.md, creandolo si no existe."""
        archivo_perfil = self.perfil_dir / "Perfil.md"
        ahora = datetime.now()

        if not archivo_perfil.exists():
            archivo_perfil.write_text(
                "# Perfil de Carlo\n\n## Hechos guardados automaticamente\n\n",
                encoding="utf-8",
            )

        with archivo_perfil.open("a", encoding="utf-8") as f:
            f.write(f"- ({ahora:%Y-%m-%d}) {hecho}\n")

        return archivo_perfil

    # ------------------------------------------------------------------ #
    # Extractor de companeros (gente nueva que le presentan a Miku)
    # ------------------------------------------------------------------ #

    PROMPT_EXTRACCION_COMPANERO = """Analiza este mensaje de Carlo y decide si menciona o describe a OTRA PERSONA (no a Carlo mismo): un amigo, companero de clase, o alguien que le estan presentando a Miku.

Busca especificamente el NOMBRE de esa persona y algun dato sobre ella: como la conoce Carlo, su relacion, sus gustos/hobbies, o donde la conocio.

IGNORA mensajes que hablan solo de Carlo mismo (sus propios gustos, su familia, su pareja -- eso lo guarda otro sistema).
IGNORA preguntas.
IGNORA mensajes que no mencionan ningun nombre de persona nueva.

Si SI hay una persona nueva siendo mencionada/descrita, responde en UNA sola linea con este formato exacto:
Nombre - dato breve en tercera persona sobre esa persona.

Si NO aplica, responde exactamente: NINGUNO

Ejemplos:
Mensaje: "Hola Miku, te presento a Alan, es mi companero de clase" -> Alan - Es companero de clase de Carlo.
Mensaje: "Fernanda me ayudo con la tarea de historia" -> Fernanda - Ayudo a Carlo con la tarea de historia.
Mensaje: "A Alan le encanta el futbol y los videojuegos" -> Alan - Le encanta el futbol y los videojuegos.
Mensaje: "Hoy comi tacos con mis amigos" -> NINGUNO
Mensaje: "Mi novia se llama Keyli" -> NINGUNO
Mensaje: "¿Como se llama mi companero de clase?" -> NINGUNO"""

    def extraer_y_guardar_companero(self, mensaje_usuario: str) -> None:
        """
        Igual que extraer_y_guardar_hechos_clave pero para gente nueva
        que Carlo le presenta a Miku (companeros, amigos): corre en un
        hilo aparte, y si encuentra un nombre + dato sobre esa persona,
        lo guarda en 03_Companeros/Companeros.md bajo una seccion con
        su nombre (una por persona, los datos nuevos se van acumulando
        debajo del mismo encabezado).
        """
        hilo = threading.Thread(
            target=self._analizar_y_guardar_companero,
            args=(mensaje_usuario,),
            daemon=True,
        )
        hilo.start()

    def _analizar_y_guardar_companero(self, mensaje_usuario: str) -> None:
        try:
            resultado = self._preguntar_si_hay_companero(mensaje_usuario)
        except Exception as error:
            print(f"[Miku-Memoria] No se pudo analizar companero en segundo plano: {error}")
            return

        if not resultado:
            return

        nombre, hecho = resultado
        if self._companero_ya_guardado(hecho):
            print(f"[Miku-Memoria] Dato de companero ya estaba guardado, no lo repito: {nombre} - {hecho}")
            return

        ruta = self._anexar_a_companeros(nombre, hecho)
        print(f"[Miku-Memoria] Companero guardado en {ruta.name}: {nombre} - {hecho}")

    def _companero_ya_guardado(self, hecho: str) -> bool:
        archivo = self.companeros_dir / "Companeros.md"
        if not archivo.exists():
            return False

        contenido_normalizado = normalizar_texto(archivo.read_text(encoding="utf-8"))
        return normalizar_texto(hecho) in contenido_normalizado

    def _preguntar_si_hay_companero(self, mensaje_usuario: str) -> tuple[str, str] | None:
        payload = {
            "model": config.MODELO_MEMORIA,
            "messages": [
                {"role": "system", "content": self.PROMPT_EXTRACCION_COMPANERO},
                {"role": "user", "content": mensaje_usuario},
            ],
            "stream": False,
            "options": {"num_predict": 60, "temperature": 0.2},
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }

        resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=20)
        resp.raise_for_status()

        texto = resp.json()["message"]["content"].strip()

        if not texto or texto.upper().startswith("NINGUNO") or " - " not in texto or len(texto) > 200:
            return None

        nombre, _, hecho = texto.partition(" - ")
        nombre = nombre.strip()
        hecho = hecho.strip()
        if not nombre or not hecho:
            return None

        return nombre, hecho

    def _anexar_a_companeros(self, nombre: str, hecho: str) -> Path:
        """
        Agrega un dato nuevo a 03_Companeros/Companeros.md, bajo una
        seccion "### Nombre" -- si la persona ya tiene seccion, el dato
        se anexa debajo de la suya en vez de crear una seccion duplicada,
        para que cada companero vaya acumulando su propio perfil.
        """
        archivo = self.companeros_dir / "Companeros.md"
        ahora = datetime.now()

        if not archivo.exists():
            archivo.write_text("# Companeros de Carlo\n\n", encoding="utf-8")

        contenido = archivo.read_text(encoding="utf-8")
        encabezado = f"### {nombre}"
        linea_nueva = f"- ({ahora:%Y-%m-%d}) {hecho}\n"

        indice_encabezado = contenido.find(encabezado)
        if indice_encabezado == -1:
            if not contenido.endswith("\n\n"):
                contenido = contenido.rstrip("\n") + "\n\n"
            contenido += f"{encabezado}\n{linea_nueva}\n"
        else:
            indice_siguiente = contenido.find("\n### ", indice_encabezado + len(encabezado))
            punto_insercion = indice_siguiente if indice_siguiente != -1 else len(contenido)
            contenido = (
                contenido[:punto_insercion].rstrip("\n") + "\n" + linea_nueva
                + contenido[punto_insercion:]
            )

        archivo.write_text(contenido, encoding="utf-8")
        return archivo
