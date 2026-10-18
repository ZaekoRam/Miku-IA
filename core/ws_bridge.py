"""
Puente WebSocket entre el backend de Python y la ventana 3D de Electron
(gui_3d/). Manda las senales de lip-sync mientras Miku habla:

    {"tipo": "hablando_inicio"}
    {"tipo": "hablando_nivel", "nivel": 0.0-1.0}
    {"tipo": "hablando_fin"}
    {"tipo": "activar"}              -- se detecto "Oye Miku": salir del modo reposo
    {"tipo": "dormir"}               -- "Miku duerme"/"modo juego": pausar el render de una vez

Y recibe los mensajes de texto que se escriben en la barra de chat de la
ventana:

    {"type": "user_message", "text": "..."}

El servidor WebSocket es asincronico (libreria `websockets`), pero corre
en su propio hilo con su propio event loop para que el resto del
programa -- que es sincronico -- pueda mandar eventos con metodos
normales (notificar_inicio(), etc.) sin tener que volverse async.

Uso como script (prueba manual, sin GUI): levanta el servidor y manda
un ciclo de eventos de ejemplo cada pocos segundos.
    python -m core.ws_bridge

Uso como modulo:
    from core.ws_bridge import PuenteWebSocket
    puente = PuenteWebSocket()
    puente.al_recibir_mensaje_usuario = lambda texto: print("Chat:", texto)
    puente.iniciar()
    puente.notificar_inicio()
    puente.notificar_nivel(0.8)
    puente.notificar_fin()
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Callable

import websockets

import config


class PuenteWebSocket:
    def __init__(self, host: str = config.WEBSOCKET_HOST, puerto: int = config.WEBSOCKET_PUERTO):
        self.host = host
        self.puerto = puerto
        self._clientes: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listo = threading.Event()

        # Lo asigna quien use el puente (main.py) despues de crearlo:
        # se llama con el texto cuando llega {"type": "user_message"}
        # desde la barra de chat de la ventana.
        self.al_recibir_mensaje_usuario: Callable[[str], None] | None = None

    def iniciar(self) -> None:
        """Arranca el servidor WebSocket en un hilo de fondo (no bloquea)."""
        hilo = threading.Thread(target=self._correr_servidor, daemon=True)
        hilo.start()

        # Espera a que el event loop este listo antes de devolver el
        # control, para que quien llamo pueda mandar eventos justo
        # despues sin una condicion de carrera.
        if not self._listo.wait(timeout=10):
            print("[Miku-WS] El servidor WebSocket no arranco a tiempo.")

    def _correr_servidor(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._arrancar_y_servir())
        except Exception as error:
            print(f"[Miku-WS] El servidor WebSocket se cayo: {error}")

    async def _arrancar_y_servir(self) -> None:
        async with websockets.serve(self._manejar_cliente, self.host, self.puerto):
            print(f"[Miku-WS] Servidor WebSocket escuchando en ws://{self.host}:{self.puerto}")
            self._listo.set()
            await asyncio.Future()  # se queda corriendo hasta que se mate el hilo

    async def _manejar_cliente(self, websocket) -> None:
        self._clientes.add(websocket)
        print(f"[Miku-WS] Cliente conectado ({len(self._clientes)} activo(s)).")
        try:
            async for mensaje_crudo in websocket:
                self._procesar_mensaje_entrante(mensaje_crudo)
        finally:
            self._clientes.discard(websocket)
            print(f"[Miku-WS] Cliente desconectado ({len(self._clientes)} activo(s)).")

    def _procesar_mensaje_entrante(self, mensaje_crudo: str) -> None:
        try:
            mensaje = json.loads(mensaje_crudo)
        except (json.JSONDecodeError, TypeError):
            print(f"[Miku-WS] Mensaje entrante invalido (no es JSON): {mensaje_crudo!r}")
            return

        if mensaje.get("type") != "user_message":
            return

        texto = str(mensaje.get("text", "")).strip()
        if not texto or not self.al_recibir_mensaje_usuario:
            return

        print(f"[Miku-WS] Mensaje de texto desde la ventana: {texto!r}")

        # El callback (Ollama + Edge-TTS + Obsidian) es sincronico y
        # bloqueante -- correrlo aqui trabaria el event loop del
        # servidor WebSocket (y con el, las senales de lip-sync y
        # cualquier otro mensaje entrante) mientras dura la respuesta.
        # Se manda a un hilo aparte para no bloquear.
        threading.Thread(
            target=self.al_recibir_mensaje_usuario,
            args=(texto,),
            daemon=True,
        ).start()

    async def _difundir(self, mensaje: dict) -> None:
        if not self._clientes:
            return
        datos = json.dumps(mensaje)
        # list(...) porque un cliente se puede desconectar mientras mandamos.
        await asyncio.gather(
            *(cliente.send(datos) for cliente in list(self._clientes)),
            return_exceptions=True,
        )

    def _mandar(self, mensaje: dict) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._difundir(mensaje), self._loop)

    # ------------------------------------------------------------------ #
    # API sincronica para el resto del programa
    # ------------------------------------------------------------------ #

    def notificar_inicio(self) -> None:
        self._mandar({"tipo": "hablando_inicio"})

    def notificar_nivel(self, nivel: float) -> None:
        self._mandar({"tipo": "hablando_nivel", "nivel": max(0.0, min(1.0, nivel))})

    def notificar_fin(self) -> None:
        self._mandar({"tipo": "hablando_fin"})

    def notificar_activacion(self, asentir: bool = True) -> None:
        """
        Se manda apenas se detecta la palabra clave ("Oye Miku"), antes
        de grabar el comando (o al escribir en el chat/despertar de modo
        juego). gui_3d/renderer.js lo usa para salir del "modo reposo"
        (reanudar el loop de render) de inmediato, en vez de esperar a
        que la ventana vuelva a ser visible.

        asentir=False evita el gesto de cabeza (y en main.py, tambien la
        frase hablada de "¿Si?") -- solo tiene sentido cuando "oye miku"
        se dijo SOLA y esta esperando instruccion; si ya vino con una
        instruccion pegada (voz en la misma respiracion, o cualquier
        mensaje de texto, que siempre trae contenido) se salta.
        """
        self._mandar({"tipo": "activar", "asentir": asentir})

    def notificar_respuesta_texto(self, texto: str) -> None:
        """
        Modo hibrido (Universidad/Silencioso): la respuesta de Miku a un
        mensaje ESCRITO en el chat de la ventana, para mostrarla ahi en
        vez de (o ademas de) decirla en voz alta -- ver main.py,
        al_recibir_comando(via_texto=True).
        """
        self._mandar({"tipo": "chat_respuesta", "texto": texto})

    def notificar_dormir(self) -> None:
        """
        Modo juego ("Miku duerme"): le dice al avatar que pause su loop
        de render de una vez, sin importar si la ventana esta visible o
        no (a diferencia del modo reposo automatico, que depende de que
        la ventana este oculta/minimizada).
        """
        self._mandar({"tipo": "dormir"})

    def notificar_escuchando_inicio(self) -> None:
        """
        Se manda apenas empieza a grabarse el comando (justo despues de
        "Oye Miku"). gui_3d/renderer.js lo usa para prender el "halo" de
        escuchando alrededor de la barra de chat y cambiar su texto a
        "Escuchando...", como en otros asistentes de voz.
        """
        self._mandar({"tipo": "escuchando_inicio"})

    def notificar_escuchando_fin(self) -> None:
        """
        Se manda apenas termina de grabarse el comando (haya quedado
        texto o no -- ver MikuListener.escuchar/on_fin_escucha). Apaga
        el halo de escuchando.
        """
        self._mandar({"tipo": "escuchando_fin"})

    def notificar_respuesta_completa(self) -> None:
        """
        Se manda cuando Miku termino de decir TODA la respuesta (la
        cola de oraciones se vacio por completo), no cada vez que
        termina una oracion suelta. gui_3d/renderer.js lo usa para
        programar la animacion de salida (fade-out) del avatar unos
        segundos despues, en vez de ocultarlo a mitad de una respuesta
        larga con pausas cortas entre oraciones.
        """
        self._mandar({"tipo": "respuesta_completa"})


if __name__ == "__main__":
    puente = PuenteWebSocket()
    puente.iniciar()

    print("[Miku-WS] Prueba manual: abre gui_3d (npm start) y deberias ver la boca moverse cada pocos segundos.")
    try:
        while True:
            puente.notificar_inicio()
            for _ in range(10):
                puente.notificar_nivel(0.2 + 0.6 * (_ % 2))
                time.sleep(0.15)
            puente.notificar_fin()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[Miku-WS] Detenido por el usuario.")
