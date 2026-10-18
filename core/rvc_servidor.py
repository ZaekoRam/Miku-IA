"""
Servidor RVC persistente: carga el modelo UNA sola vez y despues
convierte audio bajo demanda, en vez de recargar PyTorch + fairseq +
HuBERT + RMVPE + el modelo en cada frase (eso es lo que hacia lenta
cada respuesta -- unos 7s solo de carga -- con core/rvc_convertir.py,
que arrancaba un proceso nuevo por turno).

IMPORTANTE: corre con el Python del venv aislado rvc_env/ (Python
3.10), igual que rvc_convertir.py -- ver ese archivo para el porque.

Protocolo: JSON por linea sobre stdin/stdout (texto, no binario). Lo
arranca y lo maneja core/tts_speaker.py como un subprocess de larga
vida; no se llama a mano normalmente.

    Peticion  (stdin):  {"entrada": "...", "salida": "...", "semitonos": 0}
    Respuesta (stdout): MARCADOR + {"ok": true} o {"ok": false, "error": "..."}
    {"_salir": true}      -> apaga el servidor

Al arrancar imprime una linea MARCADOR + {"listo": true} (o
{"listo": false, ...}) para que quien lo lanzo sepa cuando ya esta
listo para recibir trabajo.

Para el modo juego ("Miku duerme"/"despierta"), core/tts_speaker.py NO
le manda un comando de "descargar el modelo" a este proceso -- en vez
de eso, MATA este proceso entero y lo vuelve a lanzar de cero cuando
hace falta reactivarlo. Se probo primero recargar el modelo llamando
rvc.load_model() de nuevo sobre la misma instancia (sin matar el
proceso), pero eso dispara un subprocess interno de alguna libreria
(fairseq/numba/torch -- no se aislo cual exactamente) que en Windows a
veces se lanza con el Python equivocado y se queda colgado, trabando
todo el servidor. Matar y relanzar el proceso completo es mas lento
pero usa el mismo camino de arranque que ya se probo muchas veces y
funciona siempre; ademas libera el 100% de la VRAM (no solo la cache),
que es justo el objetivo del modo juego.

Por que el MARCADOR: rvc_python/fairseq/torch imprimen cosas sueltas a
stdout por su cuenta (ej. "is_half:True, device:cuda:0" durante la
carga, o logs de fairseq durante cada inferencia) que no tienen nada
que ver con nuestro protocolo. Sin un marcador, tts_speaker.py podia
leer una de esas lineas sueltas pensando que era la respuesta JSON y
tronar. Con el marcador, solo nos importan las lineas que empiezan con
el, el resto de la salida se ignora sin problema.
"""

import argparse
import json
import sys

MARCADOR = "###RVC_MIKU### "


def _emitir(datos: dict) -> None:
    print(MARCADOR + json.dumps(datos), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor RVC persistente (JSON por stdin/stdout).")
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--indice", default="")
    parser.add_argument("--metodo-pitch", default="rmvpe", dest="metodo_pitch")
    parser.add_argument("--dispositivo", default="cuda:0")
    args = parser.parse_args()

    try:
        from rvc_python.infer import RVCInference

        rvc = RVCInference(
            device=args.dispositivo,
            model_path=args.modelo,
            index_path=args.indice,
            version="v2",
        )
        rvc.set_params(f0method=args.metodo_pitch)
    except Exception as error:
        _emitir({"listo": False, "error": str(error)})
        return

    _emitir({"listo": True})

    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue

        try:
            peticion = json.loads(linea)
        except json.JSONDecodeError as error:
            _emitir({"ok": False, "error": f"JSON invalido: {error}"})
            continue

        if peticion.get("_salir"):
            break

        try:
            rvc.set_params(f0up_key=int(peticion.get("semitonos", 0)))
            rvc.infer_file(peticion["entrada"], peticion["salida"])
            _emitir({"ok": True})
        except Exception as error:
            _emitir({"ok": False, "error": str(error)})

    rvc.unload_model()


if __name__ == "__main__":
    main()
