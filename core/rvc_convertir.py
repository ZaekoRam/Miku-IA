"""
Script standalone de conversion RVC (voz de Edge-TTS -> voz de Miku).

IMPORTANTE: este script corre con el Python del venv aislado rvc_env/
(Python 3.10), NO con el Python principal del proyecto (3.12). Las
dependencias de RVC (sobre todo `fairseq`) no se pueden instalar en el
Python principal -- por eso viven separadas, en su propio venv, y
core/tts_speaker.py lo invoca como subprocess pasandole todo por
argumentos de linea de comandos. No importa nada de este proyecto
(config.py, core/*) a proposito: los dos entornos no comparten paquetes.

Uso (normalmente lo llama tts_speaker.py, no a mano):
    rvc_env\\Scripts\\python.exe core\\rvc_convertir.py ^
        --entrada audio_original.mp3 --salida audio_rvc.wav ^
        --modelo models\\rvc\\miku.pth --indice models\\rvc\\miku.index ^
        --semitonos 0 --metodo-pitch rmvpe --dispositivo cuda:0
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte audio a la voz de Miku con RVC.")
    parser.add_argument("--entrada", required=True, help="Audio de entrada (la voz de Edge-TTS)")
    parser.add_argument("--salida", required=True, help="Donde escribir el .wav ya convertido")
    parser.add_argument("--modelo", required=True, help="Ruta al .pth del modelo RVC")
    parser.add_argument("--indice", default="", help="Ruta al .index del modelo RVC (opcional)")
    parser.add_argument("--semitonos", type=int, default=0, help="Transposicion de pitch en semitonos")
    parser.add_argument(
        "--metodo-pitch",
        default="rmvpe",
        dest="metodo_pitch",
        help="Extractor de f0: rmvpe, harvest, pm, crepe...",
    )
    parser.add_argument("--dispositivo", default="cuda:0", help="cuda:0 o cpu:0")
    args = parser.parse_args()

    from rvc_python.infer import RVCInference

    rvc = RVCInference(
        device=args.dispositivo,
        model_path=args.modelo,
        index_path=args.indice,
        version="v2",
    )
    try:
        rvc.set_params(f0method=args.metodo_pitch, f0up_key=args.semitonos)
        rvc.infer_file(args.entrada, args.salida)
    finally:
        rvc.unload_model()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[RVC] Error durante la conversion: {error}", file=sys.stderr)
        sys.exit(1)
