"""
Prueba manual del ObsidianManager.

No requiere microfono ni GPU: solo escribe una entrada de prueba en el
Vault de Miku para confirmar que la ruta y el metodo funcionan.

Uso:
    python tests/test_obsidian_manager.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.obsidian_manager import ObsidianManager


def main() -> None:
    print("=== Verificando ruta del Vault de Miku ===")
    if not config.verificar_rutas_vaults():
        print("El Vault de Miku no existe. Revisa config.py.")
        return

    obsidian = ObsidianManager(config.VAULT_MIKU_IA)

    print("\n=== Probando escribir_diario_miku ===")
    ruta_diario = obsidian.escribir_diario_miku(
        "Prueba automatica del ObsidianManager: todo funcionando correctamente."
    )
    print(f"Entrada anadida en: {ruta_diario}")

    print("\nListo. Revisa el archivo en Obsidian para confirmar el cambio.")


if __name__ == "__main__":
    main()
