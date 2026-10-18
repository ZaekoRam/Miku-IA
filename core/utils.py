"""Utilidades compartidas por los modulos de Miku."""

import unicodedata


def normalizar_texto(texto: str) -> str:
    """Pasa a minusculas y quita acentos/diacriticos para comparar texto de forma robusta."""
    texto = texto.strip().lower()
    texto_sin_acentos = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto_sin_acentos if not unicodedata.combining(c))
