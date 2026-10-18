# Modelos (no están en el repo)

Pesan varios GB y GitHub no acepta archivos de más de 100 MB, así que se guardan aparte. Para que Miku funcione,
estos archivos deben estar aquí con estos nombres (ver las rutas en `config.py`):

| Archivo | Para qué | Tamaño aprox. |
|---|---|---|
| `llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Cerebro: el modelo de lenguaje local (llama-cpp-python) | 4.7 GB |
| `llm/qwen2.5-1.5b-instruct-q8_0.gguf` | Modelo chico alternativo | 1.9 GB |
| `kokoro/kokoro-v1.0.fp16-gpu.onnx` | Voz: TTS Kokoro (GPU) | 177 MB |
| `kokoro/voices-v1.0.bin` | Voces de Kokoro | 28 MB |
| `rvc/miku.pth` y `rvc/miku.index` | Conversión a la voz de Miku (RVC) | 55 MB + 368 MB |
| `rvc/hatsune-miku-Latin-Spanish.pth` y `.index` | Variante en español latino | 55 MB + 8 MB |

El avatar `gui_3d/assets/miku.vrm` tampoco va en el repo (ver `gui_3d/assets/LEEME.txt`).
