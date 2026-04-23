# Wan2.1 Image-to-Video API

API HTTP para generación de video a partir de una imagen y un prompt de texto, basada en el modelo [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) de Alibaba.

## Requisitos

- Docker con `nvidia-container-toolkit`
- GPU NVIDIA con al menos 20GB de VRAM (probado en RTX 4000 Ada)
- Acceso a internet en el primer arranque (descarga ~30GB del modelo)

## Librerías principales

| Librería | Versión | Función |
|---|---|---|
| `diffusers` | 0.33.0 | Pipeline I2V (WanImageToVideoPipeline) |
| `transformers` | 4.46.0 | Encoder de texto (T5) |
| `accelerate` | 0.34.0 | CPU offload para gestión de VRAM |
| `torch` | 2.7.0 | Backend CUDA |
| `pillow` | - | Procesado de imagen de entrada |
| `fastapi` | 0.111.0 | Servidor HTTP |

## Arranque

```bash
mkdir -p /home/esit/projects/wan-i2v/{modelos,outputs}
docker compose up --build
```

El primer arranque descarga ~30GB. Puede tardar bastante.

## Uso

### Health check

```bash
curl http://IP:9000/health
```

### Generar video (mínimo)

```bash
curl -X POST http://IP:9000/generate \
  -F "image=@foto.jpg" \
  -F "prompt=La persona camina hacia la cámara sonriendo" \
  --output video.mp4
```

### Generar video (parámetros completos)

```bash
curl -X POST http://IP:9000/generate \
  -F "image=@foto.jpg" \
  -F "prompt=La persona camina hacia la cámara sonriendo, luz natural, cinematográfico" \
  -F "negative_prompt=baja calidad, estático, borroso" \
  -F "num_frames=49" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=6.0" \
  --output video.mp4
```

### Parámetros disponibles

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `image` | fichero | — | Imagen de entrada (JPG/PNG) — requerido |
| `prompt` | string | — | Descripción del movimiento a generar — requerido |
| `negative_prompt` | string | (predefinido) | Lo que el modelo debe evitar |
| `num_frames` | int | 49 | Número de frames (~3s a 16fps) |
| `num_inference_steps` | int | 20 | Pasos de difusión. Más = más calidad, más tiempo |
| `guidance_scale` | float | 5.0 | Fidelidad al prompt. Rango recomendado: 4.0–7.0 |

## Notas

- La imagen se redimensiona automáticamente a 480P manteniendo el ratio de aspecto.
- Se usa `enable_model_cpu_offload()` para gestionar los 20GB de VRAM.
- La generación tarda entre 5 y 15 minutos según los parámetros.
- No hay autenticación — se asume red interna/VPN.