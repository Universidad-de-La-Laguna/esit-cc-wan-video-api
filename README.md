# esit-cc-wan-video-api
Minimal HTTP API for text-to-video generation using Wan2.1, served via FastAPI and Docker with GPU support.


# Wan2.1 Video API

API HTTP minimalista para generación de video a partir de texto, basada en el modelo [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) de Alibaba. Desplegada como contenedor Docker con acceso GPU.

## Requisitos

- Docker con `nvidia-container-toolkit` instalado
- GPU NVIDIA con al menos 16GB de VRAM (probado en RTX 5060 Ti)
- Acceso a internet en el primer arranque (descarga ~5GB del modelo desde HuggingFace)

## Librerías principales

| Librería | Versión | Función |
|---|---|---|
| `diffusers` | 0.33.0 | Pipeline de generación de video (WanPipeline) |
| `transformers` | 4.46.0 | Encoder de texto (T5) |
| `accelerate` | 0.34.0 | CPU offload para gestión de VRAM |
| `torch` | 2.7.0 | Backend de cómputo CUDA |
| `fastapi` | 0.111.0 | Servidor HTTP |
| `uvicorn` | 0.30.1 | Servidor ASGI |
| `imageio` | - | Exportación de frames a MP4 |
| `ffmpeg` | - | Codificación de video |

## Estructura del proyecto

```
wan-video/
├── main.py              # API FastAPI
├── Dockerfile           # Imagen Docker
├── docker-compose.yml   # Configuración del servicio
└── modelos/             # Caché del modelo (HuggingFace, se crea en el primer arranque)
```

## Configuración

Antes de arrancar, ajusta las rutas de volúmenes en `docker-compose.yml`:

```yaml
volumes:
  - /ruta/en/host/modelos:/models    # caché HuggingFace
  - /ruta/en/host/outputs:/tmp       # videos generados
```

## Arranque

```bash
# Primera vez (descarga modelo ~5GB, puede tardar varios minutos)
docker compose up --build

# Arranques posteriores
docker compose up

# En segundo plano
docker compose up -d
```

El servicio queda disponible en el puerto `9000`.

## Uso

### Health check

```bash
curl http://IP:9000/health
```

Respuesta:
```json
{"status": "ok", "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}
```

### Generar video (parámetros mínimos)

```bash
curl -X POST http://IP:9000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Un amanecer en Marte"}' \
  --output video.mp4
```

### Generar video (parámetros completos)

```bash
curl -X POST http://IP:9000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Un amanecer en Marte, cielo rojizo, dunas de arena, luz cálida, cinematográfico, alta calidad",
    "negative_prompt": "baja calidad, estático, borroso",
    "num_frames": 49,
    "num_inference_steps": 40,
    "guidance_scale": 7.5,
    "height": 480,
    "width": 832
  }' \
  --output video.mp4
```

### Parámetros disponibles

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `prompt` | string | — | Descripción del video a generar (requerido) |
| `negative_prompt` | string | (predefinido) | Lo que el modelo debe evitar |
| `num_frames` | int | 33 | Número de frames (~2s a 16fps). Máx recomendado: 49 |
| `num_inference_steps` | int | 20 | Pasos de difusión. Más pasos = más calidad, más tiempo |
| `guidance_scale` | float | 6.0 | Fidelidad al prompt. Rango recomendado: 5.0–8.0 |
| `height` | int | 480 | Alto del video en píxeles |
| `width` | int | 832 | Ancho del video en píxeles |

## Notas

- El modelo se carga en GPU al arrancar el contenedor, no en cada petición. El primer `docker compose up` puede tardar varios minutos.
- Se usa `enable_model_cpu_offload()` para mover al CPU los componentes inactivos del pipeline y liberar VRAM durante la inferencia.
- La generación tarda entre 2 y 10 minutos dependiendo de `num_frames` y `num_inference_steps`.
- El video se devuelve directamente en la respuesta HTTP como `video/mp4`.
- No hay autenticación — se asume despliegue en red interna/VPN.
