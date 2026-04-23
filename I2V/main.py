import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import torch
from PIL import Image
import io
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print("Cargando modelo Wan2.1-I2V-14B-480P...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    print("Modelo cargado.")
    yield
    del pipe


app = FastAPI(title="Wan2.1 Image-to-Video API", lifespan=lifespan)


@app.post("/generate")
async def generate_video(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
        "residual, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, "
        "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
    ),
    num_frames: int = Form(49),
    guidance_scale: float = Form(5.0),
    num_inference_steps: int = Form(20),
):
    try:
        # Leer y convertir la imagen
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Redimensionar a 480p manteniendo ratio (el modelo lo requiere)
        max_area = 480 * 832
        w, h = pil_image.size
        scale = (max_area / (w * h)) ** 0.5
        new_w = round(w * scale / 16) * 16
        new_h = round(h * scale / 16) * 16
        pil_image = pil_image.resize((new_w, new_h))

        output = pipe(
            image=pil_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=new_h,
            width=new_w,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        tmp_path = f"/tmp/{uuid.uuid4()}.mp4"
        export_to_video(output.frames[0], tmp_path, fps=16)
        return FileResponse(tmp_path, media_type="video/mp4", filename="output.mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
