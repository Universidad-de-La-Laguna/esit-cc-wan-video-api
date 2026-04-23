import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print("Cargando modelo Wan2.1-T2V-1.3B...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=8.0
    )
    # Offload al CPU los componentes no activos para ahorrar VRAM
    pipe.enable_model_cpu_offload()
    print("Modelo cargado.")
    yield
    del pipe


app = FastAPI(title="Wan2.1 Video API", lifespan=lifespan)


class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residual, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_frames: int = 33        # reducido: ~2 segundos a 16fps
    height: int = 480
    width: int = 832
    guidance_scale: float = 6.0
    num_inference_steps: int = 20


@app.post("/generate")
async def generate_video(req: VideoRequest):
    try:
        output = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
        )
        tmp_path = f"/tmp/{uuid.uuid4()}.mp4"
        export_to_video(output.frames[0], tmp_path, fps=16)
        return FileResponse(tmp_path, media_type="video/mp4", filename="output.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
