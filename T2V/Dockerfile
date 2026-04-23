FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
 
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    "diffusers==0.33.0" \
    "transformers==4.46.0" \
    "accelerate==0.34.0" \
    "imageio[ffmpeg]" \
    sentencepiece \
    ftfy \
    pydantic==2.7.1
 
COPY main.py .
 
ENV HF_HOME=/models
 
EXPOSE 8000
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
 
