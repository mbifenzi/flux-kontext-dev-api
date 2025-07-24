FROM python:3.10-slim

RUN apt-get update && apt-get install -y git libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install git+https://github.com/huggingface/diffusers.git \
                transformers accelerate pillow

ENV PYTHONUNBUFFERED=1