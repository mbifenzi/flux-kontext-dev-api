FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git lfs install

COPY . .

CMD ["python3", "handler.py"]
