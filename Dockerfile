FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git git-lfs python3.10 python3.10-venv python3-pip libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip

# Clone and install flux
WORKDIR /app
RUN git clone https://github.com/black-forest-labs/flux.git
WORKDIR /app/flux
RUN pip install -e ".[all]"

# Back to root dir
WORKDIR /app

COPY handler.py .

CMD ["python3", "handler.py"]
