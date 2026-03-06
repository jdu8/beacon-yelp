FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY setup.py .
COPY beacon/ beacon/
COPY scripts/ scripts/
COPY configs/ configs/

RUN pip install -e .

ENTRYPOINT ["python", "scripts/train.py"]