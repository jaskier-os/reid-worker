FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python 3.12 + runtime deps, then remove build-only packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

# Fetch + export ALL model weights at build time (not committed to git):
#   - YOLOv8n person detector (yolov8n.pt at /app)
#   - SCRFD face detectors + ArcFace recognizers (det_*, w600k_*)
#   - Deep3DFaceRecon (epoch_20.pth, deep3d_recon_epoch20.pth, deep3d_recon.onnx
#     + .onnx.data, deep3d_recon.npz, BFM_Fitting/)
#   - 3DDFA_V2 (3ddfa_v2.onnx + .onnx.data + .npz)
# Deps (torch, onnx, huggingface_hub) come from requirements.txt installed above.
RUN bash fetch_models.sh --3d

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

RUN if [ ! -f config.ini ]; then cp config.ini.example config.ini; fi

EXPOSE 9284

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:9284/health || exit 1

CMD ["python", "main.py", "--mode", "server", "--host", "0.0.0.0", "--port", "9284"]
