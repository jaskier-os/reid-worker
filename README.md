# reid-worker

> **Docs & wiki:** [github.com/jaskier-os/docs/wiki](https://github.com/jaskier-os/docs/wiki)

Multi-camera person re-identification. Runs face recognition (SCRFD detect +
ArcFace embed) and optional gait (OpenGait), matches against a FAISS index, and
persists results through the reid-db-handler HTTP API. It never touches the
database directly. Runs either as an HTTP stream-ingest server or against local
cameras.

## Build / run

Local dev:

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# weights are not committed; fetch the public pretrained models
bash fetch_models.sh          # core face + person detectors
bash fetch_models.sh --3d     # add 3D face reconstruction weights
bash fetch_models.sh --all    # everything (3D + gait)

cp config.ini.example config.ini   # then edit

python main.py --mode server --host 0.0.0.0 --port 9284   # default
python main.py                                            # local cameras, headless
python main.py --debug                                    # GUI overlays
```

Docker:

```bash
docker build -t reid-worker .
```

The image is based on `nvidia/cuda:12.1.0-cudnn8`. The build runs `bash
fetch_models.sh --3d`, pulling ~1GB of ONNX / 3D weights, so it is slow and
large. The container starts the server on port 9284 with a `/health`
healthcheck. CI builds this Dockerfile and bumps the image tag in the deploy repo.

## Configuration

Config lives in `config.ini` (copy `config.ini.example`) and can be overridden by
env vars, which take precedence. The ones that matter:

- `REID_DB_API_URL` - reid-db-handler base URL (no trailing slash).
- `REID_DB_API_KEY` - API key for reid-db-handler (secret, don't commit).
- `REID_DB_ENABLED` - `false` to skip the API and use a local on-disk FAISS index.
- `REID_ANALYTICS_URL` - optional analytics endpoint.
- `HF_TOKEN` - optional HuggingFace token for `fetch_models.sh --3d`.

Everything else (model variants, thresholds, frame skip) is in the `config.ini`
sections; see the example file's inline comments.

## Dependencies

- Python 3.12, 64-bit.
- `onnxruntime-gpu` with NVIDIA CUDA 12.x + cuDNN for accelerated inference. No
  GPU: install plain `onnxruntime` and set `device = cpu` in config.
- Model weights fetched at build/setup time, not committed. Standard public
  pretrained models (YOLOv8, SCRFD, ArcFace, Deep3DFaceRecon, 3DDFA_V2, OpenGait).
- A reachable reid-db-handler instance (unless running with a local FAISS index).
- OpenCV system libs on headless Linux (`libgl1`, `libglib2.0-0`).
