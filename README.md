# reid-worker

Multi-camera person re-identification worker. It combines **face recognition**
(SCRFD face detection -> ArcFace 512-dim embeddings -> FAISS matching) with
**full-body person detection** (YOLOv8) and an optional **gait** pipeline
(OpenGait), and persists all results through the **reid-db-handler API**.

The worker never connects to a database directly. All person, trait, and
sighting data is read and written exclusively through the reid-db-handler HTTP
API, so this service stays stateless with respect to the data store.

## Features

- Face detection (SCRFD) + recognition (ArcFace) with FAISS similarity search.
- Full-body person detection (YOLOv8).
- Auto-merge of person identities when a face is temporarily occluded.
- Optional anti-spoofing / liveness gate before registering a new person.
- Optional 3D face reconstruction (Deep3DFaceRecon / 3DDFA_V2) at registration.
- Optional gait recognition framework (OpenGait, vendored under `OpenGait/`).
- HTTP stream-ingest server mode plus a local-camera mode.

## Prerequisites

- Python 3.12 (64-bit).
- A running reid-db-handler API instance (provides `/health`, `/api/persons`,
  `/api/traits`, `/api/sightings`).
- Optional NVIDIA GPU with CUDA 12.x + cuDNN 9.x for accelerated inference.
  Without a GPU, install the CPU `onnxruntime` build instead of `onnxruntime-gpu`
  and set `device = cpu` in `config.ini`.
- System libs for OpenCV (`libgl1`, `libglib2.0-0`) on headless Linux.

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download model weights (none are committed; all are public pretrained models):

   ```bash
   bash fetch_models.sh          # core face + person detectors
   bash fetch_models.sh --3d     # add 3D face reconstruction weights
   bash fetch_models.sh --gait   # add OpenGait pretrained checkpoints
   bash fetch_models.sh --all    # everything
   ```

3. Configure the service. Two interchangeable mechanisms are supported; env vars
   take precedence over the file.

   - **File:** copy the example and edit values:

     ```bash
     cp config.ini.example config.ini
     ```

   - **Environment:** copy `.env.example` to `.env` (or export the vars) and set
     at minimum the database connection.

   ### Configuration reference

   `config.ini` sections (see `config.ini.example` for inline comments):

   - `[database]` `api_url`, `api_key`, `analytics_url`, `enabled` -- the
     reid-db-handler API endpoint and key. With `enabled = false` the worker uses
     a local on-disk FAISS index instead of the centralized API.
   - `[person_detection]` -- YOLOv8 model variant, confidence, image size.
   - `[face_detection]` -- SCRFD model, confidence, IoU threshold.
   - `[face_recognition]` -- ArcFace model path, similarity threshold, samples.
   - `[person_tracking]` -- auto-merge threshold and time window.
   - `[anti_spoofing]` -- liveness model and threshold.
   - `[face_3d]` -- 3D reconstruction model, device, output format.
   - `[performance]` -- frame skip and queue size.
   - `[system]` -- debug flag and log level.

   ### Environment variables (override config.ini)

   | Variable              | Description                                             |
   | --------------------- | ------------------------------------------------------- |
   | `REID_DB_ENABLED`     | `true`/`false` -- use the API backend or local FAISS.   |
   | `REID_DB_API_URL`     | reid-db-handler API base URL (no trailing slash).       |
   | `REID_DB_API_KEY`     | reid-db-handler API key (secret -- never commit).       |
   | `REID_ANALYTICS_URL`  | Optional analytics endpoint URL.                        |
   | `REID_LOG_LEVEL`      | `DEBUG` / `INFO` / `WARNING` / `ERROR`.                 |
   | `HF_TOKEN`            | Optional HuggingFace token for `fetch_models.sh --3d`.  |

   No host IPs are hardcoded anywhere; point `REID_DB_API_URL` /
   `config.ini api_url` at wherever your reid-db-handler runs.

## How to build

There is no compile step (pure Python). "Building" means installing the pinned
dependencies and fetching weights, as in Setup above. A container image is also
provided:

```bash
docker build -t reid-worker .
```

## How to run

```bash
# HTTP stream-ingest server mode (default), listens on --host/--port
python main.py --mode server --host 0.0.0.0 --port 9284

# Local cameras, headless
python main.py

# Debug GUI with detection overlays
python main.py --debug

# Specific camera indices / face-only
python main.py --cameras 0 1
python main.py --no-person-detection

# Convenience launcher (creates venv, sets CUDA lib paths, then runs main.py)
bash run.sh
```

### Testing

A smoke test verifies the modules import and the config loads:

```bash
python -c "import main; print('import OK')"
python test_centralized_faiss.py   # exercises the reid-db-handler API client
```

`test_centralized_faiss.py` requires a reachable reid-db-handler instance and a
valid `config.ini` / env configuration.

## Database architecture

This worker communicates with the data store **only** through the
reid-db-handler API (`src/db_client.py` -> `TraitsDBClient`, and
`src/remote_face_db.py`). It does not import any database driver and opens no
direct PostgreSQL connection. To change where data lives, change the
reid-db-handler deployment, not this service.

## TLS / VPN

This service needs no TLS certificate and no VPN to run. It speaks plain HTTP to
whatever `REID_DB_API_URL` points at. If you front reid-db-handler with HTTPS,
simply use an `https://` URL; the standard `requests` trust store applies. No
certificate or key files are bundled or required.

## Model weights

All weights are standard, publicly available pretrained models and are fetched
by `fetch_models.sh` -- none are committed to this repository:

| Model                          | Source                                                            |
| ------------------------------ | ----------------------------------------------------------------- |
| YOLOv8n (`yolov8n.pt`)         | ultralytics assets release                                        |
| SCRFD detectors (`det_*.onnx`) | github.com/yakhyo/face-reidentification release v0.0.1            |
| ArcFace (`w600k_*.onnx`)       | github.com/yakhyo/face-reidentification release v0.0.1            |
| Deep3DFaceRecon + BFM09        | HuggingFace `vinthony/SadTalker` (via `scripts/setup_deep3d.py`)  |
| 3DDFA_V2                       | exported by `scripts/export_3ddfa_v2.py`                          |
| OpenGait checkpoints           | OpenGait model zoo (`docs/1.model_zoo.md`)                        |

If you train or finetune your own weights, place them under
`face-reidentification/weights/` (gitignored) and point the relevant config
entry at the file. The anti-spoofing model (`antispoof_minifas.onnx`) is only
needed when `[anti_spoofing] enabled = true`; supply your own MiniFASNet ONNX.
