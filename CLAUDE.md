# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

IMPORTANT: NEVER USE EMOJIS ANYWHERE IN LOGGING, CODE, OR OTHER TEXT.

IMPORTANT: Treat this codebase as work in progress. Never add backwards-compatibility
or legacy support unless explicitly asked. When a change may break a dependency, go
refactor every dependency rather than preserving the old path. Remove code that
becomes redundant.

## What this is

reid-worker is the GPU person re-identification worker. It takes camera feeds (or
HTTP stream-ingest), recognizes people with face recognition (SCRFD face detection +
ArcFace 512-dim embeddings) and optional gait (OpenGait), matches against a FAISS
index, and persists results. It scales horizontally: many workers feed one
reid-db-handler.

This is one of three tightly coupled ReID services. For the whole-system map (every
service, port map, architecture), see the orchestrator repo's CLAUDE.md / docs
(jaskier-os/orchestrator) rather than duplicating it here.

- **reid-db-handler repo** -- the only service that talks to the database directly;
  stores unique ids, metadata, sightings, relations, and FAISS data.
- **reid-worker** (this repo) -- recognizes people from camera feeds and sends new
  data to reid-db-handler.
- **reid-analytics repo** -- backend aggregates reid-db-handler data and proxies
  this worker for photo identification; frontend renders the catalogue/sightings UI.

## Architecture rule (KEEP PROMINENT)

reid-worker MUST NEVER connect to any database directly. ALL database operations go
through the reid-db-handler HTTP API. The client lives in `src/db_client.py`
(see also `src/remote_face_db.py` for the remote FAISS-backed face DB). Never add a
direct MongoDB or FAISS-DB connection here.

## Coupling rule

These three repos are bound by shared protocols. When changing one, check the
others before assuming the contract holds. The dependency chain is:
reid-db-handler schema -> reid-analytics export -> frontend. Concretely: if you
change the data this worker POSTs (sighting fields, trait/embedding format), inspect
reid-db-handler to confirm the schema accepts it, then check that reid-analytics
still aggregates it correctly. A change touching the wire format almost always
touches all three.

## Build / run

Weights are NOT committed. Every model is a standard publicly available pretrained
model fetched at build time by `fetch_models.sh` (YOLOv8n person detector, SCRFD +
ArcFace, optional Deep3DFaceRecon/3DDFA_V2/BFM 3D weights, optional OpenGait
checkpoints).

Local dev:

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

bash fetch_models.sh          # core face + person detectors
bash fetch_models.sh --3d     # add 3D face reconstruction weights
bash fetch_models.sh --gait   # add OpenGait gait checkpoints
bash fetch_models.sh --all    # everything

cp config.ini.example config.ini   # then edit (API URL + key for reid-db-handler)

python main.py --mode server --host 0.0.0.0 --port 9284   # HTTP stream-ingest (default)
python main.py                                            # local cameras, headless
python main.py --debug                                    # GUI overlays
```

Docker: `docker build -t reid-worker .` (base `nvidia/cuda:12.1.0-cudnn8`; the build
runs `fetch_models.sh --3d`, pulling ~1GB, so it is slow and large; container starts
the server on port 9284 with a `/health` healthcheck).

## Deploy

Deploy is via CI: commit + push, let GitLab CI build the Dockerfile and bump the
image tag in the deployment catalog repo. Never build/push the image by hand and
never edit Kubernetes deployments/env/secrets directly (`kubectl set env`/`edit`/
`patch`) -- those get overwritten by Flux. All deployed-service changes go through
the deployment catalog manifests.

## GPU

Requires CUDA 12.x + cuDNN. Auto-detects GPU via ONNX Runtime, falls back to CPU.
Verify: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`.
64-bit Python 3.12 required.

## Key files

- `main.py` -- entrypoint (server mode and local-camera mode).
- `src/db_client.py`, `src/remote_face_db.py` -- the ONLY path to reid-db-handler.
- `src/sighting_aggregator.py` -- batches/aggregates detections into sightings.
- `src/stream_source.py`, `src/unified_source.py`, `src/camera_calibration.py` --
  input sources and calibration.
- `src/depth_estimation.py`, `src/head_pose.py` -- per-frame analysis helpers.
- `face-reidentification/` -- SCRFD + ArcFace models and inference.
- `OpenGait/` -- gait recognition framework (PyTorch).
- `fetch_models.sh`, `scripts/setup_deep3d.py`, `scripts/export_3ddfa_v2.py` --
  weight fetching / export.

## Service management

Never restart any deployed service unless the user explicitly asks. When a restart
is needed, tell the user which service and why.

Do not write documentation unless asked to.
