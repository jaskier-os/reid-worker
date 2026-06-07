#!/usr/bin/env bash
#
# fetch_models.sh - download all pretrained model weights for reid-worker.
#
# Every model used by this project is a STANDARD, publicly available pretrained
# model. None are committed to the repository. This script fetches:
#   - YOLOv8n person detector            (ultralytics)
#   - SCRFD face detectors + ArcFace     (yakhyo/face-reidentification release)
#   - Deep3DFaceRecon + 3DDFA_V2 + BFM   (HuggingFace, via scripts/setup_deep3d.py)
#   - OpenGait pretrained gait models    (optional, official OpenGait sources)
#
# Usage:
#   bash fetch_models.sh           # core face + person models
#   bash fetch_models.sh --3d      # also fetch 3D face reconstruction weights
#   bash fetch_models.sh --gait    # also fetch OpenGait pretrained checkpoints
#   bash fetch_models.sh --all     # everything
#
set -euo pipefail
cd "$(dirname "$0")"

WANT_3D=0
WANT_GAIT=0
for arg in "$@"; do
  case "$arg" in
    --3d)   WANT_3D=1 ;;
    --gait) WANT_GAIT=1 ;;
    --all)  WANT_3D=1; WANT_GAIT=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

WEIGHTS_DIR="face-reidentification/weights"
mkdir -p "$WEIGHTS_DIR"

echo "[fetch_models] Downloading YOLOv8n person detector..."
# Ultralytics auto-downloads on first use, but fetch it explicitly here.
curl -fL -o yolov8n.pt \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

echo "[fetch_models] Downloading SCRFD face detectors + ArcFace recognizers..."
SCRFD_BASE="https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1"
for f in det_2.5g.onnx det_500m.onnx det_10g.onnx w600k_mbf.onnx w600k_r50.onnx; do
  echo "  - $f"
  curl -fL -o "$WEIGHTS_DIR/$f" "$SCRFD_BASE/$f"
done

if [ "$WANT_3D" -eq 1 ]; then
  echo "[fetch_models] Downloading 3D face reconstruction weights (Deep3DFaceRecon + BFM)..."
  # Pulls epoch_20.pth + BFM09 basis from vinthony/SadTalker on HuggingFace,
  # then exports the ONNX graphs. Requires: pip install huggingface_hub
  python scripts/setup_deep3d.py
  echo "[fetch_models] Exporting 3DDFA_V2 ONNX..."
  python scripts/export_3ddfa_v2.py
fi

if [ "$WANT_GAIT" -eq 1 ]; then
  echo "[fetch_models] OpenGait pretrained gait models."
  echo "  The official OpenGait pretrained checkpoints (CASIA-B: GaitSet, GaitPart,"
  echo "  GaitGL, GLN, Baseline; OUMVLP) are distributed by the OpenGait project."
  echo "  Obtain them from the OpenGait model zoo and place under OpenGait/output/:"
  echo "    https://github.com/ShiqiYu/OpenGait/blob/master/docs/1.model_zoo.md"
  echo "  Then unzip pretrained_oumvlp_model.zip into OpenGait/ if you use OUMVLP."
fi

echo "[fetch_models] Done. Core models are in $WEIGHTS_DIR and ./yolov8n.pt"
