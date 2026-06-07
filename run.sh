#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "[reid-worker] Creating venv..."
  python3 -m venv venv
fi

echo "[reid-worker] Installing dependencies..."
venv/bin/python -m pip install -q -r requirements.txt

NVIDIA_LIBS="$(dirname "$(find venv -name 'libcublas.so*' -print -quit 2>/dev/null)")"
CUDNN_LIBS="$(dirname "$(find venv -name 'libcudnn.so*' -print -quit 2>/dev/null)")"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${CUDNN_LIBS}:${LD_LIBRARY_PATH}"

echo "[reid-worker] Starting..."
exec venv/bin/python main.py "$@"
