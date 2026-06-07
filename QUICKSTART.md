# ReID Worker - Quick Start Guide

## Modes

### 1. Stream Server Mode (Default)
HTTP API for receiving video frames from remote sources.

```bash
# Start on default port 5000
python main.py

# Custom port
python main.py --port 8080
```

### 2. Camera Mode
Process local webcams/cameras.

```bash
# Auto-detect cameras
python main.py --mode camera --debug

# Specific cameras
python main.py --mode camera --cameras 0 1 --debug
```

### 3. Depth Test Mode
Test depth perception and save outputs.

```bash
python main.py --mode test-depth
```

## Debug Hotkeys

When running with `--debug` flag:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle depth visualization |
| `g` | Toggle distance/rotation lines |
| `s` | Save screenshot |
| `r` | Reset statistics |
| `SPACE` | Pause/Resume |
| `h` | Toggle help overlay |
| `1-9` | Toggle cameras |

## Debug Visualizations

### Distance Lines
- Yellow line from camera center to person
- Distance in meters at midpoint
- Angle in degrees from camera axis

### Heading Arrows
- Magenta arrow showing movement direction
- Calculated from position history
- Heading angle displayed

### Depth Overlay
- Color-coded depth map (blue=far, red=close)
- Overlaid on face bounding boxes
- Toggle with 'd' key

## Docker

```bash
# Start entire stack
docker-compose up

# Reid worker only
docker-compose up reid-worker

# Rebuild after changes
docker-compose up --build reid-worker
```

## Quick Tests

```bash
# Test with webcam
python main.py --mode camera --cameras 0 --debug

# Test depth perception
python main.py --mode test-depth

# Start API server
python main.py --mode server --port 5000
```

## File Locations

- **Main script**: `main.py`
- **Config**: `config.ini` (copy from `config.ini.example`)
- **Depth outputs**: `depth_test_output/`
- **Screenshots**: Current directory with timestamp
- **Docker config**: `Dockerfile` and `/docker-compose.yml`

## API Endpoints (Stream Server Mode)

- `GET /health` - Health check
- `GET /api/streams` - List streams
- `POST /api/streams/<id>` - Create stream
- `DELETE /api/streams/<id>` - Delete stream
- `POST /api/streams/<id>/frame` - Push frame
- `GET /api/stats` - Statistics

## Common Issues

**No cameras detected**
```bash
# Check cameras
python -c "import cv2; [print(f'Camera {i}') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

**Depth estimation not working**
- Check config.ini [depth] section
- Verify depth model path
- Try different depth method (monocular/ir/hybrid)

**Port already in use**
```bash
# Use different port
python main.py --port 8080
```

## Help

```bash
# Full help
python main.py --help

# Mode-specific examples
python main.py --help | grep -A20 "Examples:"
```
