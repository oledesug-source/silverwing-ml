# SilverWing Gesture OS

> *“I am the Storm. I am the Eye. I am SilverWing.”*

A J.A.R.V.I.S.-style gesture control system that turns a webcam into a real-time
interface for your desktop — fusing hand tracking, AR overlays, object detection,
system control, and an IoT bridge into one aggressive pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SilverWing Gesture OS                 │
├──────────┬──────────┬──────────────┬──────────┬───────────┤
│  Hand    │  Vision  │    HUD/AR    │  Control │   IoT     │  ← 5 subsystems
│ Controller│ Detector │   Overlay    │  System  │  Bridge    │
├──────────┴──────────┴──────────────┴──────────┴───────────┤
│              gesture_os.py  (single file, ~1500 LOC)       │
├───────────────────────────────────────────────────────────┤
│     MediaPipe  →  OpenCV  →  YOLOv8  →  PyAutoGUI  →      │
│     WebSocket / MQTT                                        │
└───────────────────────────────────────────────────────────┘
```

### Subsystems

| # | Subsystem | Technology | Responsibility |
|---|-----------|-----------|----------------|
| 1 | **Hand Controller** | MediaPipe Hands / Tasks API | Detects 21 hand landmarks per frame, classifies 5 gestures |
| 2 | **Vision / AR** | YOLOv8 (Ultralytics) | Object detection (cars, planes, people) in a background thread |
| 3 | **HUD Overlay** | OpenCV | Semi-transparent AR layer: system stats, detected objects, gesture feedback |
| 4 | **System Control** | PyAutoGUI | Mouse cursor mapping, clicks, volume, window management |
| 5 | **IoT Bridge** | WebSocket / MQTT | Relays high-level commands to external devices (drones, robots, smart home) |

### Gesture Map

| Gesture | Action |
|---------|--------|
| **Pinch** (thumb + index touch) | Select / confirm / click |
| **Fist** | Lock system (freeze cursor) |
| **Open Palm** | Summon main dashboard / HUD |
| **Swipe Right** | Open Google Maps / directions |
| **Swipe Left** | Open live data feed (weather/stocks) |

---

## Quick Start

### Prerequisites
- Python 3.11+ (tested on 3.13)
- Webcam
- Windows / macOS / Linux

### Installation

```bash
# Clone and install the gesture extras
pip install -e ".[gesture]"

# Or install from the requirements file
pip install -r requirements-gesture.txt

# For development (includes pytest, ruff, mypy)
pip install -e ".[gesture,dev]"
```

### Usage

```bash
# Run with defaults (uses configs/gesture.yaml if present)
python scripts/gesture_os.py

# Custom camera and resolution
python scripts/gesture_os.py --camera 1 --width 1920 --height 1080

# Disable subsystems
python scripts/gesture_os.py --no-detection --no-iot --no-system-control

# Debug: draw raw landmarks on the feed
python scripts/gesture_os.py --debug-landmarks

# List gestures and exit
python scripts/gesture_os.py --list-gestures

# Verbose logging
python scripts/gesture_os.py -v
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML config file (default: `configs/gesture.yaml`) |
| `--camera N` | Webcam index (default: 0) |
| `--width N` | Frame width (default: 1280) |
| `--height N` | Frame height (default: 720) |
| `--no-detection` | Disable YOLOv8 object detection |
| `--no-hud` | Disable AR overlay |
| `--no-system-control` | Disable PyAutoGUI system control |
| `--no-iot` | Disable IoT WebSocket/MQTT bridge |
| `--no-video` | Disable video window (headless mode) |
| `--debug-landmarks` | Draw raw MediaPipe hand landmarks |
| `-v` / `--verbose` | Enable debug logging |
| `--list-gestures` | Print gesture → action mapping and exit |

---

## Configuration

All settings live in `configs/gesture.yaml`:

```yaml
# Camera
camera_index: 0
frame_width: 1280
frame_height: 720

# Hand tracking
min_detection_confidence: 0.8
min_tracking_confidence: 0.5
max_hands: 2

# Gesture thresholds
pinch_distance_threshold: 0.04   # normalised coordinate units
swipe_min_distance: 0.15         # fraction of frame width
swipe_max_duration: 1.5          # seconds — swipe must complete within this window
gesture_cooldown: 0.8            # seconds between gesture actions

# Mouse / cursor
enable_mouse_control: true
mouse_sensitivity: 1.0
smooth_factor: 0.3   # EMA smoothing factor (0 = no smoothing)

# Object detection (YOLOv8)
enable_detection: true
yolo_model: yolov8n.pt
detection_interval: 0.5
detection_confidence: 0.45

# HUD
enable_hud: true
hud_alpha: 0.55

# IoT bridge
enable_iot: true
iot_protocol: websocket   # or "mqtt"
iot_host: 127.0.0.1
iot_port: 8765
```

---

## MediaPipe API Compatibility

This script supports **both** MediaPipe API generations:

| API | Mediapipe Version | Backend |
|-----|-------------------|---------|
| `solutions.hands` | `< 1.0` (0.10.x) | Legacy — simplest, no model download needed |
| `tasks.python.vision` | `>= 1.0` | Modern — requires downloading a `.task` model file |

**Detection logic** (`scripts/gesture_os.py`):
1. If `mediapipe.solutions` is importable → use the legacy `solutions` API.
2. Otherwise, fall back to the `tasks` API:
   - A `HandLandmark` constants class provides the same integer indices
     (e.g. `THUMB_TIP = 4`, `INDEX_FINGER_TIP = 8`) so the classification
     code does not depend on either API's enum.
   - The model file (`hand_landmarker.task`, ~8 MB) is searched for in
     `models/` and `~/.cache/silverwing/mediapipe/`, or downloaded
     automatically from the MediaPipe model repository.
   - On Windows, a `_patch_windows_free()` shim resolves the `free()`
     symbol from the UCRT (the bundled `libmediapipe.dll` doesn't export it),
     which prevents an `AttributeError` during detector initialisation.

**Recommendation:** Install `mediapipe<1.0` (`pip install "mediapipe>=0.10,<1.0"`)
to use the simpler `solutions` API without needing a model file.

---

## Testing

```bash
# Run gesture OS unit tests only
python -m pytest tests/test_gesture_os.py -v

# Run the full suite
python -m pytest tests/ -v

# Lint
python -m ruff check scripts/gesture_os.py tests/test_gesture_os.py

# Type-check
python -m mypy scripts/gesture_os.py
```

The test suite (`tests/test_gesture_os.py`) has 44 tests covering:

- `_dist` — Euclidean distance between landmarks
- `_screen_coords` — normalised-to-pixel coordinate mapping
- `_is_finger_extended` — finger extension detection logic
- `GestureConfig` — dataclass construction, `to_dict`/`from_dict`, YAML loading
- `_detect_swipe` — timestamped position history, `swipe_max_duration` enforcement,
  cooldown, horizontal-dominance check, direction detection
- `_classify` — priority order (pinch → swipe → open_palm → fist)
- `_is_pinch`, `_is_open_palm`, `_is_fist` — individual gesture detectors
- `HandLandmark` constants — all 21 landmark indices verified
- `load_config` — YAML parsing, missing file fallback, None handling

All tests use mock `MockLandmark` dataclass instances — no webcam or GPU required.

---

## Project Layout

```
Silverwing-ML/
├── configs/
│   └── gesture.yaml          # Gesture OS configuration
├── scripts/
│   └── gesture_os.py         # Main script (single file, all subsystems)
├── tests/
│   ├── test_gesture_os.py    # Unit tests for gesture detection logic
│   └── test_scripts.py       # Existing script tests
├── requirements-gesture.txt  # Gesture-specific dependencies
├── pyproject.toml            # Project config (ruff, mypy, pytest, optional-deps)
└── models/                   # Place hand_landmarker.task here (if using tasks API)
```

---

## Platform Integration

Gesture OS is wired into the Silverwing Tactical Command Platform as a
first-class capability provider, not a standalone script.

### Capability Registration

`sw_platform/tools/gesture.py` defines `GestureCapabilityProvider` and
`register_gesture_capabilities()`, which are called during platform startup
(`scripts/serve_platform.py → setup_platform()`).  Five capabilities are
registered into the platform's `CapabilityRegistry`:

| Capability | Method | Risk | Permission |
|---|---|---|---|
| `gesture_list` | `list_gestures()` | Low | L0 |
| `gesture_status` | `get_status()` | Low | L0 |
| `gesture_system_stats` | `get_system_stats()` | Low | L1 |
| `iot_send_command` | `send_iot_command(cmd, payload)` | Medium | L2 |
| `gesture_execute` | `execute_gesture(gesture)` | High | L2 |

All capabilities flow through the standard **propose → policy → permission →
sandbox → audit** lifecycle.  Missing dependencies are handled lazily — the
provider still registers, but execution returns a descriptive error.

### REST API Endpoints

The served platform (`scripts/serve_platform.py`) exposes three read-only
endpoints for the UI:

| Endpoint | Description |
|---|---|
| `GET /v1/gestures` | Static gesture → action mapping table |
| `GET /v1/gestures/status` | Subsystem availability (MediaPipe, OpenCV, YOLOv8, PyAutoGUI, IoT) + config snapshot |
| `GET /v1/gestures/stats` | Live system metrics: CPU, memory, battery, network, IP |

### UI Panel

The "Gesture Control" panel in `silverwing_platform/frontend/index.html`
renders:

- **Gesture mapping table** — gesture name, action description, risk badge
- **Subsystem status chips** — green/red dots with backend indicator
- **System stats** — CPU, memory, battery, network, IP (temperature-colored)
- **Action controls** — dropdown to execute a gesture, IoT command form

The header HUD bar shows a `GESTURE: ONLINE/OFFLINE/DEGRADED` chip updated
every 15 s.  Stats auto-refresh every 5 s.

```bash
# Start the platform with the mock LLM (no GPU/model required)
python scripts/serve_platform.py --mock --no-browser --port 9876

# In a second terminal — verify endpoints
python -c "import urllib.request, json; print(json.dumps(json.loads(urllib.request.urlopen('http://localhost:9876/v1/gestures').read()), indent=2))"
```

---

## Graceful Degradation

Every subsystem checks for its dependencies at import time and degrades
gracefully:

| Missing Dependency | Effect |
|---|---|
| `cv2` | Video feed and HUD disabled — script runs in console-only mode |
| `mediapipe` | Hand tracking disabled — no gesture recognition |
| `mediapipe.solutions` (mediapipe 1.0+) | Falls back to `tasks` API |
| `pyautogui` | System control disabled (no mouse/click/volume) |
| `ultralytics` | Object detection disabled |
| `socketio` / `paho-mqtt` | IoT bridge disabled |
| `psutil` | Falls back to `/proc` or `wmic` for system stats |
| Model file unavailable | Gesture recognition disabled with a clear error message |

The script always starts and reports which subsystems are online — pressing
**Esc** or `Ctrl+C` stops it cleanly.
