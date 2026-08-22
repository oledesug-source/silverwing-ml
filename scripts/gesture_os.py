"""SilverWing Gesture OS — aggressive, J.A.R.V.I.S.-style hand & vision control.

This script turns a webcam into a real-time gesture interface for the
SilverWing platform.  It fuses five subsystems into a single processing
loop:

    1. **Hand Controller** — MediaPipe Hands detects 21 landmarks per hand
       and classifies five canonical gestures: *pinch*, *fist*,
       *open_palm*, *swipe_right*, *swipe_left*.

    2. **Vision / AR** — YOLOv8 (Ultralytics) runs in a background thread
       for environment awareness (car, plane, person, …).  Detected
       objects are tagged with bounding boxes and overlay labels on the
       video feed.

    3. **HUD Overlay** — a semi-transparent AR layer draws system stats,
       detected objects, active gesture, and a targeting reticle.

    4. **System Control** — PyAutoGUI maps the index-finger tip to screen
       coordinates for cursor control; pinch = click, open_palm = summons
       the main dashboard, swipe_right = open Maps.

    5. **IoT Bridge** — a WebSocket / MQTT client relays high-level
       commands (e.g. *launch_drone*, *lock_down*) to external devices
       such as drones, robots, or smart-home hubs.

All subsystems degrade gracefully: if an optional dependency is missing
the relevant subsystem is disabled with a warning rather than crashing
the whole pipeline.

Usage::

    python scripts/gesture_os.py                              # defaults
    python scripts/gesture_os.py --camera 1 --width 1280      # custom cam
    python scripts/gesture_os.py --config configs/gesture.yaml
    python scripts/gesture_os.py --no-detection --no-iot       # disable
    python scripts/gesture_os.py --list-gestures              # quick lookup
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
#  Project-root bootstrap (same pattern as other scripts in this repo)
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
#  Console encoding — allow Unicode glyphs (arrows, box-drawing, emoji) on
#  Windows where the default codepage is cp1252.
# --------------------------------------------------------------------------- #
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

# --------------------------------------------------------------------------- #
#  Optional-dependency import guards
# --------------------------------------------------------------------------- #

try:
    import cv2  # type: ignore[import-untyped]
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

try:
    import mediapipe as mp  # type: ignore[import-untyped]
    HAS_MEDIPIPE = True
except ImportError:  # pragma: no cover
    HAS_MEDIPIPE = False

# MediaPipe API detection — the *solutions* module was removed in
# mediapipe 1.0; the replacement *tasks* API needs a model asset file.
HAS_MP_SOLUTIONS = False
if HAS_MEDIPIPE:
    try:
        mp.solutions.hands.Hands  # noqa: B018 — probe for solutions sub-module
        HAS_MP_SOLUTIONS = True
    except (AttributeError, ImportError):
        HAS_MP_SOLUTIONS = False

try:
    import numpy as np  # type: ignore[import-untyped]  # noqa: F401
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

try:
    import pyautogui  # type: ignore[import-untyped]
    HAS_PYAUTOGUI = True
except ImportError:  # pragma: no cover
    HAS_PYAUTOGUI = False

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    HAS_ULTRALYTICS = False

try:
    import socketio  # type: ignore[import-untyped]  # noqa: F401
    HAS_SOCKETIO = True
except ImportError:  # pragma: no cover
    HAS_SOCKETIO = False

try:
    import tkinter as tk
    from tkinter import ttk  # noqa: F401
    HAS_TK = True
except ImportError:  # pragma: no cover
    HAS_TK = False

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger("silverwing.gesture_os")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  [\033[34m%-8s\033[0m]  %(message)s"
        if os.name == "nt" or sys.stdout.isatty()
        else "%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("absl").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
#  Landmark index constants — work with any MediaPipe version
# --------------------------------------------------------------------------- #


class HandLandmark:
    """Integer indices for the 21 hand landmarks.

    Mirrors ``mp.solutions.hands.HandLandmark`` so the rest of the code
    does not depend on which MediaPipe API version is installed.
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


def _get_hand_landmark_constants() -> Any:
    """Return the HandLandmark enum from whichever MediaPipe API is active."""
    if HAS_MP_SOLUTIONS:
        return mp.solutions.hands.HandLandmark
    return HandLandmark


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #


@dataclass
class GestureConfig:
    """Central configuration for every subsystem.

    Defaults are tuned for a typical laptop webcam + 1080p display.
    All values can be overridden via ``--config <yaml>`` or CLI flags.
    """

    # --- Camera ---------------------------------------------------------
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # --- Hand tracking --------------------------------------------------
    min_detection_confidence: float = 0.8
    min_tracking_confidence: float = 0.5
    max_hands: int = 2

    # --- Gesture thresholds ---------------------------------------------
    pinch_distance_threshold: float = 0.04  # normalised image diagonal
    swipe_min_distance: float = 0.15        # fraction of frame width
    swipe_max_duration: float = 1.5         # seconds to register a swipe
    gesture_cooldown: float = 0.8           # seconds between actions

    # --- Mouse / cursor -------------------------------------------------
    enable_mouse_control: bool = True
    mouse_sensitivity: float = 1.0
    click_on_pinch: bool = True
    smooth_factor: float = 0.3   # exponential moving average (0 = no smoothing)

    # --- Object detection -----------------------------------------------
    enable_detection: bool = True
    yolo_model: str = "yolov8n.pt"          # auto-downloaded by Ultralytics
    detection_interval: float = 0.5        # seconds between inference runs
    detection_confidence: float = 0.45

    # --- HUD / AR overlay -----------------------------------------------
    enable_hud: bool = True
    hud_alpha: float = 0.55       # overlay transparency
    hud_show_stats: bool = True
    hud_show_objects: bool = True
    hud_show_gesture: bool = True

    # --- System control -------------------------------------------------
    enable_system_control: bool = True
    maps_url: str = "https://g.co/maps"        # swipe_right destination
    live_data_url: str = "https://www.google.com/finance"  # swipe_left destination

    # --- Gadgets --------------------------------------------------------
    enable_gadgets: bool = True
    gadget_refresh_interval: float = 2.0

    # --- IoT bridge -----------------------------------------------------
    enable_iot: bool = True
    iot_protocol: str = "websocket"   # or "mqtt"
    iot_host: str = "127.0.0.1"
    iot_port: int = 8765
    iot_reconnect_delay: float = 5.0

    # --- Misc -----------------------------------------------------------
    show_video_feed: bool = True
    debug_landmarks: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GestureConfig:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


def load_config(config_path: str | None) -> GestureConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    cfg = GestureConfig()
    if not config_path:
        return cfg
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s — using defaults", path)
        return cfg
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        merged = cfg.to_dict()
        merged.update(data)
        cfg = GestureConfig.from_dict(merged)
        logger.info("Loaded config from %s", path)
    except ImportError:
        logger.warning("PyYAML not installed — cannot parse YAML config")
    except Exception as exc:
        logger.warning("Config parse error (%s) — using defaults", exc)
    return cfg


# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #


def _dist(p1: Any, p2: Any) -> float:
    """Euclidean distance between two normalised landmark points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _landmark_to_pixel(lm: Any, h: int, w: int) -> tuple[int, int]:
    """Convert a normalised MediaPipe landmark to pixel coordinates."""
    return int(lm.x * w), int(lm.y * h)


def _screen_coords(landmark: Any, screen_w: int, screen_h: int, sensitivity: float = 1.0) -> tuple[int, int]:
    """Map a normalised landmark position to absolute screen coordinates."""
    x = int(landmark.x * screen_w * sensitivity)
    y = int(landmark.y * screen_h * sensitivity)
    return max(0, min(x, screen_w - 1)), max(0, min(y, screen_h - 1))


def _is_finger_extended(landmarks: Any, tip_idx: int, dip_idx: int, pip_idx: int) -> bool:
    """Return *True* when the finger whose tip is *tip_idx* is extended.

    A finger is considered extended when the tip is above both the PIP
    and DIP joints (i.e. the finger is straight, not curled) **and**
    the tip sits at least 2 % of the image height above the wrist.
    """
    tip = landmarks[tip_idx]
    dip = landmarks[dip_idx]
    pip = landmarks[pip_idx]
    wrist = landmarks[0]
    # Straight finger: tip is above DIP above PIP; validate against wrist
    # for a minimum extension threshold.
    return tip.y < pip.y and tip.y < dip.y and (wrist.y - tip.y) > 0.02


def get_system_stats() -> dict[str, str]:
    """Collect lightweight system metrics for the HUD / gadget window."""
    stats: dict[str, str] = {}

    # CPU load via /proc or psutil fallback
    try:
        import psutil  # type: ignore[import-untyped]
        stats["cpu"] = f"{psutil.cpu_percent(interval=0.2):.0f}%"
        stats["mem"] = f"{psutil.virtual_memory().percent:.0f}%"
        stats["net"] = f"{psutil.net_io_counters().bytes_recv / 1_048_576:.1f} MB↓"
    except ImportError:
        # Fallback without psutil
        try:
            load1, _, _ = os.getloadavg()
            stats["load"] = f"{load1:.2f}"
        except (AttributeError, OSError):
            stats["load"] = "n/a"

        # Memory on Windows / Linux
        if os.name == "nt":
            try:
                out = subprocess.check_output(
                    ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/Value"],
                    text=True, timeout=2,
                )
                vals = {}
                for line in out.strip().splitlines():
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        vals[k.strip()] = int(v.strip())
                if "TotalVisibleMemorySize" in vals and vals["TotalVisibleMemorySize"]:
                    used_pct = 100 - (vals.get("FreePhysicalMemory", 0) / vals["TotalVisibleMemorySize"] * 100)
                    stats["mem"] = f"{used_pct:.0f}%"
                else:
                    stats["mem"] = "n/a"
            except Exception:
                stats["mem"] = "n/a"
        else:
            try:
                with open("/proc/meminfo", encoding="utf-8") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            meminfo[parts[0]] = int(parts[1].strip().split()[0])
                total = meminfo.get("MemTotal", 1)
                avail = meminfo.get("MemAvailable", total)
                stats["mem"] = f"{100 - avail / total * 100:.0f}%"
            except Exception:
                stats["mem"] = "n/a"

        stats["cpu"] = stats.get("load", "n/a")

    # Battery (if available)
    try:
        import psutil  # type: ignore[import-untyped]
        batt = psutil.sensors_battery()
        if batt:
            stats["bat"] = f"{batt.percent:.0f}%{' ⚡' if batt.power_plugged else ' 🔋'}"
    except Exception:
        pass

    # Network public IP (non-blocking, best-effort)
    try:
        # quick local check
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        stats["ip"] = ip
    except Exception:
        stats["ip"] = "n/a"

    return stats


def _patch_windows_free() -> None:
    """Monkey-patch the mediapipe C bindings to resolve ``free()``
    from the C runtime on Windows (libmediapipe.dll doesn't export it).

    This is a no-op on non-Windows or when the patch has already been
    applied.
    """
    import platform as _platform

    if os.name != "nt":
        return

    bindings = sys.modules.get("mediapipe.tasks.python.core.mediapipe_c_bindings")
    if bindings is None:
        return
    if getattr(bindings, "_free_patched", False):
        return

    import ctypes
    import importlib.resources as resources

    # Load the C runtime DLL to get ``free``
    try:
        ucrt = ctypes.CDLL("ucrtbase.dll")
    except OSError:
        ucrt = ctypes.CDLL("msvcrt.dll")
    ucrt.free.argtypes = [ctypes.c_void_p]
    ucrt.free.restype = None

    _orig_load = bindings.load_raw_library
    _mod = bindings

    def _patched_load_raw_library(signatures=()):
        """Replacement for load_raw_library with a free() fallback."""
        if _mod._shared_lib is None:
            if os.name == "posix":
                lib_name = "libmediapipe.so"
            elif _platform.system() == "Darwin":
                lib_name = "libmediapipe.dylib"
            else:
                lib_name = "libmediapipe.dll"
            ctx = resources.files("mediapipe.tasks.c")
            abs_path = str(ctx / lib_name)
            _mod._shared_lib = ctypes.CDLL(abs_path)

        for signature in signatures:
            c_func = getattr(_mod._shared_lib, signature.func_name)
            c_func.argtypes = signature.argtypes
            c_func.restype = signature.restype

        # Register "free()" — fall back to UCRT on Windows
        try:
            _mod._shared_lib.free.argtypes = [ctypes.c_void_p]
            _mod._shared_lib.free.restype = None
        except AttributeError:
            _mod._shared_lib.__dict__["free"] = ucrt.free

        _mod._shared_lib.__dict__["_free_patched"] = True
        return _mod._shared_lib

    _mod.load_raw_library = _patched_load_raw_library

    # If load_shared_library was already imported by hand_landmarker,
    # it calls load_raw_library() by module-global lookup — our patch
    # on _mod.load_raw_library covers that path.
    _mod._free_patched = True


# --------------------------------------------------------------------------- #
#  1 — Hand Controller (MediaPipe)
# --------------------------------------------------------------------------- #


class GestureController:
    """Real-time hand tracking and gesture classification.

    Wraps MediaPipe Hands, classifies five canonical gestures, and
    tracks swipe direction over a rolling history of hand positions.
    """

    GESTURE_NAMES = ("pinch", "fist", "open_palm", "swipe_right", "swipe_left")

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._hands: Any = None
        self._backend: str | None = None
        self._init_mediapipe()

        # Rolling history for swipe detection — (timestamp, x, y)
        self._pos_history: deque[tuple[float, float, float]] = deque(maxlen=20)
        self._last_gesture_time: float = 0.0
        self._current_gesture: str | None = None

    def _init_mediapipe(self) -> None:
        if not HAS_MEDIPIPE:
            logger.warning("mediapipe not installed — gesture recognition disabled")
            return

        if HAS_MP_SOLUTIONS:
            # ---- Legacy solutions API (mediapipe < 1.0) ----
            self._backend = "solutions"
            self._hands = mp.solutions.hands.Hands(
                min_detection_confidence=self.cfg.min_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
                max_num_hands=self.cfg.max_hands,
            )
            logger.info("MediaPipe Hands (solutions API) initialised (max_hands=%d)",
                        self.cfg.max_hands)
        else:
            # ---- New tasks API (mediapipe >= 1.0) ----
            self._backend = self._init_tasks_backend()
            if self._backend == "tasks":
                logger.info("MediaPipe HandLandmarker (tasks API) initialised (max_hands=%d)",
                            self.cfg.max_hands)

    # ------------------------------------------------------------------ #
    #  Tasks API backend (mediapipe >= 1.0)
    # ------------------------------------------------------------------ #

    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    _MODEL_FILENAME = "hand_landmarker.task"

    def _init_tasks_backend(self) -> str | None:
        """Initialise the MediaPipe *tasks* API backend.

        Returns ``"tasks"`` on success or ``None`` if the model asset
        could not be obtained (the detector is then gracefully disabled).
        """
        try:
            from mediapipe.tasks.python.core.base_options import (  # type: ignore[import-untyped]
                BaseOptions,
            )
            from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]
                HandLandmarker,
                HandLandmarkerOptions,
            )
        except ImportError:
            logger.warning("mediapipe.tasks API unavailable — gesture recognition disabled")
            return None

        # On Windows the mediapipe C DLL doesn't export ``free()`` —
        # it lives in the C runtime (ucrtbase.dll). Patch the binding
        # so HandLandmarker can initialise.
        _patch_windows_free()

        # Resolve the model file: look in project, then download.
        model_path = self._resolve_model_path()
        if model_path is None:
            return None

        try:
            opts = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                num_hands=self.cfg.max_hands,
                min_hand_detection_confidence=self.cfg.min_detection_confidence,
                min_hand_presence_confidence=self.cfg.min_tracking_confidence,
            )
            self._hands = HandLandmarker.create_from_options(opts)
            return "tasks"
        except Exception as exc:
            logger.warning("HandLandmarker init failed (%s) — gesture recognition disabled", exc)
            return None
        except ImportError:
            logger.warning("mediapipe.tasks API unavailable — gesture recognition disabled")
            return None

        # Resolve the model file: look in project, then download.
        model_path = self._resolve_model_path()
        if model_path is None:
            return None

        try:
            opts = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                num_hands=self.cfg.max_hands,
                min_hand_detection_confidence=self.cfg.min_detection_confidence,
                min_hand_presence_confidence=self.cfg.min_tracking_confidence,
            )
            self._hands = HandLandmarker.create_from_options(opts)
            return "tasks"
        except Exception as exc:
            logger.warning("HandLandmarker init failed (%s) — gesture recognition disabled", exc)
            return None

    def _resolve_model_path(self) -> Path | None:
        """Find or download the hand-landmarker ``.task`` model file."""
        # 1 — check explicit config / local paths
        candidates = [
            PROJECT_ROOT / "models" / self._MODEL_FILENAME,
            PROJECT_ROOT / "third_party" / self._MODEL_FILENAME,
        ]
        for c in candidates:
            if c.exists():
                return c

        # 2 — try downloading to the user cache dir
        cache_dir = Path.home() / ".cache" / "silverwing" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / self._MODEL_FILENAME
        if dest.exists():
            return dest

        logger.info("Downloading MediaPipe hand-landmarker model (~10 MB) ...")
        try:
            import urllib.request
            urllib.request.urlretrieve(self._MODEL_URL, dest)
            logger.info("Model downloaded to %s", dest)
            return dest
        except Exception as exc:
            logger.warning(
                "Could not download hand-landmarker model (%s).\n"
                "  Gesture recognition will be disabled.\n"
                "  Fix: install mediapipe<1.0 (pip install 'mediapipe>=0.10,<1.0')\n"
                "  or manually place hand_landmarker.task in models/",
                exc,
            )
            return None

    @property
    def available(self) -> bool:
        return self._hands is not None

    def __enter__(self) -> GestureController:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:  # pragma: no cover
                pass
            self._hands = None
        self._backend = None

    # -- public API ------------------------------------------------------

    # Predefined hand connections (index pairs) for drawing without the
    # solutions API — works with both mediapipe < 1.0 and >= 1.0.
    _HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
        # Wrist to pinky/thumb base
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index finger
        (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17),
    )

    def _draw_landmarks(self, frame: Any, hand_lms: Any) -> None:
        """Draw hand landmarks on the frame (fallback for missing solutions API)."""
        if not HAS_CV2:
            return
        h, w = frame.shape[:2]
        for lm in hand_lms.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        for a, b in self._HAND_CONNECTIONS:
            la, lb = hand_lms.landmark[a], hand_lms.landmark[b]
            cv2.line(frame,
                     (int(la.x * w), int(la.y * h)),
                     (int(lb.x * w), int(lb.y * h)),
                     (255, 255, 255), 1)

    def process(self, frame: Any) -> tuple[str | None, Any | None]:
        """Process a BGR frame and return ``(gesture, hand_landmarks)``.

        *gesture* is one of :attr:`GESTURE_NAMES` or ``None``.
        *hand_landmarks* is the primary hand's 21-point set (or ``None``).
        """
        if not self.available:
            return None, None

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if HAS_CV2 else frame

        # Dispatch through the appropriate backend (solutions vs tasks API)
        if self._backend == "solutions":
            results = self._hands.process(rgb)
            hand_list = results.multi_hand_landmarks
        else:
            # tasks API — wrap in mp.Image and call detect()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self._hands.detect(mp_image)
            hand_list = results.hand_landmarks

        if not hand_list:
            self._current_gesture = None
            self._pos_history.clear()
            return None, None

        # Use the first detected hand as the primary controller
        hand_lms = hand_list[0]
        # Normalise to a plain list so landmarks[i] works regardless of API
        landmarks = list(hand_lms.landmark)

        # Track position for swipe detection (index MCP — stable point)
        _now = time.monotonic()
        mcp = landmarks[HandLandmark.INDEX_FINGER_MCP]
        self._pos_history.append((_now, mcp.x, mcp.y))

        gesture = self._classify(landmarks)
        self._current_gesture = gesture
        return gesture, hand_lms

    def get_index_tip(self, hand_lms: Any) -> Any:
        """Return the index-finger-tip landmark (landmark #8)."""
        if hand_lms is None:
            return None
        return hand_lms.landmark[HandLandmark.INDEX_FINGER_TIP]

    # -- classification internals ---------------------------------------

    def _classify(self, landmarks: Any) -> str | None:
        """Classify the current hand pose into a named gesture.

        Priority order: pinch → swipe → open_palm → fist.
        Swipe is detected only if there has been enough horizontal
        motion; otherwise the static pose determines the gesture.
        """
        now = time.monotonic()

        # --- Pinch: thumb-tip (4) close to index-tip (8) ---
        if self._is_pinch(landmarks):
            self._set_gesture("pinch", now)
            return "pinch"

        # --- Swipe: enough history and dominant direction ---
        swipe = self._detect_swipe()
        if swipe:
            self._set_gesture(swipe, now)
            return swipe

        # --- Open palm: all fingers extended ---
        if self._is_open_palm(landmarks):
            self._set_gesture("open_palm", now)
            return "open_palm"

        # --- Fist: all fingers curled ---
        if self._is_fist(landmarks):
            self._set_gesture("fist", now)
            return "fist"

        return None

    def _set_gesture(self, gesture: str, now: float) -> None:
        self._current_gesture = gesture
        self._last_gesture_time = now

    def _is_pinch(self, landmarks: Any) -> bool:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        d = _dist(thumb_tip, index_tip)
        return d < self.cfg.pinch_distance_threshold

    def _is_open_palm(self, landmarks: Any) -> bool:
        fingers = [8, 12, 16, 20]   # tips
        dips  = [7, 11, 15, 19]     # DIP joints
        pips  = [6, 10, 14, 18]     # PIP joints
        return all(_is_finger_extended(landmarks, t, d, p) for t, d, p in zip(fingers, dips, pips))

    def _is_fist(self, landmarks: Any) -> bool:
        fingers = [8, 12, 16, 20]
        pips  = [6, 10, 14, 18]
        # Fingers are curled when tips are below PIP joints
        return all(landmarks[t].y > landmarks[p].y for t, p in zip(fingers, pips))

    def _detect_swipe(self) -> str | None:
        """Detect a left/right swipe from the rolling position history.

        Position samples are stored as ``(timestamp, x, y)`` tuples.
        A swipe is registered only when *all* of the following hold:

        * At least 5 history samples exist.
        * The motion occurred entirely within ``swipe_max_duration`` seconds
          (slow/dragging motion is ignored — only a genuine swipe counts).
        * The net horizontal travel exceeds ``swipe_min_distance``
          (a fraction of the normalised frame width).
        * Horizontal movement dominates the total path (≥ 70 %).
        """
        if len(self._pos_history) < 5:
            return None

        # Debounce — respect the global gesture cooldown
        if time.monotonic() - self._last_gesture_time < self.cfg.gesture_cooldown:
            return None

        # Enforce the maximum time window for a swipe gesture.
        # Samples outside this window should have already been evicted
        # by the bounded deque, but we check defensively.
        time_span = self._pos_history[-1][0] - self._pos_history[0][0]
        if time_span > self.cfg.swipe_max_duration:
            return None

        span = len(self._pos_history)
        if span < 2:
            return None

        # Extract normalised x positions (index 1 in each tuple)
        xs = [p[1] for p in self._pos_history]
        delta_x = xs[-1] - xs[0]

        # Total path length (frame-to-frame Euclidean distance)
        total_dist = sum(
            math.hypot(
                xs[i + 1] - xs[i],
                self._pos_history[i + 1][2] - self._pos_history[i][2],
            )
            for i in range(span - 1)
        )
        if total_dist < 1e-6:
            return None

        # Horizontal component must dominate
        if abs(delta_x) / total_dist < 0.7:
            return None

        if abs(delta_x) < self.cfg.swipe_min_distance:
            return None

        # Direction — positive delta_x means the hand moved rightward
        if delta_x > 0:
            return "swipe_right"
        return "swipe_left"


# --------------------------------------------------------------------------- #
#  2 — Object Detection (YOLOv8, background thread)
# --------------------------------------------------------------------------- #


class ObjectDetector:
    """Runs YOLOv8 object detection in a background thread.

    A single inference is triggered every ``detection_interval`` seconds
    and results are pushed to a thread-safe queue for the main loop to
    consume.
    """

    DEFAULT_CLASSES = {"car", "truck", "bus", "person", "airplane",
                       "boat", "bicycle", "motorcycle", "dog", "cat"}

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._results_queue: queue.Queue[list[dict[str, Any]]] = queue.Queue(maxsize=2)
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_frame: Any = None
        self._frame_lock = threading.Lock()
        self._available = False

        if not HAS_ULTRALYTICS:
            logger.warning("ultralytics not installed — object detection disabled")
            return

        try:
            logger.info("Loading YOLO model '%s' ...", cfg.yolo_model)
            self._model = YOLO(cfg.yolo_model)
            self._available = True
            logger.info("YOLOv8 model loaded")
        except Exception as exc:
            logger.warning("YOLO model load failed (%s) — detection disabled", exc)

    @property
    def available(self) -> bool:
        return self._available

    def set_frame(self, frame: Any) -> None:
        """Provide the latest frame for the next inference cycle."""
        if not self._available or frame is None:
            return
        with self._frame_lock:
            self._last_frame = frame

    def get_results(self) -> list[dict[str, Any]]:
        """Drain the results queue and return the most recent detections."""
        results: list[dict[str, Any]] = []
        while True:
            try:
                results = self._results_queue.get_nowait()
            except queue.Empty:
                break
        return results

    def start(self) -> None:
        if not self._available:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="YOLO-Detector", daemon=True)
        self._thread.start()
        logger.info("Object detection thread started")

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _loop(self) -> None:
        """Background inference loop."""
        while self._running.is_set():
            with self._frame_lock:
                frame = self._last_frame.copy() if self._last_frame is not None else None

            if frame is not None:
                try:
                    # Resize for speed
                    h, w = frame.shape[:2]
                    scale = min(640 / max(w, h), 1.0) if w > 0 else 1.0
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                    results = self._model(frame, conf=self.cfg.detection_confidence, verbose=False)[0]
                    detections: list[dict[str, Any]] = []

                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        label = self._model.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()  # x1, y1, x2, y2 (in resized frame)

                        # Scale bounding box back to original frame
                        if scale < 1.0:
                            xyxy = [v / scale for v in xyxy]

                        detections.append({
                            "label": label,
                            "confidence": conf,
                            "bbox": [int(xyxy[0]), int(xyxy[1]),
                                     int(xyxy[2]), int(xyxy[3])],
                        })

                    # Keep only the most recent result (drop stale)
                    try:
                        self._results_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._results_queue.put(detections)

                except Exception as exc:
                    logger.debug("YOLO inference error: %s", exc)

            time.sleep(self.cfg.detection_interval)


# --------------------------------------------------------------------------- #
#  3 — HUD / AR Overlay
# --------------------------------------------------------------------------- #


class HUDOverlay:
    """Renders a semi-transparent AR overlay on the video frame."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._visible = cfg.enable_hud
        self._font = cv2.FONT_HERSHEY_SIMPLEX if HAS_CV2 else None

    @property
    def available(self) -> bool:
        return self._visible and HAS_CV2

    def render(
        self,
        frame: Any,
        *,
        gesture: str | None = None,
        detections: list[dict[str, Any]] | None = None,
        stats: dict[str, str] | None = None,
    ) -> Any:
        """Draw the overlay onto *frame* and return the composited image."""
        if not self.available:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        # --- Top bar: SilverWing branding + system stats ----------------
        if self.cfg.hud_show_stats and stats:
            bar_h = 40
            overlay = cv2.rectangle(overlay, (0, 0), (w, bar_h),
                                    (0, 0, 0), -1)
            stat_str = "  ".join(f"{k}: {v}" for k, v in stats.items())
            cv2.putText(overlay, stat_str, (10, 25),
                        self._font, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(overlay, "SILVERWING HUB", (w - 160, 25),
                        self._font, 0.55, (255, 100, 0), 1, cv2.LINE_AA)

        # --- Left sidebar: detected objects ----------------------------
        if self.cfg.hud_show_objects and detections:
            obj_y = 60
            for det in detections:
                label = f"{det['label']} {det['confidence']:.2f}"
                cv2.putText(overlay, label, (10, obj_y),
                            self._font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                obj_y += 20

        # --- Target reticle (centre of frame) --------------------------
        cx, cy = w // 2, h // 2
        cv2.circle(overlay, (cx, cy), 20, (0, 255, 255), 1)
        cv2.line(overlay, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 1)
        cv2.line(overlay, (cx, cy - 30), (cx, cy + 30), (0, 255, 255), 1)

        # --- Active gesture indicator -----------------------------------
        if self.cfg.hud_show_gesture:
            if gesture:
                box_w = 140
                gx, gy = w - box_w - 10, h - 50
                overlay = cv2.rectangle(overlay, (gx, gy), (gx + box_w, gy + 35),
                                        (30, 30, 60), -1)
                overlay = cv2.rectangle(overlay, (gx, gy), (gx + box_w, gy + 35),
                                        (0, 255, 200), 2)
                cv2.putText(overlay, f"Gesture: {gesture.upper()}", (gx + 8, gy + 22),
                            self._font, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(overlay, "No gesture", (w - 120, h - 25),
                            self._font, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        # --- Bottom status strip ---------------------------------------
        status_h = 26
        overlay = cv2.rectangle(overlay, (0, h - status_h), (w, h),
                                (0, 0, 0), -1)
        status = "READY" if gesture is None else f"ACTIVE: {gesture}"
        cv2.putText(overlay, f"  {status}", (5, h - 7),
                    self._font, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(overlay, "Esc=quit  F=freeze  Space=click", (w // 2 - 80, h - 7),
                    self._font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        # Alpha blend
        alpha = self.cfg.hud_alpha
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# --------------------------------------------------------------------------- #
#  4 — System Controller (PyAutoGUI)
# --------------------------------------------------------------------------- #


class SystemController:
    """Translates gestures into real system actions via PyAutoGUI."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._available = HAS_PYAUTOGUI
        self._screen_w, self._screen_h = (0, 0)
        self._smoothed: tuple[float, float] | None = None

        if self._available:
            self._screen_w, self._screen_h = pyautogui.size()
            pyautogui.FAILSAFE = True
            # Move to centre on first frame to avoid accidental jumps
            pyautogui.moveTo(self._screen_w // 2, self._screen_h // 2)
            logger.info("PyAutoGUI ready — screen %dx%d", self._screen_w, self._screen_h)
        else:
            logger.warning("pyautogui not installed — system control disabled")

    @property
    def available(self) -> bool:
        return self._available

    # -- cursor ----------------------------------------------------------

    def move_cursor(self, landmark: Any) -> None:
        """Smoothly move the cursor to follow the index-finger tip."""
        if not self._available or landmark is None:
            return
        sx, sy = _screen_coords(landmark, self._screen_w, self._screen_h,
                                sensitivity=self.cfg.mouse_sensitivity)

        if self._smoothed is None:
            self._smoothed = (sx, sy)

        sf = self.cfg.smooth_factor
        nx = sf * self._smoothed[0] + (1 - sf) * sx
        ny = sf * self._smoothed[1] + (1 - sf) * sy
        self._smoothed = (nx, ny)
        pyautogui.moveTo(int(nx), int(ny), duration=0)

    def click(self) -> None:
        if self._available:
            pyautogui.click()
            logger.debug("Click executed")

    # -- gestures → actions ---------------------------------------------

    def execute_gesture(self, gesture: str) -> bool:
        """Map a recognised gesture to a system-level action.

        Returns ``True`` if the gesture was handled.
        """
        if not self._available:
            return False

        handler = {
            "pinch": self._handle_pinch,
            "fist": self._handle_fist,
            "open_palm": self._handle_open_palm,
            "swipe_right": self._handle_swipe_right,
            "swipe_left": self._handle_swipe_left,
        }.get(gesture)

        if handler:
            handler()
            return True
        return False

    # -- individual handlers --------------------------------------------

    def _handle_pinch(self) -> None:
        """Pinch = select / confirm (single click)."""
        if self.cfg.click_on_pinch:
            self.click()
        logger.info("Gesture: PINCH → click/select")

    def _handle_fist(self) -> None:
        """Fist = lock system (freeze all movement, minimise windows)."""
        logger.info("Gesture: FIST → lock system")
        if os.name == "nt":
            self._run("rundll32.exe user32.dll,LockWorkStation")
        else:
            self._run("loginctl lock-sessions")

    def _handle_open_palm(self) -> None:
        """Open palm = summon the main dashboard / HUD."""
        logger.info("Gesture: OPEN PALM → summon dashboard")
        if self.cfg.enable_gadgets and HAS_TK:
            self._spawn_gadget_window(
                title="SilverWing Dashboard",
                content=self._format_stats(get_system_stats()),
                width=320, height=180,
            )
        elif self.cfg.enable_system_control and self._available:
            # Fallback: toggle fullscreen via PyAutoGUI
            logger.info("Open palm → toggling fullscreen (no tkinter)")
            try:
                pyautogui.press("f11")
            except Exception:
                pass
        else:
            logger.warning("Open palm: no gadget window or system control available")

    def _handle_swipe_right(self) -> None:
        """Swipe right = open Google Maps / directions."""
        logger.info("Gesture: SWIPE RIGHT → open maps")
        self._open_url(self.cfg.maps_url)

    def _handle_swipe_left(self) -> None:
        """Swipe left = live data feed (weather / stocks)."""
        logger.info("Gesture: SWIPE LEFT → live data feed")
        self._open_url(self.cfg.live_data_url)

    # -- helpers ---------------------------------------------------------

    def _open_url(self, url: str) -> None:
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Failed to open URL %s: %s", url, exc)

    def _run(self, cmd: str) -> None:
        try:
            subprocess.Popen(cmd, shell=True)
        except Exception as exc:
            logger.warning("Command failed: %s — %s", cmd, exc)

    def _format_stats(self, stats: dict[str, str]) -> str:
        lines = ["SilverWing System Dashboard", "=" * 30]
        for k, v in stats.items():
            lines.append(f"  {k.upper():>6} : {v}")
        lines.append("=" * 30)
        lines.append("Built-in gesture: pinch = select | fist = lock")
        lines.append("open_palm = dashboard | swipe = maps/data")
        return "\n".join(lines)

    def _spawn_gadget_window(self, title: str, content: str,
                             width: int = 300, height: int = 200) -> None:
        """Create a borderless, always-on-top floating window (a 'gadget').

        The window's ``mainloop`` runs in a **daemon thread** so it never
        blocks the gesture-processing loop.
        """
        if not HAS_TK:
            return

        root = tk.Tk()
        root.title(title)
        root.geometry(f"{width}x{height}+20+20")
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", "white")  # click-through-ish
        root.configure(bg="white")

        text = tk.Text(root, bg="white", fg="black", font=("Consolas", 9),
                       wrap="word", relief="flat")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("1.0", content)
        text.config(state="disabled")

        root.after(int(self.cfg.gadget_refresh_interval * 1000),
                   lambda: self._refresh_gadget(root, text))

        # Run the Tk event loop in a daemon thread so it doesn't block
        # the main gesture pipeline.
        t = threading.Thread(target=root.mainloop, name="GadgetWindow", daemon=True)
        t.start()

    def _refresh_gadget(self, root: Any, text_widget: Any) -> None:
        """Periodically update the gadget window contents."""
        try:
            stats = get_system_stats()
            new_content = self._format_stats(stats)
            text_widget.config(state="normal")
            text_widget.delete("1.0", "end")
            text_widget.insert("1.0", new_content)
            text_widget.config(state="disabled")
        except Exception:
            pass
        root.after(int(self.cfg.gadget_refresh_interval * 1000),
                   lambda: self._refresh_gadget(root, text_widget))


# --------------------------------------------------------------------------- #
#  5 — IoT Bridge (WebSocket / MQTT)
# --------------------------------------------------------------------------- #


class IoTBridge:
    """Relay high-level commands to external devices (drones, robots, etc.).

    Supports two transports:

    * **websocket** — uses ``python-socketio`` to emit events to a
      SilverWing IoT gateway.
    * **mqtt** — uses ``paho-mqtt`` to publish to a topic.

    The bridge starts a background thread that reconnects automatically.
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._connected = False
        self._available = False
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: Any = None
        self._mqtt_client: Any = None

        if not cfg.enable_iot:
            logger.info("IoT bridge disabled by config")
            return

        protocol = cfg.iot_protocol.lower()
        if protocol == "websocket" and not HAS_SOCKETIO:
            logger.warning("socketio not installed — WebSocket IoT bridge disabled")
            return
        if protocol == "mqtt":
            try:
                import paho.mqtt.client as mqtt  # type: ignore[import-untyped]  # noqa: F401
            except ImportError:
                logger.warning("paho-mqtt not installed — MQTT IoT bridge disabled")
                return

        self._available = True
        self._connect_transport(protocol)

    @property
    def available(self) -> bool:
        return self._available

    def _connect_transport(self, protocol: str) -> None:
        """Open a background connection to the IoT gateway."""
        self._running.set()
        self._thread = threading.Thread(target=self._run, args=(protocol,),
                                        name="IoT-Bridge", daemon=True)
        self._thread.start()

    def _run(self, protocol: str) -> None:
        if protocol == "websocket" and HAS_SOCKETIO:
            self._run_socketio()
        elif protocol == "mqtt":
            self._run_mqtt()

    def _run_socketio(self) -> None:
        """WebSocket client loop using python-socketio."""
        while self._running.is_set():
            try:
                import socketio  # type: ignore[import-untyped]
                self._socket = socketio.Client()
                self._socket.connect(
                    f"http://{self.cfg.iot_host}:{self.cfg.iot_port}",
                    wait=True, timeout=5,
                )
                self._connected = True
                logger.info("IoT WebSocket bridge connected to %s:%d",
                            self.cfg.iot_host, self.cfg.iot_port)
                # Keep the socket alive
                while self._running.is_set():
                    self._socket.wait()
                    time.sleep(1)
            except Exception as exc:
                logger.debug("IoT WebSocket reconnect: %s", exc)
            finally:
                self._connected = False
                if self._socket:
                    try:
                        self._socket.disconnect()
                    except Exception:
                        pass
                    self._socket = None
            time.sleep(self.cfg.iot_reconnect_delay)

    def _run_mqtt(self) -> None:
        """MQTT client loop."""
        import paho.mqtt.client as mqtt  # type: ignore[import-untyped]

        def _on_connect(client, _userdata, _flags, rc):
            if rc == 0:
                self._connected = True
                logger.info("IoT MQTT bridge connected to %s:%d",
                            self.cfg.iot_host, self.cfg.iot_port)
            else:
                logger.warning("IoT MQTT connect failed (rc=%d)", rc)

        while self._running.is_set():
            try:
                self._mqtt_client = mqtt.Client()
                self._mqtt_client.on_connect = _on_connect
                self._mqtt_client.connect(self.cfg.iot_host, self.cfg.iot_port, 60)
                self._mqtt_client.loop_forever()
            except Exception as exc:
                logger.debug("IoT MQTT reconnect: %s", exc)
            finally:
                self._connected = False
            time.sleep(self.cfg.iot_reconnect_delay)

    def stop(self) -> None:
        self._running.clear()
        if self._socket:
            try:
                self._socket.disconnect()
            except Exception:
                pass
        if self._mqtt_client:
            try:
                self._mqtt_client.disconnect()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def send_command(self, command: str, payload: dict[str, Any] | None = None) -> bool:
        """Emit a high-level command to connected IoT devices."""
        if not self._connected:
            logger.warning("IoT bridge not connected — command '%s' dropped", command)
            return False

        event = {
            "command": command,
            "payload": payload or {},
            "timestamp": time.time(),
        }

        try:
            if self._socket is not None:
                self._socket.emit("command", event)
            elif self._mqtt_client is not None:
                topic = "silverwing/iot/command"
                self._mqtt_client.publish(topic, json.dumps(event))
            logger.info("IoT: %s → %s", command, event["payload"])
            return True
        except Exception as exc:
            logger.warning("IoT send failed: %s", exc)
            return False


# --------------------------------------------------------------------------- #
#  Main orchestrator
# --------------------------------------------------------------------------- #


class GestureOS:
    """Top-level orchestrator wiring all subsystems together.

    The main loop runs at ~30 FPS (configurable), reading frames from
    the webcam, running gesture recognition, optionally launching object
    detection, rendering the AR HUD, and dispatching system / IoT actions.
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._screen_w, self._screen_h = self._get_screen_size()

        # Initialise subsystems
        self.gesture = GestureController(cfg)
        self.detector = ObjectDetector(cfg) if cfg.enable_detection else None
        self.hud = HUDOverlay(cfg) if cfg.enable_hud else None
        self.system = SystemController(cfg) if cfg.enable_system_control else None
        self.iot = IoTBridge(cfg) if cfg.enable_iot else None

        self._cap: Any = None
        self._running = False
        self._freeze = False
        self._frame_time = 0.0

        # Cached system stats (refreshed every 2 s to avoid per-frame overhead)
        self._stats_cache: dict[str, str] = {}
        self._stats_last_update: float = 0.0

        # Gesture action callbacks (extensible)
        self.gesture_actions: dict[str, Callable[[], None]] = {}

    # -- utilities -------------------------------------------------------

    @staticmethod
    def _get_screen_size() -> tuple[int, int]:
        if HAS_PYAUTOGUI:
            try:
                return pyautogui.size()
            except Exception:
                pass
        return 1920, 1080

    def _open_camera(self) -> bool:
        if not HAS_CV2:
            logger.error("OpenCV (cv2) is required to open the camera")
            return False
        self._cap = cv2.VideoCapture(self.cfg.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        if not self._cap.isOpened():
            logger.error("Cannot open camera index %d", self.cfg.camera_index)
            return False
        logger.info("Camera opened — %dx%d @ %d",
                    self.cfg.frame_width, self.cfg.frame_height, self.cfg.camera_index)
        return True

    def _close_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- main loop -------------------------------------------------------

    def run(self) -> None:
        """Start the gesture-OS event loop."""
        if not HAS_CV2:
            logger.error("OpenCV is not installed.  Install with:  pip install opencv-python-headless")
            return
        if not self._open_camera():
            return
        if self.detector is not None:
            self.detector.start()

        self._running = True
        logger.info("=== SilverWing Gesture OS — ACTIVE ===")
        logger.info("Gestures: %s", ", ".join(GestureController.GESTURE_NAMES))

        try:
            while self._running:
                t0 = time.monotonic()
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    continue

                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                self._loop_frame(frame)

                if self.cfg.show_video_feed:
                    cv2.imshow("SilverWing HUD", frame)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)
                else:
                    time.sleep(0.01)

                # Maintain target ~30 FPS
                self._frame_time = time.monotonic() - t0
                target = 1 / 30
                remaining = target - self._frame_time
                if remaining > 0 and not self.cfg.show_video_feed:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    def _loop_frame(self, frame: Any) -> None:
        """Process one video frame through all subsystems."""
        h, w = frame.shape[:2]

        # 1 — Update object detector with current frame
        if self.detector is not None:
            self.detector.set_frame(frame)

        # 2 — Gesture recognition
        gesture, hand_lms = self.gesture.process(frame)

        # 3 — Mouse control (index-finger tip follows cursor)
        if self.system is not None and hand_lms is not None and not self._freeze:
            idx_tip = self.gesture.get_index_tip(hand_lms)
            if idx_tip is not None:
                self.system.move_cursor(idx_tip)

        # 4 — Dispatch gesture action
        if gesture is not None:
            handled = self.system.execute_gesture(gesture) if self.system else False
            # Forward to IoT: always for high-impact gestures, or as
            # fallback if the system controller couldn't handle it.
            if self.iot is not None:
                if gesture in ("swipe_right", "swipe_left", "open_palm") or not handled:
                    self.iot.send_command(gesture)

        # 5 — Get latest detections
        detections = []
        if self.detector is not None:
            detections = self.detector.get_results()

        # 6 — System stats (cached, refreshed every 2 s)
        if self.hud is not None and self.hud.available and self.cfg.hud_show_stats:
            now = time.monotonic()
            if now - self._stats_last_update > 2.0:
                self._stats_cache = get_system_stats()
                self._stats_last_update = now
        stats = self._stats_cache or None

        # 7 — Render AR overlay
        if self.hud is not None and self.hud.available:
            frame[:] = self.hud.render(
                frame, gesture=gesture, detections=detections, stats=stats,
            )

        # 8 — Draw hand landmarks (debug)
        if self.cfg.debug_landmarks and hand_lms is not None:
            self._draw_landmarks(frame, hand_lms)

    def _handle_key(self, key: int) -> None:
        """Handle keyboard input during the video loop."""
        if key == 27:  # Esc
            self._running = False
        elif key == ord("f") or key == ord("F"):
            self._freeze = not self._freeze
            logger.info("Freeze: %s", "ON" if self._freeze else "OFF")
        elif key == ord(" "):  # Space — manual click
            if self.system is not None:
                self.system.click()

    def shutdown(self) -> None:
        """Clean shutdown of all subsystems."""
        logger.info("Shutting down SilverWing Gesture OS ...")
        self._running = False
        if self.detector is not None:
            self.detector.stop()
        if self.iot is not None:
            self.iot.stop()
        if self.gesture is not None:
            self.gesture.close()
        self._close_camera()
        if HAS_CV2:
            cv2.destroyAllWindows()
        logger.info("All subsystems offline.  ✉  SilverWing signing off.")


# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #


def _list_gestures() -> None:
    """Print the gesture → action mapping table."""
    table = [
        ("pinch",        "Select / confirm (mouse click)"),
        ("fist",         "Lock system (freeze)"),
        ("open_palm",    "Summon main dashboard / HUD"),
        ("swipe_right",  "Open Google Maps / directions"),
        ("swipe_left",   "Open live data feed (weather/stocks)"),
    ]
    print(f"\n{'Gesture':<16} {'Action':<40}")
    print("─" * 56)
    for g, a in table:
        print(f"  {g:<14} {a}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="gesture_os",
        description="SilverWing Gesture OS — hand & vision control interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/gesture_os.py                              # run with defaults
  python scripts/gesture_os.py --camera 1 --width 1920      # custom camera
  python scripts/gesture_os.py --no-detection               # skip YOLO
  python scripts/gesture_os.py --no-iot                     # skip IoT bridge
  python scripts/gesture_os.py --list-gestures              # show mapping
""",
    )
    parser.add_argument("--config", default="configs/gesture.yaml",
                        help="YAML config file (default: configs/gesture.yaml)")
    parser.add_argument("--camera", type=int, default=None,
                        help="Webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=None,
                        help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=None,
                        help="Frame height (default: 720)")
    parser.add_argument("--no-detection", action="store_true",
                        help="Disable YOLOv8 object detection")
    parser.add_argument("--no-hud", action="store_true",
                        help="Disable AR overlay / HUD")
    parser.add_argument("--no-system-control", action="store_true",
                        help="Disable PyAutoGUI system control")
    parser.add_argument("--no-iot", action="store_true",
                        help="Disable IoT WebSocket / MQTT bridge")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable the cv2 video window (headless mode)")
    parser.add_argument("--debug-landmarks", action="store_true",
                        help="Draw raw MediaPipe hand landmarks on the feed")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--list-gestures", action="store_true",
                        help="Print gesture → action mapping and exit")

    args = parser.parse_args()

    if args.list_gestures:
        _list_gestures()
        return 0

    _setup_logging(verbose=args.verbose)

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.width is not None:
        cfg.frame_width = args.width
    if args.height is not None:
        cfg.frame_height = args.height
    cfg.enable_detection = cfg.enable_detection and not args.no_detection
    cfg.enable_hud = cfg.enable_hud and not args.no_hud
    cfg.enable_system_control = cfg.enable_system_control and not args.no_system_control
    cfg.enable_iot = cfg.enable_iot and not args.no_iot
    cfg.show_video_feed = cfg.show_video_feed and not args.no_video
    cfg.debug_landmarks = cfg.debug_landmarks or args.debug_landmarks

    logger.info("SilverWing Gesture OS — configuration:")
    logger.info("  Camera:            idx=%d, %dx%d",
                cfg.camera_index, cfg.frame_width, cfg.frame_height)
    logger.info("  Hand tracking:     %s", "OK" if HAS_MEDIPIPE else "MISSING")
    logger.info("  Object detection:  %s (model=%s)",
                "OK" if (cfg.enable_detection and HAS_ULTRALYTICS) else "disabled",
                cfg.yolo_model)
    logger.info("  HUD overlay:       %s", "OK" if HAS_CV2 else "MISSING")
    logger.info("  System control:    %s", "OK" if HAS_PYAUTOGUI else "MISSING")
    logger.info("  IoT bridge:        %s (%s)",
                cfg.iot_protocol if cfg.enable_iot else "disabled",
                "OK" if HAS_SOCKETIO or cfg.iot_protocol != "websocket" else "MISSING")
    logger.info("  Video feed:        %s", "ON" if cfg.show_video_feed else "OFF (headless)")

    # Warn about missing critical dependencies
    missing = []
    if not HAS_CV2:
        missing.append("opencv-python-headless")
    if not HAS_MEDIPIPE:
        missing.append("mediapipe")
    if missing:
        logger.error("Missing critical dependencies: %s", ", ".join(missing))
        logger.error("Install with: pip install %s", " ".join(missing))
        return 1

    # Launch
    gesture_os = GestureOS(cfg)
    gesture_os.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
