"""Unit tests for SilverWing gesture detection logic.

These tests exercise the *pure-Python* parts of ``gesture_os.py`` —
configuration parsing, distance helpers, finger-extension detection,
gesture classification (pinch, open-palm, fist, swipe), and the
swipe time-window guard.  No webcam or GPU is required.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import pytest

# Ensure the project root is on sys.path so ``scripts.gesture_os`` resolves
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gesture_os import (  # noqa: E402
    GestureConfig,
    HandLandmark,
    _dist,
    _is_finger_extended,
    _screen_coords,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# --------------------------------------------------------------------------- #
#  Mock landmark helpers
# --------------------------------------------------------------------------- #


@dataclass
class MockLandmark:
    """A simple stand-in for a MediaPipe NormalizedLandmark."""

    x: float
    y: float
    z: float = 0.0


def make_landmarks(coords: dict[int, tuple[float, float]]) -> list[MockLandmark]:
    """Build a 21-element landmark list from a {index: (x, y)} mapping.

    Unspecified landmarks default to (0, 0).
    """
    pts: list[MockLandmark] = []
    for i in range(21):
        x, y = coords.get(i, (0.5, 0.5))
        pts.append(MockLandmark(x=x, y=y))
    return pts


# --------------------------------------------------------------------------- #
#  _dist
# --------------------------------------------------------------------------- #


class TestDist:
    def test_zero_distance(self):
        a = MockLandmark(1.0, 2.0)
        assert _dist(a, a) == 0.0

    def test_horizontal(self):
        a = MockLandmark(0.0, 0.0)
        b = MockLandmark(3.0, 0.0)
        assert _dist(a, b) == pytest.approx(3.0)

    def test_vertical(self):
        a = MockLandmark(0.0, 0.0)
        b = MockLandmark(0.0, 4.0)
        assert _dist(a, b) == pytest.approx(4.0)

    def test_diagonal(self):
        a = MockLandmark(0.0, 0.0)
        b = MockLandmark(3.0, 4.0)
        assert _dist(a, b) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
#  _screen_coords
# --------------------------------------------------------------------------- #


class TestScreenCoords:
    def test_basic_mapping(self):
        lm = MockLandmark(0.5, 0.5)
        x, y = _screen_coords(lm, 1920, 1080)
        assert x == 960
        assert y == 540

    def test_clipping_to_screen(self):
        lm = MockLandmark(1.5, 2.0)
        x, y = _screen_coords(lm, 1920, 1080)
        assert x == 1919
        assert y == 1079

    def test_sensitivity_scaling(self):
        lm = MockLandmark(0.5, 0.5)
        x, y = _screen_coords(lm, 1920, 1080, sensitivity=2.0)
        # 0.5 * 1920 * 2 = 1920, clipped to 1919
        assert x == 1919
        assert y == 1079

    def test_origin(self):
        lm = MockLandmark(0.0, 0.0)
        x, y = _screen_coords(lm, 1920, 1080)
        assert x == 0
        assert y == 0


# --------------------------------------------------------------------------- #
#  _is_finger_extended
# --------------------------------------------------------------------------- #


class TestIsFingerExtended:
    def test_extended_finger(self):
        # tip above DIP above PIP, and tip above wrist
        lm = make_landmarks({
            0: (0.5, 0.8),   # wrist (low y)
            6: (0.5, 0.4),   # PIP
            7: (0.5, 0.3),   # DIP
            8: (0.5, 0.2),   # TIP (highest y — wait, lowest y means top of image)
        })
        # tip.y=0.2 < pip.y=0.4 ✓, tip.y=0.2 < dip.y=0.3 ✓, wrist.y-tip.y=0.8-0.2=0.6 > 0.02 ✓
        assert _is_finger_extended(lm, 8, 7, 6)

    def test_curl_finger_too_bent(self):
        lm = make_landmarks({
            0: (0.5, 0.8),   # wrist
            6: (0.5, 0.5),   # PIP
            7: (0.5, 0.6),   # DIP (below PIP — wrong direction)
            8: (0.5, 0.7),   # TIP (below DIP)
        })
        # tip.y=0.7 > pip.y=0.5 → not extended
        assert not _is_finger_extended(lm, 8, 7, 6)

    def test_tip_below_wrist(self):
        lm = make_landmarks({
            0: (0.5, 0.1),   # wrist (high up)
            6: (0.5, 0.5),   # PIP
            7: (0.5, 0.4),   # DIP
            8: (0.5, 0.9),   # TIP (way below wrist)
        })
        # tip.y=0.9, wrist.y=0.1 → wrist.y - tip.y = -0.8 < 0.02 → not extended
        assert not _is_finger_extended(lm, 8, 7, 6)


# --------------------------------------------------------------------------- #
#  GestureConfig
# --------------------------------------------------------------------------- #


class TestGestureConfig:
    def test_defaults(self):
        cfg = GestureConfig()
        assert cfg.camera_index == 0
        assert cfg.frame_width == 1280
        assert cfg.frame_height == 720
        assert cfg.swipe_max_duration == 1.5
        assert cfg.pinch_distance_threshold == 0.04
        assert cfg.swipe_min_distance == 0.15

    def test_to_dict_roundtrip(self):
        cfg = GestureConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["camera_index"] == 0
        assert len(d) == len(cfg.__dataclass_fields__)

    def test_from_dict_with_extra_keys(self):
        cfg = GestureConfig.from_dict({"camera_index": 1, "unknown_key": True})
        assert cfg.camera_index == 1

    def test_from_dict_partial(self):
        cfg = GestureConfig.from_dict({"frame_width": 1920, "frame_height": 1080})
        assert cfg.frame_width == 1920
        assert cfg.frame_height == 1080
        assert cfg.camera_index == 0  # default preserved

    def test_from_dict_empty(self):
        cfg = GestureConfig.from_dict({})
        assert cfg.camera_index == 0
        assert cfg.swipe_max_duration == 1.5


# --------------------------------------------------------------------------- #
#  Swipe detection
# --------------------------------------------------------------------------- #


class TestDetectSwipe:
    """Test _detect_swipe with timestamped position history."""

    def _make_controller(self, **kwargs) -> object:
        """Create a GestureController with a mock _hands (no MediaPipe needed)."""
        from scripts.gesture_os import GestureController
        cfg = GestureConfig(**kwargs) if kwargs else GestureConfig()
        gc = GestureController.__new__(GestureController)
        gc.cfg = cfg
        gc._pos_history = deque(maxlen=20)
        gc._last_gesture_time = 0.0
        gc._current_gesture = None
        gc._backend = None
        gc._hands = None  # no real backend needed for swipe tests
        return gc

    def test_no_history_returns_none(self):
        gc = self._make_controller()
        assert gc._detect_swipe() is None

    def test_too_few_samples(self):
        """Fewer than 5 samples should return None."""
        gc = self._make_controller()
        now = time.monotonic()
        for i in range(4):
            gc._pos_history.append((now + i * 0.1, 0.5 + i * 0.01, 0.5))
        assert gc._detect_swipe() is None

    def test_swipe_right_detected(self):
        """Hand moves rightward past swipe_min_distance within swipe_max_duration."""
        gc = self._make_controller()
        now = time.monotonic()
        # Simulate hand moving from x=0.3 to x=0.6 (delta_x=0.3 > 0.15)
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.3 + i * 0.06, 0.5))
        result = gc._detect_swipe()
        assert result == "swipe_right"

    def test_swipe_left_detected(self):
        """Hand moves leftward past swipe_min_distance within swipe_max_duration."""
        gc = self._make_controller()
        now = time.monotonic()
        # Simulate hand moving from x=0.6 to x=0.3 (delta_x=-0.3)
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.6 - i * 0.06, 0.5))
        result = gc._detect_swipe()
        assert result == "swipe_left"

    def test_swipe_too_slow_returns_none(self):
        """Motion spread over more than swipe_max_duration should be ignored."""
        gc = self._make_controller(swipe_max_duration=0.5)
        now = time.monotonic()
        # 6 samples spread over 2 seconds — exceeds 0.5s window
        for i in range(6):
            t = now + i * 0.4  # 2.0s total span
            gc._pos_history.append((t, 0.3 + i * 0.06, 0.5))
        assert gc._detect_swipe() is None

    def test_swipe_too_short_returns_none(self):
        """Horizontal distance below swipe_min_distance should be ignored."""
        gc = self._make_controller(swipe_min_distance=0.5)
        now = time.monotonic()
        # Only 0.1 travel — below threshold
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.5 + i * 0.02, 0.5))
        assert gc._detect_swipe() is None

    def test_cooldown_blocks_swipe(self):
        """If a gesture happened within cooldown, swipe should be suppressed."""
        gc = self._make_controller(gesture_cooldown=2.0)
        gc._last_gesture_time = time.monotonic()  # just now
        now = time.monotonic()
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.3 + i * 0.06, 0.5))
        assert gc._detect_swipe() is None

    def test_vertical_motion_not_swipe(self):
        """Purely vertical motion (delta_x = 0) should not register as a swipe."""
        gc = self._make_controller()
        now = time.monotonic()
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.5, 0.3 + i * 0.05))
        assert gc._detect_swipe() is None

    def test_diagonal_motion_dominantly_horizontal(self):
        """Horizontal > 70 % of total path should still register."""
        gc = self._make_controller()
        now = time.monotonic()
        # delta_x = 0.3, delta_y = 0.05 → 0.3 / 0.304 ≈ 0.98 > 0.7
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.3 + i * 0.06, 0.5 + i * 0.01))
        result = gc._detect_swipe()
        assert result == "swipe_right"

    def test_diagonal_motion_dominantly_vertical(self):
        """Vertical > 30 % of path should NOT register as horizontal swipe."""
        gc = self._make_controller()
        now = time.monotonic()
        # delta_x = 0.3, delta_y = 0.3 → 0.3 / 0.42 ≈ 0.71 > 0.7 — borderline
        # Use delta_y=0.4 to make it clearly vertical-dominant
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.3 + i * 0.06, 0.3 + i * 0.08))
        assert gc._detect_swipe() is None

    def test_timestamp_is_first_element(self):
        """Each history entry must be (timestamp, x, y)."""
        gc = self._make_controller()
        now = time.monotonic()
        gc._pos_history.append((now, 0.5, 0.5))
        entry = gc._pos_history[0]
        assert len(entry) == 3
        assert entry[0] == now  # timestamp is first
        assert entry[1] == 0.5  # x is second
        assert entry[2] == 0.5  # y is third


# --------------------------------------------------------------------------- #
#  Gesture classification (priority order)
# --------------------------------------------------------------------------- #


class TestClassify:
    """Test _classify priority: pinch > swipe > open_palm > fist."""

    def _make_controller(self) -> object:
        from scripts.gesture_os import GestureController
        gc = GestureController.__new__(GestureController)
        gc.cfg = GestureConfig()
        gc._pos_history = deque(maxlen=20)
        gc._last_gesture_time = 0.0
        gc._current_gesture = None
        gc._backend = None
        gc._hands = None
        return gc

    def test_pinch_takes_priority(self):
        """Even if position history shows a swipe, pinch should win."""
        gc = self._make_controller()
        now = time.monotonic()
        # Fill position history with a rightward swipe
        for i in range(6):
            t = now + i * 0.1
            gc._pos_history.append((t, 0.3 + i * 0.06, 0.5))

        # Pinch: thumb tip (4) close to index tip (8)
        lm = make_landmarks({
            0: (0.5, 0.5),  # wrist
            4: (0.5, 0.5),  # thumb tip — at same spot as index
            8: (0.5, 0.5),  # index tip — at same spot as thumb
            6: (0.5, 0.4),  # index PIP
            7: (0.5, 0.45), # index DIP
            12: (0.5, 0.8), # middle PIP (for fist detection — curled)
            11: (0.5, 0.8),  # middle DIP
            16: (0.5, 0.8),
            15: (0.5, 0.8),
            20: (0.5, 0.8),
            19: (0.5, 0.8),
        })
        result = gc._classify(lm)
        assert result == "pinch"

    def test_open_palm_takes_priority_over_fist(self):
        """If all fingers are extended, open_palm wins over fist."""
        gc = self._make_controller()
        # No position history → no swipe
        # All fingers extended → open_palm
        lm = make_landmarks({
            0: (0.5, 0.8),   # wrist (low)
            4: (0.3, 0.2),   # thumb tip
            5: (0.3, 0.3),   # index MCP
            6: (0.5, 0.5),   # index PIP
            7: (0.5, 0.35),  # index DIP
            8: (0.5, 0.2),   # index tip (above PIP and DIP, above wrist)
            10: (0.7, 0.5),  # middle PIP
            11: (0.7, 0.35), # middle DIP
            12: (0.7, 0.2),  # middle tip
            14: (0.8, 0.5),  # ring PIP
            15: (0.8, 0.35), # ring DIP
            16: (0.8, 0.2),  # ring tip
            18: (0.9, 0.5),  # pinky PIP
            19: (0.9, 0.35), # pinky DIP
            20: (0.9, 0.2),  # pinky tip
        })
        result = gc._classify(lm)
        assert result == "open_palm"

    def test_fist_when_fingers_curled(self):
        """All fingers curled → fist."""
        gc = self._make_controller()
        # Fingers curled: tips below PIPs
        lm = make_landmarks({
            0: (0.5, 0.8),  # wrist
            4: (0.3, 0.5),  # thumb tip — not extended (curled)
            8: (0.5, 0.85), # index tip below PIP
            6: (0.5, 0.7),  # index PIP
            7: (0.5, 0.8),  # index DIP
            12: (0.7, 0.85),# middle tip below PIP
            10: (0.7, 0.7), # middle PIP
            11: (0.7, 0.8), # middle DIP
            16: (0.8, 0.85),# ring tip below PIP
            14: (0.8, 0.7), # ring PIP
            15: (0.8, 0.8), # ring DIP
            20: (0.9, 0.85),# pinky tip below PIP
            18: (0.9, 0.7), # pinky PIP
            19: (0.9, 0.8), # pinky DIP
        })
        result = gc._classify(lm)
        assert result == "fist"

    def test_no_gesture_when_hand_not_recognized(self):
        """Hand in ambiguous position returns None."""
        gc = self._make_controller()
        # No pinch, no swipe, not all extended, not all curled
        lm = make_landmarks({
            0: (0.5, 0.8),
            4: (0.3, 0.5),  # thumb not touching index
            8: (0.5, 0.7),  # index tip below PIP (curled)
            6: (0.5, 0.5),  # index PIP
            12: (0.7, 0.7), # middle tip below PIP (curled)
            10: (0.7, 0.5), # middle PIP
            # Some fingers extended, some not → neither open_palm nor fist
        })
        result = gc._classify(lm)
        assert result is None


# --------------------------------------------------------------------------- #
#  _is_pinch (via GestureController)
# --------------------------------------------------------------------------- #


class TestIsPinch:
    def _make_controller(self, threshold: float = 0.04) -> object:
        from scripts.gesture_os import GestureController
        gc = GestureController.__new__(GestureController)
        gc.cfg = GestureConfig(pinch_distance_threshold=threshold)
        gc._pos_history = deque(maxlen=20)
        gc._last_gesture_time = 0.0
        gc._current_gesture = None
        gc._backend = None
        gc._hands = None
        return gc

    def test_pinch_close(self):
        gc = self._make_controller(threshold=0.04)
        lm = make_landmarks({4: (0.5, 0.5), 8: (0.5, 0.5)})
        assert gc._is_pinch(lm)

    def test_not_pinch_far(self):
        gc = self._make_controller(threshold=0.04)
        lm = make_landmarks({4: (0.0, 0.0), 8: (0.5, 0.5)})
        d = _dist(lm[4], lm[8])
        assert d > 0.04
        assert not gc._is_pinch(lm)


# --------------------------------------------------------------------------- #
#  _is_open_palm (via GestureController)
# --------------------------------------------------------------------------- #


class TestIsOpenPalm:
    def _make_controller(self) -> object:
        from scripts.gesture_os import GestureController
        gc = GestureController.__new__(GestureController)
        gc.cfg = GestureConfig()
        gc._pos_history = deque(maxlen=20)
        gc._last_gesture_time = 0.0
        gc._current_gesture = None
        gc._backend = None
        gc._hands = None
        return gc

    def test_all_extended(self):
        gc = self._make_controller()
        lm = make_landmarks({
            0: (0.5, 0.8),   # wrist
            6: (0.5, 0.5), 7: (0.5, 0.45), 8: (0.5, 0.2),   # index
            10: (0.7, 0.5), 11: (0.7, 0.45), 12: (0.7, 0.2), # middle
            14: (0.8, 0.5), 15: (0.8, 0.45), 16: (0.8, 0.2), # ring
            18: (0.9, 0.5), 19: (0.9, 0.45), 20: (0.9, 0.2), # pinky
        })
        assert gc._is_open_palm(lm)

    def test_one_curled(self):
        gc = self._make_controller()
        lm = make_landmarks({
            0: (0.5, 0.8),   # wrist
            6: (0.5, 0.5), 7: (0.5, 0.45), 8: (0.5, 0.2),   # index — extended
            10: (0.7, 0.5), 11: (0.7, 0.65), 12: (0.7, 0.7), # middle — curled
            14: (0.8, 0.5), 15: (0.8, 0.45), 16: (0.8, 0.2), # ring — extended
            18: (0.9, 0.5), 19: (0.9, 0.45), 20: (0.9, 0.2), # pinky — extended
        })
        assert not gc._is_open_palm(lm)


# --------------------------------------------------------------------------- #
#  _is_fist (via GestureController)
# --------------------------------------------------------------------------- #


class TestIsFist:
    def _make_controller(self) -> object:
        from scripts.gesture_os import GestureController
        gc = GestureController.__new__(GestureController)
        gc.cfg = GestureConfig()
        gc._pos_history = deque(maxlen=20)
        gc._last_gesture_time = 0.0
        gc._current_gesture = None
        gc._backend = None
        gc._hands = None
        return gc

    def test_all_curled(self):
        gc = self._make_controller()
        # Tips below PIPs (y increases downward in image coords)
        lm = make_landmarks({
            8: (0.5, 0.9), 6: (0.5, 0.7),   # index
            12: (0.7, 0.9), 10: (0.7, 0.7),  # middle
            16: (0.8, 0.9), 14: (0.8, 0.7),  # ring
            20: (0.9, 0.9), 18: (0.9, 0.7),  # pinky
        })
        assert gc._is_fist(lm)

    def test_not_fist_when_extended(self):
        gc = self._make_controller()
        lm = make_landmarks({
            8: (0.5, 0.2), 6: (0.5, 0.5),   # index — extended (tip above PIP)
            12: (0.7, 0.9), 10: (0.7, 0.7),  # middle — curled
            16: (0.8, 0.9), 14: (0.8, 0.7),  # ring — curled
            20: (0.9, 0.9), 18: (0.9, 0.7),  # pinky — curled
        })
        assert not gc._is_fist(lm)


# --------------------------------------------------------------------------- #
#  HandLandmark constants
# --------------------------------------------------------------------------- #


class TestHandLandmarkConstants:
    def test_all_21_landmarks_present(self):
        indices = [
            HandLandmark.WRIST,
            HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP,
            HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP,
            HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP,
            HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP,
        ]
        assert len(indices) == 21
        assert sorted(indices) == list(range(21))

    def test_thmb_tip_is_4(self):
        assert HandLandmark.THUMB_TIP == 4

    def test_index_tip_is_8(self):
        assert HandLandmark.INDEX_FINGER_TIP == 8

    def test_index_mcp_is_5(self):
        assert HandLandmark.INDEX_FINGER_MCP == 5


# --------------------------------------------------------------------------- #
#  load_config with YAML
# --------------------------------------------------------------------------- #


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path):
        from scripts.gesture_os import load_config
        cfg_file = tmp_path / "gesture.yaml"
        cfg_file.write_text(
            "camera_index: 1\n"
            "frame_width: 1920\n"
            "swipe_max_duration: 2.0\n"
            "unknown_field: true\n",
            encoding="utf-8",
        )
        cfg = load_config(str(cfg_file))
        assert cfg.camera_index == 1
        assert cfg.frame_width == 1920
        assert cfg.swipe_max_duration == 2.0
        # Unknown fields are silently ignored
        assert not hasattr(cfg, "unknown_field")

    def test_load_missing_file(self, tmp_path):
        from scripts.gesture_os import load_config
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg.camera_index == 0  # defaults preserved
        assert cfg.swipe_max_duration == 1.5

    def test_load_none_returns_defaults(self):
        from scripts.gesture_os import load_config
        cfg = load_config(None)
        assert isinstance(cfg, GestureConfig)
        assert cfg.camera_index == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
