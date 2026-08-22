"""Gesture OS capabilities for the Silverwing Platform.

Registers the gesture-processing features from ``scripts/gesture_os`` as
platform capabilities so they flow through the standard
propose → policy → permission → sandbox → audit lifecycle.

Every function degrades gracefully: if the GestureOS dependency chain
is not importable the capabilities are still registered (with a
descriptive error message returned at execution time) rather than
preventing the platform from booting.
"""

from __future__ import annotations

import logging
from typing import Any

from sw_platform.capabilities.schema import CapabilitySchema

logger = logging.getLogger("silverwing.gesture")

__all__ = [
    "GestureCapabilityProvider",
    "register_gesture_capabilities",
    "get_gesture_registry",
]


# ── static gesture mapping (mirrors _list_gestures in gesture_os.py) ──

GESTURE_ACTIONS: dict[str, str] = {
    "pinch": "Select / confirm (mouse click)",
    "fist": "Lock system (freeze all movement)",
    "open_palm": "Summon main dashboard / HUD",
    "swipe_right": "Open Google Maps / directions",
    "swipe_left": "Open live data feed (weather/stocks)",
}

GESTURE_RISK: dict[str, str] = {
    "pinch": "low",
    "fist": "medium",
    "open_palm": "medium",
    "swipe_right": "low",
    "swipe_left": "low",
}


class GestureCapabilityProvider:
    """Thin wrapper around ``scripts.gesture_os`` with lazy initialisation.

    All imports from the gesture module happen inside methods so that
    importing this module never fails even if optional dependencies
    (mediapipe, cv2, pyautogui, ultralytics) are missing.
    """

    def __init__(self) -> None:
        self._gesture_os: Any = None
        self._controller: Any = None
        self._detector: Any = None
        self._hud: Any = None
        self._system: Any = None
        self._iot: Any = None
        self._initialized: bool = False

    # ── lazy import ──

    def _ensure_imported(self) -> bool:
        """Import GestureOS and subsystems.  Returns ``True`` on success."""
        if self._initialized:
            return True
        try:
            # Import the gesture OS module
            import scripts.gesture_os as gesture_os  # type: ignore[import-not-found]
            self._gesture_os = gesture_os
            self._initialized = True
            return True
        except Exception as exc:
            logger.warning("GestureOS module not importable: %s", exc)
            self._initialized = False
            return False

    # ── public API ──

    def get_gesture_mapping(self) -> list[dict[str, str]]:
        """Return the static gesture → action mapping table."""
        return [
            {
                "gesture": g,
                "action": a,
                "risk_level": GESTURE_RISK.get(g, "low"),
            }
            for g, a in GESTURE_ACTIONS.items()
        ]

    def get_status(self) -> dict[str, Any]:
        """Return subsystem availability and configuration snapshot."""
        if not self._ensure_imported():
            return {
                "available": False,
                "reason": "gesture_os module not importable",
                "subsystems": {},
                "config": {},
            }

        g = self._gesture_os
        subsystems = {
            "mediapipe": {
                "available": g.HAS_MEDIPIPE,
                "backend": g.HAS_MP_SOLUTIONS
                and "solutions"
                or (g.HAS_MEDIPIPE and "tasks" or "missing"),
            },
            "opencv": g.HAS_CV2,
            "numpy": g.HAS_NUMPY,
            "pyautogui": g.HAS_PYAUTOGUI,
            "yolo": g.HAS_ULTRALYTICS,
            "socketio": g.HAS_SOCKETIO,
            "tkinter": g.HAS_TK,
        }

        cfg = g.GestureConfig()
        return {
            "available": True,
            "subsystems": subsystems,
            "config": cfg.to_dict(),
        }

    def get_system_stats(self) -> dict[str, str]:
        """Collect live system metrics (CPU, memory, battery, IP)."""
        if not self._ensure_imported():
            return {"error": "gesture_os not available"}
        try:
            return self._gesture_os.get_system_stats()
        except Exception as exc:
            return {"error": str(exc)}

    def send_iot_command(self, command: str = "", payload: str = "{}") -> str:
        """Send a command to the IoT bridge."""
        if not self._ensure_imported():
            return "IoT bridge unavailable — gesture_os not importable"
        try:
            import json as _json
            iot = self._gesture_os.IoTBridge(self._gesture_os.GestureConfig())
            if not iot.available:
                return "IoT bridge not available (WebSocket/MQTT libraries missing)"
            payload_dict = _json.loads(payload) if payload else {}
            success = iot.send_command(command, payload_dict)
            if success:
                return f"Command '{command}' sent successfully"
            return f"Command '{command}' dropped (bridge not connected)"
        except Exception as exc:
            return f"IoT error: {exc}"

    def execute_gesture(self, gesture: str = "") -> str:
        """Execute a gesture action through the SystemController."""
        if not self._ensure_imported():
            return f"Gesture OS not available — cannot execute '{gesture}'"
        if gesture not in GESTURE_ACTIONS:
            return f"Unknown gesture: {gesture}. Available: {list(GESTURE_ACTIONS)}"
        try:
            sys_ctrl = self._gesture_os.SystemController(
                self._gesture_os.GestureConfig()
            )
            if not sys_ctrl.available:
                return f"System control unavailable (pyautogui not installed) — gesture '{gesture}' noted"
            handled = sys_ctrl.execute_gesture(gesture)
            if handled:
                return f"Gesture '{gesture}' executed: {GESTURE_ACTIONS[gesture]}"
            return f"Gesture '{gesture}' not handled"
        except Exception as exc:
            return f"Gesture execution error: {exc}"

    def list_gestures(self) -> str:
        """Return a formatted text table of gestures (for CLI capability)."""
        lines = ["Gesture".ljust(16) + "Action".ljust(40)]
        lines.append("-" * 56)
        for g, a in GESTURE_ACTIONS.items():
            lines.append(f"  {g:<14} {a}")
        return "\n".join(lines)


def register_gesture_capabilities(registry: Any) -> GestureCapabilityProvider:
    """Register all GestureOS capabilities into the platform registry.

    Each capability carries:
    - A name, description, and input schema
    - A risk level and permission requirement that the policy engine
      uses to gate execution
    - Tags for discovery (e.g. ``gesture``, ``iot``, ``system``)

    Returns the ``GestureCapabilityProvider`` instance (also stored
    on the registry for status queries).
    """
    provider = GestureCapabilityProvider()

    # --- list gestures (static mapping, no deps needed) ---
    registry.register(
        CapabilitySchema(
            name="gesture_list",
            description="List all gesture-to-action mappings (pinch, fist, "
            "open_palm, swipe_right, swipe_left).",
            input_schema={},
            fn=provider.list_gestures,
            tags=["gesture", "info"],
            risk_level="low",
            permissions_required=["L0"],
            timeout_seconds=5,
        )
    )

    # --- gesture status (subsystem availability) ---
    registry.register(
        CapabilitySchema(
            name="gesture_status",
            description="Query subsystem availability: MediaPipe backend, "
            "OpenCV, YOLOv8, PyAutoGUI, IoT bridge, and current config.",
            input_schema={},
            fn=provider.get_status,
            tags=["gesture", "info", "status"],
            risk_level="low",
            permissions_required=["L0"],
            timeout_seconds=10,
        )
    )

    # --- system stats ---
    registry.register(
        CapabilitySchema(
            name="gesture_system_stats",
            description="Collect live system metrics: CPU, memory, battery, "
            "network IP.  Used by the HUD overlay and gadget window.",
            input_schema={},
            fn=provider.get_system_stats,
            tags=["gesture", "system", "stats"],
            risk_level="low",
            permissions_required=["L1"],
            timeout_seconds=5,
        )
    )

    # --- send IoT command ---
    registry.register(
        CapabilitySchema(
            name="iot_send_command",
            description="Relay a high-level command (e.g. lock_down, "
            "launch_drone) to external IoT devices via WebSocket/MQTT.",
            input_schema={
                "command": {"type": "string",
                            "description": "Command name (e.g. lock_down)"},
                "payload": {"type": "string",
                            "description": "JSON payload for the command"},
            },
            fn=provider.send_iot_command,
            tags=["gesture", "iot", "network"],
            risk_level="medium",
            permissions_required=["L2"],
            timeout_seconds=10,
        )
    )

    # --- execute gesture ---
    registry.register(
        CapabilitySchema(
            name="gesture_execute",
            description="Execute a gesture action through the SystemController "
            "(pinch=click, fist=lock, open_palm=dashboard, etc.).",
            input_schema={
                "gesture": {"type": "string",
                            "description": "Gesture name to execute"},
            },
            fn=provider.execute_gesture,
            tags=["gesture", "system", "control"],
            risk_level="high",
            permissions_required=["L2"],
            timeout_seconds=10,
        )
    )

    # Stash the provider on the registry for status queries
    registry._gesture_provider = provider  # type: ignore[attr-defined]
    return provider


def get_gesture_registry() -> dict[str, Any]:
    """Return a serialisable snapshot of gesture feature state.

    Used by REST endpoints and the UI to display the current wiring
    without executing any capability.
    """
    provider = GestureCapabilityProvider()
    status = provider.get_status()
    mapping = provider.get_gesture_mapping()
    return {
        "gestures": mapping,
        "status": status,
    }
