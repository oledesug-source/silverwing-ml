"""Evaluation report writing.

Writes a reproducible evaluation_report.json capturing model, benchmark,
metrics, git commit, environment and config hash — the artifact a promotion
decision requires (foundation.yaml evaluation requirements).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .runner import EvalResult


def _environment() -> dict:
    torch_version: str | None = None
    try:
        from importlib import import_module

        torch = import_module("torch")
        torch_version = torch.__version__
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": f"{sys.platform}",
        "torch": torch_version,
    }


def write_evaluation_report(
    result: EvalResult,
    output_dir: str | Path,
    config_digest: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write result.to_dict() plus environment info as evaluation_report.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["environment"] = _environment()
    payload["config_digest"] = config_digest
    payload["written_at"] = datetime.now(UTC).isoformat()
    if extra:
        payload["extra"] = extra
    path = output_dir / "evaluation_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
