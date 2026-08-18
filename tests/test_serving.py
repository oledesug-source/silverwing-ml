"""Tests for serving layer (api + runtime)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from serving.api import ApiResponse, SilverwingHandler
from serving.runtime import Runtime, RuntimeConfig


def test_api_response_success():
    resp = ApiResponse(success=True, data={"key": "value"})
    body = resp.to_json()
    parsed = json.loads(body)
    assert parsed["success"] is True
    assert parsed["data"]["key"] == "value"
    assert parsed["error"] == ""


def test_api_response_error():
    resp = ApiResponse(success=False, error="bad request")
    parsed = json.loads(resp.to_json())
    assert parsed["success"] is False
    assert parsed["error"] == "bad request"


def test_api_response_empty():
    resp = ApiResponse(success=True)
    parsed = json.loads(resp.to_json())
    assert parsed["success"] is True
    assert parsed["data"] is None


def test_runtime_config_defaults():
    cfg = RuntimeConfig()
    assert cfg.device == "cpu"
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.max_new_tokens == 128


def test_runtime_config_to_dict():
    cfg = RuntimeConfig(port=9000, device="cuda")
    d = cfg.to_dict()
    assert d["port"] == 9000
    assert d["device"] == "cuda"


def test_runtime_config_from_yaml(tmp_path: Path):
    cfg_path = tmp_path / "serving.yaml"
    cfg_path.write_text(
        "serving:\n  port: 9000\n  host: 127.0.0.1\n",
        encoding="utf-8",
    )
    cfg = RuntimeConfig.from_yaml(cfg_path)
    assert cfg.port == 9000
    assert cfg.host == "127.0.0.1"


def test_runtime_not_loaded():
    rt = Runtime(RuntimeConfig())
    assert not rt.is_loaded
    assert rt.health()["status"] == "no_model"


def test_runtime_generator_raises_before_load():
    rt = Runtime(RuntimeConfig())
    try:
        _ = rt.generator
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()


def test_runtime_health():
    rt = Runtime(RuntimeConfig())
    h = rt.health()
    assert h["loaded"] is False
    assert h["status"] == "no_model"
    assert "config" in h


def test_runtime_unload():
    rt = Runtime(RuntimeConfig())
    rt.unload()
    assert not rt.is_loaded
