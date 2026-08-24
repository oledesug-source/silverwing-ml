"""Tests for the SilverWing platform frontend controller.

Verifies that the modular, server-rendered frontend built on the
``intelligence.webdev`` framework correctly:
  - Renders the dashboard template with partials.
  - Falls back to direct static URLs when no dist build exists.
  - Fingerprints assets when the dist build is present.
  - Serves static assets through the StaticFileServer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Importing sw_platform first resolves the circular import between
# silverwing_platform/__init__.py and sw_platform/__init__.py.
import sw_platform  # noqa: F401
from silverwing_platform.frontend import FrontendController
from silverwing_platform.frontend.controller import _CSS_FILES, _JS_FILES
from silverwing_platform.frontend.controller import _CSS_FILES, _JS_FILES


@pytest.fixture()
def controller(tmp_path: Path) -> FrontendController:
    """Build a controller rooted in the real templates/static dirs,
    but isolated through the real FrontendController defaults.
    """
    return FrontendController()


class _FakeCap:
    name = "test_tool"
    version = "1.2.3"
    description = "A test capability"
    risk_level = "low"
    tags = ["test"]
    timeout_seconds = 15
    input_schema = {"type": "object"}


# ------------------------------------------------------------------ #
# render_dashboard
# ------------------------------------------------------------------ #

def test_render_dashboard_returns_html(controller: FrontendController) -> None:
    html = controller.render_dashboard()
    assert html.startswith("<!DOCTYPE html>")
    assert "<html lang=\"en\">" in html
    assert "</html>" in html


def test_render_dashboard_includes_css_links(controller: FrontendController) -> None:
    html = controller.render_dashboard()
    for name in _CSS_FILES:
        # Fingerprinted names look like "tactical-base.<hash>.css"
        stem = name.split("/")[-1].rsplit(".css", 1)[0]
        assert stem in html


def test_render_dashboard_includes_js_scripts(controller: FrontendController) -> None:
    html = controller.render_dashboard()
    for name in _JS_FILES:
        stem = name.split("/")[-1].rsplit(".js", 1)[0]
        assert stem in html


def test_render_dashboard_with_platform_context(
    controller: FrontendController,
) -> None:
    from silverwing_platform.frontend.controller import PlatformContext

    ctx = PlatformContext(
        model_name="gpt-4",
        capability_count=5,
        clearance_level="L2",
        capabilities=[FrontendController.cap_to_summary(_FakeCap())],
        cap_json=json.dumps([{"name": "test_tool"}]),
    )
    html = controller.render_dashboard(ctx)
    assert "gpt-4" in html
    assert "5" in html
    assert "L2" in html


def test_render_dashboard_with_dict_context(
    controller: FrontendController,
) -> None:
    html = controller.render_dashboard({"model_name": "dict-model"})
    assert "dict-model" in html


# ------------------------------------------------------------------ #
# Asset fingerprinting
# ------------------------------------------------------------------ #

def test_fingerprint_url_fallback_with_empty_static(tmp_path: Path) -> None:
    """When source file doesn't exist, fingerprint_url returns the raw path."""
    static_dir = tmp_path / "static"
    templates_dir = tmp_path / "templates"
    static_dir.mkdir()
    templates_dir.mkdir()

    ctrl = FrontendController(
        templates_dir=templates_dir,
        static_dir=static_dir,
    )
    url = ctrl.fingerprint_url("css/nonexistent.css")
    assert url == "/static/css/nonexistent.css"


def test_fingerprint_url_with_manifest(
    controller: FrontendController, tmp_path: Path
) -> None:
    # Simulate a dist build by creating a fake dist entry
    dist_dir = controller.static_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    fake_name = "css/tactical-base.a1b2c3.css"
    (dist_dir / fake_name.split("/")[-1]).write_text("/* minified */")

    # Override manifest
    from silverwing_platform.frontend.controller import AssetEntry

    controller._asset_manifest = {  # noqa: SLF001
        "css/tactical-base.css": AssetEntry(
            name="css/tactical-base.css",
            key=fake_name,
            url="/static/dist/" + fake_name,
        )
    }
    url = controller.fingerprint_url("css/tactical-base.css")
    assert url == "/static/dist/css/tactical-base.a1b2c3.css"


def test_default_context_has_asset_urls(controller: FrontendController) -> None:
    ctx = controller._default_context()  # noqa: SLF001
    assert len(ctx["css_urls"]) == 4
    assert len(ctx["js_urls"]) == 8
    for url in ctx["css_urls"]:
        assert url.startswith("/static/")
    for url in ctx["js_urls"]:
        assert url.startswith("/static/")


def test_fingerprint_url_fallback_with_empty_static(tmp_path: Path) -> None:
    """When source file doesn't exist, fingerprint_url returns the raw path."""
    static_dir = tmp_path / "static"
    templates_dir = tmp_path / "templates"
    static_dir.mkdir()
    templates_dir.mkdir()

    ctrl = FrontendController(
        templates_dir=templates_dir,
        static_dir=static_dir,
    )
    url = ctrl.fingerprint_url("css/nonexistent.css")
    assert url == "/static/css/nonexistent.css"


# ------------------------------------------------------------------ #
# Static file serving
# ------------------------------------------------------------------ #

def test_static_file_server_serves_css(
    controller: FrontendController,
) -> None:
    server = controller._route_static  # noqa: SLF001
    body, status, headers = server("css/tactical-base.css")
    assert status == 200
    assert isinstance(body, (str, bytes))


# ------------------------------------------------------------------ #
# Capability helpers
# ------------------------------------------------------------------ #

def test_cap_to_summary_extracts_attributes() -> None:
    cap = _FakeCap()
    summary = FrontendController.cap_to_summary(cap)
    assert summary.name == "test_tool"
    assert summary.version == "1.2.3"
    assert summary.description == "A test capability"
    assert summary.risk_level == "low"
    assert summary.tags == ["test"]
    assert summary.timeout_seconds == 15


def test_caps_to_json_serialises() -> None:
    result = FrontendController.caps_to_json([_FakeCap()])
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["name"] == "test_tool"
    assert data[0]["input_schema"] == {"type": "object"}


# ------------------------------------------------------------------ #
# Router wiring
# ------------------------------------------------------------------ #

def test_router_has_dashboard_route(controller: FrontendController) -> None:
    paths = [r.path for r in controller.router._routes]  # noqa: SLF001
    assert "/" in paths


def test_router_has_static_route(controller: FrontendController) -> None:
    paths = [r.path for r in controller.router._routes]  # noqa: SLF001
    assert any("/static" in p for p in paths)
