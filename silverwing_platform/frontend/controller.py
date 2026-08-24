"""Frontend controller for the SilverWing Tactical Command Interface.

Upgrades the legacy monolithic ``index.html`` to a modular, server-rendered
frontend powered by the project's own ``intelligence.webdev`` framework:

  - **TemplateEngine** — Jinja2-style templates with ``{% include %}``,
    ``{{ var }}``, ``{% if %}``, and ``{% for %}`` support.
  - **AssetPipeline** — CSS / JS concatenation, minification, and content
    fingerprinting for cache-busting.
  - **StaticFileServer** — MIME-type detection, ETag/Last-Modified headers,
    and 304 Not-Modified handling.
  - **Router** — typed URL routes with named reversal.

All public symbols are re-exported from ``silverwing_platform.frontend``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from intelligence.webdev.router import Route, Router
from intelligence.webdev.static_files import AssetPipeline, StaticFileServer
from intelligence.webdev.templates import TemplateEngine

__all__ = [
    "AssetEntry",
    "CapabilitySummary",
    "FrontendController",
    "PlatformContext",
]

_FRONTEND_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _FRONTEND_DIR / "templates"
_PARTIALS_DIR = _TEMPLATES_DIR / "partials"
_STATIC_DIR = _FRONTEND_DIR / "static"

_CSS_FILES: list[str] = [
    "css/tactical-base.css",
    "css/tactical-layout.css",
    "css/tactical-components.css",
    "css/tactical-ui.css",
]

_JS_FILES: list[str] = [
    "js/utils.js",
    "js/init.js",
    "js/capabilities.js",
    "js/chat.js",
    "js/tools.js",
    "js/audit.js",
    "js/gestures.js",
    "js/main.js",
]


@dataclass
class CapabilitySummary:
    """Minimal capability descriptor surfaced to the template layer."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    risk_level: str = "low"
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 30


@dataclass
class AssetEntry:
    """A fingerprinted static asset with its original and fingerprinted URLs."""

    name: str
    key: str
    url: str


@dataclass
class PlatformContext:
    """Typed context bag passed to the dashboard template.

    The template engine receives this as a flat dict (``asdict``),
    so attribute order and naming must match the template variables
    referenced by ``{% include %}`` partials and ``{{ }}`` tags.
    """

    static_prefix: str = "/static"
    model_name: str = "STANDBY"
    capability_count: int = 0
    clearance_level: str = "L0"
    max_rounds: int = 5
    adapt_enabled: bool = True
    cap_enabled: bool = True
    capabilities: list[CapabilitySummary] = field(default_factory=list)
    cap_json: str = "[]"
    theme: str = "tactical"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for the template engine."""
        ctx = asdict(self)
        ctx["adapt_text"] = "ON" if self.adapt_enabled else "OFF"
        ctx["adapt_class"] = "" if self.adapt_enabled else "off"
        return ctx


class FrontendController:
    """Render and serve the Tactical Command frontend.

    Encapsulates template loading, partial registration, asset
    fingerprinting, and static-file path resolution behind a single
    Pythonic interface so the FastAPI layer stays thin.

    Parameters
    ----------
    templates_dir
        Directory containing ``index.html`` and the ``partials/`` sub-folder.
    static_dir
        Directory containing ``css/`` and ``js/`` sub-folders.
    static_url_prefix
        URL prefix under which static assets are served by the host app.
    """

    def __init__(
        self,
        templates_dir: Path | str | None = None,
        static_dir: Path | str | None = None,
        static_url_prefix: str = "/static",
    ) -> None:
        self.templates_dir: Path = Path(templates_dir) if templates_dir else _TEMPLATES_DIR
        self.static_dir: Path = Path(static_dir) if static_dir else _STATIC_DIR
        self.static_url_prefix: str = static_url_prefix.rstrip("/")
        self._engine: TemplateEngine = TemplateEngine()
        self._partials_loaded: bool = False
        self._asset_manifest: dict[str, AssetEntry] = {}
        self._router: Router = Router()
        self._wire_routes()

    # ------------------------------------------------------------------
    # Partial registration
    # ------------------------------------------------------------------

    def _ensure_partials(self) -> None:
        """Load and register every ``_*.html`` partial from the partials dir."""
        if self._partials_loaded:
            return
        partials_path = self.templates_dir / "partials"
        if partials_path.is_dir():
            for partial_file in sorted(partials_path.iterdir()):
                if partial_file.is_file() and partial_file.suffix == ".html":
                    name = partial_file.stem
                    if name.startswith("_"):
                        name = name[1:]
                    source = partial_file.read_text(encoding="utf-8")
                    self._engine.add_partial(name, source)
        self._partials_loaded = True

    def _register_partial(self, name: str, source: str) -> None:
        """Register a single inline partial (used in tests)."""
        self._engine.add_partial(name, source)
        self._partials_loaded = True

    # ------------------------------------------------------------------
    # Asset pipeline
    # ------------------------------------------------------------------

    def build_asset_manifest(self) -> dict[str, AssetEntry]:
        """Run CSS and JS through the AssetPipeline, returning fingerprinted URLs.

        The result is cached on the controller instance; call again (or
        clear_cache first) when source files change during development.
        """
        if self._asset_manifest:
            return self._asset_manifest

        pipeline = AssetPipeline(output_dir=str(self.static_dir / "dist"))
        dist_dir = self.static_dir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)

        css_result: dict[str, str] = {}
        js_result: dict[str, str] = {}

        for name in _CSS_FILES:
            src = self.static_dir / name
            if src.exists():
                fingerprinted = pipeline.fingerprint(str(src))
                css_result[name] = fingerprinted

        for name in _JS_FILES:
            src = self.static_dir / name
            if src.exists():
                fingerprinted = pipeline.fingerprint(str(src))
                js_result[name] = fingerprinted

        for name, fp_name in {**css_result, **js_result}.items():
            entry = AssetEntry(
                name=name,
                key=fp_name,
                url=f"{self.static_url_prefix}/dist/{fp_name}",
            )
            self._asset_manifest[name] = entry

        return self._asset_manifest

    def fingerprint_url(self, asset_path: str) -> str:
        """Return the fingerprinted URL for a known asset, or the raw path."""
        full = asset_path.lstrip("/")
        manifest = self.build_asset_manifest()
        if full in manifest:
            return manifest[full].url
        return f"{self.static_url_prefix}/{asset_path}"

    def clear_cache(self) -> None:
        """Invalidate template partials, asset manifest, and engine cache."""
        self._partials_loaded = False
        self._asset_manifest.clear()
        self._engine._cache.clear()  # noqa: SLF001

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _default_context(self) -> dict[str, Any]:
        """Build the base template context with static asset URLs.

        Uses fingerprinted URLs from the asset pipeline when available,
        falling back to direct ``/static/<file>`` paths when the dist
        build has not yet run (e.g. during development or testing).
        """
        manifest = self.build_asset_manifest()
        css_urls: list[str] = []
        for name in _CSS_FILES:
            if name in manifest:
                css_urls.append(manifest[name].url)
            else:
                css_urls.append(f"{self.static_url_prefix}/{name}")
        js_urls: list[str] = []
        for name in _JS_FILES:
            if name in manifest:
                js_urls.append(manifest[name].url)
            else:
                js_urls.append(f"{self.static_url_prefix}/{name}")
        return {
            "static_prefix": self.static_url_prefix,
            "css_urls": css_urls,
            "js_urls": js_urls,
        }

    def render_dashboard(
        self,
        ctx: PlatformContext | dict[str, Any] | None = None,
    ) -> str:
        """Render the full dashboard HTML page.

        Parameters
        ----------
        ctx
            Either a :class:`PlatformContext` dataclass, a plain dict,
            or ``None`` for the default context.
        """
        self._ensure_partials()

        base = self._default_context()

        if ctx is None:
            merged: dict[str, Any] = {**base}
        elif isinstance(ctx, PlatformContext):
            merged = {**base, **ctx.to_dict()}
        else:
            merged = {**base, **ctx}

        index_path = self.templates_dir / "index.html"
        source = index_path.read_text(encoding="utf-8")
        return self._engine.render(source, merged)

    def render_partial(self, name: str, context: dict[str, Any] | None = None) -> str:
        """Render a single registered partial by name."""
        self._ensure_partials()
        self._engine.add_partial  # no-op, already loaded
        return self._engine._render(  # noqa: SLF001
            self._engine._partials.get(name, ""), dict(context or {})
        )

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def _wire_routes(self) -> None:
        """Register typed routes on the internal ``intelligence.webdev.Router``.

        These can be mounted by an ASGI/WSGI adapter or used directly by
        ``intelligence.networking.server.BaseHTTPServer``.
        """
        self._router.get("/", self._route_index, name="dashboard")
        self._router.get("/static/<path:filepath>", self._route_static, name="static")

    @property
    def router(self) -> Router:
        """The internal router (read-only access for inspection / testing)."""
        return self._router

    def _route_index(self, req: Any = None) -> tuple[str, int, dict[str, str]]:
        """Handler for ``GET /`` — returns (html_body, status, headers)."""
        try:
            html = self.render_dashboard()
            return html, 200, {"Content-Type": "text/html; charset=utf-8"}
        except Exception as exc:
            return f"<html><body>ERROR: {exc}</body></html>", 500, {"Content-Type": "text/html"}

    def _route_static(self, filepath: str, req: Any = None) -> tuple[bytes | str, int, dict[str, str]]:
        """Handler for ``GET /static/<path:filepath>`` — delegates to StaticFileServer."""
        server = StaticFileServer(root_dir=str(self.static_dir))
        result = server.serve("/" + filepath)
        body = result["body"]
        if isinstance(body, bytes):
            return body, result["status"], result["headers"]
        return str(body), result["status"], result["headers"]

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cap_to_summary(cap: Any) -> CapabilitySummary:
        """Extract a :class:`CapabilitySummary` from a registry capability object."""
        return CapabilitySummary(
            name=getattr(cap, "name", "?"),
            version=getattr(cap, "version", "1.0.0"),
            description=getattr(cap, "description", ""),
            risk_level=getattr(cap, "risk_level", "low"),
            tags=getattr(cap, "tags", []) or [],
            timeout_seconds=getattr(cap, "timeout_seconds", 30),
        )

    @staticmethod
    def caps_to_summaries(caps: list[Any]) -> list[CapabilitySummary]:
        """Map a list of registry capability objects to summaries."""
        return [FrontendController.cap_to_summary(c) for c in caps]

    @staticmethod
    def caps_to_json(caps: list[Any]) -> str:
        """Serialise capability objects to a JSON array string for inline JS."""
        import json

        summaries = [
            {
                "name": getattr(c, "name", "?"),
                "version": getattr(c, "version", "1.0.0"),
                "description": getattr(c, "description", ""),
                "risk_level": getattr(c, "risk_level", "low"),
                "tags": getattr(c, "tags", []) or [],
                "timeout_seconds": getattr(c, "timeout_seconds", 30),
                "input_schema": getattr(c, "input_schema", {}),
            }
            for c in caps
        ]
        return json.dumps(summaries, default=str, ensure_ascii=False)
