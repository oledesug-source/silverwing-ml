"""
Static file serving, asset pipeline, and cache header management.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

__all__ = [
    "StaticFileServer",
    "AssetPipeline",
    "CacheHeaders",
]

MIME_TYPES: dict[str, str] = {
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".json": "application/json",
    ".xml": "application/xml",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".eot": "application/vnd.ms-fontobject",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".gz": "application/gzip",
    ".wasm": "application/wasm",
    ".map": "application/json",
    ".py": "text/x-python",
    ".rs": "text/plain",
    ".go": "text/plain",
    ".java": "text/plain",
    ".kt": "text/plain",
    ".c": "text/plain",
    ".cpp": "text/plain",
    ".h": "text/plain",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".conf": "text/plain",
    ".env": "text/plain",
}


class CacheHeaders:
    """HTTP cache header generation and validation for static resources."""

    def __init__(self, max_age: int = 3600, use_etag: bool = True) -> None:
        self.max_age = max_age
        self.use_etag = use_etag

    def generate_etag(self, content: bytes) -> str:
        """Generate an ETag from content hash."""
        return f'"{hashlib.md5(content).hexdigest()}"'

    def last_modified(self, file_path: str) -> str:
        """Get the last modified time of a file in HTTP format."""
        try:
            mtime = os.path.getmtime(file_path)
            return time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(mtime))
        except OSError:
            return time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())

    def apply_headers(self, headers: dict[str, str], content: bytes, file_path: str) -> dict[str, str]:
        """Apply cache-related headers to a response headers dict."""
        if self.use_etag:
            headers["ETag"] = self.generate_etag(content)
        headers["Last-Modified"] = self.last_modified(file_path)
        headers["Cache-Control"] = f"public, max-age={self.max_age}"
        return headers

    def check_not_modified(self, request_headers: dict[str, str], content: bytes, file_path: str) -> bool:
        """Check if the content matches If-None-Match or If-Modified-Since."""
        if_none_match = request_headers.get("If-None-Match", "")
        if if_none_match and self.use_etag:
            etag = self.generate_etag(content)
            if if_none_match == etag:
                return True
        if_modified_since = request_headers.get("If-Modified-Since", "")
        if if_modified_since:
            last_mod = self.last_modified(file_path)
            if if_modified_since == last_mod:
                return True
        return False


class StaticFileServer:
    """Serve static files from a directory with MIME detection and caching."""

    def __init__(self, root_dir: str = "static", index_file: str = "index.html") -> None:
        self.root_dir = Path(root_dir)
        self.index_file = index_file
        self.cache = CacheHeaders()
        self._directory_listing = True

    def serve(self, path: str, request_headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Serve a static file, returning status, content, and headers."""
        clean_path = path.lstrip("/")
        if not clean_path:
            clean_path = self.index_file
        file_path = self.root_dir / clean_path
        if not file_path.exists() or not file_path.is_file():
            if file_path.is_dir():
                return self._serve_directory(file_path)
            return {"status": 404, "body": b"File not found", "headers": {"Content-Type": "text/plain"}}
        try:
            content = file_path.read_bytes()
        except PermissionError:
            return {"status": 403, "body": b"Forbidden", "headers": {"Content-Type": "text/plain"}}
        if request_headers and self.cache.check_not_modified(request_headers, content, str(file_path)):
            return {"status": 304, "body": b"", "headers": {}}
        mime_type = self._detect_mime(str(file_path))
        headers: dict[str, str] = {"Content-Type": mime_type, "Content-Length": str(len(content))}
        headers = self.cache.apply_headers(headers, content, str(file_path))
        return {"status": 200, "body": content, "headers": headers}

    def _detect_mime(self, file_path: str) -> str:
        """Detect MIME type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in MIME_TYPES:
            return MIME_TYPES[ext]
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    def _serve_directory(self, dir_path: Path) -> dict[str, Any]:
        """Generate a directory listing page."""
        if not self._directory_listing:
            return {"status": 403, "body": b"Forbidden", "headers": {"Content-Type": "text/plain"}}
        entries = sorted(os.listdir(dir_path))
        html_parts = [
            "<!DOCTYPE html><html><head><title>Directory Listing</title></head><body>",
            f"<h1>Directory: {dir_path.name}</h1><ul>",
        ]
        for entry in entries:
            entry_path = dir_path / entry
            suffix = "/" if entry_path.is_dir() else ""
            html_parts.append(f'<li><a href="{entry}{suffix}">{entry}{suffix}</a></li>')
        html_parts.append("</ul></body></html>")
        body = "\n".join(html_parts).encode("utf-8")
        return {"status": 200, "body": body, "headers": {"Content-Type": "text/html; charset=utf-8"}}


class AssetPipeline:
    """Asset pipeline for concatenation, minification, and fingerprinting."""

    def __init__(self, output_dir: str = "static/dist") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def concat(self, files: list[str], output_name: str) -> str:
        """Concatenate multiple files into a single output file."""
        combined = b""
        for f in files:
            path = Path(f)
            if path.exists():
                combined += path.read_bytes()
                combined += b"\n"
        output_path = self.output_dir / output_name
        output_path.write_bytes(combined)
        return str(output_path)

    def minify(self, content: str, content_type: str = "css") -> str:
        """Basic minification by stripping whitespace and comments."""
        if content_type == "css":
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
            content = re.sub(r"\s+", " ", content)
            content = re.sub(r"\s*([{}:;,])\s*", r"\1", content)
            content = content.strip()
        elif content_type == "js":
            content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
            content = re.sub(r"\s+", " ", content)
            content = re.sub(r"\s*([{}:;,=+\-<>!&|])\s*", r"\1", content)
            content = content.strip()
        elif content_type == "html":
            content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
            content = re.sub(r">\s+<", "> <", content)
            content = re.sub(r"\s+", " ", content)
            content = re.sub(r"\s*\n\s*", "", content)
            content = content.strip()
        return content

    def fingerprint(self, file_path: str) -> str:
        """Generate a fingerprinted filename based on content hash."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        content = path.read_bytes()
        content_hash = hashlib.md5(content).hexdigest()[:8]
        ext = path.suffix
        name = path.stem
        fingerprinted_name = f"{name}.{content_hash}{ext}"
        output_path = self.output_dir / fingerprinted_name
        output_path.write_bytes(content)
        return fingerprinted_name

    def process(self, files: list[str], minify: bool = True, fingerprint: bool = True) -> dict[str, str]:
        """Process a list of files through the pipeline, returning a map of original to processed paths."""
        results: dict[str, str] = {}
        for f in files:
            path = Path(f)
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            ext = path.suffix.lstrip(".")
            if minify and ext in ("css", "js", "html"):
                content = self.minify(content, ext)
            output_name = path.name
            if fingerprint:
                output_name = self.fingerprint(f)
            output_path = self.output_dir / output_name
            if minify:
                output_path.write_text(content, encoding="utf-8")
            else:
                output_path.write_bytes(path.read_bytes())
            results[f] = str(output_path)
        return results
