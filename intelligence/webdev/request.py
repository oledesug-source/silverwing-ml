"""
HTTP Request and Response objects for the web framework.
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import parse_qs

__all__ = [
    "Headers",
    "QueryParams",
    "Request",
    "Response",
]


class Headers:
    """Dictionary-like container for HTTP headers with case-insensitive access."""

    def __init__(self, raw_headers: list[tuple[str, str]] | None = None) -> None:
        self._store: dict[str, str] = {}
        self._list_store: dict[str, list[str]] = {}
        if raw_headers:
            for name, value in raw_headers:
                lower = name.lower()
                self._store[lower] = value
                self._list_store.setdefault(lower, []).append(value)

    def get(self, key: str, default: str = "") -> str:
        return self._store.get(key.lower(), default)

    def getlist(self, key: str) -> list[str]:
        return self._list_store.get(key.lower(), [])

    def items(self) -> list[tuple[str, str]]:
        return list(self._store.items())

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._store

    def __getitem__(self, key: str) -> str:
        return self._store[key.lower()]

    def __setitem__(self, key: str, value: str) -> None:
        self._store[key.lower()] = value

    def __iter__(self):
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)


class QueryParams:
    """Parsed URL query string parameters."""

    def __init__(self, raw: str = "") -> None:
        self._raw = raw
        self._parsed: dict[str, list[str]] = parse_qs(raw) if raw else {}

    def get(self, key: str, default: str = "") -> str:
        values = self._parsed.get(key, [])
        return values[0] if values else default

    def getlist(self, key: str) -> list[str]:
        return self._parsed.get(key, [])

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, values in self._parsed.items():
            result[key] = values[0] if len(values) == 1 else values
        return result

    def __contains__(self, key: str) -> bool:
        return key in self._parsed

    def __getitem__(self, key: str) -> str:
        values = self._parsed.get(key, [])
        if not values:
            raise KeyError(key)
        return values[0]


@dataclass
class Request:
    """HTTP request object encapsulating method, URL, headers, body, and parsed data."""

    method: str = "GET"
    url: str = "/"
    path: str = "/"
    query_params: QueryParams = field(default_factory=QueryParams)
    headers: Headers = field(default_factory=Headers)
    body: bytes = b""
    content_type: str = ""
    remote_addr: str = "127.0.0.1"
    user_agent: str = ""
    _cookies: dict[str, str] | None = field(default=None, repr=False)

    def json(self) -> dict[str, Any]:
        """Parse the request body as JSON."""
        if not self.body:
            return {}
        return _json.loads(self.body.decode("utf-8"))

    def form(self) -> dict[str, str]:
        """Parse URL-encoded form data."""
        if not self.body:
            return {}
        return parse_qs(self.body.decode("utf-8"), keep_blank_values=True)

    def files(self) -> dict[str, Any]:
        """Parse multipart form data (simplified parser)."""
        if not self.body:
            return {}
        if "multipart/form-data" not in self.content_type:
            return {}
        boundary = ""
        match = re.search(r"boundary=(.+?)(?:;|$)", self.content_type)
        if match:
            boundary = match.group(1).strip()
        if not boundary:
            return {}
        return _parse_multipart(self.body, boundary)

    def cookies(self) -> dict[str, str]:
        """Parse cookies from the Cookie header."""
        if self._cookies is not None:
            return self._cookies
        cookie_header = self.headers.get("Cookie", "")
        self._cookies = {}
        if cookie_header:
            for part in cookie_header.split(";"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    self._cookies[k.strip()] = v.strip()
        return self._cookies

    def is_ajax(self) -> bool:
        """Check if this is an AJAX (XMLHttpRequest) request."""
        return self.headers.get("X-Requested-With", "").lower() == "xmlhttprequest"


@dataclass
class Response:
    """HTTP response object with status code, body, headers, and convenience constructors."""

    status_code: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str = "text/plain"

    @classmethod
    def json(cls, data: Any, status: int = 200) -> Response:
        """Create a JSON response."""
        body = _json.dumps(data, default=str).encode("utf-8")
        return cls(
            status_code=status,
            body=body,
            content_type="application/json",
            headers={"Content-Type": "application/json"},
        )

    @classmethod
    def html(cls, content: str, status: int = 200) -> Response:
        """Create an HTML response."""
        body = content.encode("utf-8")
        return cls(
            status_code=status,
            body=body,
            content_type="text/html",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    @classmethod
    def text(cls, content: str, status: int = 200) -> Response:
        """Create a plain text response."""
        body = content.encode("utf-8")
        return cls(
            status_code=status,
            body=body,
            content_type="text/plain",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

    @classmethod
    def redirect(cls, url: str, status: int = 302) -> Response:
        """Create a redirect response."""
        return cls(
            status_code=status,
            body=b"",
            headers={"Location": url, "Content-Type": "text/plain"},
            content_type="text/plain",
        )

    def set_cookie(self, name: str, value: str, **kwargs: Any) -> None:
        """Set a cookie on the response."""
        cookie = SimpleCookie()
        cookie[name] = value
        morsel = cookie[name]
        for k, v in kwargs.items():
            if k == "max_age":
                morsel["max-age"] = str(v)
            elif k == "path":
                morsel[k] = v
            elif k == "domain":
                morsel[k] = v
            elif k == "secure":
                morsel["secure"] = str(v).lower()
            elif k == "httponly":
                morsel["httponly"] = str(v).lower()
        existing = self.headers.get("Set-Cookie", "")
        cookie_str = morsel.OutputString()
        if existing:
            self.headers["Set-Cookie"] = f"{existing}; {cookie_str}"
        else:
            self.headers["Set-Cookie"] = cookie_str

    def delete_cookie(self, name: str) -> None:
        """Delete a cookie by setting it with max-age 0."""
        self.set_cookie(name, "", max_age=0)


def _parse_multipart(body: bytes, boundary: str) -> dict[str, Any]:
    """Simplified multipart form data parser."""
    boundary_bytes = boundary.encode("utf-8")
    parts: dict[str, Any] = {}
    segments = body.split(b"--" + boundary_bytes)
    for segment in segments:
        if not segment or segment.strip() == b"" or segment.strip() == b"--":
            continue
        if b"\r\n\r\n" in segment:
            header_block, content = segment.split(b"\r\n\r\n", 1)
        elif b"\n\n" in segment:
            header_block, content = segment.split(b"\n\n", 1)
        else:
            continue
        if content.endswith(b"\r\n"):
            content = content[:-2]
        elif content.endswith(b"\n"):
            content = content[:-1]
        name_match = re.search(rb'Content-Disposition: form-data; name="([^"]+)"', header_block)
        filename_match = re.search(rb'filename="([^"]+)"', header_block)
        if name_match:
            field_name = name_match.group(1).decode("utf-8")
            if filename_match:
                parts[field_name] = {
                    "filename": filename_match.group(1).decode("utf-8"),
                    "content": content,
                }
            else:
                parts[field_name] = content.decode("utf-8", errors="replace")
    return parts
