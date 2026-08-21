"""HTTP client for ML API communication.

Provides a lightweight HTTP client for interacting with model serving
endpoints, external APIs, and microservices.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HttpResponse:
    """Parsed HTTP response."""

    status_code: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True if status code is 2xx."""
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        """Parse response body as JSON."""
        return json.loads(self.body)


class HttpClient:
    """Simple HTTP client for REST API calls.

    Usage::

        client = HttpClient(base_url="http://localhost:8000")
        resp = client.get("/health")
        if resp.ok:
            print(resp.json())
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_headers = headers or {}

    def get(
        self,
        path: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        """Send a GET request."""
        url = self._build_url(path, params)
        return self._request("GET", url, headers=headers)

    def post(
        self,
        path: str,
        data: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        """Send a POST request with JSON body."""
        url = self._build_url(path)
        body = json.dumps(data).encode() if data is not None else None
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        return self._request("POST", url, body=body, headers=h)

    def put(
        self,
        path: str,
        data: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        """Send a PUT request with JSON body."""
        url = self._build_url(path)
        body = json.dumps(data).encode() if data is not None else None
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        return self._request("PUT", url, body=body, headers=h)

    def delete(
        self, path: str, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        """Send a DELETE request."""
        url = self._build_url(path)
        return self._request("DELETE", url, headers=headers)

    def _build_url(
        self, path: str, params: dict[str, str] | None = None
    ) -> str:
        url = f"{self._base_url}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"
        return url

    def _request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        import time

        h = dict(self._default_headers)
        if headers:
            h.update(headers)

        req = urllib.request.Request(
            url, data=body, headers=h, method=method
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                elapsed = (time.time() - start) * 1000
                return HttpResponse(
                    status_code=resp.status,
                    body=resp.read().decode(),
                    headers=dict(resp.headers),
                    elapsed_ms=elapsed,
                )
        except urllib.error.HTTPError as exc:
            elapsed = (time.time() - start) * 1000
            return HttpResponse(
                status_code=exc.code,
                body=exc.read().decode() if exc.fp else "",
                elapsed_ms=elapsed,
            )
        except urllib.error.URLError as exc:
            elapsed = (time.time() - start) * 1000
            return HttpResponse(
                status_code=0,
                body=str(exc.reason),
                elapsed_ms=elapsed,
            )
