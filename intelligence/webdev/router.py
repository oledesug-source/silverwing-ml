"""
URL Router with type-safe parameter extraction and named route reversal.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Route",
    "RouteMatch",
    "RouteParamConverter",
    "Router",
]


@dataclass
class Route:
    """A registered route definition."""

    method: str
    path: str
    handler: Callable[..., Any]
    name: str | None = None
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class RouteMatch:
    """Result of matching a URL against registered routes."""

    route: Route
    params: dict[str, Any]
    query_params: dict[str, Any]


class RouteParamConverter:
    """Type converters for route parameters."""

    CONVERTERS: dict[str, tuple[re.Pattern[str], type]] = {}

    @classmethod
    def register(cls, name: str, pattern: re.Pattern[str], target_type: type) -> None:
        cls.CONVERTERS[name] = (pattern, target_type)

    @classmethod
    def convert(cls, type_name: str, value: str) -> Any:
        if type_name not in cls.CONVERTERS:
            return value
        pattern, target_type = cls.CONVERTERS[type_name]
        if pattern.fullmatch(value):
            return target_type(value)
        raise ValueError(f"Cannot convert '{value}' to {type_name}")

    @classmethod
    def param_regex(cls, type_name: str) -> str:
        if type_name in cls.CONVERTERS:
            pattern, _ = cls.CONVERTERS[type_name]
            return pattern.pattern
        return r"[^/]+"

    @classmethod
    def extract_path_params(cls, pattern: str) -> list[tuple[str, str]]:
        return re.findall(r"<(\w+):(\w+)>", pattern)


RouteParamConverter.register("int", re.compile(r"-?\d+"), int)
RouteParamConverter.register("str", re.compile(r"[^/]+"), str)
RouteParamConverter.register("slug", re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*"), str)
RouteParamConverter.register("float", re.compile(r"-?\d+(?:\.\d+)?"), float)
RouteParamConverter.register("uuid", re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), str)


class Router:
    """URL router supporting typed parameters, named routes, middleware, and grouping."""

    def __init__(self) -> None:
        self._routes: list[Route] = []
        self._named_routes: dict[str, Route] = {}
        self._middlewares: list[Callable[..., Any]] = []

    def add_route(self, method: str, path: str, handler: Callable[..., Any], name: str | None = None) -> Route:
        """Register a new route with optional name and typed parameters."""
        param_specs = RouteParamConverter.extract_path_params(path)
        params = {pname: ptype for ptype, pname in param_specs}
        route = Route(method=method.upper(), path=path, handler=handler, name=name, params=params)
        self._routes.append(route)
        if name:
            self._named_routes[name] = route
        return route

    def get(self, path: str, handler: Callable[..., Any], name: str | None = None) -> Route:
        return self.add_route("GET", path, handler, name)

    def post(self, path: str, handler: Callable[..., Any], name: str | None = None) -> Route:
        return self.add_route("POST", path, handler, name)

    def put(self, path: str, handler: Callable[..., Any], name: str | None = None) -> Route:
        return self.add_route("PUT", path, handler, name)

    def delete(self, path: str, handler: Callable[..., Any], name: str | None = None) -> Route:
        return self.add_route("DELETE", path, handler, name)

    def patch(self, path: str, handler: Callable[..., Any], name: str | None = None) -> Route:
        return self.add_route("PATCH", path, handler, name)

    def _build_regex(self, path: str) -> re.Pattern[str]:
        """Convert a route path with <type:name> placeholders into a compiled regex."""
        param_specs = RouteParamConverter.extract_path_params(path)
        regex_path = path
        for ptype, pname in param_specs:
            inner = RouteParamConverter.param_regex(ptype)
            regex_path = regex_path.replace(f"<{ptype}:{pname}>", f"(?P<{pname}>{inner})")
        regex_path = regex_path.replace("/", r"\/")
        return re.compile(r"^" + regex_path + r"$")

    def match(self, method: str, url: str) -> RouteMatch | None:
        """Match a method+URL against registered routes, returning RouteMatch with extracted params."""
        path = url.split("?")[0]
        query_string = url.split("?")[1] if "?" in url else ""
        query_params = _parse_query_string(query_string)

        for route in self._routes:
            if route.method != method.upper():
                continue
            regex = self._build_regex(route.path)
            m = regex.match(path)
            if m:
                params: dict[str, Any] = {}
                for pname, ptype in route.params.items():
                    raw = m.group(pname)
                    params[pname] = RouteParamConverter.convert(ptype, raw)
                return RouteMatch(route=route, params=params, query_params=query_params)
        return None

    def reverse(self, name: str, **kwargs: Any) -> str:
        """Generate a URL path from a named route and keyword arguments."""
        if name not in self._named_routes:
            raise KeyError(f"Route '{name}' not found")
        route = self._named_routes[name]
        path = route.path
        for pname, ptype in route.params.items():
            if pname in kwargs:
                path = path.replace(f"<{ptype}:{pname}>", str(kwargs[pname]))
            else:
                raise KeyError(f"Missing parameter '{pname}' for route '{name}'")
        remaining = re.findall(r"<\w+:\w+>", path)
        if remaining:
            raise KeyError(f"Missing parameters {remaining} for route '{name}'")
        return path

    def middleware(self, middleware_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Add a middleware function applied to matched routes."""
        self._middlewares.append(middleware_fn)
        return middleware_fn

    def group(self, prefix: str, routes_fn: Callable[..., Any]) -> None:
        """Create a route group with a common path prefix."""
        original_add = self.add_route
        prefix.rstrip("/").split("/")
        list(self._routes)

        def patched_add(method: str, path: str, handler: Callable[..., Any], name: str | None = None) -> Route:
            full_path = prefix.rstrip("/") + path
            route = original_add(method, full_path, handler, name)
            return route

        self.add_route = patched_add
        try:
            routes_fn(self)
        finally:
            self.add_route = original_add


def _parse_query_string(query_string: str) -> dict[str, Any]:
    """Parse a URL query string into a dictionary."""
    from urllib.parse import parse_qs
    result: dict[str, Any] = {}
    if not query_string:
        return result
    parsed = parse_qs(query_string)
    for key, values in parsed.items():
        result[key] = values[0] if len(values) == 1 else values
    return result
