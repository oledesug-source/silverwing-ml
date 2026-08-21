"""
Template engine with inheritance, filters, macros, and file-based loading.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "Template",
    "TemplateEngine",
    "TemplateLoader",
]

_DEFAULT_FILTERS: dict[str, Callable[[Any], Any]] = {
    "upper": lambda v: str(v).upper(),
    "lower": lambda v: str(v).lower(),
    "title": lambda v: str(v).title(),
    "capitalize": lambda v: str(v).capitalize(),
    "strip": lambda v: str(v).strip(),
    "length": lambda v: len(v) if hasattr(v, "__len__") else 0,
    "reverse": lambda v: str(v)[::-1] if isinstance(v, str) else list(reversed(v)),
    "join": lambda v: ", ".join(str(i) for i in v) if isinstance(v, (list, tuple, set)) else str(v),
    "int": lambda v: int(v),
    "float": lambda v: float(v),
    "bool": lambda v: bool(v),
    "json": lambda v: __import__("json").dumps(v, default=str),
    "urlencode": lambda v: __import__("urllib.parse", fromlist=["quote"]).quote(str(v)),
}


class Template:
    """Compiled template that can be rendered with a context dictionary."""

    def __init__(self, compiled_fn: Callable[[dict[str, Any]], str]) -> None:
        self._compiled = compiled_fn

    def render(self, context: dict[str, Any] | None = None) -> str:
        """Render the template with the given context."""
        return self._compiled(context or {})


class TemplateEngine:
    """Template engine supporting variables, control flow, inheritance, filters, and macros."""

    def __init__(self) -> None:
        self._filters: dict[str, Callable[[Any], Any]] = dict(_DEFAULT_FILTERS)
        self._cache: dict[str, Template] = {}
        self._partials: dict[str, str] = {}
        self._macros: dict[str, tuple[list[str], str]] = {}

    def register_filter(self, name: str, fn: Callable[[Any], Any]) -> None:
        """Register a custom filter function."""
        self._filters[name] = fn

    def add_partial(self, name: str, template_string: str) -> None:
        """Register an inline partial template."""
        self._partials[name] = template_string

    def render(self, template_string: str, context: dict[str, Any] | None = None) -> str:
        """Parse and render a template string with the given context."""
        template_string = self._extract_macros(template_string)
        template_string = self._process_includes(template_string)
        template_string, base = self._extract_extends(template_string)
        compiled = self._compile(template_string)
        result = compiled(context or {})
        if base:
            base_source = self._partials.get(base, "")
            base_compiled = self._compile(base_source)
            result = base_compiled({**(context or {}), "__child_content__": result})
        return result

    def render_file(self, path: str, context: dict[str, Any] | None = None) -> str:
        """Render a template from a file path."""
        key = os.path.abspath(path)
        if key in self._cache:
            return self._cache[key].render(context)
        source = Path(path).read_text(encoding="utf-8")
        template = Template(self._compile(source))
        self._cache[key] = template
        return template.render(context or {})

    def _extract_macros(self, source: str) -> str:
        macro_pattern = re.compile(r"\{%\s*macro\s+(\w+)\(([^)]*)\)\s*%\}(.*?)\{%\s*endmacro\s*%\}", re.DOTALL)
        for match in macro_pattern.finditer(source):
            name = match.group(1)
            params = [p.strip() for p in match.group(2).split(",") if p.strip()]
            body = match.group(3)
            self._macros[name] = (params, body)
        return macro_pattern.sub("", source)

    def _process_includes(self, source: str) -> str:
        include_pattern = re.compile(r"\{%\s*include\s+[\"'](\w+)[\"']\s*%\}")

        def replace_include(m: re.Match[str]) -> str:
            partial_name = m.group(1)
            if partial_name in self._partials:
                return self._partials[partial_name]
            return ""

        return include_pattern.sub(replace_include, source)

    def _extract_extends(self, source: str) -> tuple[str, str | None]:
        extends_pattern = re.compile(r"\{%\s*extends\s+[\"']([^\"']+)[\"']\s*%\}")
        match = extends_pattern.search(source)
        if match:
            base = match.group(1)
            cleaned = extends_pattern.sub("", source)
            return cleaned, base
        return source, None

    def _compile(self, source: str) -> Callable[[dict[str, Any]], str]:
        """Compile a template source into a callable rendering function."""
        def compiled(context: dict[str, Any]) -> str:
            return self._render(source, dict(context))
        return compiled

    def _render(self, source: str, context: dict[str, Any]) -> str:
        """Recursively render a template source with context."""
        result = source
        result = self._render_if_blocks(result, context)
        result = self._render_for_blocks(result, context)
        result = self._render_macro_calls(result, context)
        result = self._render_variables(result, context)
        return result

    def _resolve_variable(self, name: str, context: dict[str, Any]) -> Any:
        """Resolve a variable name against the context, supporting dot notation."""
        if "." in name:
            parts = name.split(".", 1)
            obj = self._resolve_variable(parts[0], context)
            if isinstance(obj, dict) and parts[1] in obj:
                return obj[parts[1]]
            return ""
        if name in context:
            return context[name]
        return ""

    def _apply_filters(self, value: Any, filter_expr: str) -> Any:
        """Apply a chain of pipe-separated filters to a value."""
        filters = [f.strip() for f in filter_expr.split("|")]
        for filt in filters:
            if filt.startswith("default(") and filt.endswith(")"):
                default_val = filt[8:-1].strip().strip("\"'")
                if not value and value != 0:
                    value = default_val
            elif filt in self._filters:
                value = self._filters[filt](value)
        return value

    def _render_variables(self, source: str, context: dict[str, Any]) -> str:
        """Replace all {{ variable }} and {{ variable | filter }} expressions."""
        def replace_var(m: re.Match[str]) -> str:
            expr = m.group(1).strip()
            parts = expr.split("|", 1)
            var_name = parts[0].strip()
            value = self._resolve_variable(var_name, context)
            if len(parts) > 1:
                value = self._apply_filters(value, parts[1])
            return str(value)
        return re.sub(r"\{\{(.+?)\}\}", replace_var, source)

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """Evaluate a condition string against the context."""
        condition = condition.strip()
        if condition.startswith("not "):
            inner = condition[4:].strip()
            return not self._evaluate_condition(inner, context)
        if " and " in condition:
            parts = condition.split(" and ", 1)
            return self._evaluate_condition(parts[0], context) and self._evaluate_condition(parts[1], context)
        if " or " in condition:
            parts = condition.split(" or ", 1)
            return self._evaluate_condition(parts[0], context) or self._evaluate_condition(parts[1], context)
        if "==" in condition:
            left, right = condition.split("==", 1)
            left_val = self._resolve_variable(left.strip().strip("'\""), context) if left.strip().startswith("'") or left.strip().startswith('"') else self._resolve_variable(left.strip(), context)
            right_val = right.strip().strip("'\"")
            return str(left_val) == right_val
        if "!=" in condition:
            left, right = condition.split("!=", 1)
            left_val = self._resolve_variable(left.strip(), context)
            right_val = right.strip().strip("'\"")
            return str(left_val) != right_val
        value = self._resolve_variable(condition, context)
        return bool(value)

    def _render_if_blocks(self, source: str, context: dict[str, Any]) -> str:
        """Process all {% if %}...{% endif %} blocks."""
        re.compile(
            r"\{%\s*if\s+(.+?)\s*%\}"
            r"(.*?)"
            r"(?:\{%\s*elif\s+(.+?)\s*%\}(.*?))*"
            r"\{%\s*endif\s*%\}",
            re.DOTALL,
        )

        def process_if(m: re.Match[str]) -> str:
            condition = m.group(1)
            if_body = m.group(2)
            if self._evaluate_condition(condition, context):
                return self._render(if_body, context)
            return ""

        result = source
        while re.search(r"\{%\s*if\s+", result):
            result = re.sub(
                r"\{%\s*if\s+(.+?)\s*%\}(.*?)\{%\s*endif\s*%\}",
                process_if,
                result,
                flags=re.DOTALL,
            )
        return result

    def _render_for_blocks(self, source: str, context: dict[str, Any]) -> str:
        """Process all {% for item in items %}...{% endfor %} blocks."""
        def process_for(m: re.Match[str]) -> str:
            var_name = m.group(1)
            iterable_name = m.group(2)
            body = m.group(3)
            iterable = self._resolve_variable(iterable_name, context)
            if not hasattr(iterable, "__iter__"):
                return ""
            parts = []
            for item in iterable:
                local_ctx = dict(context)
                local_ctx[var_name] = item
                parts.append(self._render(body, local_ctx))
            return "".join(parts)

        result = source
        while re.search(r"\{%\s*for\s+", result):
            result = re.sub(
                r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}",
                process_for,
                result,
                flags=re.DOTALL,
            )
        return result

    def _render_macro_calls(self, source: str, context: dict[str, Any]) -> str:
        """Process all {% call macro(args) %} expressions."""
        def process_call(m: re.Match[str]) -> str:
            call_expr = m.group(1).strip()
            paren_idx = call_expr.find("(")
            if paren_idx == -1:
                return m.group(0)
            macro_name = call_expr[:paren_idx].strip()
            args_str = call_expr[paren_idx + 1:].rstrip(")")
            if macro_name not in self._macros:
                return m.group(0)
            params, body = self._macros[macro_name]
            args = self._parse_macro_args(args_str)
            local_ctx = dict(context)
            for i, p in enumerate(params):
                if i < len(args):
                    local_ctx[p] = args[i]
            sub_engine = TemplateEngine()
            sub_engine._filters = dict(self._filters)
            sub_engine._macros = dict(self._macros)
            return sub_engine._render(body, local_ctx)

        return re.sub(r"\{%\s*call\s+(.+?)\s*%\}", process_call, source)

    def _parse_macro_args(self, args_str: str) -> list[str]:
        """Parse comma-separated macro arguments, respecting quoted strings."""
        if not args_str.strip():
            return []
        args: list[str] = []
        current = ""
        in_quote = False
        quote_char = ""
        for ch in args_str:
            if ch in ('"', "'") and not in_quote:
                in_quote = True
                quote_char = ch
            elif ch == quote_char and in_quote:
                in_quote = False
                quote_char = ""
            elif ch == "," and not in_quote:
                args.append(current.strip().strip("\"'"))
                current = ""
                continue
            current += ch
        if current.strip():
            args.append(current.strip().strip("\"'"))
        return args


class TemplateLoader:
    """File system based template loading and caching."""

    def __init__(self, base_dir: str = "templates") -> None:
        self.base_dir = Path(base_dir)
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        """Load a template file by name."""
        if name in self._cache:
            return self._cache[name]
        path = self.base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Template '{name}' not found at {path}")
        source = path.read_text(encoding="utf-8")
        self._cache[name] = source
        return source

    def render(self, name: str, context: dict[str, Any] | None = None, engine: TemplateEngine | None = None) -> str:
        """Load and render a template file."""
        if engine is None:
            engine = TemplateEngine()
        source = self.load(name)
        return engine.render(source, context)

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()
