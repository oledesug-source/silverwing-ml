"""Web automation tools — browse the web, scrape structured data, auto-fill forms.

Uses Playwright if available. Falls back to httpx for basic HTTP requests
when Playwright is not installed, so the agent can still make HTTP calls.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from sw_platform.harness.core import ExecutionResult, ToolProvider, ToolSpec

logger = logging.getLogger(__name__)

_HAS_PLAYWRIGHT = False
try:
    from playwright.sync_api import sync_playwright  # type: ignore[import]

    _HAS_PLAYWRIGHT = True
except ImportError:
    logger.info("Playwright not installed; web automation will use httpx fallback")
    _HAS_PLAYWRIGHT = False


class WebAutomationProvider(ToolProvider):
    """Provider for web browsing and automation tools."""

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless

    def get_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="web_fetch",
                description="Fetch HTML content from a URL. Returns page title and body text.",
                parameters={"url": "str - URL to fetch"},
                tags=["web", "fetch", "http"],
                risk_level="low",
                permission_required="network",
            ),
            ToolSpec(
                name="web_scrape",
                description="Extract structured data from a webpage using CSS selectors.",
                parameters={
                    "url": "str - URL to scrape",
                    "selector": "str - CSS selector to extract elements",
                },
                tags=["web", "scrape", "parsing"],
                risk_level="low",
                permission_required="network",
            ),
            ToolSpec(
                name="web_form_fill",
                description="Fill and submit a web form by field name/ID.",
                parameters={
                    "url": "str - form page URL",
                    "fields": "str - JSON mapping of field names to values",
                    "submit_selector": "str - CSS selector for submit button",
                },
                tags=["web", "automation", "form"],
                risk_level="medium",
                permission_required="network",
            ),
        ]

    def execute(self, name: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        if name == "web_fetch":
            return self._web_fetch(**kwargs)
        elif name == "web_scrape":
            return self._web_scrape(**kwargs)
        elif name == "web_form_fill":
            return self._web_form_fill(**kwargs)
        return ExecutionResult(
            tool_name=name, success=False, error=f"Unknown tool: {name}",
            elapsed_seconds=time.monotonic() - t0,
        )

    def _web_fetch(self, url: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        if _HAS_PLAYWRIGHT:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self._headless)
                    page = browser.new_page()
                    page.goto(url, timeout=15000)
                    page.wait_for_timeout(1000)
                    title = page.title()
                    content = page.inner_text("body")
                    browser.close()
                    return ExecutionResult(
                        tool_name="web_fetch", success=True,
                        output=f"<title>{title}</title>\n\n{content[:5000]}",
                        elapsed_seconds=time.monotonic() - t0,
                    )
            except Exception as exc:
                return ExecutionResult(tool_name="web_fetch", success=False,
                    error=str(exc), elapsed_seconds=time.monotonic() - t0)
        else:
            try:
                import httpx
                with httpx.Client(timeout=15) as client:
                    resp = client.get(url)
                import re
                title = ""
                match = re.search(r"<title>(.*?)</title>", resp.text, re.IGNORECASE | re.DOTALL)
                if match:
                    title = match.group(1).strip()
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()
                return ExecutionResult(tool_name="web_fetch", success=True,
                    output=f"<title>{title}</title>\n\n{text[:5000]}",
                    elapsed_seconds=time.monotonic() - t0,
                    metadata={"status_code": resp.status_code, "url": str(resp.url)})
            except ImportError:
                return ExecutionResult(tool_name="web_fetch", success=False,
                    error="Neither Playwright nor httpx is installed",
                    elapsed_seconds=time.monotonic() - t0)
            except Exception as exc:
                return ExecutionResult(tool_name="web_fetch", success=False,
                    error=str(exc), elapsed_seconds=time.monotonic() - t0)

    def _web_scrape(self, url: str, selector: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        if _HAS_PLAYWRIGHT:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self._headless)
                    page = browser.new_page()
                    page.goto(url, timeout=15000)
                    page.wait_for_timeout(1000)
                    elements = page.query_selector_all(selector)
                    if not elements:
                        return ExecutionResult(tool_name="web_scrape", success=True,
                            output=f"No elements found matching: {selector}",
                            elapsed_seconds=time.monotonic() - t0)
                    results = [{"text": el.inner_text().strip(), "html": el.inner_html()[:500]} for el in elements]
                    browser.close()
                    return ExecutionResult(tool_name="web_scrape", success=True,
                        output=json.dumps(results, indent=2),
                        elapsed_seconds=time.monotonic() - t0)
            except Exception as exc:
                return ExecutionResult(tool_name="web_scrape", success=False,
                    error=str(exc), elapsed_seconds=time.monotonic() - t0)
        else:
            try:
                import re

                import httpx
                with httpx.Client(timeout=15) as client:
                    resp = client.get(url)
                tag_match = re.match(r"^(\w+)", selector)
                tag = tag_match.group(1) if tag_match else "div"
                pattern = re.compile(rf"<{tag}[^>]*>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
                matches = pattern.findall(resp.text)
                results = [{"text": re.sub(r"<[^>]+>", "", m).strip()} for m in matches]
                return ExecutionResult(tool_name="web_scrape", success=True,
                    output=json.dumps(results, indent=2),
                    elapsed_seconds=time.monotonic() - t0)
            except ImportError:
                return ExecutionResult(tool_name="web_scrape", success=False,
                    error="Neither Playwright nor httpx is installed",
                    elapsed_seconds=time.monotonic() - t0)
            except Exception as exc:
                return ExecutionResult(tool_name="web_scrape", success=False,
                    error=str(exc), elapsed_seconds=time.monotonic() - t0)

    def _web_form_fill(self, url: str, fields: str,
                       submit_selector: str = "button[type=submit]",
                       **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        if not _HAS_PLAYWRIGHT:
            return ExecutionResult(tool_name="web_form_fill", success=False,
                error="Playwright is required for web_form_fill. Install: pip install playwright && playwright install",
                elapsed_seconds=time.monotonic() - t0)
        import json
        try:
            field_dict = json.loads(fields) if isinstance(fields, str) else fields
        except (json.JSONDecodeError, TypeError):
            return ExecutionResult(tool_name="web_form_fill", success=False,
                error="fields must be a JSON string mapping field names to values",
                elapsed_seconds=time.monotonic() - t0)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self._headless)
                page = browser.new_page()
                page.goto(url, timeout=15000)
                page.wait_for_timeout(1000)
                for name, value in field_dict.items():
                    page.fill(f'input[name="{name}"], #{name}, input[id="{name}"]', str(value))
                page.click(submit_selector)
                page.wait_for_timeout(2000)
                result = page.inner_text("body")
                browser.close()
                return ExecutionResult(tool_name="web_form_fill", success=True,
                    output=f"Form submitted. Page content:\n{result[:3000]}",
                    elapsed_seconds=time.monotonic() - t0)
        except Exception as exc:
            return ExecutionResult(tool_name="web_form_fill", success=False,
                error=str(exc), elapsed_seconds=time.monotonic() - t0)
