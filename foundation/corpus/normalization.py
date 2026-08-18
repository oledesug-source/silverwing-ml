"""Text normalization for the corpus pipeline.

Pure deterministic transformations: encoding cleanup, unicode normalization,
control-character removal, HTML entity decoding and whitespace collapsing.
Normalization must be byte-stable so content hashes are comparable across runs.
"""

from __future__ import annotations

import html
import re
import unicodedata

_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_SPACES = re.compile(r"[ \t\f\v]+")
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_MULTI_BLANK = re.compile(r"[ \t]+\n")
_RE_HEADING_WS = re.compile(r"\n[ \t]+")


def normalize_text(text: str) -> str:
    """Return a deterministic, cleaned version of the input text."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = _RE_CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_HEADING_WS.sub("\n", text)
    text = _RE_MULTI_BLANK.sub("\n", text)
    text = _RE_SPACES.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def normalize_document_id(value: str) -> str:
    """Sanitize an arbitrary string into a safe document_id."""
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    return value.strip("-") or "document"
