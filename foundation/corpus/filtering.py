"""Quality, language and domain filtering.

Quality filters prune low-value documents (too short, non-textual, repetitive,
spammy). Language detection is a lightweight heuristic based on unicode script
blocks plus stopword scoring; it is intentionally dependency-free and suited
to corpus triage rather than production-grade detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .schema import DocumentRecord

FilterFn = Callable[[DocumentRecord], bool]

_WORD_RE = re.compile(r"\w+")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")

_LATIN_STOPWORDS = {
    "en": {"the", "and", "of", "to", "a", "in", "is", "it", "that", "for", "on", "with", "as", "this", "by", "at"},
    "de": {"der", "die", "das", "und", "in", "ist", "zu", "den", "mit", "von", "des", "ein", "eine"},
    "fr": {"le", "la", "les", "de", "des", "et", "est", "un", "une", "que", "qui", "dans", "pour", "pas"},
    "es": {"el", "la", "los", "las", "de", "que", "y", "a", "en", "un", "una", "es", "por", "para"},
    "it": {"il", "lo", "la", "di", "che", "e", "e", "a", "in", "un", "una", "è", "per", "non", "con"},
    "pt": {"o", "a", "os", "as", "de", "que", "e", "um", "uma", "em", "é", "para", "com", "não"},
    "nl": {"de", "het", "een", "van", "en", "in", "is", "dat", "die", "met", "voor", "op", "te"},
}


@dataclass
class QualityFilter:
    min_chars: int = 200
    min_words: int = 40
    min_alpha_ratio: float = 0.4
    max_punct_ratio: float = 0.3
    max_url_ratio: float = 0.05
    max_email_count: int = 10
    max_duplicate_line_ratio: float = 0.6

    def keep(self, record: DocumentRecord) -> bool:
        text = record.text
        chars = len(text)
        if chars < self.min_chars:
            return False
        words = len(_WORD_RE.findall(text))
        if words < self.min_words:
            return False
        alpha = len(_ALPHA_RE.findall(text))
        if chars and (alpha / chars) < self.min_alpha_ratio:
            return False
        punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if chars and (punct / chars) > self.max_punct_ratio:
            return False
        url_chars = sum(len(m) for m in _URL_RE.findall(text))
        if chars and (url_chars / chars) > self.max_url_ratio:
            return False
        if len(_EMAIL_RE.findall(text)) > self.max_email_count:
            return False
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            seen: dict[str, int] = {}
            for ln in lines:
                seen[ln] = seen.get(ln, 0) + 1
            dup_ratio = 1.0 - (len(seen) / len(lines))
            if dup_ratio > self.max_duplicate_line_ratio:
                return False
        return True


def detect_script(text: str) -> str:
    """Return a coarse script label for the dominant unicode block."""
    scores = {
        "cjk": len(_CJK_RE.findall(text)),
        "cyrillic": len(_CYRILLIC_RE.findall(text)),
        "arabic": len(_ARABIC_RE.findall(text)),
        "devanagari": len(_DEVANAGARI_RE.findall(text)),
        "latin": len(_ALPHA_RE.findall(text)),
    }
    best, best_score = "latin", 0
    for script, score in scores.items():
        if score > best_score:
            best, best_score = script, score
    return best


def detect_language(text: str, hint: Optional[str] = None) -> str:
    script = detect_script(text)
    if script != "latin":
        return script
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return hint or "unknown"
    samples = min(len(words), 400)
    if hint is not None and hint in _LATIN_STOPWORDS:
        hint_stops = _LATIN_STOPWORDS[hint]
        hint_score = sum(1 for w in words[:samples] if w in hint_stops)
        if hint_score / samples > 0.03:
            return hint
    best_lang, best_hit = None, -1
    for lang, stops in _LATIN_STOPWORDS.items():
        hits = sum(1 for w in words[:400] if w in stops)
        if hits > best_hit:
            best_lang, best_hit = lang, hits
    return best_lang if best_lang and best_hit > 0 else hint or "unknown"


def detect_domain(text: str, hint: Optional[str] = None) -> str:
    lowered = text.lower()
    markers = {
        "code": ("def ", "class ", "import ", "func ", "return ", "int main", "const ", "function "),
        "math": ("theorem", "lemma", "proof", "equation", "sqrt", "integral", "derivative", "frac{"),
        "science": ("experiment", "hypothesis", "molecule", "species", "quantum", "hypothesis"),
        "news": ("reuters", "breaking news", "reported", "officials said", "according to"),
    }
    for domain, keywords in markers.items():
        if any(k in lowered for k in keywords):
            return domain
    return hint or "general"


@dataclass
class LanguageFilter:
    allowed: tuple[str, ...] = ("en",)

    def keep(self, record: DocumentRecord) -> bool:
        return record.provenance.language in self.allowed


class FilterChain:
    """Applies a sequence of filters; records must pass every filter."""

    def __init__(self, filters: list[FilterFn]) -> None:
        self.filters = filters

    def evaluate(self, record: DocumentRecord) -> bool:
        for fn in self.filters:
            if not fn(record):
                return False
        return True

    def apply(self, records) -> tuple[list[DocumentRecord], int]:
        kept: list[DocumentRecord] = []
        dropped = 0
        for record in records:
            if self.evaluate(record):
                kept.append(record)
            else:
                dropped += 1
        return kept, dropped
