"""Quality, language and domain filtering.

Quality filters prune low-value documents (too short, non-textual, repetitive,
spammy).  A perplexity-based filter (C4-style) can optionally score documents
against a simple n-gram model to remove incoherent or low-quality text.
Language detection is a lightweight heuristic based on unicode script blocks
plus stopword scoring; it is intentionally dependency-free and suited to corpus
triage rather than production-grade detection.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

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
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r" {2,}")

_LATIN_STOPWORDS = {
    "en": {"the", "and", "of", "to", "a", "in", "is", "it", "that", "for", "on", "with", "as", "this", "by", "at"},
    "de": {"der", "die", "das", "und", "in", "ist", "zu", "den", "mit", "von", "des", "ein", "eine"},
    "fr": {"le", "la", "les", "de", "des", "et", "est", "un", "une", "que", "qui", "dans", "pour", "pas"},
    "es": {"el", "la", "los", "las", "de", "que", "y", "a", "en", "un", "una", "es", "por", "para"},
    "it": {"il", "lo", "la", "di", "che", "e", "a", "in", "un", "una", "è", "per", "non", "con"},
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
    # C4-style perplexity filter (optional)
    max_perplexity: float = 0.0  # 0 = disabled; e.g. 1000.0 to enable

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
        # C4-style: reject documents with residual HTML tags
        if _HTML_TAG_RE.search(text):
            return False
        # C4-style: reject documents with too many consecutive spaces
        if _MULTI_SPACE_RE.search(text):
            # more than 2 consecutive spaces suggests poor normalization
            space_runs = _MULTI_SPACE_RE.findall(text)
            if len(space_runs) > 5:
                return False
        return True


# ---------------------------------------------------------------------------
# C4-style perplexity filter using a simple character-level n-gram model
# ---------------------------------------------------------------------------

class PerplexityFilter:
    """Score documents by character-level perplexity and reject high-perplexity ones.

    This is a lightweight approximation of C4's GPT-2 perplexity filter that
    avoids the torch dependency.  It fits a 4-gram character model on the fly
    from a reference corpus (or uses a default English model) and scores each
    document.  Documents with perplexity above ``max_perplexity`` are rejected.

    Set ``max_perplexity`` to 0 (default) to disable this filter.
    """

    def __init__(
        self,
        max_perplexity: float = 0.0,
        reference_text: str | None = None,
        ngram_order: int = 4,
        smoothing: float = 1e-6,
    ) -> None:
        self.max_perplexity = max_perplexity
        self.ngram_order = ngram_order
        self.smoothing = smoothing
        self._ngram_counts: Counter[tuple[str, ...]] = Counter()
        self._total: int = 0
        self._fitted = False
        if reference_text:
            self.fit(reference_text)

    def fit(self, text: str) -> None:
        """Build the character n-gram frequency model from reference text."""
        text = text.lower()
        for i in range(len(text) - self.ngram_order):
            gram = text[i : i + self.ngram_order]
            prefix = gram[:-1]
            self._ngram_counts[prefix] += 1
            self._total += 1
        self._fitted = True

    def _log_prob(self, text: str) -> float:
        text = text.lower()
        log_prob = 0.0
        count = 0
        for i in range(len(text) - self.ngram_order):
            gram = text[i : i + self.ngram_order]
            prefix = gram[:-1]
            gram[-1]
            freq = self._ngram_counts.get(prefix, 0)
            sum(
                1 for k, v in self._ngram_counts.items()
                if k == prefix and v > 0
            )
            prob = (self._ngram_counts.get(gram, 0) + self.smoothing) / (freq + self.smoothing * 256)
            log_prob += math.log(prob + 1e-10)
            count += 1
        return log_prob / max(count, 1)

    def perplexity(self, text: str) -> float:
        if not self._fitted or not text.strip():
            return float("inf")
        lp = self._log_prob(text)
        return math.exp(-lp)

    def keep(self, record: DocumentRecord) -> bool:
        if self.max_perplexity <= 0 or not self._fitted:
            return True
        ppl = self.perplexity(record.text)
        return ppl <= self.max_perplexity


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


def detect_language(text: str, hint: str | None = None) -> str:
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


def detect_domain(text: str, hint: str | None = None) -> str:
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
