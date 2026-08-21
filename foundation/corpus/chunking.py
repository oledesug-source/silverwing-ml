"""Document chunking for the training pipeline.

Chunks are produced per-document with a hard token budget and optional overlap,
and inherit the parent's provenance. Token counts use a lightweight heuristic
(~1.3 tokens per word for Latin scripts, characters / 2 for CJK) until the
real tokenizer (M05) is wired in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .schema import DocumentRecord, Provenance

_CJK_RANGES = (
    ("\u4e00", "\u9fff"),
    ("\u3400", "\u4dbf"),
    ("\u3040", "\u30ff"),
    ("\uac00", "\ud7af"),
)


def _has_cjk(text: str) -> bool:
    for ch in text:
        for lo, hi in _CJK_RANGES:
            if lo <= ch <= hi:
                return True
    return False


def estimate_tokens(text: str) -> int:
    """Fast token estimate: word count * 1.3 for Latin, len/2 for CJK."""
    if not text:
        return 0
    if _has_cjk(text):
        return len(text) // 2
    return max(1, int(len(text.split()) * 1.3))


def _split_into_sentences(text: str) -> list[str]:
    parts = text.replace("\n", " ")
    sentences = []
    for chunk in parts.split("."):
        chunk = chunk.strip()
        if chunk:
            sentences.append(chunk + ". ")
    return sentences


def _word_budget_pieces(text: str, max_tokens: int) -> list[str]:
    """Split text into pieces each within max_tokens by word budget.

    Uses incremental word-count estimation to avoid O(n²) string joins.
    """
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    group: list[str] = []
    group_words = 0
    for word in words:
        group_words += 1
        # Incremental estimate: (total_chars + spaces) / ~3.08 ≈ words * 1.3
        # We use word count directly since we're in Latin-dominant mode.
        est = int(group_words * 1.3)
        if group and est > max_tokens:
            # Emit current group minus this word
            group_words -= 1
            pieces.append(" ".join(group))
            group = []
            est = 0
        group.append(word)
    if group:
        pieces.append(" ".join(group))
    return pieces


@dataclass
class Chunker:
    max_tokens: int = 1024
    overlap_tokens: int = 128

    def chunk(self, record: DocumentRecord) -> list[DocumentRecord]:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        sentences = _split_into_sentences(record.text)
        if not sentences:
            return []
        chunks: list[DocumentRecord] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            for piece in self._piece_sentence(sentence):
                sentence_tokens = estimate_tokens(piece)
                if current and current_tokens + sentence_tokens > self.max_tokens:
                    chunks.extend(self._make_chunk(record, current))
                    overlap = self._take_overlap(current, self.overlap_tokens)
                    current = list(overlap)
                    current_tokens = sum(estimate_tokens(s) for s in current)
                current.append(piece)
                current_tokens += sentence_tokens
        if current:
            chunks.extend(self._make_chunk(record, current))
        return chunks

    def _piece_sentence(self, sentence: str) -> list[str]:
        """Split a single oversized sentence into word-budget pieces."""
        if estimate_tokens(sentence) <= self.max_tokens:
            return [sentence]
        return _word_budget_pieces(sentence, self.max_tokens)

    @staticmethod
    def _take_overlap(sentences: list[str], tokens: int) -> list[str]:
        total, taken = 0, []
        for sentence in reversed(sentences):
            cost = estimate_tokens(sentence)
            if total + cost > tokens:
                break
            taken.insert(0, sentence)
            total += cost
        return taken

    def _make_chunk(self, parent: DocumentRecord, sentences: list[str]) -> list[DocumentRecord]:
        text = " ".join(sentences).strip()
        if not text:
            return []
        provenance = Provenance(
            source_id=parent.provenance.source_id,
            source_type=parent.provenance.source_type,
            domain=parent.provenance.domain,
            language=parent.provenance.language,
            collection_timestamp=parent.provenance.collection_timestamp,
            parent_document=parent.document_id,
            processing_version=parent.provenance.processing_version,
        )

        def build(piece_text: str) -> DocumentRecord:
            document_id = hashlib.sha256((parent.document_id + piece_text[:64]).encode("utf-8")).hexdigest()[:24]
            record = DocumentRecord.build(document_id=document_id, text=piece_text, provenance=provenance)
            record.token_count = estimate_tokens(piece_text)
            return record

        if estimate_tokens(text) <= self.max_tokens:
            return [build(text)]
        return [build(piece) for piece in _word_budget_pieces(text, self.max_tokens)]
