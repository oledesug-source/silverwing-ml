"""Document chunking for the training pipeline.

Chunks are produced per-document with a hard token budget and optional overlap,
and inherit the parent's provenance. Token counts use a lightweight heuristic
(words for latin scripts, characters / 1.5 for CJK) until the real tokenizer
(M05) is wired in.
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


def _is_cjk(char: str) -> bool:
    return any(lo <= char <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    cjk_count = sum(1 for c in text if _is_cjk(c))
    non_cjk = len(text) - cjk_count
    return cjk_count // 2 + non_cjk // 4


def _split_into_sentences(text: str) -> list[str]:
    parts = text.replace("\n", " ")
    sentences = []
    for chunk in parts.split("."):
        chunk = chunk.strip()
        if chunk:
            sentences.append(chunk + ". ")
    return sentences


def _word_budget_pieces(text: str, max_tokens: int) -> list[str]:
    """Split text into pieces each within max_tokens by word budget."""
    words = text.split()
    pieces, group = [], []
    for word in words:
        candidate = estimate_tokens(" ".join(group + [word]))
        if group and candidate > max_tokens:
            pieces.append(" ".join(group))
            group = []
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
