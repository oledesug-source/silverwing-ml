"""Byte-level BPE core (GPT-2 style), dependency-free.

IDs are laid out as:
    0..NUM_SPECIALS-1        special tokens
    NUM_SPECIALS..+255       single UTF-8 bytes (byte b -> id b + NUM_SPECIALS)
    +256..                   one id per BPE merge, in merge order

Training tracks each adjacent pair's occurrence positions in a linked symbol
stream, so each merge visits only its real occurrences (never a full-list
scan) and updates a local neighbourhood. Pair selection uses a bucket queue
(count -> set of pair keys) so the most frequent pair is found without
re-sorting or scanning all counts every merge. Token strings are stored as
latin-1 strings (one JSON-safe character per byte) and converted back with
.encode("latin-1").
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

SPECIAL_TOKENS = ("<|endoftext|>", "<|pad|>", "<|unk|>", "<|bos|>")
NUM_SPECIALS = len(SPECIAL_TOKENS)
BYTE_VOCAB = 256

MERGE_BASE = NUM_SPECIALS + BYTE_VOCAB


def _to_latin1(char: str) -> str:
    return char


def train_bpe(
    corpus: Iterable[str],
    vocab_size: int,
    min_frequency: int = 2,
) -> tuple[list[tuple[str, str]], dict]:
    """Train byte-level BPE merges over an iterable of texts.

    Each merge step picks the most frequent adjacent pair and merges *all* of
    its occurrences in a single pass over a doubly-linked symbol list, updating
    only the local pair counts. Returns (merges, stats) where merges is an
    ordered list of (left_token, right_token) latin-1 token pairs.
    """
    num_merges = vocab_size - BYTE_VOCAB - NUM_SPECIALS
    if num_merges <= 0:
        raise ValueError(f"vocab_size must exceed {BYTE_VOCAB + NUM_SPECIALS}, got {vocab_size}")

    ids: list[int] = []
    n_bytes = 0
    n_documents = 0
    for text in corpus:
        encoded = text.encode("utf-8")
        ids.extend(b + NUM_SPECIALS for b in encoded)
        n_bytes += len(encoded)
        n_documents += 1
    n = len(ids)
    if n < 2:
        raise ValueError("corpus too small for BPE training")

    next_arr = list(range(1, n)) + [-1]
    prev_arr = [-1] + list(range(0, n - 1))

    # occ[key] holds the positions of the left element of each adjacent pair,
    # so a merge only visits its real occurrences instead of scanning the
    # whole symbol list. buckets[count] holds the pair keys with that count.
    occ: dict[tuple[int, int], set[int]] = {}
    buckets: dict[int, set[tuple[int, int]]] = {}
    max_count = 0
    for i in range(n - 1):
        key = (ids[i], ids[i + 1])
        occ.setdefault(key, set()).add(i)
    for key, positions in occ.items():
        count = len(positions)
        buckets.setdefault(count, set()).add(key)
        if count > max_count:
            max_count = count

    token_by_id: dict[int, str] = {b + NUM_SPECIALS: chr(b) for b in range(BYTE_VOCAB)}
    merges: list[tuple[str, str]] = []
    next_id = MERGE_BASE
    produced = 0

    def _set_count(key: tuple[int, int], count: int) -> None:
        nonlocal max_count
        old = counts.get(key)
        if old is not None:
            buckets[old].discard(key)
            if not buckets[old]:
                del buckets[old]
            counts.pop(key, None)
        if count > 0:
            counts[key] = count
            buckets.setdefault(count, set()).add(key)
            if count > max_count:
                max_count = count

    def _move_occurrence(key: tuple[int, int], position: int, delta: int) -> None:
        positions = occ.setdefault(key, set())
        if delta > 0:
            positions.add(position)
        else:
            positions.discard(position)
        _set_count(key, len(positions))

    counts: dict[tuple[int, int], int] = {key: len(positions) for key, positions in occ.items()}

    for _ in range(num_merges):
        while max_count > 0 and not buckets.get(max_count, ()):
            max_count -= 1
        if max_count == 0 or max_count < min_frequency:
            break
        pair = min(buckets[max_count])
        a_id, b_id = pair
        new_id = next_id
        next_id += 1
        produced += 1

        a_tok = token_by_id[a_id]
        b_tok = token_by_id[b_id]
        merges.append((a_tok, b_tok))
        token_by_id[new_id] = a_tok + b_tok

        positions = sorted(occ.get(pair, ()))
        for p in positions:
            if p not in occ.get(pair, ()):
                continue
            q = next_arr[p]
            r = next_arr[q]
            h = prev_arr[p]
            _move_occurrence((a_id, b_id), p, -1)
            if r != -1:
                _move_occurrence((b_id, ids[r]), q, -1)
            if h != -1:
                _move_occurrence((ids[h], a_id), h, -1)
            ids[p] = new_id
            next_arr[p] = r
            if r != -1:
                prev_arr[r] = p
            if r != -1:
                _move_occurrence((new_id, ids[r]), p, +1)
            if h != -1:
                _move_occurrence((ids[h], new_id), h, +1)

    stats = {
        "documents": n_documents,
        "bytes": n_bytes,
        "symbols": n,
        "requested_merges": num_merges,
        "produced_merges": produced,
        "early_stopped": produced < num_merges,
    }
    return merges, stats


def build_ranks(merges: list[tuple[str, str]]) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    """Return (pair->rank, merged_token->rank) mappings."""
    pair_rank: dict[tuple[str, str], int] = {}
    token_rank: dict[str, int] = {}
    for rank, (a, b) in enumerate(merges):
        pair_rank[(a, b)] = rank
        token_rank[a + b] = rank
    return pair_rank, token_rank


def encode_with_merges(text: str, pair_rank: dict[tuple[str, str], int], token_rank: dict[str, int]) -> list[int]:
    """Encode text to ids applying all available merges.

    Greedily merges the lowest-rank adjacent pair until none remain. Uses a
    min-heap over adjacent pairs with lazy invalidation over a linked list of
    live elements, so each merge costs O(log n) instead of an O(n) rescan of
    every adjacent pair (the reference GPT-2 encoder is O(n^2)). Produces the
    same token ids as the reference greedy algorithm.
    """
    raw = [chr(b) for b in text.encode("utf-8")]
    n = len(raw)
    if n < 2:
        return [ord(t) + NUM_SPECIALS if len(t) == 1 else MERGE_BASE + token_rank[t] for t in raw]

    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(0, n - 1))
    heap = []
    for i in range(n - 1):
        rank = pair_rank.get((raw[i], raw[i + 1]))
        if rank is not None:
            heapq.heappush(heap, (rank, i))

    while heap:
        rank, i = heapq.heappop(heap)
        j = nxt[i]
        if j == -1:
            continue
        if pair_rank.get((raw[i], raw[j])) != rank:
            continue
        merged = raw[i] + raw[j]
        k = nxt[j]
        h = prv[i]
        raw[i] = merged
        nxt[i] = k
        if k != -1:
            prv[k] = i
        if h != -1 and k != -1:
            left_rank = pair_rank.get((raw[h], merged))
            if left_rank is not None:
                heapq.heappush(heap, (left_rank, h))
            right_rank = pair_rank.get((merged, raw[k]))
            if right_rank is not None:
                heapq.heappush(heap, (right_rank, i))

    ids: list[int] = []
    i = 0
    while i != -1:
        token = raw[i]
        if len(token) == 1:
            ids.append(ord(token) + NUM_SPECIALS)
        else:
            ids.append(MERGE_BASE + token_rank[token])
        i = nxt[i]
    return ids


def token_to_text(token: str) -> str:
    """Convert a latin-1 token string back to real text bytes."""
    return token.encode("latin-1").decode("utf-8", errors="replace")
