"""Working memory and context management.

Provides:

- **MemoryStore**: a flat key-value store with importance scores
- **WorkingMemory**: a sliding window that fits within the model's block_size,
  automatically evicting low-importance entries when full
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """A single memory entry."""

    key: str
    content: str
    importance: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Age in seconds since creation."""
        return time.time() - self.timestamp


class MemoryStore:
    """Flat key-value memory store with importance weighting.

    Usage::

        store = MemoryStore()
        store.add(MemoryEntry(key="fact1", content="Water boils at 100C", importance=0.9))
        entry = store.get("fact1")
        relevant = store.search("temperature")
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    def add(self, entry: MemoryEntry) -> None:
        """Add or update a memory entry."""
        self._entries[entry.key] = entry

    def get(self, key: str) -> MemoryEntry | None:
        """Retrieve a memory entry by key."""
        return self._entries.get(key)

    def remove(self, key: str) -> bool:
        """Remove a memory entry. Returns True if it existed."""
        return self._entries.pop(key, None) is not None

    def search(self, query: str) -> list[MemoryEntry]:
        """Simple substring search across all entries, sorted by importance."""
        query_lower = query.lower()
        matches = [
            e for e in self._entries.values()
            if query_lower in e.content.lower() or query_lower in e.key.lower()
        ]
        return sorted(matches, key=lambda e: e.importance, reverse=True)

    def most_important(self, n: int = 10) -> list[MemoryEntry]:
        """Return the N most important entries."""
        return sorted(
            self._entries.values(),
            key=lambda e: e.importance,
            reverse=True,
        )[:n]

    def most_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the N most recent entries."""
        return sorted(
            self._entries.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )[:n]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def keys(self) -> list[str]:
        return list(self._entries.keys())


class WorkingMemory:
    """Sliding-window working memory that fits within the model's block size.

    Maintains a list of entries that represent the current conversation context.
    When the context exceeds ``max_tokens``, low-importance entries are evicted
    first, followed by oldest entries.

    Usage::

        wm = WorkingMemory(max_tokens=512)
        wm.add(MemoryEntry(key="user", content="Hello!"))
        wm.add(MemoryEntry(key="assistant", content="Hi there!"))
        context = wm.build_context()
    """

    def __init__(self, max_tokens: int = 512, reserve_for_response: int = 128) -> None:
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self._entries: list[MemoryEntry] = []

    @property
    def available_tokens(self) -> int:
        """Tokens available for new entries."""
        return self.max_tokens - self.reserve_for_response

    @property
    def used_tokens(self) -> int:
        """Approximate tokens used (rough estimate: 1 token per 4 chars)."""
        return sum(max(1, len(e.content) // 4) for e in self._entries)

    @property
    def fits(self) -> bool:
        """Whether all entries fit within the budget."""
        return self.used_tokens <= self.available_tokens

    def add(self, entry: MemoryEntry) -> None:
        """Add an entry, evicting old entries if necessary."""
        self._entries.append(entry)
        self._evict()

    def _evict(self) -> None:
        """Evict entries until we fit within budget."""
        while self.used_tokens > self.available_tokens and self._entries:
            least_important_idx = min(
                range(len(self._entries)),
                key=lambda i: (self._entries[i].importance, self._entries[i].timestamp),
            )
            self._entries.pop(least_important_idx)

    def build_context(self) -> str:
        """Build the context string from all entries."""
        return "\n".join(e.content for e in self._entries)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    def entries(self) -> list[MemoryEntry]:
        """Return a copy of the current entries."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
