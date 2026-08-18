"""Tests for M15.4: memory and context management."""

from __future__ import annotations

import time

from intelligence.memory import MemoryEntry, MemoryStore, WorkingMemory


def test_memory_store_add_and_get():
    store = MemoryStore()
    store.add(MemoryEntry(key="k1", content="hello"))
    entry = store.get("k1")
    assert entry is not None
    assert entry.content == "hello"


def test_memory_store_get_missing():
    store = MemoryStore()
    assert store.get("missing") is None


def test_memory_store_remove():
    store = MemoryStore()
    store.add(MemoryEntry(key="k1", content="hello"))
    assert store.remove("k1") is True
    assert store.get("k1") is None
    assert store.remove("k1") is False


def test_memory_store_search():
    store = MemoryStore()
    store.add(MemoryEntry(key="a", content="water boils at 100C"))
    store.add(MemoryEntry(key="b", content="ice melts at 0C"))
    store.add(MemoryEntry(key="c", content="python is a language"))
    results = store.search("0C")
    assert len(results) == 2
    results = store.search("water")
    assert len(results) == 1
    results = store.search("ice")
    assert len(results) == 1


def test_memory_store_search_by_key():
    store = MemoryStore()
    store.add(MemoryEntry(key="temp_fact", content="some content"))
    results = store.search("temp")
    assert len(results) == 1


def test_memory_store_most_important():
    store = MemoryStore()
    store.add(MemoryEntry(key="low", content="low", importance=0.1))
    store.add(MemoryEntry(key="high", content="high", importance=0.9))
    store.add(MemoryEntry(key="mid", content="mid", importance=0.5))
    top = store.most_important(2)
    assert top[0].key == "high"
    assert top[1].key == "mid"


def test_memory_store_most_recent():
    store = MemoryStore()
    store.add(MemoryEntry(key="old", content="old", timestamp=1000))
    store.add(MemoryEntry(key="new", content="new", timestamp=2000))
    recent = store.most_recent(1)
    assert recent[0].key == "new"


def test_memory_store_len_and_contains():
    store = MemoryStore()
    assert len(store) == 0
    assert "k1" not in store
    store.add(MemoryEntry(key="k1", content="v"))
    assert len(store) == 1
    assert "k1" in store


def test_memory_store_clear():
    store = MemoryStore()
    store.add(MemoryEntry(key="k1", content="v"))
    store.clear()
    assert len(store) == 0


def test_memory_store_keys():
    store = MemoryStore()
    store.add(MemoryEntry(key="a", content="1"))
    store.add(MemoryEntry(key="b", content="2"))
    assert sorted(store.keys()) == ["a", "b"]


def test_memory_store_overwrite():
    store = MemoryStore()
    store.add(MemoryEntry(key="k1", content="old"))
    store.add(MemoryEntry(key="k1", content="new"))
    assert store.get("k1").content == "new"
    assert len(store) == 1


# --- WorkingMemory ---


def test_working_memory_add_and_build():
    wm = WorkingMemory(max_tokens=200)
    wm.add(MemoryEntry(key="u", content="Hello"))
    wm.add(MemoryEntry(key="a", content="Hi there!"))
    ctx = wm.build_context()
    assert "Hello" in ctx
    assert "Hi there!" in ctx


def test_working_memory_eviction():
    wm = WorkingMemory(max_tokens=20, reserve_for_response=0)
    for i in range(20):
        wm.add(MemoryEntry(key=f"e{i}", content="x" * 40, importance=float(i)))
    assert wm.fits
    assert len(wm) > 0


def test_working_memory_fits():
    wm = WorkingMemory(max_tokens=100, reserve_for_response=50)
    wm.add(MemoryEntry(key="k", content="short"))
    assert wm.fits


def test_working_memory_clear():
    wm = WorkingMemory()
    wm.add(MemoryEntry(key="k", content="v"))
    wm.clear()
    assert len(wm) == 0


def test_working_memory_bool():
    wm = WorkingMemory()
    assert not wm
    wm.add(MemoryEntry(key="k", content="v"))
    assert wm


def test_working_memory_entries():
    wm = WorkingMemory()
    wm.add(MemoryEntry(key="a", content="1"))
    wm.add(MemoryEntry(key="b", content="2"))
    entries = wm.entries()
    assert len(entries) == 2


def test_working_memory_available_tokens():
    wm = WorkingMemory(max_tokens=512, reserve_for_response=128)
    assert wm.available_tokens == 384


def test_working_memory_used_tokens():
    wm = WorkingMemory(max_tokens=512)
    wm.add(MemoryEntry(key="k", content="a" * 40))
    assert wm.used_tokens > 0


def test_memory_entry_age():
    e = MemoryEntry(key="k", content="v", timestamp=time.time() - 10)
    assert e.age >= 9.0
