"""Git guard for reproducible runs (M01 rule).

A training run may only start from a committed repository state: the run pins
the commit hash, so the exact code that produced a checkpoint is recoverable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARDED_PATHS = [
    "foundation",
    "configs",
    "scripts",
    "tests",
    "benchmarks",
    "pyproject.toml",
    "requirements.txt",
]


def _run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)


def git_commit() -> str:
    out = _run_git(["rev-parse", "HEAD"])
    if out.returncode != 0:
        raise RuntimeError(f"not a git repository at {ROOT}: {out.stderr.strip()}")
    commit = out.stdout.strip()
    if not commit:
        raise RuntimeError("git HEAD has no commit")
    return commit


def git_is_clean(paths: list[str] | None = None) -> bool:
    args = ["status", "--porcelain", "--", *(paths or GUARDED_PATHS)]
    out = _run_git(args)
    if out.returncode != 0:
        raise RuntimeError(f"git status failed: {out.stderr.strip()}")
    return out.stdout.strip() == ""


def require_clean_repo(paths: list[str] | None = None) -> str:
    """Raise unless the guarded paths are committed; return the HEAD commit."""
    if not git_is_clean(paths):
        raise RuntimeError(
            "training requires a committed repository (M01 rule); "
            "commit or stash uncommitted changes under the guarded paths first"
        )
    return git_commit()
