"""Self-awareness capabilities (M23): the served LLM can inspect ITS OWN
project - modules, milestones, gaps - so enhancement targets come from
evidence, not guesses. Read-only by design; mutations stay behind the
existing gated write_file / python_exec capabilities.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

SCAN_ROOTS = ["foundation", "intelligence", "serving", "silverwing_platform", "sw_platform", "scripts"]


def _git(*args: str) -> str:
    try:
        r = subprocess.run(
            ["git", *args], cwd=str(REPO), capture_output=True, text=True, timeout=10
        )
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
    except OSError:
        return 0


def project_status() -> str:
    head = _git("rev-parse", "--short", "HEAD") or "unknown"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    dirty = len(_git("status", "--porcelain").splitlines())
    last_msg = _git("log", "-1", "--format=%s")[:90]

    modules = []
    total_loc = 0
    for root_name in SCAN_ROOTS:
        root = REPO / root_name
        if not root.is_dir():
            continue
        py_files = [p for p in root.rglob("*.py") if "__pycache__" not in str(p)]
        loc = sum(_loc(p) for p in py_files)
        total_loc += loc
        modules.append(f"{root_name}:{len(py_files)}files/{loc}loc")

    sft = REPO / "experiments" / "sft" / "sft-v3-all.jsonl"
    n_sft = sum(1 for _ in sft.open(encoding="utf-8")) if sft.exists() else 0

    ckpt = None
    for candidate in [
        REPO / "experiments" / "checkpoints" / "alignment" / "best.pt",
        REPO / "experiments" / "checkpoints" / "sft-v2" / "best.pt",
    ]:
        if candidate.exists():
            ckpt = f"{candidate.parent.name}/best.pt ({candidate.stat().st_size / 1e9:.2f}GB)"
            break

    return json.dumps({
        "identity": "Silverwing Decoder V2 SFT - served LLM of this repository",
        "repo_head": head,
        "branch": branch,
        "dirty_files": dirty,
        "last_commit": last_msg,
        "code": {"modules": ", ".join(modules), "total_loc": total_loc},
        "training_data": {"sft_v3_records": n_sft},
        "active_checkpoint": ckpt or "external (F:/AI/models/sft/best.pt)",
    }, ensure_ascii=False)


def project_gaps() -> str:
    """Evidence-based list of where enhancements are needed."""
    gaps: list[dict[str, str]] = []

    # 1. hollow packages: __init__.py present but no real module content
    for root_name in SCAN_ROOTS:
        root = REPO / root_name
        if not root.is_dir():
            continue
        for pkg in {p.parent for p in root.rglob("__init__.py")}:
            rel = pkg.relative_to(REPO).as_posix()
            py_files = [
                p for p in pkg.glob("*.py")
                if p.name != "__init__.py" and "__pycache__" not in str(p)
            ]
            if not py_files and pkg.name != "__pycache__":
                gaps.append({
                    "gap": f"hollow package {rel}/ - no implementation modules",
                    "action": f"implement core module(s) in {rel}/",
                })

    # 2. unfinished markers
    todo_count = 0
    todo_samples: list[str] = []
    for root_name in SCAN_ROOTS:
        root = REPO / root_name
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in str(p):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in re.finditer(r"#\s*(TODO|FIXME)\b(.*)", text):
                todo_count += 1
                if len(todo_samples) < 5:
                    rel = p.relative_to(REPO)
                    todo_samples.append(f"{rel}: {m.group(0).strip()[:80]}")
    if todo_count:
        gaps.append({
            "gap": f"{todo_count} TODO/FIXME markers",
            "action": "resolve highest-impact markers first",
            "samples": "; ".join(todo_samples),
        })

    # 3. open roadmap items in milestones doc
    miles = REPO / "docs" / "milestones.md"
    if miles.exists():
        open_items = [
            ln.strip()[:110]
            for ln in miles.read_text(encoding="utf-8", errors="replace").splitlines()
            if re.search(r"\b(In progress|Pending|Planned)\b", ln, re.IGNORECASE)
        ]
        for item in open_items[:5]:
            gaps.append({"gap": "roadmark open", "action": item})

    # 4. datasets without manifests / benchmarks not generated
    checks = [
        (REPO / "benchmarks" / "unified" / "unified-v2.jsonl", "generate unified-benchmark-v2"),
        (REPO / "experiments" / "checkpoints" / "alignment" / "best.pt",
         "run DPO stage (GPU) to produce aligned checkpoint"),
    ]
    for missing, action in checks:
        if not missing.exists():
            gaps.append({"gap": f"missing artifact {missing.name}", "action": action})

    summary = {
        "total_gaps": len(gaps),
        "gaps": gaps[:12],
        "note": "prioritise items unblocking training (GPU artifacts) before cosmetic ones",
    }
    return json.dumps(summary, ensure_ascii=False, indent=1)


def register_self_capabilities(schema_cls: type) -> list[Any]:
    return [
        schema_cls(
            name="project.status",
            description="Self-model: this repo's identity, git state, module inventory, dataset size, active checkpoint",
            input_schema={},
            tags=["self", "safe"],
            fn=project_status,
        ),
        schema_cls(
            name="project.gaps",
            description="Where the project needs enhancement: hollow packages, TODOs, open roadmap items, missing GPU artifacts",
            input_schema={},
            tags=["self", "safe"],
            fn=project_gaps,
        ),
    ]


__all__ = ["register_self_capabilities", "project_status", "project_gaps"]
