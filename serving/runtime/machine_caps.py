"""Machine-control capabilities - lets the served LLM query and (gated)
operate the host machine through audited platform tools.

Read-only tools are always available. Mutating access (``system.shell``)
is double-gated: requires BOTH
    SILVERWING_SHELL_ALLOW=1
and the command matching an allowlist prefix (SILVERWING_SHELL_ALLOWLIST,
comma-separated, sensible read-only defaults).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from typing import Any

DEFAULT_SHELL_ALLOWLIST = (
    "echo,whoami,hostname,dir,ls,pwd,python --version,pip --version,"
    "ipconfig,ifconfig,ip a,tasklist,ps aux,systeminfo,uname,ping -n 1,ping -c 1"
)

SHELL_TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 4000


def system_info() -> str:
    import platform

    import psutil

    vm = psutil.virtual_memory()
    du = psutil.disk_usage(os.sep if os.name == "nt" else "/")
    boot = time.strftime("%Y-%m-%d %H:%M", time.localtime(psutil.boot_time()))
    return (
        f"host={socket.gethostname()} os={platform.system()} {platform.release()} "
        f"cpu={platform.processor() or 'unknown'} cores={psutil.cpu_count(logical=True)} "
        f"cpu_percent={psutil.cpu_percent(interval=0.3)}% "
        f"ram_used={vm.percent}% ({vm.used // 2**30}G/{vm.total // 2**30}G) "
        f"disk_used={du.percent}% boot={boot}"
    )


def processes(top_n: int = 10) -> str:
    import psutil

    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            procs.append((p.info["pid"], p.info["name"], p.memory_info().rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda x: x[2], reverse=True)
    lines = [f"{pid:>7}  {name[:32]:<32} {rss // 2**20:>6} MB" for pid, name, rss in procs[: max(1, top_n)]]
    return "PID      NAME                             RSS\n" + "\n".join(lines)


def disk_usage() -> str:
    import psutil

    parts = psutil.disk_partitions()
    out = []
    for part in parts:
        try:
            u = psutil.disk_usage(part.mountpoint)
            out.append(
                f"{part.device:<12} total={u.total / 1e9:6.1f}GB used={u.percent:3d}% "
                f"free={u.free / 1e9:.1f}GB"
            )
        except PermissionError:
            continue
    return "\n".join(out) or "no readable partitions"


def net_status() -> str:
    import psutil

    counters = psutil.net_io_counters()
    addrs = psutil.net_if_addrs()
    ips = []
    for iface, items in addrs.items():
        for item in items:
            if item.family == socket.AF_INET and not item.address.startswith("127."):
                ips.append(f"{iface}={item.address}")
    conns = psutil.net_connections(kind="inet")
    established = sum(1 for c in conns if c.status == psutil.CONN_ESTABLISHED)
    return (
        "interfaces: " + ("; ".join(ips) or "none") +
        f" | sent={counters.bytes_sent / 1e6:.1f}MB recv={counters.bytes_recv / 1e6:.1f}MB "
        f"| established_conns={established}"
    )


def shell(command: str) -> str:
    """Execute an allow-listed shell command. See module docstring for gates."""
    if os.environ.get("SILVERWING_SHELL_ALLOW") != "1":
        raise PermissionError(
            "shell disabled - start server with SILVERWING_SHELL_ALLOW=1 to enable"
        )
    allowlist = [
        p.strip().lower()
        for p in os.environ.get("SILVERWING_SHELL_ALLOWLIST", DEFAULT_SHELL_ALLOWLIST).split(",")
        if p.strip()
    ]
    cmd = command.strip()
    lowered = cmd.lower()
    if not any(lowered.startswith(prefix) for prefix in allowlist):
        raise PermissionError(
            f"command not in allowlist: {cmd!r}. allowed prefixes: {', '.join(allowlist)}"
        )
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=SHELL_TIMEOUT_SECONDS,
    )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated, rc={result.returncode}]"
    return output or f"(no output, rc={result.returncode})"


def python_exec(code: str) -> str:
    """Run a short Python snippet (gated like system.shell)."""
    import sys as _sys

    if os.environ.get("SILVERWING_CODE_ALLOW") != "1":
        raise PermissionError(
            "code execution disabled - start server with SILVERWING_CODE_ALLOW=1"
        )
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated, rc={result.returncode}]"
    return output or f"(no output, rc={result.returncode})"


def machine_capabilities(schema_cls: type) -> list[Any]:
    """Build CapabilitySchema instances (class injected to avoid import cycle)."""
    return [
        schema_cls(
            name="system.info",
            description="Host machine snapshot: OS, CPU load, RAM, disk, boot time",
            input_schema={},
            tags=["machine", "safe"],
            fn=system_info,
        ),
        schema_cls(
            name="system.processes",
            description="Top processes by memory usage",
            input_schema={"top_n": {"type": "integer"}},
            tags=["machine", "safe"],
            fn=processes,
        ),
        schema_cls(
            name="system.disk",
            description="Per-partition disk usage",
            input_schema={},
            tags=["machine", "safe"],
            fn=disk_usage,
        ),
        schema_cls(
            name="net.status",
            description="Network interfaces, traffic counters, connection count",
            input_schema={},
            tags=["machine", "network", "safe"],
            fn=net_status,
        ),
        schema_cls(
            name="system.shell",
            description=(
                "Run a shell command (ALLOWLISTED only; server must be started "
                "with SILVERWING_SHELL_ALLOW=1)"
            ),
            input_schema={"command": {"type": "string"}},
            tags=["machine", "admin", "dangerous"],
            fn=shell,
        ),
        schema_cls(
            name="python_exec",
            description=(
                "Execute a short Python snippet and return its output "
                "(server must be started with SILVERWING_CODE_ALLOW=1)"
            ),
            input_schema={"code": {"type": "string"}},
            tags=["machine", "code", "dangerous"],
            fn=python_exec,
        ),
    ]


__all__ = ["machine_capabilities"]
