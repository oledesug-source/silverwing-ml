"""Tool registry for the Silverwing agent harness.

Aggregates all tool providers (code execution, web automation, database,
filesystem, git) into a single registry that the HarnessAgent can consume.
"""

from __future__ import annotations

import logging

from sw_platform.harness.core import ToolProvider
from sw_platform.tools.code_execution import CodeExecutionProvider
from sw_platform.tools.database import DatabaseProvider
from sw_platform.tools.filesystem import FilesystemProvider
from sw_platform.tools.git import GitProvider
from sw_platform.tools.web_automation import WebAutomationProvider

logger = logging.getLogger(__name__)

__all__ = [
    "create_tool_registry",
    "ToolProvider",
]


def create_tool_registry(
    sandbox: object | None = None,
    allowed_paths: list[str] | None = None,
    database_path: str | None = None,
    git_repo: str | None = None,
) -> list[ToolProvider]:
    """Create a list of all registered tool providers.

    Parameters:
        sandbox: Optional SandboxExecutor for resource-limited execution.
        allowed_paths: Path prefixes the sandbox permits for filesystem tools.
        database_path: Path to SQLite database file for DatabaseProvider.
        git_repo: Root directory for GitProvider operations.

    Returns:
        A list of ToolProvider instances ready for registration with
        HarnessAgent.
    """
    providers: list[ToolProvider] = []

    # Layer 3: Code execution engine
    providers.append(CodeExecutionProvider(sandbox=sandbox))

    # Layer 3: Web automation (Playwright or httpx)
    providers.append(WebAutomationProvider())

    # Layer 3: Database & file system tools
    providers.append(FilesystemProvider(
        allowed_paths=allowed_paths or [],
        sandbox=sandbox,
    ))

    if database_path:
        providers.append(DatabaseProvider(database_path=database_path))

    if git_repo:
        providers.append(GitProvider(repo_path=git_repo))

    logger.info("Registered %d tool providers", len(providers))
    return providers
