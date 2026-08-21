"""Database access tools — direct access to local SQLite databases.

Provides query execution, table listing, and schema introspection
capabilities with read-only enforcement by default.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from sw_platform.harness.core import ExecutionResult, ToolProvider, ToolSpec

logger = logging.getLogger(__name__)


class DatabaseProvider(ToolProvider):
    """Provider for SQLite database operation tools.

    Parameters:
        database_path: Path to the SQLite database file.
        read_only: If True, only SELECT queries are allowed.
    """

    def __init__(
        self,
        database_path: str = ":memory:",
        read_only: bool = True,
    ) -> None:
        self._database_path = database_path
        self._read_only = read_only

    def get_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="sql_query",
                description=(
                    "Execute a SQL query against the local SQLite database. "
                    "Supports SELECT, INSERT, UPDATE, DELETE, CREATE TABLE. "
                    "Returns results as rows. "
                    "Read-only mode blocks INSERT/UPDATE/DELETE/CREATE."
                ),
                parameters={
                    "query": "str — SQL query to execute",
                    "params": "list — optional query parameters (for parameterized queries)",
                },
                tags=["database", "sqlite", "query"],
                risk_level="high",
                permission_required="execute",
            ),
            ToolSpec(
                name="sql_list_tables",
                description="List all tables in the SQLite database.",
                parameters={},
                tags=["database", "sqlite", "schema"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="sql_schema",
                description="Get the schema (CREATE statement) for a table.",
                parameters={"table_name": "str — name of the table"},
                tags=["database", "sqlite", "schema"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="sql_explain",
                description=(
                    "Explain a SQL query execution plan using EXPLAIN QUERY PLAN. "
                    "Useful for query optimization."
                ),
                parameters={"query": "str — SQL query to explain"},
                tags=["database", "sqlite", "performance"],
                risk_level="low",
                permission_required="read",
            ),
        ]

    def execute(self, name: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()

        if name == "sql_query":
            return self._sql_query(**kwargs)
        elif name == "sql_list_tables":
            return self._sql_list_tables(**kwargs)
        elif name == "sql_schema":
            return self._sql_schema(**kwargs)
        elif name == "sql_explain":
            return self._sql_explain(**kwargs)

        return ExecutionResult(
            tool_name=name,
            success=False,
            error=f"Unknown tool: {name}",
            elapsed_seconds=time.monotonic() - t0,
        )

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection (read-only or read-write)."""
        if self._read_only:
            uri = f"file:{self._database_path}?mode=ro"
            return sqlite3.connect(uri, uri=True, check_same_thread=False)
        return sqlite3.connect(self._database_path, check_same_thread=False)

    def _sql_query(
        self, query: str, params: list[Any] | None = None, **kwargs: Any
    ) -> ExecutionResult:
        t0 = time.monotonic()
        params = params or []

        if self._read_only:
            query_upper = query.strip().upper()
            forbidden = ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
                         "ALTER", "ATTACH", "DETACH", "REPLACE", "VACUUM")
            first_word = query_upper.split()[0] if query_upper.split() else ""
            if first_word in forbidden or any(
                query_upper.startswith(fw) for fw in forbidden
            ):
                return ExecutionResult(
                    tool_name="sql_query",
                    success=False,
                    error="Read-only mode: write operations are blocked",
                    elapsed_seconds=time.monotonic() - t0,
                )

        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(query, params)

            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                if rows:
                    header = "| " + " | ".join(columns) + " |"
                    separator = "|" + "|".join(
                        "---" for _ in columns
                    ) + "|"
                    lines = [header, separator]
                    for row in rows:
                        lines.append("| " + " | ".join(str(v) for v in row) + " |")
                    output = "\n".join(lines)
                    # Include row count
                    output += f"\n\n({len(rows)} rows)"
                else:
                    output = f"Columns: {', '.join(columns)}\n(No rows returned)"
            else:
                conn.commit()
                output = f"Query executed successfully. Rows affected: {cursor.rowcount}"

            cursor.close()
            conn.close()
            return ExecutionResult(
                tool_name="sql_query",
                success=True,
                output=output,
                elapsed_seconds=time.monotonic() - t0,
            )
        except sqlite3.Error as exc:
            return ExecutionResult(
                tool_name="sql_query",
                success=False,
                error=f"SQLite error: {exc}",
                elapsed_seconds=time.monotonic() - t0,
            )
        finally:
            try:
                conn.close()
            except NameError:
                pass

    def _sql_list_tables(self, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return ExecutionResult(
                tool_name="sql_list_tables",
                success=True,
                output="\n".join(tables) if tables else "No tables found",
                elapsed_seconds=time.monotonic() - t0,
            )
        except sqlite3.Error as exc:
            return ExecutionResult(
                tool_name="sql_list_tables",
                success=False,
                error=f"SQLite error: {exc}",
                elapsed_seconds=time.monotonic() - t0,
            )

    def _sql_schema(self, table_name: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if row and row[0]:
                return ExecutionResult(
                    tool_name="sql_schema",
                    success=True,
                    output=row[0],
                    elapsed_seconds=time.monotonic() - t0,
                )
            return ExecutionResult(
                tool_name="sql_schema",
                success=False,
                error=f"Table not found: {table_name}",
                elapsed_seconds=time.monotonic() - t0,
            )
        except sqlite3.Error as exc:
            return ExecutionResult(
                tool_name="sql_schema",
                success=False,
                error=f"SQLite error: {exc}",
                elapsed_seconds=time.monotonic() - t0,
            )

    def _sql_explain(self, query: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            plan_lines = [str(row) for row in rows]
            return ExecutionResult(
                tool_name="sql_explain",
                success=True,
                output="\n".join(plan_lines),
                elapsed_seconds=time.monotonic() - t0,
            )
        except sqlite3.Error as exc:
            return ExecutionResult(
                tool_name="sql_explain",
                success=False,
                error=f"SQLite error: {exc}",
                elapsed_seconds=time.monotonic() - t0,
            )
