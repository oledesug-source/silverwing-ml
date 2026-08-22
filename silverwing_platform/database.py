"""Platform database layer.

Extends the existing :class:`foundation.database.Database` (SQLite-backed,
migrated, context-managed) with the relational tables required by the
SilverWing platform metadata:

    users, sessions, conversations, messages, projects, memory_records,
    capabilities, agents, tasks, workflows, workflow_runs, policies,
    approvals, audit_events, model_registry, platform_experiments.

Design:
    ``PlatformDatabase`` *wraps* a ``foundation.database.Database`` and runs a
    single additive migration (``platform_v1``) that creates the platform
    tables on top of the existing experiment/corpus/benchmark schema.  This
    keeps the existing experiment-tracking tables untouched and preserves the
    baseline test suite.

Storage policy (see PHASE 27):
    * Relational DB -> metadata/state (this file).
    * Artifact storage -> checkpoints, weights, corpora, experiment artifacts
      (left on the filesystem; only paths are stored here).
    * Vector/semantic store -> embeddings (see ``platform.memory``).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from foundation.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "PlatformDatabase",
    "UserRecord",
    "SessionRecord",
    "ProjectRecord",
    "ConversationRecord",
    "MessageRecord",
    "MemoryRecord",
    "CapabilityRecord",
    "AgentRecord",
    "TaskRecord",
    "WorkflowRecord",
    "WorkflowRunRecord",
    "PolicyRecord",
    "ApprovalRecord",
    "AuditRecord",
    "ModelArtifactRecord",
    "PLATFORM_SCHEMA_VERSION",
]

PLATFORM_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Record data classes
# ---------------------------------------------------------------------------

@dataclass
class UserRecord:
    name: str
    email: str = ""
    role: str = "user"
    user_id: str | None = None
    created_at: float | None = None


@dataclass
class SessionRecord:
    session_id: str
    user_id: str | None = None
    project_id: str | None = None
    started_at: float | None = None
    ended_at: float | None = None


@dataclass
class ProjectRecord:
    name: str
    description: str = ""
    project_id: str | None = None
    created_at: float | None = None


@dataclass
class ConversationRecord:
    session_id: str
    title: str = ""
    conversation_id: str | None = None
    created_at: float | None = None


@dataclass
class MessageRecord:
    conversation_id: str
    role: str
    content: str
    sequence: int = 0
    message_id: str | None = None
    created_at: float | None = None


@dataclass
class MemoryRecord:
    scope: str
    key: str
    content: str
    memory_id: str | None = None
    importance: float = 1.0
    created_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityRecord:
    name: str
    version: str = "1.0.0"
    description: str = ""
    input_schema: str = ""
    output_schema: str = ""
    risk_level: str = "low"
    permissions_required: str = "L0"
    timeout_seconds: float = 30.0
    enabled: bool = True
    capability_type: str = "tool"
    tags: str = ""
    source: str = "builtin"


@dataclass
class AgentRecord:
    agent_id: str
    name: str
    model: str = "silverwing-v2"
    instructions: str = ""
    capabilities: str = ""
    memory_scope: str = "session"
    permission_level: str = "L0"
    policies: str = ""
    max_steps: int = 10
    max_runtime: float = 60.0
    created_at: float | None = None


@dataclass
class TaskRecord:
    task_id: str
    agent_id: str
    project_id: str | None
    status: str = "pending"
    input: str = ""
    output: str = ""
    created_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class WorkflowRecord:
    name: str
    description: str = ""
    steps: str = ""
    permissions: str = ""
    max_runtime: float = 300.0
    workflow_id: str | None = None


@dataclass
class WorkflowRunRecord:
    workflow_id: str
    status: str = "pending"
    input: str = ""
    output: str = ""
    run_id: str | None = None
    created_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class PolicyRecord:
    category: str
    decision: str = "allow"
    constraints: str = ""
    policy_id: str | None = None


@dataclass
class ApprovalRecord:
    request_id: str
    capability_id: str
    action: str
    target: str
    risk_level: str
    reason: str
    user_id: str = ""
    session_id: str = ""
    status: str = "pending"
    created_at: float | None = None
    expires_at: float | None = None
    decision: str = ""


@dataclass
class AuditRecord:
    request_id: str
    user_id: str = ""
    session_id: str = ""
    project_id: str = ""
    agent_id: str = ""
    capability_id: str = ""
    action: str = ""
    decision: str = ""
    approval_status: str = ""
    execution_status: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None


@dataclass
class ModelArtifactRecord:
    model_id: str
    version: str
    checkpoint: str = ""
    training_run: str = ""
    dataset_version: str = ""
    configuration: str = ""
    evaluation_metrics: str = ""
    status: str = "experimental"
    created_at: float | None = None
    artifact_id: str | None = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_PLATFORM_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT DEFAULT '',
    role         TEXT DEFAULT 'user',
    created_at   REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT REFERENCES users(user_id),
    project_id  TEXT REFERENCES projects(project_id),
    started_at  REAL,
    ended_at    REAL
);
CREATE TABLE IF NOT EXISTS projects (
    project_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT DEFAULT '',
    created_at   REAL
);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(session_id),
    title         TEXT DEFAULT '',
    created_at    REAL
);
CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(conversation_id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sequence        INTEGER DEFAULT 0,
    created_at      REAL
);
CREATE TABLE IF NOT EXISTS memory_records (
    memory_id    TEXT PRIMARY KEY,
    scope        TEXT NOT NULL,
    key          TEXT NOT NULL,
    content      TEXT NOT NULL,
    importance   REAL DEFAULT 1.0,
    created_at   REAL,
    metadata     TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS capabilities (
    name              TEXT PRIMARY KEY,
    version           TEXT DEFAULT '1.0.0',
    description       TEXT,
    input_schema      TEXT DEFAULT '{}',
    output_schema     TEXT DEFAULT '{}',
    risk_level        TEXT DEFAULT 'low',
    permissions_required TEXT DEFAULT 'L0',
    timeout_seconds   REAL DEFAULT 30.0,
    enabled           INTEGER DEFAULT 1,
    capability_type   TEXT DEFAULT 'tool',
    tags              TEXT DEFAULT '',
    source            TEXT DEFAULT 'builtin'
);
CREATE TABLE IF NOT EXISTS agents (
    agent_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    model           TEXT DEFAULT 'silverwing-v2',
    instructions    TEXT DEFAULT '',
    capabilities    TEXT DEFAULT '[]',
    memory_scope    TEXT DEFAULT 'session',
    permission_level TEXT DEFAULT 'L0',
    policies        TEXT DEFAULT '[]',
    max_steps       INTEGER DEFAULT 10,
    max_runtime     REAL DEFAULT 60.0,
    created_at      REAL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id       TEXT PRIMARY KEY,
    agent_id      TEXT REFERENCES agents(agent_id),
    project_id    TEXT REFERENCES projects(project_id),
    status        TEXT DEFAULT 'pending',
    input_text    TEXT DEFAULT '',
    output_text   TEXT DEFAULT '',
    created_at    REAL,
    started_at    REAL,
    completed_at  REAL
);
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT DEFAULT '',
    steps         TEXT DEFAULT '[]',
    permissions   TEXT DEFAULT '[]',
    max_runtime   REAL DEFAULT 300.0
);
CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id        TEXT PRIMARY KEY,
    workflow_id   TEXT REFERENCES workflows(workflow_id),
    status        TEXT DEFAULT 'pending',
    input_text    TEXT DEFAULT '',
    output_text   TEXT DEFAULT '',
    created_at    REAL,
    started_at    REAL,
    completed_at  REAL
);
CREATE TABLE IF NOT EXISTS policies (
    policy_id      TEXT PRIMARY KEY,
    category       TEXT NOT NULL,
    decision       TEXT DEFAULT 'allow',
    constraints    TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS approvals (
    request_id    TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL,
    action        TEXT NOT NULL,
    target        TEXT DEFAULT '',
    risk_level    TEXT NOT NULL,
    reason        TEXT DEFAULT '',
    user_id       TEXT DEFAULT '',
    session_id    TEXT DEFAULT '',
    status        TEXT DEFAULT 'pending',
    created_at    REAL,
    expires_at    REAL,
    decision      TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS platform_audit_events (
    event_id        TEXT PRIMARY KEY,
    request_id      TEXT,
    user_id         TEXT,
    session_id      TEXT,
    project_id      TEXT,
    agent_id        TEXT,
    capability_id   TEXT,
    action          TEXT,
    decision        TEXT,
    approval_status TEXT,
    execution_status TEXT,
    error           TEXT DEFAULT '',
    metadata        TEXT DEFAULT '{}',
    timestamp       REAL
);
CREATE TABLE IF NOT EXISTS model_registry (
    artifact_id       TEXT PRIMARY KEY,
    model_id          TEXT NOT NULL,
    version           TEXT NOT NULL,
    checkpoint        TEXT DEFAULT '',
    training_run      TEXT DEFAULT '',
    dataset_version   TEXT DEFAULT '',
    configuration     TEXT DEFAULT '{}',
    evaluation_metrics TEXT DEFAULT '{}',
    status            TEXT DEFAULT 'experimental',
    created_at        REAL
);
CREATE TABLE IF NOT EXISTS schema_version (
    key   TEXT PRIMARY KEY,
    value INTEGER
);
"""


class PlatformDatabase:
    """SQLite persistence for platform metadata.

    Wraps :class:`foundation.database.Database` and applies a single
    additive migration that creates all platform tables.  The underlying
    experiment-tracking schema is left intact.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path("platform_data/platform.db")
        self._path = Path(path)
        self._db = Database(str(self._path))
        self._db.initialize()
        self._apply_platform_schema()

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    @property
    def db(self) -> Database:
        return self._db

    @property
    def conn(self):
        return self._db.conn

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> PlatformDatabase:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _apply_platform_schema(self) -> None:
        # Check if the platform schema has already been applied
        try:
            cur = self._db.conn.execute(
                "SELECT value FROM schema_version WHERE key='platform_schema'"
            )
            row = cur.fetchone()
            if row and row[0]:
                return
        except Exception:
            pass
        # Create the platform schema (idempotent via IF NOT EXISTS)
        self._db.conn.executescript(_PLATFORM_SCHEMA)
        self._db.conn.execute(
            "INSERT OR REPLACE INTO schema_version (key, value) VALUES (?, ?)",
            ("platform_schema", PLATFORM_SCHEMA_VERSION),
        )
        self._db.conn.commit()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def upsert_user(self, user: UserRecord) -> str:
        user_id = user.user_id or f"usr-{uuid.uuid4().hex[:12]}"
        now = user.created_at or time.time()
        self._db.conn.execute(
            "INSERT OR REPLACE INTO users (user_id, name, email, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, user.name, user.email, user.role, now),
        )
        self._db.conn.commit()
        return user_id

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, session: SessionRecord) -> str:
        sid = session.session_id or f"sess-{uuid.uuid4().hex[:12]}"
        now = session.started_at or time.time()
        user_id = session.user_id or None
        project_id = session.project_id or None
        # Ensure referenced user/project exist to satisfy FK constraints
        if user_id:
            self._ensure_user_exists(user_id)
        if project_id:
            self._ensure_project_exists(project_id)
        self._db.conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, user_id, project_id, started_at) "
            "VALUES (?, ?, ?, ?)",
            (sid, user_id, project_id, now),
        )
        self._db.conn.commit()
        return sid

    def _ensure_user_exists(self, user_id: str) -> None:
        row = self._db.conn.execute(
            "SELECT 1 FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if row is None:
            self._db.conn.execute(
                "INSERT OR IGNORE INTO users (user_id, name, email, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, "anonymous", "", "user", time.time()),
            )

    def _ensure_project_exists(self, project_id: str) -> None:
        row = self._db.conn.execute(
            "SELECT 1 FROM projects WHERE project_id=?", (project_id,)
        ).fetchone()
        if row is None:
            self._db.conn.execute(
                "INSERT OR IGNORE INTO projects (project_id, name, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (project_id, "unnamed", "", time.time()),
            )

    def end_session(self, session_id: str) -> None:
        self._db.conn.execute(
            "UPDATE sessions SET ended_at=? WHERE session_id=?",
            (time.time(), session_id),
        )
        self._db.conn.commit()

    def list_sessions(self, user_id: str | None = None) -> list[dict[str, Any]]:
        if user_id:
            rows = self._db.conn.execute(
                "SELECT * FROM sessions WHERE user_id=? ORDER BY started_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Conversations & messages
    # ------------------------------------------------------------------

    def create_conversation(self, session_id: str, title: str = "") -> str:
        cid = f"conv-{uuid.uuid4().hex[:12]}"
        self._db.conn.execute(
            "INSERT INTO conversations (conversation_id, session_id, title, created_at) "
            "VALUES (?, ?, ?, ?)",
            (cid, session_id, title, time.time()),
        )
        self._db.conn.commit()
        return cid

    def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> str:
        mid = f"msg-{uuid.uuid4().hex[:12]}"
        seq = self._db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()[0]
        self._db.conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, content, sequence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, role, content, seq, time.time()),
        )
        self._db.conn.commit()
        return mid

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY sequence",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Approvals
    # ------------------------------------------------------------------

    def insert_approval(self, rec: ApprovalRecord) -> str:
        now = rec.created_at or time.time()
        self._db.conn.execute(
            "INSERT INTO approvals (request_id, capability_id, action, target, "
            "risk_level, reason, user_id, session_id, status, created_at, "
            "expires_at, decision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.request_id, rec.capability_id, rec.action, rec.target,
                rec.risk_level, rec.reason, rec.user_id, rec.session_id,
                rec.status, now, rec.expires_at, rec.decision,
            ),
        )
        self._db.conn.commit()
        return rec.request_id

    def update_approval(self, request_id: str, status: str, decision: str = "") -> None:
        self._db.conn.execute(
            "UPDATE approvals SET status=?, decision=? WHERE request_id=?",
            (status, decision, request_id),
        )
        self._db.conn.commit()

    def get_approval(self, request_id: str) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM approvals WHERE request_id=?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_approvals(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._db.conn.execute(
                "SELECT * FROM approvals WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                "SELECT * FROM approvals ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def insert_audit(self, rec: AuditRecord) -> str:
        event_id = f"evt-{uuid.uuid4().hex[:12]}"
        ts = rec.timestamp or time.time()
        self._db.conn.execute(
            "INSERT INTO platform_audit_events "
            "(event_id, request_id, user_id, session_id, project_id, agent_id, "
            "capability_id, action, decision, approval_status, execution_status, "
            "error, metadata, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id, rec.request_id, rec.user_id, rec.session_id,
                rec.project_id, rec.agent_id, rec.capability_id, rec.action,
                rec.decision, rec.approval_status, rec.execution_status,
                rec.error, json.dumps(rec.metadata), ts,
            ),
        )
        self._db.conn.commit()
        return event_id

    def query_audit(
        self,
        request_id: str | None = None,
        capability_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM platform_audit_events WHERE 1=1"
        params: list[Any] = []
        if request_id:
            query += " AND request_id = ?"
            params.append(request_id)
        if capability_id:
            query += " AND capability_id = ?"
            params.append(capability_id)
        query += f" ORDER BY timestamp DESC LIMIT {int(limit)}"
        rows = self._db.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(self, rec: ModelArtifactRecord) -> str:
        aid = rec.artifact_id or f"mdl-{uuid.uuid4().hex[:12]}"
        now = rec.created_at or time.time()
        self._db.conn.execute(
            "INSERT INTO model_registry "
            "(artifact_id, model_id, version, checkpoint, training_run, "
            "dataset_version, configuration, evaluation_metrics, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                aid, rec.model_id, rec.version, rec.checkpoint, rec.training_run,
                rec.dataset_version, rec.configuration, rec.evaluation_metrics,
                rec.status, now,
            ),
        )
        self._db.conn.commit()
        return aid

    def update_model_status(self, artifact_id: str, status: str) -> None:
        self._db.conn.execute(
            "UPDATE model_registry SET status=? WHERE artifact_id=?",
            (status, artifact_id),
        )
        self._db.conn.commit()

    def get_model(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM model_registry WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_promoted_model(self) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM model_registry WHERE status='promoted' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def list_models(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._db.conn.execute(
                "SELECT * FROM model_registry WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                "SELECT * FROM model_registry ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def register_capability(self, rec: CapabilityRecord) -> str:
        self._db.conn.execute(
            "INSERT OR REPLACE INTO capabilities "
            "(name, version, description, input_schema, output_schema, "
            "risk_level, permissions_required, timeout_seconds, enabled, "
            "capability_type, tags, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.name, rec.version, rec.description, rec.input_schema,
                rec.output_schema, rec.risk_level, rec.permissions_required,
                rec.timeout_seconds, int(rec.enabled), rec.capability_type,
                rec.tags, rec.source,
            ),
        )
        self._db.conn.commit()
        return rec.name

    def list_capabilities(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        if enabled_only:
            rows = self._db.conn.execute(
                "SELECT * FROM capabilities WHERE enabled=1"
            ).fetchall()
        else:
            rows = self._db.conn.execute("SELECT * FROM capabilities").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def register_agent(self, rec: AgentRecord) -> str:
        now = rec.created_at or time.time()
        self._db.conn.execute(
            "INSERT OR REPLACE INTO agents "
            "(agent_id, name, model, instructions, capabilities, memory_scope, "
            "permission_level, policies, max_steps, max_runtime, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.agent_id, rec.name, rec.model, rec.instructions,
                rec.capabilities, rec.memory_scope, rec.permission_level,
                rec.policies, rec.max_steps, rec.max_runtime, now,
            ),
        )
        self._db.conn.commit()
        return rec.agent_id

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM agents WHERE agent_id=?", (agent_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_agents(self) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT * FROM agents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def create_task(self, rec: TaskRecord) -> str:
        tid = rec.task_id or f"task-{uuid.uuid4().hex[:12]}"
        now = rec.created_at or time.time()
        agent_id = rec.agent_id or None
        project_id = rec.project_id or None
        # Ensure referenced agent and project exist to satisfy FK constraints
        if agent_id:
            self._ensure_agent_exists(agent_id)
        if project_id:
            self._ensure_project_exists(project_id)
        self._db.conn.execute(
            "INSERT INTO tasks (task_id, agent_id, project_id, status, input_text, "
            "output_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, agent_id, project_id, rec.status, rec.input, "", now),
        )
        self._db.conn.commit()
        return tid

    def _ensure_agent_exists(self, agent_id: str) -> None:
        row = self._db.conn.execute(
            "SELECT 1 FROM agents WHERE agent_id=?", (agent_id,)
        ).fetchone()
        if row is None:
            self._db.conn.execute(
                "INSERT OR IGNORE INTO agents "
                "(agent_id, name, model, instructions, capabilities, memory_scope, "
                "permission_level, policies, max_steps, max_runtime, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (agent_id, "unnamed", "default", "", "[]", "session", "L0", "[]", 10, 60.0, time.time()),
            )

    def update_task(
        self, task_id: str, status: str, output: str = ""
    ) -> None:
        now = time.time()
        self._db.conn.execute(
            "UPDATE tasks SET status=?, output_text=?, completed_at=? "
            "WHERE task_id=?",
            (status, output, now if status in ("done", "failed", "completed") else None, task_id),
        )
        self._db.conn.commit()

    def list_tasks(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        if agent_id:
            rows = self._db.conn.execute(
                "SELECT * FROM tasks WHERE agent_id=? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    def register_workflow(self, rec: WorkflowRecord) -> str:
        wid = rec.workflow_id or f"wf-{uuid.uuid4().hex[:12]}"
        self._db.conn.execute(
            "INSERT OR REPLACE INTO workflows "
            "(workflow_id, name, description, steps, permissions, max_runtime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (wid, rec.name, rec.description, rec.steps, rec.permissions, rec.max_runtime),
        )
        self._db.conn.commit()
        return wid

    def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_workflows(self) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT * FROM workflows ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Memory records
    # ------------------------------------------------------------------

    def store_memory(self, rec: MemoryRecord) -> str:
        mid = rec.memory_id or f"mem-{uuid.uuid4().hex[:12]}"
        now = rec.created_at or time.time()
        self._db.conn.execute(
            "INSERT INTO memory_records (memory_id, scope, key, content, "
            "importance, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, rec.scope, rec.key, rec.content, rec.importance, now,
             json.dumps(rec.metadata)),
        )
        self._db.conn.commit()
        return mid

    def search_memory(self, query: str, scope: str | None = None) -> list[dict[str, Any]]:
        q = "%" + query + "%"
        if scope:
            rows = self._db.conn.execute(
                "SELECT * FROM memory_records WHERE scope=? AND "
                "(key LIKE ? OR content LIKE ?) ORDER BY importance DESC, created_at DESC",
                (scope, q, q),
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                "SELECT * FROM memory_records WHERE key LIKE ? OR content LIKE ? "
                "ORDER BY importance DESC, created_at DESC",
                (q, q),
            ).fetchall()
        return [dict(r) for r in rows]
