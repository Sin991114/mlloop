"""SQLite-backed experiment ledger with an append-only JSONL event mirror.

The SQLite database is the source of truth; the JSONL file is a replayable,
human-greppable audit trail. Both live under ``<workspace>/.mlloop/``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS goal (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    created_at TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN ('classification', 'regression')),
    dataset_path TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    target_column TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    metric_direction TEXT NOT NULL CHECK (metric_direction IN ('maximize', 'minimize')),
    target_value REAL,
    monitor_metrics TEXT NOT NULL DEFAULT '[]',
    constraints TEXT NOT NULL DEFAULT '{}',
    policy TEXT NOT NULL DEFAULT '{}',
    metric_script TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    statement TEXT NOT NULL,
    rationale TEXT NOT NULL,
    prediction TEXT NOT NULL,
    test_plan TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'testing', 'confirmed', 'refuted', 'inconclusive')),
    resolved_at TEXT,
    resolution_narrative TEXT,
    evidence_run_ids TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    parent_run_id TEXT REFERENCES runs (id),
    hypothesis_id TEXT REFERENCES hypotheses (id),
    kind TEXT NOT NULL DEFAULT 'experiment'
        CHECK (kind IN ('baseline', 'experiment', 'forensics')),
    intent TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'finished', 'abandoned')),
    abandon_reason TEXT,
    artifact_dir TEXT NOT NULL,
    metrics TEXT,
    meta TEXT,
    artifact_report TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence TEXT,
    next_action TEXT
);

CREATE TABLE IF NOT EXISTS feature_context (
    feature TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    meaning TEXT NOT NULL,
    source TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS fe_probes (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    results TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ensemble_probes (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    results TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnoses (
    run_id TEXT PRIMARY KEY REFERENCES runs (id),
    created_at TEXT NOT NULL,
    results TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forensics (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    quick INTEGER NOT NULL DEFAULT 0,
    results TEXT NOT NULL,
    verdict TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    """Owns the ``.mlloop`` directory: database, event mirror, run artifact dirs."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / ".mlloop"
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "ledger.db"
        self.events_path = self.root / "events.jsonl"
        with self.connect() as con:
            con.executescript(SCHEMA)
            # Lightweight migration for ledgers created before the column existed.
            goal_columns = [row[1] for row in con.execute("PRAGMA table_info(goal)")]
            if "metric_script" not in goal_columns:
                con.execute("ALTER TABLE goal ADD COLUMN metric_script TEXT")

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def emit(self, con: sqlite3.Connection, kind: str, payload: dict) -> None:
        ts = utcnow()
        con.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (ts, kind, json.dumps(payload, ensure_ascii=False)),
        )
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**payload, "ts": ts, "kind": kind}, ensure_ascii=False) + "\n")

    def next_id(self, con: sqlite3.Connection, table: str, prefix: str) -> str:
        # Rows are never deleted, so COUNT is a safe monotonic counter.
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return f"{prefix}{n + 1}"
