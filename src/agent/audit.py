from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("data") / "nexusops_audit.db"


class AuditRepository:
    """SQLite-backed persistence for NexusOps investigations and audit events."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    investigation_id INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    step TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (investigation_id)
                        REFERENCES investigations(id)
                )
                """
            )

            connection.commit()

    def create_investigation(
        self,
        question: str,
        status: str = "started",
    ) -> int:
        """Create a persistent investigation and return its ID."""

        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO investigations (
                    question,
                    status,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    question,
                    status,
                    created_at,
                ),
            )

            connection.commit()
            return int(cursor.lastrowid)

    def record_event(
        self,
        investigation_id: int,
        event: str,
        step: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Persist one investigation audit event."""

        timestamp = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    investigation_id,
                    event,
                    step,
                    details,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    investigation_id,
                    event,
                    step,
                    json.dumps(
                        details or {},
                        default=str,
                    ),
                    timestamp,
                ),
            )

            connection.commit()

    def update_investigation_status(
        self,
        investigation_id: int,
        status: str,
    ) -> None:
        """Update the final status of an investigation."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE investigations
                SET status = ?
                WHERE id = ?
                """,
                (
                    status,
                    investigation_id,
                ),
            )

            connection.commit()

    def get_investigation(
        self,
        investigation_id: int,
    ) -> dict[str, Any] | None:
        """Return one investigation."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    question,
                    status,
                    created_at
                FROM investigations
                WHERE id = ?
                """,
                (investigation_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_events(
        self,
        investigation_id: int,
    ) -> list[dict[str, Any]]:
        """Return all audit events for an investigation."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    investigation_id,
                    event,
                    step,
                    details,
                    timestamp
                FROM audit_events
                WHERE investigation_id = ?
                ORDER BY id ASC
                """,
                (investigation_id,),
            ).fetchall()

        events: list[dict[str, Any]] = []

        for row in rows:
            event = dict(row)

            try:
                event["details"] = json.loads(
                    event["details"]
                )
            except json.JSONDecodeError:
                pass

            events.append(event)

        return events