"""Lightweight SQLite helper for STEAD artifacts."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SteadDatabase:
    """SQLite store for paths to evidence, VLM review, and alarm JSON files."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stead_events (
                    event_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    evidence_path TEXT,
                    vlm_review_path TEXT,
                    alarm_result_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

