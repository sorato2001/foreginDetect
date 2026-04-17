"""
Database module for event persistence.
Handles SQLite operations for storing and retrieving event records.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import EventRecord


logger = logging.getLogger(__name__)


class EventDatabase:
    """Manager for SQLite event database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Create a connection using Row objects for stable column access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema if not exists and migrate missing columns."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                media_type TEXT NOT NULL DEFAULT 'video',
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL,
                event_start_time TEXT NOT NULL,
                event_end_time TEXT NOT NULL,
                image_path TEXT,
                video_path TEXT,
                annotated_image_path TEXT,
                analysis_result TEXT,
                status TEXT NOT NULL DEFAULT 'completed',
                alarm_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(camera_id, event_start_time)
            )
            """
        )

        cursor.execute("PRAGMA table_info(events)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "analysis_result" not in columns:
            cursor.execute("ALTER TABLE events ADD COLUMN analysis_result TEXT")
        if "media_type" not in columns:
            cursor.execute(
                "ALTER TABLE events ADD COLUMN media_type TEXT NOT NULL DEFAULT 'video'"
            )
        if "annotated_image_path" not in columns:
            cursor.execute("ALTER TABLE events ADD COLUMN annotated_image_path TEXT")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_camera_id ON events(camera_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_time ON events(event_time)"
        )

        conn.commit()
        conn.close()
        logger.info("Database initialized at %s", self.db_path)

    def add_event(self, event: EventRecord) -> int:
        """Add a new event record."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO events (
                    camera_id, media_type, event_type, event_time, event_start_time,
                    event_end_time, image_path, video_path, annotated_image_path,
                    analysis_result, status, alarm_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.camera_id,
                    event.media_type,
                    event.event_type,
                    event.event_time.isoformat(),
                    event.event_start_time.isoformat(),
                    event.event_end_time.isoformat(),
                    event.image_path,
                    event.video_path,
                    event.annotated_image_path,
                    json.dumps(event.analysis_result) if event.analysis_result is not None else None,
                    event.status,
                    event.alarm_count,
                    event.created_at.isoformat(),
                    event.updated_at.isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_event(self, event_id: int, event: EventRecord) -> None:
        """Update an existing event record."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE events SET
                    camera_id=?, media_type=?, event_type=?, event_time=?, event_start_time=?,
                    event_end_time=?, image_path=?, video_path=?, annotated_image_path=?,
                    analysis_result=?, status=?, alarm_count=?, updated_at=?
                WHERE id=?
                """,
                (
                    event.camera_id,
                    event.media_type,
                    event.event_type,
                    event.event_time.isoformat(),
                    event.event_start_time.isoformat(),
                    event.event_end_time.isoformat(),
                    event.image_path,
                    event.video_path,
                    event.annotated_image_path,
                    json.dumps(event.analysis_result) if event.analysis_result is not None else None,
                    event.status,
                    event.alarm_count,
                    event.updated_at.isoformat(),
                    event_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_event(self, event_id: int) -> Optional[EventRecord]:
        """Get event by ID."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM events WHERE id=?", (event_id,))
            row = cursor.fetchone()
            return self._row_to_event(row) if row else None
        finally:
            conn.close()

    def get_events_by_camera(self, camera_id: str, limit: int = 100) -> List[EventRecord]:
        """Get events for a specific camera."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT * FROM events WHERE camera_id=?
                ORDER BY event_time DESC
                LIMIT ?
                """,
                (camera_id, limit),
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_all_events(self, limit: int = 1000) -> List[EventRecord]:
        """Get all events ordered by descending event_time."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT * FROM events
                ORDER BY event_time DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_event(self, event_id: int) -> None:
        """Delete an event record."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM events WHERE id=?", (event_id,))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> EventRecord:
        """Convert database row to EventRecord."""
        analysis_result = None
        try:
            if row["analysis_result"]:
                analysis_result = json.loads(row["analysis_result"])
        except Exception:
            analysis_result = None

        media_type = row["media_type"] if "media_type" in row.keys() and row["media_type"] else "video"
        annotated_image_path = (
            row["annotated_image_path"]
            if "annotated_image_path" in row.keys()
            else None
        )

        return EventRecord(
            id=row["id"],
            camera_id=row["camera_id"],
            media_type=media_type,
            event_type=row["event_type"],
            event_time=datetime.fromisoformat(row["event_time"]),
            event_start_time=datetime.fromisoformat(row["event_start_time"]),
            event_end_time=datetime.fromisoformat(row["event_end_time"]),
            image_path=row["image_path"],
            video_path=row["video_path"],
            annotated_image_path=annotated_image_path,
            analysis_result=analysis_result,
            status=row["status"],
            alarm_count=row["alarm_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
