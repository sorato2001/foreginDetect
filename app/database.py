"""
Database module for event persistence.
Handles SQLite operations for storing and retrieving event records.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from .models import EventRecord


logger = logging.getLogger(__name__)


class EventDatabase:
    """Manager for SQLite event database."""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema if not exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL,
                event_start_time TEXT NOT NULL,
                event_end_time TEXT NOT NULL,
                image_path TEXT,
                video_path TEXT,
                analysis_result TEXT,
                status TEXT NOT NULL DEFAULT 'completed',
                alarm_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(camera_id, event_start_time)
            )
        """)

        # Ensure analysis_result column exists (for older databases)
        cursor.execute("PRAGMA table_info(events)")
        columns = [row[1] for row in cursor.fetchall()]
        if "analysis_result" not in columns:
            cursor.execute("ALTER TABLE events ADD COLUMN analysis_result TEXT")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_camera_id ON events(camera_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_time ON events(event_time)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def add_event(self, event: EventRecord) -> int:
        """
        Add a new event record.
        
        Args:
            event: EventRecord to store
            
        Returns:
            ID of inserted record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO events (
                    camera_id, event_type, event_time, event_start_time,
                    event_end_time, image_path, video_path, analysis_result, status,
                    alarm_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.camera_id,
                event.event_type,
                event.event_time.isoformat(),
                event.event_start_time.isoformat(),
                event.event_end_time.isoformat(),
                event.image_path,
                event.video_path,
                json.dumps(event.analysis_result) if event.analysis_result is not None else None,
                event.status,
                event.alarm_count,
                event.created_at.isoformat(),
                event.updated_at.isoformat(),
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
            logger.debug(f"Event recorded: camera={event.camera_id}, id={record_id}")
            return record_id
            
        finally:
            conn.close()
    
    def update_event(self, event_id: int, event: EventRecord) -> None:
        """
        Update an existing event record.
        
        Args:
            event_id: ID of record to update
            event: Updated EventRecord
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE events SET
                    camera_id=?, event_type=?, event_time=?, event_start_time=?,
                    event_end_time=?, image_path=?, video_path=?, analysis_result=?, status=?,
                    alarm_count=?, updated_at=?
                WHERE id=?
            """, (
                event.camera_id,
                event.event_type,
                event.event_time.isoformat(),
                event.event_start_time.isoformat(),
                event.event_end_time.isoformat(),
                event.image_path,
                event.video_path,
                json.dumps(event.analysis_result) if event.analysis_result is not None else None,
                event.status,
                event.alarm_count,
                event.updated_at.isoformat(),
                event_id,
            ))
            
            conn.commit()
            logger.debug(f"Event updated: id={event_id}")
            
        finally:
            conn.close()
    
    def get_event(self, event_id: int) -> Optional[EventRecord]:
        """
        Get event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            EventRecord if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM events WHERE id=?", (event_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_event(row)
            
        finally:
            conn.close()
    
    def get_events_by_camera(self, camera_id: str, limit: int = 100) -> List[EventRecord]:
        """
        Get events for a specific camera.
        
        Args:
            camera_id: Camera identifier
            limit: Maximum number of records to return
            
        Returns:
            List of EventRecord objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM events WHERE camera_id=?
                ORDER BY event_time DESC
                LIMIT ?
            """, (camera_id, limit))
            
            rows = cursor.fetchall()
            return [self._row_to_event(row) for row in rows]
            
        finally:
            conn.close()
    
    def get_all_events(self, limit: int = 1000) -> List[EventRecord]:
        """
        Get all events.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of EventRecord objects sorted by event_time descending
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM events
                ORDER BY event_time DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_event(row) for row in rows]
            
        finally:
            conn.close()
    
    def delete_event(self, event_id: int) -> None:
        """
        Delete an event record.
        
        Args:
            event_id: Event ID to delete
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM events WHERE id=?", (event_id,))
            conn.commit()
            logger.debug(f"Event deleted: id={event_id}")
            
        finally:
            conn.close()
    
    @staticmethod
    def _row_to_event(row: tuple) -> EventRecord:
        """Convert database row tuple to EventRecord."""
        analysis_result = None
        try:
            if row[8]:
                analysis_result = json.loads(row[8])
        except Exception:
            analysis_result = None

        return EventRecord(
            id=row[0],
            camera_id=row[1],
            event_type=row[2],
            event_time=datetime.fromisoformat(row[3]),
            event_start_time=datetime.fromisoformat(row[4]),
            event_end_time=datetime.fromisoformat(row[5]),
            image_path=row[6],
            video_path=row[7],
            analysis_result=analysis_result,
            status=row[9],
            alarm_count=row[10],
            created_at=datetime.fromisoformat(row[11]),
            updated_at=datetime.fromisoformat(row[12]),
        )
