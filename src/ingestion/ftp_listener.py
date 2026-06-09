"""Compatibility placeholder for FTP alarm ingestion.

The existing FTP image listener remains in ``app.event_listener``.
"""

from __future__ import annotations


class FTPListener:
    """Adapter shell for future FTP trigger integration."""

    def __init__(self, watch_dir: str) -> None:
        self.watch_dir = watch_dir

