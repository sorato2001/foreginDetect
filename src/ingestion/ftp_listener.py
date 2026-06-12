"""External alarm ingestion adapter shell."""

from __future__ import annotations


class FTPListener:
    """Adapter shell for future file/event trigger integration."""

    def __init__(self, watch_dir: str) -> None:
        self.watch_dir = watch_dir
