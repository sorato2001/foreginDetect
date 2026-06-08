"""Railway perimeter security analysis package.

This package implements the demo pipeline:
YOLO candidate detection -> region rules -> optional VLM review -> final alarm decision.
"""

from .event_analyzer import EventAnalyzer

__all__ = ["EventAnalyzer"]
