"""FastAPI app for STEAD research artifacts."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes_analysis import router as analysis_router
from src.api.routes_events import router as events_router


def create_app() -> FastAPI:
    """Create the STEAD API application."""
    app = FastAPI(title="STEAD Analysis API", version="0.1.0")
    app.include_router(analysis_router)
    app.include_router(events_router)
    return app


app = create_app()

