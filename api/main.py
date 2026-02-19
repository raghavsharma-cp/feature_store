"""
API application entry: FastAPI app, lifespan, and router mounting.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import get_settings
from api.routers import health, patients


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: e.g. init DB pool, cleanup."""
    yield
    # Shutdown cleanup if needed


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="Feature Store API",
        description="APIs that read and update the PostgreSQL feature store.",
        env=settings.api_env,
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(patients.router, prefix="/patients", tags=["patients"])
    return app


app = create_app()
