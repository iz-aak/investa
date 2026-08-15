"""FastAPI application entrypoint.

This process serves the HTTP API only. Background scans and scheduled jobs
run in a separate worker process (see ``investa/worker.py``); the two share
state via the ``data/`` directory (SQLite + DuckDB files).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.health import router as health_router
from .config import get_settings
from .data.duckdb_store import init_duckdb
from .database import init_db
from .logging_setup import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    log = get_logger("investa.api")

    settings.ensure_dirs()
    init_db(settings)
    init_duckdb(settings)

    log.info(
        "api.startup",
        env=settings.app_env.value,
        sqlite=str(settings.sqlite_path),
        duckdb=str(settings.duckdb_path),
    )
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Investa",
        version="0.1.0",
        description="AI-powered reversal stock screener.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    return app


app = create_app()
