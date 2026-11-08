"""FastAPI application entrypoint.

This process serves the HTTP API only. Background scans and scheduled jobs
run in a separate worker process (see ``investa/worker.py``); the two share
state via the ``data/`` directory (SQLite + DuckDB files).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.backtest import router as backtest_router
from .api.health import router as health_router
from .api.screener import router as screener_router
from .api.universe import router as universe_router
from .backtest_v2.api_routes import router as backtest_v2_router
from .backtest_v2 import models as _v2_models  # noqa: F401 — registers BacktestV2Job
from .config import get_settings
from .data import models as _base_models  # noqa: F401 — registers UniverseSnapshot etc.
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
    app.include_router(screener_router)
    app.include_router(backtest_router)
    app.include_router(backtest_v2_router)
    app.include_router(universe_router)
    return app


app = create_app()
