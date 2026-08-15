"""Background worker entrypoint.

Runs APScheduler jobs (periodic market scans, weekly market-context refresh)
in a process separate from the FastAPI app. Kept as a skeleton for now; scan
and market-context jobs are wired in later build steps.

Run with:  ``python -m investa.worker``
"""

from __future__ import annotations

import signal
import time

from apscheduler.schedulers.background import BackgroundScheduler

from .config import get_settings
from .data.duckdb_store import init_duckdb
from .database import init_db
from .logging_setup import configure_logging, get_logger


def build_scheduler() -> BackgroundScheduler:
    """Construct the scheduler. Jobs are registered by subsequent build steps."""
    scheduler = BackgroundScheduler(timezone="UTC")
    # Jobs will be registered here in later steps:
    #   - periodic scans (pre-market / open / close)
    #   - weekly market context refresh
    return scheduler


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    log = get_logger("investa.worker")

    settings.ensure_dirs()
    init_db(settings)
    init_duckdb(settings)

    scheduler = build_scheduler()
    scheduler.start()
    log.info("worker.started", jobs=[j.id for j in scheduler.get_jobs()])

    stop = False

    def _handle(signum, _frame):  # noqa: ANN001 — signal handler signature
        nonlocal stop
        log.info("worker.stopping", signal=signum)
        stop = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    try:
        while not stop:
            time.sleep(1)
    finally:
        scheduler.shutdown(wait=False)
        log.info("worker.stopped")


if __name__ == "__main__":
    main()
