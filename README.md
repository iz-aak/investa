# Investa

AI-powered reversal stock screener. See the project plan for full context.

This repository is being built in incremental steps. The current state is
**Step 1: Skeleton** — configuration, structured logging, both databases
(SQLite + DuckDB), FastAPI app with a `/api/health` endpoint, a background
worker entrypoint (no jobs registered yet), and Docker + docker-compose.

## Layout

```
investa/
├── investa/                     # Application package
│   ├── config.py                # pydantic-settings
│   ├── logging_setup.py         # structlog
│   ├── database.py              # SQLAlchemy 2.x engine + session_scope
│   ├── main.py                  # FastAPI app (API process)
│   ├── worker.py                # APScheduler host (worker process)
│   ├── data/
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   └── duckdb_store.py      # DuckDB signal-history store
│   ├── api/
│   │   ├── envelope.py          # Shared response envelope
│   │   └── health.py            # /api/health
│   ├── pipeline/                # (filled in later steps)
│   └── backtest/                # (filled in later steps)
├── data/
│   └── universes/               # SP500.csv, NDX100.csv, ... (added in step 7)
├── tests/                       # pytest suite
├── Dockerfile                   # Multi-stage build (TA-Lib compiled in)
├── docker-compose.yml           # api + worker sharing a data volume
├── pyproject.toml
└── .env.example
```

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# TA-Lib C library (macOS):
brew install ta-lib
# TA-Lib C library (Debian/Ubuntu): compile from source — see Dockerfile.

cp .env.example .env
```

## Running

```bash
# API process
uvicorn investa.main:app --reload

# Worker process (separate terminal)
python -m investa.worker
```

Health check: `curl http://localhost:8000/api/health`

## Docker

```bash
docker compose up --build
```

Brings up both services with a shared `investa_data` volume.

## Tests

```bash
pytest -m "not integration"
```

Skeleton tests do not require TA-Lib or network access.
