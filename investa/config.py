"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class AppEnv(str, Enum):
    development = "development"
    production = "production"
    test = "test"


class ScanMode(str, Enum):
    deep_reversal = "deep_reversal"
    pullback = "pullback"


class Interval(str, Enum):
    daily = "1d"
    weekly = "1wk"


class Settings(BaseSettings):
    """Global application settings.

    Values are populated from environment variables or a local ``.env`` file.
    Access via :func:`get_settings` (memoised) rather than instantiating directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────
    app_env: AppEnv = AppEnv.development
    log_level: str = "INFO"
    log_json: bool = False

    # ── Paths ───────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    sqlite_path: Path = Path("./data/investa.sqlite")
    duckdb_path: Path = Path("./data/signals.duckdb")
    universes_dir: Path = Path("./data/universes")

    # ── Scanner ─────────────────────────────────────────────────
    default_universe: str = "sp500"
    default_mode: ScanMode = ScanMode.deep_reversal
    default_interval: Interval = Interval.daily
    min_score_threshold: float = 60.0
    scan_batch_size: int = 50
    scan_batch_delay_seconds: float = 0.5
    scan_lookback_days: int = 260

    # ── Scheduler ───────────────────────────────────────────────
    scan_cron_premarket: str = "0 6 * * 1-5"
    scan_cron_open: str = "30 9 * * 1-5"
    scan_cron_close: str = "5 16 * * 1-5"
    market_context_cron: str = "0 7 * * 1"

    # ── Universe (Stage 1) ──────────────────────────────────────
    # Filtering thresholds applied when building the eligible US ticker
    # universe from the NASDAQ symbol directory. All overridable per-call
    # via POST /api/universe/rebuild.
    universe_min_avg_volume_usd: int = 1_000_000
    universe_min_share_price: float = 3.0
    universe_min_market_cap: int = 50_000_000
    universe_min_history_days: int = 250
    universe_max_sector_pct: float = 0.20
    universe_batch_size: int = 50
    universe_batch_delay_seconds: float = 0.3
    universe_nasdaq_listed_url: str = (
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    )
    universe_other_listed_url: str = (
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    )

    # ── Cache ───────────────────────────────────────────────────
    yf_cache_ttl_seconds: int = 900
    ticker_validation_ttl_seconds: int = 86_400

    # ── API ─────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Annotated with NoDecode so pydantic-settings passes the raw env string
    # to our validator instead of trying to JSON-parse it as a list.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v: object) -> object:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_path}"

    def ensure_dirs(self) -> None:
        """Create data directories if they do not yet exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.universes_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached global settings instance."""
    return Settings()
