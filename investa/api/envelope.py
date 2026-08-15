"""Shared response envelope for all Investa API routes.

Every endpoint returns ``{"success": bool, "data": ..., "meta": {...}}`` so
the frontend can rely on a single shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Meta(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total: int | None = None
    scanned_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Envelope(BaseModel, Generic[T]):
    success: bool = True
    data: T | None = None
    meta: Meta = Field(default_factory=Meta)


def ok(data: Any = None, **meta: Any) -> dict[str, Any]:
    """Build a success envelope as a plain dict (fast, avoids generic serialisation)."""
    return {
        "success": True,
        "data": data,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **meta,
        },
    }


def fail(error: str, **meta: Any) -> dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": error,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **meta,
        },
    }
