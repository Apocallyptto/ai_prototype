"""
tools.db

Small SQLAlchemy helper used by services.

- Reads DB_URL (preferred) or DATABASE_URL.
- Accepts either SQLAlchemy style (postgresql+psycopg://...) or plain postgresql://...
- Caches the Engine.
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_ENGINE: Optional[Engine] = None


def _normalize_sqlalchemy_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        raise RuntimeError("DB_URL / DATABASE_URL not set")

    # If user provides plain postgresql:// URL, prefer psycopg (v3) driver.
    # (psycopg[binary] is in requirements.)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


def get_engine() -> Engine:
    """Return a cached SQLAlchemy Engine."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    raw = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    url = _normalize_sqlalchemy_url(raw or "")

    _ENGINE = create_engine(
        url,
        pool_pre_ping=True,
        future=True,
    )
    return _ENGINE
