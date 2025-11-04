"""
Misc helpers used by the executor: pg_connect() and market_is_open().
"""

from __future__ import annotations
import os
import psycopg2
from typing import Optional

from alpaca.trading.client import TradingClient


def pg_connect():
    """
    Connect to Postgres using env var DB_URL like:
    postgresql://postgres:postgres@postgres:5432/trader
    """
    url = os.getenv("DB_URL", "")
    if not url:
        # Fallback to individual envs if provided
        host = os.getenv("DB_HOST", "localhost")
        port = int(os.getenv("DB_PORT", "5432"))
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        db = os.getenv("DB_NAME", "trader")
        dsn = f"host={host} port={port} user={user} password={password} dbname={db}"
        return psycopg2.connect(dsn)
    return psycopg2.connect(url)


def market_is_open() -> bool:
    """
    Ask Alpaca clock; be conservative on error.
    """
    try:
        cli = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)
        clk = cli.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return False
