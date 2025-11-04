"""
Misc helpers used by the executor: pg_connect(), market_is_open(), retry().
"""

from __future__ import annotations
import os
import time
from typing import Callable, TypeVar, Any, Optional, cast

import psycopg2
from alpaca.trading.client import TradingClient

T = TypeVar("T")

def pg_connect():
    """
    Connect to Postgres using env var DB_URL like:
      postgresql://postgres:postgres@postgres:5432/trader

    If DB_URL is not set, fall back to discrete DB_* env vars.
    """
    url = os.getenv("DB_URL", "")
    if not url:
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
    Ask Alpaca market clock; be conservative on error.
    """
    try:
        cli = TradingClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET"),
            paper=True,
        )
        clk = cli.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return False


def retry(tries: int = 3, delay: float = 1.0, exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Minimal retry decorator. Retries `tries` times on `exceptions` with `delay` seconds.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for i in range(max(1, tries)):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if i == tries - 1:
                        raise
                    time.sleep(delay)
            # mypy/pyright appeasement; we always raise above
            if last_exc:
                raise last_exc
            return cast(T, None)
        return wrapped
    return deco
