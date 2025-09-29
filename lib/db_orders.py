# lib/db_orders.py
"""
Safe, single-purpose helpers for logging orders into Postgres.

- Ensures a symbol row exists (creates it if missing).
- Uses an independent transaction per insert.
- Never poisons a long-lived transaction; rolls back on error.
"""

from __future__ import annotations
from typing import Optional
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from lib.db import make_engine

_engine = None


def _eng():
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def ensure_symbol(conn, ticker: str) -> int:
    """
    Ensure a row exists in symbols for `ticker` and return its id.
    Requires a unique index/constraint on symbols.ticker.
    """
    return conn.execute(
        sa.text(
            """
            INSERT INTO symbols (ticker)
            VALUES (:t)
            ON CONFLICT (ticker) DO UPDATE SET ticker = EXCLUDED.ticker
            RETURNING id
            """
        ),
        {"t": ticker},
    ).scalar()


def log_order_row(
    *,
    ts_iso: str,
    ticker: str,
    side: str,
    qty: float,
    order_type: str,
    limit_price: Optional[float],
    status: str,
    logger=None,
) -> bool:
    """
    Insert one row into `orders` (fresh transaction). Returns True on success.
    """
    eng = _eng()
    try:
        with eng.begin() as conn:
            sym_id = ensure_symbol(conn, ticker)
            conn.execute(
                sa.text(
                    """
                    INSERT INTO orders
                        (ts, symbol_id, ticker, side, qty, order_type, limit_price, status)
                    VALUES
                        (:ts, :sym_id, :ticker, :side, :qty, :otype, :lim, :status)
                    """
                ),
                {
                    "ts": ts_iso,
                    "sym_id": sym_id,
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "otype": order_type,
                    "lim": limit_price,
                    "status": status,
                },
            )
        return True
    except (IntegrityError, SQLAlchemyError) as e:
        if logger:
            logger.error("DB insert failed for %s %s: %s", ticker, side, e)
        return False
