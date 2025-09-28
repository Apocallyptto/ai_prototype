# jobs/make_signals.py
"""
Generate demo signals into the DB.

- Ensures each ticker exists in `symbols` and grabs its `id`.
- Inserts recent signals into `signals` with `symbol_id` set.
- Fills both plain columns (side, strength) and JSONB `signal` (if present).

Env:
  SIGNAL_TICKERS  e.g. "AAPL,MSFT,SPY"   (default)
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa

# Use your repo's helper that already knows how to read env/secrets
from lib.db import make_engine as get_engine


def parse_tickers() -> list[tuple[str, str]]:
    """
    Returns a list of (ticker, name). If you don't know names, you can keep them = ticker.
    """
    env_val = os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY")
    tickers = [t.strip().upper() for t in env_val.split(",") if t.strip()]
    # very light name mapping; adjust if you like
    name_map = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "SPY": "SPDR S&P 500 ETF",
    }
    return [(t, name_map.get(t, t)) for t in tickers]


def upsert_symbol(conn: sa.Connection, ticker: str, name: str) -> int:
    """
    Ensure a row exists in `symbols` and return its id.
    Requires a UNIQUE/PK on symbols.ticker (typical).
    """
    sql = sa.text(
        """
        INSERT INTO symbols (ticker, name)
        VALUES (:t, :n)
        ON CONFLICT (ticker) DO UPDATE
            SET name = EXCLUDED.name
        RETURNING id
        """
    )
    return conn.execute(sql, {"t": ticker, "n": name}).scalar_one()


def insert_signal_row(
    conn: sa.Connection,
    ts: datetime,
    symbol_id: int,
    ticker: str,
    timeframe: str,
    model: str,
    side: str,
    strength: float,
) -> None:
    """
    Insert one row into `signals`. Tries with JSONB `signal`, falls back to plain cols
    if that column isn't present in your schema.
    """
    params = {
        "ts": ts,
        "symbol_id": symbol_id,
        "ticker": ticker,
        "timeframe": timeframe,
        "model": model,
        "side": side,
        "strength": strength,
    }

    try:
        # Preferred (if your table has a JSONB 'signal' column)
        conn.execute(
            sa.text(
                """
                INSERT INTO signals
                    (ts, symbol_id, ticker, timeframe, model, side, strength, signal)
                VALUES
                    (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength,
                     jsonb_build_object('side', :side, 'strength', :strength))
                """
            ),
            params,
        )
    except Exception:
        # Fallback if `signal` JSONB column doesn't exist in your DB
        conn.execute(
            sa.text(
                """
                INSERT INTO signals
                    (ts, symbol_id, ticker, timeframe, model, side, strength)
                VALUES
                    (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength)
                """
            ),
            params,
        )


def generate_demo_for_ticker(
    conn: sa.Connection, symbol_id: int, ticker: str, n: int = 12
) -> int:
    """
    Create n demo signals per ticker over the last ~2 hours (spaced 10 min).
    """
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    inserted = 0

    for k in range(n):
        ts = now - timedelta(minutes=10 * (n - 1 - k))
        # toy logic for demo signals
        side = "buy" if (k % 2 == 0) else "sell"
        strength = round(random.uniform(0.15, 0.95), 3)
        insert_signal_row(
            conn=conn,
            ts=ts,
            symbol_id=symbol_id,
            ticker=ticker,
            timeframe="1d",
            model="demo_crossover",
            side=side,
            strength=strength,
        )
        inserted += 1

    return inserted


def main() -> None:
    engine = get_engine()
    tickers = parse_tickers()

    total_inserted = 0
    with engine.begin() as conn:
        for ticker, name in tickers:
            sym_id = upsert_symbol(conn, ticker, name)
            total_inserted += generate_demo_for_ticker(conn, sym_id, ticker, n=12)

    print(f"✔ inserted {total_inserted} signals")


if __name__ == "__main__":
    main()
