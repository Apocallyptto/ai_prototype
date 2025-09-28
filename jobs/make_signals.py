# jobs/make_signals.py
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from lib.db import make_engine as get_engine  # your helper

# -------- schema feature detection --------
def has_json_signal_column(conn: sa.Connection) -> bool:
    sql = sa.text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = 'signals'
          AND column_name  = 'signal'
        LIMIT 1
    """)
    return conn.execute(sql).first() is not None

# -------- symbols helpers --------
def upsert_symbol(conn: sa.Connection, ticker: str, name: str) -> int:
    sql = sa.text("""
        INSERT INTO symbols (ticker, name)
        VALUES (:t, :n)
        ON CONFLICT (ticker) DO UPDATE
          SET name = EXCLUDED.name
        RETURNING id
    """)
    return conn.execute(sql, {"t": ticker, "n": name}).scalar_one()

# -------- signals insert (two SQL variants, selected once) --------
SQL_INSERT_JSON = sa.text("""
    INSERT INTO signals
      (ts, symbol_id, ticker, timeframe, model, side, strength, signal)
    VALUES
      (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength,
       jsonb_build_object('side', :side, 'strength', :strength))
""")

SQL_INSERT_PLAIN = sa.text("""
    INSERT INTO signals
      (ts, symbol_id, ticker, timeframe, model, side, strength)
    VALUES
      (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength)
""")

def insert_signal_row(conn: sa.Connection, use_json: bool, **params) -> None:
    conn.execute(SQL_INSERT_JSON if use_json else SQL_INSERT_PLAIN, params)

# -------- demo generator --------
def parse_tickers() -> list[tuple[str, str]]:
    env_val = os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY")
    tickers = [t.strip().upper() for t in env_val.split(",") if t.strip()]
    names = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "SPY": "SPDR S&P 500 ETF"}
    return [(t, names.get(t, t)) for t in tickers]

def generate_demo_for_ticker(conn: sa.Connection, symbol_id: int, ticker: str,
                             use_json: bool, n: int = 12) -> int:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    inserted = 0
    for k in range(n):
        ts = now - timedelta(minutes=10 * (n - 1 - k))
        side = "buy" if (k % 2 == 0) else "sell"
        strength = round(random.uniform(0.15, 0.95), 3)
        insert_signal_row(
            conn,
            use_json,
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

# -------- main --------
def main() -> None:
    engine = get_engine()
    tickers = parse_tickers()

    with engine.begin() as conn:
        use_json = has_json_signal_column(conn)  # decide once, avoid throwing
        total = 0
        for ticker, name in tickers:
            sid = upsert_symbol(conn, ticker, name)
            total += generate_demo_for_ticker(conn, sid, ticker, use_json, n=12)

    print(f"✔ inserted {total} signals")

if __name__ == "__main__":
    main()
