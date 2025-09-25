# jobs/make_signals.py
from __future__ import annotations
import os, random, datetime as dt
import numpy as np
import pandas as pd
import sqlalchemy as sa

from lib.db import make_engine


def _table_has_column(conn, table: str, col: str) -> bool:
    sql = sa.text("""
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema='public' AND table_name=:t AND column_name=:c
      LIMIT 1
    """)
    return conn.execute(sql, {"t": table, "c": col}).scalar() is not None


def ensure_symbols(conn):
    # upsert 3 demo rows
    data = pd.DataFrame([
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "MSFT", "name": "Microsoft Corp."},
        {"ticker": "SPY",  "name": "SPDR S&P 500 ETF"},
    ])
    # Insert ticker if missing
    for _, r in data.iterrows():
        conn.execute(sa.text("""
          INSERT INTO symbols (ticker, name)
          VALUES (:t, :n)
          ON CONFLICT (ticker) DO UPDATE SET name=EXCLUDED.name
        """), {"t": r["ticker"], "n": r["name"]})


def get_symbol_id(conn, ticker: str) -> int:
    return conn.execute(sa.text("SELECT id FROM symbols WHERE ticker=:t"), {"t": ticker}).scalar()


def make_demo_signals(tickers: list[str], n_days: int = 12) -> pd.DataFrame:
    rows = []
    base = dt.datetime.now(dt.timezone.utc).replace(hour=17, minute=0, second=0, microsecond=0)
    for t in tickers:
        for i in range(n_days):
            ts = base - dt.timedelta(days=(n_days - 1 - i))
            side = random.choice(["buy", "sell"])
            strength = float(np.clip(np.random.normal(0.55 if side == "buy" else 0.35, 0.15), 0.05, 0.95))
            rows.append({
                "ts": ts, "ticker": t, "timeframe": "1d", "model": "baseline",
                "side": side, "strength": strength
            })
    return pd.DataFrame(rows)


def main():
    eng = make_engine()
    tickers = os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY").split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    with eng.begin() as conn:
        ensure_symbols(conn)

        df = make_demo_signals(tickers)
        # Map tickers to symbol_id
        df["symbol_id"] = [get_symbol_id(conn, t) for t in df["ticker"]]

        # Decide which insert shape to use
        has_json = _table_has_column(conn, "signals", "signal")
        if has_json:
            # Insert JSONB document in column 'signal'
            df2 = df[["ts", "symbol_id", "timeframe", "model", "side", "strength"]].copy()
            df2["signal"] = df2.apply(lambda r: {"side": r["side"], "strength": r["strength"]}, axis=1)
            df2 = df2.drop(columns=["side", "strength"])
            df2.to_sql("signals", conn, if_exists="append", index=False)
        else:
            # Insert as separate columns (your current schema)
            cols = ["ts", "symbol_id", "timeframe", "model", "side", "strength"]
            df[cols].to_sql("signals", conn, if_exists="append", index=False)

    print(f"✅ Inserted {len(df)} signals for {tickers}")


if __name__ == "__main__":
    main()
