# jobs/make_signals.py
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import sqlalchemy as sa

from lib.db import make_engine  # now importable

def fetch_symbol_ids(conn, tickers):
    q = sa.text("SELECT id, ticker FROM symbols WHERE ticker = ANY(:t)")
    df = pd.read_sql(q, conn, params={"t": tickers})
    return {r["ticker"]: int(r["id"]) for _, r in df.iterrows()}

def make_rows(ticker, n=12):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    ts = [now - timedelta(days=i) for i in range(n)][::-1]
    side = np.where(np.random.rand(n) > 0.5, "buy", "sell")
    strength = np.round(0.2 + 0.8 * np.random.rand(n), 2)
    return pd.DataFrame({
        "ts": ts,
        "timeframe": "1d",
        "model": "baseline",
        "side": side,
        "strength": strength,
        "ticker": ticker,  # temporary, will map to symbol_id
    })

def main():
    eng = make_engine()
    tickers = os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY").split(",")
    frames = [make_rows(t.strip().upper()) for t in tickers]
    df = pd.concat(frames, ignore_index=True)

    with eng.begin() as conn:
        ids = fetch_symbol_ids(conn, tickers)
        if not ids:
            raise SystemExit("No matching rows in symbols. Seed symbols first (AAPL/MSFT/SPY).")

        df["symbol_id"] = df["ticker"].map(ids)
        df = df.drop(columns=["ticker"])

        # optional: clear recent demo rows, keep it tidy
        conn.execute(sa.text("""
            DELETE FROM signals
            WHERE model = 'baseline' AND ts >= now() - interval '30 days'
        """))
        df.to_sql("signals", conn, if_exists="append", index=False)
        print(f"✔ inserted {len(df)} signals")

if __name__ == "__main__":
    main()
