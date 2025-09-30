# tools/inspect_signals.py
import os, argparse, pandas as pd, sqlalchemy as sa
from lib.db import make_engine

MIN_STRENGTH = float(os.environ.get("MIN_STRENGTH", "0.30"))
TICKERS = [t.strip().upper() for t in os.environ.get("SIGNAL_TICKERS","AAPL,MSFT,SPY").split(",") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-days", type=int, default=1)
    args = ap.parse_args()

    eng = make_engine()
    with eng.connect() as c:
        q = sa.text("""
            SELECT ts, ticker, timeframe, model, side, strength
            FROM signals
            WHERE ts >= now() - (:d || ' days')::interval
            ORDER BY ts DESC
        """)
        df = pd.read_sql(q, c, params={"d": args.since_days})
    if df.empty:
        print("No signals in window.")
        return

    print("\nAll signals in window (top 20):")
    print(df.head(20).to_string(index=False))

    # apply executor filters to show what it would use
    mask = df["strength"] >= MIN_STRENGTH
    if TICKERS:
        mask &= df["ticker"].isin(TICKERS)
    used = df[mask]

    print(f"\nAfter filters: MIN_STRENGTH>={MIN_STRENGTH}, TICKERS={TICKERS} -> {len(used)} rows")
    if not used.empty:
        print(used.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
