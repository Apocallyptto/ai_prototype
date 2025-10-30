# jobs/stats_signals.py
import os, logging
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stats_signals")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
HOURS  = int(os.getenv("STATS_LOOKBACK_HOURS","24"))
PERC   = float(os.getenv("STATS_TARGET_PERCENTILE","0.80"))  # 0.8 → 80th

SQL = """
SELECT created_at, symbol, side, strength, scaled_strength
FROM signals
WHERE created_at >= NOW() - INTERVAL :hours || ' hours'
ORDER BY created_at DESC
"""

def _fmt(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

def main():
    eng = create_engine(DB_URL)
    with eng.connect() as con:
        df = pd.read_sql_query(text(SQL), con, params={"hours": HOURS})
    if df.empty:
        print("No signals in lookback window.")
        return

    print(f"\n=== Signal stats ({HOURS}h, target percentile={int(PERC*100)}%) ===")
    rows = []
    for sym, g in df.groupby("symbol"):
        raw = g["strength"].astype(float)
        scl = g["scaled_strength"].astype(float)

        raw_p = np.nanpercentile(raw, PERC*100) if len(raw) else np.nan
        scl_p = np.nanpercentile(scl.dropna(), PERC*100) if scl.notna().sum() else np.nan

        rows.append({
            "symbol": sym,
            "n": len(g),
            "raw_mean": raw.mean(),
            "raw_p": raw_p,
            "scaled_mean": scl.mean(skipna=True),
            "scaled_p": scl_p
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No grouped stats.")
        return

    # Pretty print
    print("\nSymbol  N   raw_mean  raw_p80  scaled_mean  scaled_p80")
    for _, r in out.sort_values("symbol").iterrows():
        print(f"{r['symbol']:5} {int(r['n']):3d}  {_fmt(r['raw_mean']):>8}  {_fmt(r['raw_p']):>7}  {_fmt(r['scaled_mean']):>11}  {_fmt(r['scaled_p']):>10}")

    # Suggest a single global threshold based on ALL scaled strengths
    all_scaled = df["scaled_strength"].dropna().astype(float)
    if len(all_scaled):
        global_p = float(np.nanpercentile(all_scaled, PERC*100))
        print(f"\nSuggested MIN_STRENGTH (scaled) ≈ {global_p:.2f} (global {int(PERC*100)}th percentile of last {HOURS}h)")
    else:
        print("\nNo scaled strengths available to suggest a threshold.")

if __name__ == "__main__":
    main()
