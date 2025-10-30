# jobs/stats_signals.py  (only the SQL section and read call changed)
import os, logging
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stats_signals")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
HOURS  = int(os.getenv("STATS_LOOKBACK_HOURS","24"))
PERC   = float(os.getenv("STATS_TARGET_PERCENTILE","0.80"))

def main():
    eng = create_engine(DB_URL)
    with eng.connect() as con:
        # Build literal INTERVAL safely
        sql = text(f"""
            SELECT created_at, symbol, side, strength, scaled_strength
            FROM signals
            WHERE created_at >= NOW() - INTERVAL '{HOURS} hours'
            ORDER BY created_at DESC
        """)
        df = pd.read_sql_query(sql, con)
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
    print("\nSymbol  N   raw_mean  raw_p80  scaled_mean  scaled_p80")
    for _, r in out.sort_values("symbol").iterrows():
        fmt = lambda x: "—" if np.isnan(x) else f"{x:.3f}"
        print(f"{r['symbol']:5} {int(r['n']):3d}  {fmt(r['raw_mean']):>8}  {fmt(r['raw_p']):>7}  {fmt(r['scaled_mean']):>11}  {fmt(r['scaled_p']):>10}")

    all_scaled = df["scaled_strength"].dropna().astype(float)
    if len(all_scaled):
        global_p = float(np.nanpercentile(all_scaled, PERC*100))
        print(f"\nSuggested MIN_STRENGTH (scaled) ≈ {global_p:.2f}")
    else:
        print("\nNo scaled strengths available to suggest a threshold.")

if __name__ == "__main__":
    main()
