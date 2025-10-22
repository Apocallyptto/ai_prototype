# jobs/make_signals_ensemble.py
from __future__ import annotations

import os, sys
from datetime import datetime, timezone
import psycopg2

# Import your existing models (paths from your repo)
from jobs.make_signals_nn import predict_for_symbols as nn_predict   # re-use your nn path
from jobs.make_signals_ml import predict_for_symbols as ml_predict   # you already have ML variant

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))

def _pg_conn():
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("PGHOST","localhost"),
        user=os.getenv("PGUSER","postgres"),
        password=os.getenv("PGPASSWORD","postgres"),
        dbname=os.getenv("PGDATABASE","ai_prototype"),
        port=int(os.getenv("PGPORT","5432")),
    )

def _ensure_columns(cur):
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='public' AND table_name='signals'
    """)
    cols = {r[0] for r in cur.fetchall()}
    # ensure basic columns; schema migration is normally separate
    needed = ["symbol","side","strength","created_at","portfolio_id","source"]
    for c in needed:
        if c not in cols:
            if c in ("created_at",):
                cur.execute("ALTER TABLE public.signals ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()")
            elif c in ("portfolio_id",):
                cur.execute("ALTER TABLE public.signals ADD COLUMN IF NOT EXISTS portfolio_id INT DEFAULT 1")
            elif c in ("source",):
                cur.execute("ALTER TABLE public.signals ADD COLUMN IF NOT EXISTS source TEXT")
            else:
                cur.execute(f"ALTER TABLE public.signals ADD COLUMN IF NOT EXISTS {c} TEXT")

def _insert_signal(cur, symbol: str, side: str, strength: float, ts: datetime):
    cur.execute(
        "INSERT INTO public.signals (symbol, side, strength, created_at, portfolio_id, source) VALUES (%s,%s,%s,%s,%s,%s)",
        (symbol, side, float(strength), ts, PORTFOLIO_ID, "ensemble")
    )

def main():
    now = datetime.now(timezone.utc)
    # Get predictions from both models
    nn = nn_predict(SYMBOLS)   # expected: { "AAPL": {"side":"buy","strength":0.62}, ...}
    ml = ml_predict(SYMBOLS)   # expected: same shape

    merged = {}
    for sym in SYMBOLS:
        n = nn.get(sym); m = ml.get(sym)
        if not n and not m:
            continue
        # Average strengths when both exist; otherwise use existing one
        if n and m:
            # If sides disagree, take the one with higher strength; else average
            if n["side"] != m["side"]:
                pick = n if n["strength"] >= m["strength"] else m
                merged[sym] = pick
            else:
                s = (n["strength"] + m["strength"]) / 2.0
                merged[sym] = {"side": n["side"], "strength": s}
        else:
            merged[sym] = (n or m)

    # Gate by MIN_STRENGTH
    gated = {k:v for k,v in merged.items() if v["strength"] >= MIN_STRENGTH}

    if not gated:
        print("ensemble: no signals above threshold.")
        return

    with _pg_conn() as conn, conn.cursor() as cur:
        _ensure_columns(cur)
        for sym, sig in gated.items():
            _insert_signal(cur, sym, sig["side"], sig["strength"], now)
        conn.commit()
        for sym, sig in gated.items():
            print(f"{sym}: {sig['side']} strength={sig['strength']:.2f} at {now.isoformat()} (ensemble)")

if __name__ == "__main__":
    main()
