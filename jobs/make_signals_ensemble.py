# jobs/make_signals_ensemble.py
from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import psycopg2
import numpy as np

# ---- Config from env ----
SYMBOLS       = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MIN_STRENGTH  = float(os.getenv("MIN_STRENGTH", "0.60"))
DATABASE_URL  = os.getenv("DATABASE_URL")  # required for DB writes
PORTFOLIO_ID  = os.getenv("PORTFOLIO_ID")  # optional tag

# ---- Import model predictors ----
# NN is required; ML is optional (fallback to NN only if ML artifacts missing).
from jobs.make_signals_nn import nn_predict as nn_predict  # returns {SYM: {"side","strength"}}

def _try_ml_predict(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Try to import and run ML predictor. If model files are missing or import fails,
    return an empty dict so we can fall back to NN-only.
    """
    try:
        from jobs.make_signals_ml import predict_for_symbols as ml_predict
        preds = ml_predict(symbols)  # same shape as NN
        return preds or {}
    except FileNotFoundError:
        # models/ml_5m.pkl not found
        return {}
    except Exception as e:
        # Any other issue: be robust, run NN-only
        print(f"[WARN] ML predictor unavailable: {e}")
        return {}

# ---- DB helpers ----
def _ensure_signals_table(cur) -> None:
    cur.execute("""
    CREATE TABLE IF NOT EXISTS public.signals (
        id SERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        strength DOUBLE PRECISION NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        portfolio_id TEXT
    );
    """)

def _insert_signal(symbol: str, side: str, strength: float, ts: datetime, portfolio_id: str | None) -> None:
    if not DATABASE_URL:
        return
    with psycopg2.connect(dsn=DATABASE_URL) as conn, conn.cursor() as cur:
        _ensure_signals_table(cur)
        cur.execute(
            """INSERT INTO public.signals (symbol, side, strength, created_at, portfolio_id)
               VALUES (%s, %s, %s, %s, %s)""",
            (symbol, side, float(strength), ts, portfolio_id)
        )

# ---- Blending logic ----
def _blend(nn_val: float | None, ml_val: float | None) -> float | None:
    """
    Blend two probabilities/strengths (0..1). If one is None, return the other.
    Simple arithmetic mean for robustness.
    """
    vals = [v for v in (nn_val, ml_val) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))

def _to_buy_prob(side: str, strength: float) -> float:
    """
    Convert (side, strength) into probability of 'buy'.
    If side=='buy': prob_up = strength
    If side=='sell': prob_up = 1 - strength
    """
    return float(strength) if side == "buy" else float(1.0 - strength)

def _from_buy_prob(prob_up: float) -> Tuple[str, float]:
    """
    Convert probability of 'buy' back to (side, strength).
    """
    if prob_up >= 0.5:
        return "buy", float(prob_up)
    else:
        return "sell", float(1.0 - prob_up)

# ---- Public API: predict (no DB writes) ----
def predict_for_symbols(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Returns: { "AAPL": {"side":"buy","strength":0.62}, ... } from blended NN+ML.
    """
    nn = nn_predict(symbols)                   # required
    ml = _try_ml_predict(symbols)              # optional

    out: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        nn_r = nn.get(sym)
        ml_r = ml.get(sym) if ml else None

        nn_prob = _to_buy_prob(nn_r["side"], nn_r["strength"]) if nn_r else None
        ml_prob = _to_buy_prob(ml_r["side"], ml_r["strength"]) if ml_r else None

        blended_prob = _blend(nn_prob, ml_prob)
        if blended_prob is None:
            continue

        side, strength = _from_buy_prob(blended_prob)
        out[sym] = {"side": side, "strength": strength}
    return out

# ---- CLI entrypoint: predict + INSERT into DB ----
def main():
    now = datetime.now(timezone.utc)
    preds = predict_for_symbols(SYMBOLS)

    any_inserted = False
    for sym in SYMBOLS:
        r = preds.get(sym)
        if not r:
            continue
        side, strength = r["side"], float(r["strength"])

        if strength < MIN_STRENGTH:
            # Verbose and clear in logs:
            print(f"{sym}: below MIN_STRENGTH ({strength:.2f} < {MIN_STRENGTH}) (ensemble)")
            continue

        # Print exactly what we insert:
        print(f"{sym}: {side} strength={strength:.2f} at {now.isoformat()} (ensemble)")

        # Persist to DB so executor_bracket reads the latest side
        try:
            _insert_signal(sym, side, strength, now, PORTFOLIO_ID)
            any_inserted = True
        except Exception as e:
            print(f"[WARN] failed to insert signal for {sym}: {e}")

    if not any_inserted:
        print("ensemble: no signals above threshold.")

if __name__ == "__main__":
    main()
