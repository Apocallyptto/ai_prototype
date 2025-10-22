# jobs/make_signals_ml.py
from __future__ import annotations
import os, json
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Set, Dict

import numpy as np
import pandas as pd
import joblib

# DB: prefer psycopg3, fallback to psycopg2
try:
    import psycopg  # psycopg3
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2

# Reuse your feature recipe used by NN training
from ml.nn_train import make_features

# If you have an Alpaca bars helper, we use it and silently fallback to yfinance
try:
    from lib.broker_alpaca import get_bars
    HAVE_APCA = True
except Exception:
    HAVE_APCA = False

# ---------------- Config ---------------- #
TIMEFRAME     = os.getenv("ML_TIMEFRAME", "5Min")
OUTDIR        = os.getenv("ML_OUTDIR", "models")
MODEL_PATH    = os.path.join(OUTDIR, "ml_5m.pkl")             # joblib dump of your sklearn model
SCALER_PATH   = os.path.join(OUTDIR, "ml_5m_scaler.pkl")      # joblib dump of StandardScaler
FEAT_JSON     = os.path.join(OUTDIR, "ml_5m_features.json")   # optional: list of feature names
SYMBOLS       = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
PORTFOLIO_ID  = int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH  = float(os.getenv("MIN_STRENGTH", "0.60"))

# Use yfinance fallback for inference if Alpaca is unavailable
USE_YF = os.getenv("USE_YFINANCE_INFER", "0").lower() in {"1","true","yes","y"}

# -------------- DB helpers -------------- #
def _pg_conn():
    dsn = os.getenv("DATABASE_URL")
    if HAVE3:
        return psycopg.connect(dsn) if dsn else psycopg.connect(
            host=os.getenv("PGHOST","localhost"),
            user=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD","postgres"),
            dbname=os.getenv("PGDATABASE","ai_prototype"),
            port=os.getenv("PGPORT","5432"),
        )
    else:
        return psycopg2.connect(dsn) if dsn else psycopg2.connect(
            host=os.getenv("PGHOST","localhost"),
            user=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD","postgres"),
            dbname=os.getenv("PGDATABASE","ai_prototype"),
            port=os.getenv("PGPORT","5432"),
        )

def _signal_columns() -> Set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name='signals'
    """
    with _pg_conn() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return {r[0] for r in rows}

def _insert_signal(symbol: str, side: str, strength: float, ts: datetime):
    cols = _signal_columns()
    names, vals = ["symbol","side","strength"], [symbol, side, float(strength)]

    ts_col = "ts" if "ts" in cols else ("created_at" if "created_at" in cols else None)
    if ts_col:
        names.append(ts_col); vals.append(ts)

    if "portfolio_id" in cols:
        names.append("portfolio_id"); vals.append(PORTFOLIO_ID)

    # Optional: mark source
    if "source" in cols:
        names.append("source"); vals.append("ml")

    placeholders = ",".join(["%s"]*len(vals))
    sql = f"INSERT INTO public.signals ({','.join(names)}) VALUES ({placeholders})"
    with _pg_conn() as c:
        with c.cursor() as cur:
            cur.execute(sql, tuple(vals))
        c.commit()

# --------- Market data / features --------- #
def _bars_yf(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period="7d", interval="5m", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance empty for {symbol}")
    df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")
    df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def _fetch_bars(symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    if HAVE_APCA:
        try:
            bars = get_bars(symbol, TIMEFRAME, start, end, limit=2000)
            df = pd.DataFrame(bars)
            df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception as e:
            if not USE_YF:
                raise
            print(f"[WARN] Alpaca bars failed for {symbol}: {e}. Falling back to yfinance.")
    # fallback:
    return _bars_yf(symbol)

def _latest_features(symbol: str) -> Optional[np.ndarray]:
    df = _fetch_bars(symbol)
    feats = make_features(df)
    if feats.empty:
        return None
    # keep the latest row; ensure float32 dtype
    return feats.iloc[-1:].values.astype("float32")

# -------------- Predictor (reusable) -------------- #
def predict_for_symbols(symbols: Sequence[str]) -> Dict[str, Dict[str, float | str]]:
    """
    Returns:
        {'AAPL': {'side': 'buy', 'strength': 0.63}, ...}
    Note: does NOT write to DB (safe for import/ensemble).
    """
    # Load artifacts
    model = joblib.load(MODEL_PATH)   # e.g., GradientBoostingClassifier
    scaler = joblib.load(SCALER_PATH) # StandardScaler

    # Optional: validate features signature if you saved it
    if os.path.exists(FEAT_JSON):
        try:
            with open(FEAT_JSON, "r") as f:
                _ = json.load(f)  # reserved for future checks
        except Exception:
            pass

    out: Dict[str, Dict[str, float | str]] = {}
    for sym in [s.strip().upper() for s in symbols if s and s.strip()]:
        x = _latest_features(sym)
        if x is None:
            continue
        xs = scaler.transform(x).astype("float32")

        # Prob of "up" (class 1). If your model is not probabilistic, fallback to decision_function
        try:
            proba = float(model.predict_proba(xs)[0, 1])
        except Exception:
            # e.g., if model has no predict_proba; map decision_function to [0,1]
            from math import tanh
            d = float(model.decision_function(xs).ravel()[0])
            proba = 0.5 * (tanh(d) + 1.0)

        if proba >= 0.5:
            side, strength = "buy", proba
        else:
            side, strength = "sell", 1.0 - proba

        out[sym] = {"side": side, "strength": float(strength)}
    return out

# ------------------ CLI main ------------------ #
def main():
    preds = predict_for_symbols(SYMBOLS)
    for sym in SYMBOLS:
        p = preds.get(sym)
        if not p:
            print(f"{sym}: no features")
            continue
        if p["strength"] >= MIN_STRENGTH:
            ts = datetime.now(timezone.utc)
            _insert_signal(sym, p["side"], float(p["strength"]), ts)
            print(f"{sym}: {p['side']} strength={p['strength']:.2f} at {ts.isoformat()} (ml)")
        else:
            print(f"{sym}: below MIN_STRENGTH ({p['strength']:.2f} < {MIN_STRENGTH}) (ml)")

if __name__ == "__main__":
    main()
