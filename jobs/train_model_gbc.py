# jobs/train_model_gbc.py
import os
import io
import sys
import time
import json
import math
import joblib
import logging
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import psycopg2

# Optional fallback (only if Alpaca fails)
import yfinance as yf

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("train_model_gbc")

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Alpaca SDK (use v2 data API)
try:
    from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
except Exception:
    StockHistoricalDataClient = None  # handled below

# ---- Config ----
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "30"))
BAR_INTERVAL = os.getenv("ML_BAR_INTERVAL", "5m")  # only '5m' supported below
TARGET_HORIZON_BARS = int(os.getenv("ML_TARGET_HORIZON_BARS", "6"))
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MODEL_NAME = f"gbc_{BAR_INTERVAL}.pkl"

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---- DB helpers ----
def _conn():
    return psycopg2.connect(DB_URL)

def _migrate_models_meta():
    sql = """
    CREATE TABLE IF NOT EXISTS public.models_meta (
      id SERIAL PRIMARY KEY,
      model_name TEXT NOT NULL,
      path TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      is_active BOOLEAN NOT NULL DEFAULT false
    );
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute(sql)

def _activate_model(path: str, model_name: str):
    with _conn() as c, c.cursor() as cur:
        cur.execute("UPDATE public.models_meta SET is_active=false WHERE model_name=%s", (model_name,))
        cur.execute(
            "INSERT INTO public.models_meta(model_name, path, is_active) VALUES (%s,%s,true)",
            (model_name, path)
        )
    log.info("models_meta updated (active=True)")

# ---- Data fetch ----
def _tf_5m():
    if BAR_INTERVAL != "5m":
        raise ValueError("Only 5m supported in this trainer (set ML_BAR_INTERVAL=5m).")
    return TimeFrame(5, TimeFrame.Minute) if TimeFrame else None

def _fetch_alpaca(sym: str) -> pd.DataFrame:
    if StockHistoricalDataClient is None or not (ALPACA_API_KEY and ALPACA_API_SECRET):
        raise RuntimeError("alpaca-py not available or ALPACA creds missing")

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    req = StockBarsRequest(
        symbol_or_symbols=sym,
        timeframe=_tf_5m(),
        start=start,
        end=end,
        adjustment=None
    )
    resp = client.get_stock_bars(req)
    if sym not in resp.data or len(resp.data[sym]) == 0:
        raise RuntimeError(f"Alpaca returned no bars for {sym}")
    rows = resp.data[sym]
    df = pd.DataFrame([{
        "ts": r.timestamp,
        "open": float(r.open),
        "high": float(r.high),
        "low": float(r.low),
        "close": float(r.close),
        "volume": int(r.volume)
    } for r in rows]).sort_values("ts").reset_index(drop=True)
    return df

def _fetch_yahoo(sym: str) -> pd.DataFrame:
    # fallback only
    interval = "5m"
    period = f"{LOOKBACK_DAYS}d"
    df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False, prepost=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no bars for {sym}")
    df = df.rename(columns=str.lower).reset_index().rename(columns={"datetime": "ts"})
    return df[["ts","open","high","low","close","volume"]].dropna().reset_index(drop=True)

def _download(sym: str) -> pd.DataFrame:
    try:
        log.info(f"download {sym} (Alpaca) …")
        return _fetch_alpaca(sym)
    except Exception as e:
        log.warning(f"Alpaca failed for {sym}: {e}. Trying Yahoo fallback …")
        # Sleep a bit to reduce 429 risk
        time.sleep(1.0)
        return _fetch_yahoo(sym)

# ---- Features/labels ----
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["ret5"] = df["close"].pct_change(5)
    df["ma_fast"] = df["close"].rolling(10).mean()
    df["ma_slow"] = df["close"].rolling(50).mean()
    df["ma_diff"] = (df["ma_fast"] - df["ma_slow"]) / df["close"]
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)

    # Target: direction after H bars
    df["future_close"] = df["close"].shift(-TARGET_HORIZON_BARS)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df

# ---- Train per symbol; concatenate ----
def _aggregate_training(symbols: List[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = _download(sym)
        feat = _build_features(df)
        feat["sym"] = sym
        frames.append(feat)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

def main():
    _migrate_models_meta()

    data = _aggregate_training(SYMBOLS)
    feats = ["ret1", "hl_range", "ret5", "ma_diff", "vol_z"]
    X = data[feats].values
    y = data["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    yhat = clf.predict(X_test)
    log.info("\n" + classification_report(y_test, yhat, digits=3))

    path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump({
        "model": clf,
        "feats": feats,
        "meta": {
            "symbols": SYMBOLS,
            "bar_interval": BAR_INTERVAL,
            "lookback_days": LOOKBACK_DAYS,
            "horizon_bars": TARGET_HORIZON_BARS,
            "created_at": dt.datetime.utcnow().isoformat() + "Z"
        }
    }, path)
    log.info(f"saved model {path}")

    _activate_model(path, MODEL_NAME)

if __name__ == "__main__":
    main()
