# jobs/make_signals_ml.py

import os
import time
import logging
import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
import psycopg2
from psycopg2.extras import register_uuid

from ml.nn_train import make_features  # uses pandas/numpy only, no torch


# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("make_signals_ml")

SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
SYMBOLS = [s.strip().upper() for s in SYMBOLS if s.strip()]

MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "gbc_5m.pkl"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(MODEL_DIR, "gbc_5m_scaler.pkl"))


# DB URL – inside Docker we usually set DB_URL=postgresql://postgres:...@postgres:5432/trader
DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")

LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "30"))

# -----------------------------------------------------------------------------
# ✅ THIS IS THE CRUCIAL PART: feature columns (11) that match your StandardScaler
# -----------------------------------------------------------------------------

FEATURE_COLS = [
    "return_1",
    "return_5",
    "return_10",
    "ma_10",
    "ma_20",
    "std_10",
    "std_20",
    "hl_range",
    "oc_range",
    "vol_zscore_20",
    "rsi_14",
]


# -----------------------------------------------------------------------------
# Data loader – use Alpaca to fetch recent bars
# If you already have your own loader, you can keep it and ignore this function.
# -----------------------------------------------------------------------------

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

if ALPACA_API_KEY and ALPACA_API_SECRET:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
else:
    data_client = None
    log.warning("ALPACA_API_KEY / SECRET not set – data loader may fail.")


def load_recent_bars(symbol: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch recent minute bars from Alpaca and normalize to columns:
    [timestamp, Open, High, Low, Close, Volume]

    If you already had your own data loader in this file,
    you can replace this implementation with your original one.
    """
    if data_client is None:
        raise RuntimeError("Alpaca data client is not configured (missing API keys).")

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=lookback_days)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,  # 1-minute bars
        start=start,
        end=end,
        adjustment="raw",
    )
    bars = data_client.get_stock_bars(req).df

    if bars.empty:
        raise RuntimeError(f"No bars returned for symbol {symbol}")

    # When multiple symbols, df is MultiIndex; for a single symbol, pick that slice
    if isinstance(bars.index, pd.MultiIndex) and "symbol" in bars.index.names:
        try:
            bars = bars.xs(symbol, level="symbol")
        except KeyError:
            raise RuntimeError(f"No data in dataframe for symbol {symbol}")

    bars = bars.reset_index().rename(
        columns={
            "timestamp": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # sort just in case
    bars = bars.sort_values("Timestamp").reset_index(drop=True)
    return bars


# -----------------------------------------------------------------------------
# ✅ NEW VERSION: _latest_features returns EXACTLY FEATURE_COLS (11 features)
# -----------------------------------------------------------------------------

def _latest_features(symbol: str) -> np.ndarray:
    """
    Load recent bars for a symbol, build features with make_features(),
    and return a 2D numpy array with exactly FEATURE_COLS in the right order
    (shape: (1, len(FEATURE_COLS))).
    """
    df = load_recent_bars(symbol)  # uses Alpaca loader above (or your own, if you replace it)

    feats = make_features(df)

    # Ensure all expected feature columns exist
    missing = [c for c in FEATURE_COLS if c not in feats.columns]
    if missing:
        raise RuntimeError(
            f"Missing feature columns {missing} for symbol {symbol}. "
            f"Got columns: {list(feats.columns)}"
        )

    # Take the last row, only selected feature columns, keep it 2D
    latest = feats[FEATURE_COLS].iloc[-1:]
    return latest.values.astype("float32")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model_and_scaler():
    log.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    log.info("Loading scaler from %s", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def get_db_conn():
    if not DB_URL:
        raise RuntimeError("DB_URL (or DATABASE_URL) is not set in environment.")
    register_uuid()
    return psycopg2.connect(DB_URL)


def insert_signal(
    conn,
    symbol: str,
    side: str,
    strength: float,
    portfolio_id: str | None = None,
):
    """
    Insert a new ML signal into signals table with status='pending'.
    Assumes table columns: created_at, symbol, side, strength, portfolio_id, status.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO signals (created_at, symbol, side, strength, portfolio_id, status)
            VALUES (NOW(), %s, %s, %s, %s, 'pending');
            """,
            (symbol, side, strength, portfolio_id),
        )
    conn.commit()


# -----------------------------------------------------------------------------
# Prediction logic
# -----------------------------------------------------------------------------

def predict_for_symbols(symbols: List[str]) -> Dict[str, np.ndarray]:
    """
    Return dict: symbol -> class probabilities (e.g. [p_down, p_flat, p_up])
    """
    model, scaler = load_model_and_scaler()
    preds: Dict[str, np.ndarray] = {}

    for sym in symbols:
        try:
            x = _latest_features(sym)  # (1, len(FEATURE_COLS)) = (1, 11)
            xs = scaler.transform(x).astype("float32")
            proba = model.predict_proba(xs)[0]
            preds[sym] = proba
            log.info("Pred for %s: %s", sym, proba)
        except Exception as e:
            log.exception("Failed to build features/predict for %s: %s", sym, e)

    return preds


def pick_side_from_proba(proba: np.ndarray) -> tuple[str, float]:
    """
    Example mapping:
      index 0 -> down, 1 -> flat, 2 -> up
    You can adjust depending on how you trained the model.
    Returns (side, strength) where side is 'buy'/'sell' or 'flat'.
    """
    # assume 3-class [down, flat, up]
    if len(proba) != 3:
        raise RuntimeError(f"Expected 3-class output, got shape {proba.shape}")

    p_down, p_flat, p_up = proba
    strength = float(max(p_down, p_up))

    if strength < MIN_STRENGTH:
        return "flat", strength

    side = "buy" if p_up >= p_down else "sell"
    return side, strength


# -----------------------------------------------------------------------------
# Main job
# -----------------------------------------------------------------------------

def main():
    log.info(
        "make_signals_ml starting | SYMBOLS=%s | MIN_STRENGTH=%.2f",
        SYMBOLS,
        MIN_STRENGTH,
    )

    preds = predict_for_symbols(SYMBOLS)

    if not preds:
        log.warning("No predictions produced.")
        return

    conn = get_db_conn()
    try:
        for sym, proba in preds.items():
            if proba is None:
                continue

            side, strength = pick_side_from_proba(proba)
            if side == "flat":
                log.info("Symbol %s: signal flat (strength=%.3f), skipping insert.", sym, strength)
                continue

            log.info(
                "Inserting signal: %s %s (strength=%.3f)",
                sym,
                side,
                strength,
            )
            insert_signal(conn, symbol=sym, side=side, strength=strength, portfolio_id=None)
    finally:
        conn.close()

    log.info("make_signals_ml done.")


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            log.exception("make_signals_ml iteration failed: %s", e)
        # In Docker this will be run as a long-lived job; adjust sleep via env if you want
        sleep_sec = int(os.getenv("ML_CRON_SLEEP_SECONDS", "180"))
        log.info("Sleeping %d seconds before next run...", sleep_sec)
        time.sleep(sleep_sec)
