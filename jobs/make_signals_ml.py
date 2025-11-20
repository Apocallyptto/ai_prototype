import os
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sqlalchemy import create_engine, text

from ml.nn_train import make_features


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("make_signals_ml")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gbc_5m.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "gbc_5m_scaler.pkl")

# Features we trained on in ml.train_gbc_5m
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
# Model / scaler loading
# -----------------------------------------------------------------------------
def load_model_and_scaler():
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    logger.info("Loading scaler from %s", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# -----------------------------------------------------------------------------
# Data loading (runtime) – now using yfinance instead of Alpaca
# -----------------------------------------------------------------------------
def load_recent_bars(
    symbol: str,
    lookback_days: int = 30,
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Fetch recent OHLCV bars for one symbol using yfinance.

    This avoids Alpaca SIP subscription limits and uses the same
    data source as ml.train_gbc_5m used for training.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    df = df.copy()
    if df.index.name is None:
        df.index.name = "timestamp"

    return df


def _latest_features(symbol: str) -> np.ndarray:
    """
    Load recent bars for a symbol, build features, and return
    a 2D numpy array with exactly FEATURE_COLS in the right order
    (shape: (1, len(FEATURE_COLS))).
    """
    df = load_recent_bars(symbol)
    feats = make_features(df)

    missing = [c for c in FEATURE_COLS if c not in feats.columns]
    if missing:
        raise RuntimeError(
            f"Missing feature columns {missing} for symbol {symbol}. "
            f"Got columns: {list(feats.columns)}"
        )

    latest = feats[FEATURE_COLS].iloc[-1:]
    return latest.values.astype("float32")


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
def predict_for_symbols(symbols):
    """
    Returns a dict: {symbol: proba_array}, where proba_array is model.predict_proba(x)[0]
    """
    model, scaler = load_model_and_scaler()
    preds = {}

    for sym in symbols:
        try:
            x = _latest_features(sym)  # shape (1, len(FEATURE_COLS))
            xs = scaler.transform(x).astype("float32")
            proba = model.predict_proba(xs)[0]
            preds[sym] = proba
        except Exception as e:
            logger.error(
                "Failed to build features/predict for %s: %s", sym, e, exc_info=False
            )

    if not preds:
        logger.warning("No predictions produced.")

    return preds


# -----------------------------------------------------------------------------
# DB insert
# -----------------------------------------------------------------------------
def get_engine():
    # Try both DB_URL and DATABASE_URL to be compatible with your env/docker
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DB_URL or DATABASE_URL env var is not set.")
    return create_engine(db_url)


def insert_signals(preds, min_strength: float) -> int:
    """
    Insert ML signals into the `signals` table.

    Assumes schema:
      signals(symbol, side, strength, source, created_at, portfolio_id)
    """
    if not preds:
        return 0

    engine = get_engine()
    now = datetime.utcnow()
    source = "ml_gbc_5m"
    portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))

    inserted = 0

    with engine.begin() as conn:
        for symbol, proba in preds.items():
            # assuming binary classification: proba[0] = DOWN, proba[1] = UP
            if len(proba) < 2:
                logger.warning(
                    "Unexpected proba shape for %s: %s", symbol, proba
                )
                continue

            p_down = float(proba[0])
            p_up = float(proba[1])

            # BUY signal if model thinks "up" with sufficient confidence
            if p_up >= min_strength:
                conn.execute(
                    text(
                        """
                        INSERT INTO signals (symbol, side, strength, source, created_at, portfolio_id)
                        VALUES (:symbol, :side, :strength, :source, :created_at, :portfolio_id)
                        """
                    ),
                    {
                        "symbol": symbol,
                        "side": "buy",   # <<< dôležité: malé písmená
                        "strength": p_up,
                        "source": source,
                        "created_at": now,
                        "portfolio_id": portfolio_id,
                    },
                )
                inserted += 1

            # SELL signal if model thinks "down" with sufficient confidence
            if p_down >= min_strength:
                conn.execute(
                    text(
                        """
                        INSERT INTO signals (symbol, side, strength, source, created_at, portfolio_id)
                        VALUES (:symbol, :side, :strength, :source, :created_at, :portfolio_id)
                        """
                    ),
                    {
                        "symbol": symbol,
                        "side": "sell",  # <<< tiež malé
                        "strength": p_down,
                        "source": source,
                        "created_at": now,
                        "portfolio_id": portfolio_id,
                    },
                )
                inserted += 1

    return inserted





# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main():
    symbols_env = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
    symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]

    min_strength_str = os.getenv("MIN_STRENGTH", "0.20")
    try:
        min_strength = float(min_strength_str)
    except ValueError:
        logger.warning(
            "Invalid MIN_STRENGTH=%s, falling back to 0.20", min_strength_str
        )
        min_strength = 0.20

    logger.info(
        "make_signals_ml starting | SYMBOLS=%s | MIN_STRENGTH=%.2f",
        symbols,
        min_strength,
    )

    preds = predict_for_symbols(symbols)
    if not preds:
        return

    inserted = insert_signals(preds, min_strength)
    logger.info("Inserted %d ML signals", inserted)


if __name__ == "__main__":
    sleep_default = int(os.getenv("CRON_SLEEP_SECONDS", "180"))
    while True:
        try:
            main()
        except Exception as e:
            logger.error("make_signals_ml iteration failed: %s", e, exc_info=True)

        logger.info("Sleeping %d seconds before next run...", sleep_default)
        time.sleep(sleep_default)
