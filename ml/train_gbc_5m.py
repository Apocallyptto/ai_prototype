# ml/train_gbc_5m.py
#
# Train a GradientBoosting model + StandardScaler on 5m OHLCV data
# and save to:
#   models/gbc_5m.pkl
#   models/gbc_5m_scaler.pkl

import os
from datetime import datetime, timedelta
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.nn_train import make_features
from jobs.make_signals_ml import FEATURE_COLS


SYMBOLS: List[str] = ["AAPL", "MSFT", "SPY"]
BAR_INTERVAL = "5m"
LOOKBACK_DAYS = 60  # how much history to train on
TARGET_HORIZON_BARS = 6  # predict direction over next 6 bars (~30 minutes)


def download_bars(symbol: str) -> pd.DataFrame:
    """
    Download recent 5m OHLCV bars from Yahoo Finance using yfinance.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is a DatetimeIndex.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=BAR_INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data downloaded for {symbol}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    df["symbol"] = symbol
    return df


def load_training_data() -> pd.DataFrame:
    """
    Download and combine OHLCV data for all symbols in SYMBOLS.
    """
    all_dfs = []
    for sym in SYMBOLS:
        print(f"Downloading data for {sym}...")
        df_sym = download_bars(sym)
        all_dfs.append(df_sym)

    df_all = pd.concat(all_dfs).sort_index()
    return df_all


def build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) where X are feature vectors and y is a binary label:
      y = 1 if future return over TARGET_HORIZON_BARS is positive, else 0.
    """
    df_all = load_training_data()

    X_list = []
    y_list = []

    # Build features and targets separately for each symbol to avoid leaks
    for sym, df_sym in df_all.groupby("symbol"):
        print(f"Building features for {sym}...")

        feats = make_features(df_sym)

        # We assume feats contains FEATURE_COLS plus original price columns
        missing = [c for c in FEATURE_COLS if c not in feats.columns]
        if missing:
            raise RuntimeError(
                f"Missing feature columns {missing} for {sym}. "
                f"Got columns: {list(feats.columns)}"
            )

        # Target: future return over N bars
        close = feats["Close"].astype(float)
        future_close = close.shift(-TARGET_HORIZON_BARS)
        future_ret = (future_close - close) / close

        y = (future_ret > 0).astype(int)

        # Drop rows where we don't have future data
        valid_mask = future_ret.notna()
        feats_valid = feats.loc[valid_mask, FEATURE_COLS]
        y_valid = y.loc[valid_mask]

        X_list.append(feats_valid.values)
        y_list.append(y_valid.values)

    if not X_list:
        raise RuntimeError("No training data built (X_list is empty).")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    return X, y


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(this_dir, "..", "models")
    models_dir = os.path.normpath(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "gbc_5m.pkl")
    scaler_path = os.path.join(models_dir, "gbc_5m_scaler.pkl")

    print("Building dataset...")
    X, y = build_dataset()

    print("Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training GradientBoosting model...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    val_acc = model.score(X_val_scaled, y_val)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")

    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    print(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    print("Done.")


if __name__ == "__main__":
    main()
