# jobs/train_ml.py
from __future__ import annotations
import os, json
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

# model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# reuse feature recipe
from ml.nn_train import make_features

# optional Alpaca bars, else yfinance
try:
    from lib.broker_alpaca import get_bars
    HAVE_APCA = True
except Exception:
    HAVE_APCA = False

# ------------------ config ------------------ #
TIMEFRAME   = os.getenv("ML_TIMEFRAME", "5Min")
OUTDIR      = os.getenv("ML_OUTDIR", "models")
MODEL_PATH  = os.path.join(OUTDIR, "ml_5m.pkl")
SCALER_PATH = os.path.join(OUTDIR, "ml_5m_scaler.pkl")
FEAT_JSON   = os.path.join(OUTDIR, "ml_5m_features.json")

SYMBOLS     = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
DAYS        = int(os.getenv("ML_TRAIN_DAYS", "30"))  # lookback
USE_YF      = os.getenv("USE_YFINANCE_TRAIN", "1").lower() in {"1","true","yes","y"}

os.makedirs(OUTDIR, exist_ok=True)

def _bars_yf(symbol: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    # 5m bars for up to ~60 days (yfinance caps intraday history)
    period = "60d" if days > 30 else "30d"
    df = yf.download(symbol, period=period, interval="5m", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance empty for {symbol}")
    df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")
    df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def fetch_bars(symbol: str, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    if HAVE_APCA:
        try:
            bars = get_bars(symbol, TIMEFRAME, start, end, limit=10000)
            df = pd.DataFrame(bars)
            df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception as e:
            if not USE_YF:
                raise
            print(f"[WARN] Alpaca bars failed for {symbol}: {e}. Falling back to yfinance.")
    return _bars_yf(symbol, days)

def build_dataset(symbols: List[str], days: int) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    frames = []
    for sym in symbols:
        df = fetch_bars(sym, days)
        feats = make_features(df)  # same recipe as NN
        if feats.empty or len(feats) < 3:
            continue
        # label: next bar up? close[t+1] > close[t]
        close = df["close"].reindex(feats.index)
        next_close = close.shift(-1)
        y = (next_close > close).astype(int).iloc[:-1]
        X = feats.iloc[:-1].copy()

        X["__symbol__"] = sym  # keep sym if you want to train one model for all
        XY = X.copy()
        XY["label"] = y
        frames.append(XY)

    if not frames:
        raise RuntimeError("No training data constructed. Check data access and make_features.")
    big = pd.concat(frames).dropna()
    y = big.pop("label").values.astype("int64")
    sym_col = big.pop("__symbol__")  # not used by the model
    feature_names = list(big.columns)
    X = big.values.astype("float32")
    return X, y, feature_names

def main():
    print(f"Training ML on symbols={SYMBOLS} days={DAYS} timeframe={TIMEFRAME}")
    X, y, feat_names = build_dataset(SYMBOLS, DAYS)

    # split (simple chronological split)
    n = len(X)
    cut = int(n * 0.8)
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]

    scaler = StandardScaler()
    clf = GradientBoostingClassifier(random_state=17)

    # fit
    Xtr_s = scaler.fit_transform(Xtr)
    clf.fit(Xtr_s, ytr)

    # evaluate
    Xte_s = scaler.transform(Xte)
    yhat = clf.predict(Xte_s)
    try:
        print(classification_report(yte, yhat, digits=3))
    except Exception:
        pass

    # save artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEAT_JSON, "w") as f:
        json.dump({"features": feat_names}, f)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved feature list -> {FEAT_JSON}")

if __name__ == "__main__":
    main()
