# ml/nn_train.py
from __future__ import annotations
import os, math, json
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import torch

from lib.broker_alpaca import get_bars
from ml.nn_model import MLP, TrainConfig, train_loop

# ----------------- Config -----------------
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
TIMEFRAME = os.getenv("NN_TIMEFRAME", "5Min")
LOOKBACK_DAYS = int(os.getenv("NN_LOOKBACK_DAYS", "60"))
RET_HORIZON = int(os.getenv("NN_RET_HORIZON", "3"))
RET_THRESH_BP = float(os.getenv("NN_RET_THRESH_BP", "5"))
OUTDIR = os.getenv("NN_OUTDIR", "models")
MODEL_PATH = os.path.join(OUTDIR, "nn_5m.pt")
SCALER_PATH = os.path.join(OUTDIR, "nn_5m_scaler.pkl")
FEAT_JSON = os.path.join(OUTDIR, "nn_5m_features.json")
USE_YF = os.getenv("USE_YFINANCE_TRAIN", "0").lower() in {"1","true","yes","y"}

os.makedirs(OUTDIR, exist_ok=True)

# ----------------- Features -----------------
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _macd(close: pd.Series, f: int = 12, s: int = 26, sig: int = 9):
    macd = _ema(close, f) - _ema(close, s)
    signal = _ema(macd, sig)
    return macd, macd - signal

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = df["close"].pct_change()
    out["sma_10"] = _sma(df["close"], 10)
    out["sma_50"] = _sma(df["close"], 50)
    out["ema_20"] = _ema(df["close"], 20)
    out["rsi_14"] = _rsi(df["close"], 14)
    out["atr_14"] = _atr(df, 14)
    macd, hist = _macd(df["close"])
    out["macd"] = macd
    out["macd_hist"] = hist
    out["vol"] = df["volume"].rolling(20).mean()
    out["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    out["sma_ratio"] = out["sma_10"] / (out["sma_50"] + 1e-9)
    return out.dropna()

def label_forward_returns(df: pd.DataFrame, horizon: int, thresh_bp: float) -> pd.Series:
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    thr = thresh_bp / 10000.0
    y = pd.Series(np.nan, index=df.index)
    y[fwd > +thr] = 1.0
    y[fwd < -thr] = 0.0
    return y

# ----------------- Data fetch -----------------
def fetch_symbol_df_alpaca(symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    bars = get_bars(symbol, TIMEFRAME, start, end, limit=10_000)
    df = pd.DataFrame(bars)
    df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df[["open","high","low","close","volume"]]

def fetch_symbol_df_yf(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    # 5m interval supports up to 60d history
    df = yf.download(symbol, period=f"{LOOKBACK_DAYS}d", interval="5m", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty data for {symbol}")
    df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")
    df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def fetch_symbol_df(symbol: str) -> pd.DataFrame:
    if USE_YF:
        return fetch_symbol_df_yf(symbol)
    try:
        return fetch_symbol_df_alpaca(symbol)
    except Exception as e:
        print(f"[WARN] Alpaca fetch failed for {symbol}: {e}. Falling back to yfinance.")
        return fetch_symbol_df_yf(symbol)

def build_dataset(symbols: List[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list, y_list = [], []
    # compute feature names from first symbol after fetching
    first_df = fetch_symbol_df(symbols[0])
    feat_names = list(make_features(first_df).columns)
    for sym in symbols:
        raw = fetch_symbol_df(sym)
        feats = make_features(raw)
        y = label_forward_returns(raw.loc[feats.index], RET_HORIZON, RET_THRESH_BP)
        mask = y.notna()
        feats = feats[mask]
        y = y[mask]
        X_list.append(feats.values.astype("float32"))
        y_list.append(y.values.astype("float32"))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, feat_names

# ----------------- Train -----------------
def main():
    print(f"Training NN on {SYMBOLS} @ {TIMEFRAME}, lookback={LOOKBACK_DAYS}d, horizon={RET_HORIZON} bars (USE_YF={USE_YF})")
    X, y, feat_names = build_dataset(SYMBOLS)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype("float32")
    mask = np.isfinite(Xs).all(axis=1) & np.isfinite(y)
    Xs, y = Xs[mask], y[mask]

    pos = float((y > 0.5).mean())
    print(f"class balance: pos={pos:.3f}, neg={1-pos:.3f}")

    Xtr, Xva, ytr, yva = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=(y>0.5))
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva); yva_t = torch.from_numpy(yva)

    model = MLP(in_dim=Xtr_t.shape[1], dropout=0.10)
    cfg = TrainConfig(
        epochs=int(os.getenv("NN_EPOCHS", "12")),
        batch_size=int(os.getenv("NN_BATCH", "256")),
        lr=float(os.getenv("NN_LR", "1e-3")),
        weight_decay=float(os.getenv("NN_WD", "1e-5")),
        pos_weight=(1.0 - pos) / (pos + 1e-6) if 0 < pos < 1 else None,
    )
    train_loop(model, (Xtr_t, ytr_t), (Xva_t, yva_t), cfg)

    os.makedirs(OUTDIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": Xtr_t.shape[1]}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEAT_JSON, "w") as f:
        json.dump({"feature_names": feat_names}, f)
    print(f"saved model -> {MODEL_PATH}")
    print(f"saved scaler -> {SCALER_PATH}")

if __name__ == "__main__":
    main()
