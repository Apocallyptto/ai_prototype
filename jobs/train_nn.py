# jobs/train_nn.py
from __future__ import annotations
import os, json, math, random, time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config (env-overridable)
# =========================
TIMEFRAME        = os.getenv("NN_TIMEFRAME", "5Min")
OUTDIR           = os.getenv("NN_OUTDIR", "models")
MODEL_PATH       = os.path.join(OUTDIR, "nn_5m.pt")
SCALER_PATH      = os.path.join(OUTDIR, "nn_5m_scaler.pkl")
FEAT_JSON        = os.path.join(OUTDIR, "nn_5m_features.json")

SYMBOLS          = [s.strip().upper() for s in os.getenv("SYMBOLS","AAPL,MSFT,SPY").split(",") if s.strip()]
TRAIN_DAYS       = int(os.getenv("NN_TRAIN_DAYS", "90"))
HORIZON          = int(os.getenv("NN_LABEL_HORIZON", "3"))     # predict +3 bars
USE_YF           = os.getenv("USE_YFINANCE_TRAIN","1").lower() in {"1","true","yes","y"}

BATCH_SIZE       = int(os.getenv("NN_BATCH_SIZE", "256"))
EPOCHS           = int(os.getenv("NN_EPOCHS", "30"))
LR               = float(os.getenv("NN_LR", "1e-3"))
WEIGHT_DECAY     = float(os.getenv("NN_WEIGHT_DECAY", "1e-4"))
DROPOUT          = float(os.getenv("NN_DROPOUT", "0.25"))
EARLY_STOP       = int(os.getenv("NN_EARLY_STOP_EPOCHS", "6"))
VAL_SPLIT        = float(os.getenv("NN_VAL_SPLIT", "0.2"))
SEED             = int(os.getenv("NN_SEED", "17"))

os.makedirs(OUTDIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================
# Data access
# =========================
def _bars_yf(symbol: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    period = "60d" if days > 30 else "30d"
    df = yf.download(symbol, period=period, interval="5m", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance empty for {symbol}")
    df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")
    df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def _bars_alpaca(symbol: str, days: int) -> pd.DataFrame:
    try:
        from lib.broker_alpaca import get_bars
    except Exception as e:
        raise RuntimeError("Alpaca helper not available") from e
    from datetime import datetime, timedelta, timezone
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    bars = get_bars(symbol, TIMEFRAME, start, end, limit=10000)
    df = pd.DataFrame(bars)
    df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df[["open","high","low","close","volume"]]

def fetch_bars(symbol: str, days: int) -> pd.DataFrame:
    if not USE_YF:
        try:
            return _bars_alpaca(symbol, days)
        except Exception:
            if not USE_YF:
                raise
    return _bars_yf(symbol, days)

# =========================
# Feature engineering
# =========================
def _ema(s: pd.Series, span: int) -> pd.Series:
    s = pd.Series(s, index=s.index)  # force Series
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.Series(close, index=close.index)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = pd.Series(high, index=high.index)
    low  = pd.Series(low, index=low.index)
    close = pd.Series(close, index=close.index)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def _session_flags(index: pd.DatetimeIndex) -> pd.Series:
    # RTH for US: 13:30–20:00 UTC approx
    idx = index.tz_convert("UTC")
    hour = idx.hour
    minute = idx.minute
    hm = hour*60 + minute
    return ((hm >= 13*60+30) & (hm <= 20*60+0)).astype(float)  # 1.0 RTH, 0.0 ETH

def _time_sin_cos(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = index.tz_convert("UTC")
    mins = idx.hour*60 + idx.minute
    angle = 2*np.pi * (mins / (24*60))
    return pd.DataFrame({"tod_sin": np.sin(angle), "tod_cos": np.cos(angle)}, index=index)

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure plain Series objects for arithmetic
    open_  = pd.Series(df["open"],  index=df.index, dtype="float64")
    high   = pd.Series(df["high"],  index=df.index, dtype="float64")
    low    = pd.Series(df["low"],   index=df.index, dtype="float64")
    close  = pd.Series(df["close"], index=df.index, dtype="float64")
    volume = pd.Series(df["volume"],index=df.index, dtype="float64")

    out = pd.DataFrame(index=df.index)

    # returns
    out["ret1"] = close.pct_change(1)
    out["ret3"] = close.pct_change(3)
    out["ret6"] = close.pct_change(6)

    # EMAs
    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    out["ema9"]  = ema9
    out["ema21"] = ema21
    out["ema50"] = ema50

    # EMA gaps (use local Series to avoid DataFrame alignment surprises)
    out["ema_gap_9"]  = (close - ema9) / (close.replace(0, np.nan))
    out["ema_gap_21"] = (close - ema21) / (close.replace(0, np.nan))

    # RSI + norm
    rsi14 = _rsi(close, 14)
    out["rsi14"] = rsi14
    out["rsi_norm"] = (rsi14 - 50.0) / 50.0

    # ATR, ATR%
    atr = _atr(high, low, close, 14)
    out["atr"] = atr
    out["atr_pct"] = atr / (close.abs() + 1e-9)

    # Volume zscore
    vmean = volume.rolling(50, min_periods=10).mean()
    vstd  = volume.rolling(50, min_periods=10).std()
    out["vol_z"] = (volume - vmean) / (vstd + 1e-9)

    # Time encodings & session
    tod = _time_sin_cos(df.index)
    out = out.join(tod, how="left")
    out["is_rth"] = _session_flags(df.index)

    # Tidy
    out = out.replace([np.inf, -np.inf], np.nan).dropna().astype("float32")
    return out

# =========================
# Labeling (+HORIZON)
# =========================
def build_xy(df: pd.DataFrame, feats: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    close = pd.Series(df["close"], index=df.index)
    close = close.reindex(feats.index)
    fut = close.shift(-horizon)
    y = (fut > close).astype(np.float32).iloc[:-horizon]
    X = feats.iloc[:-horizon].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    return X.values.astype("float32"), y.values.astype("float32")

# =========================
# Torch dataset / model
# =========================
class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64], dropout: float = 0.25):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits

# =========================
# Training loop
# =========================
def train_loop(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, wd: float, early_stop: int, device: str):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val = float("inf"); best_state = None; patience = 0

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_loss += loss.item() * len(xb)
        va_loss /= len(val_loader.dataset)

        print(f"[ep {ep:02d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                print(f"Early stopping at epoch {ep}.")
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model

# =========================
# Build dataset across symbols
# =========================
def build_dataset(symbols: List[str], days: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    framesX = []; framesY = []
    feat_names: List[str] | None = None

    for sym in symbols:
        df = fetch_bars(sym, days)
        feats = make_features(df)
        X, y = build_xy(df, feats, horizon)
        if len(X) < 200:
            continue
        if feat_names is None:
            feat_names = list(feats.columns)
        framesX.append(X); framesY.append(y)

    if not framesX:
        raise RuntimeError("No training data built. Check data access & symbols.")
    X = np.vstack(framesX)
    y = np.concatenate(framesY)
    return X, y, feat_names

# =========================
# Main
# =========================
def main():
    print(f"Training NN on symbols={SYMBOLS} days={TRAIN_DAYS} tf={TIMEFRAME} horizon=+{HORIZON}")
    X, y, feat_names = build_dataset(SYMBOLS, TRAIN_DAYS, HORIZON)

    # chronological split
    n = len(X)
    cut = int(n * (1.0 - VAL_SPLIT))
    Xtr, Xva = X[:cut], X[cut:]
    ytr, yva = y[:cut], y[cut:]

    # scale
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)

    # loaders
    tr_ds = XYDataset(Xtr_s, ytr)
    va_ds = XYDataset(Xva_s, yva)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=X.shape[1], hidden=[192, 96, 48], dropout=DROPOUT).to(device)

    model = train_loop(model, tr_dl, va_dl, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP, device)

    # quick validation metrics
    model.eval()
    with torch.no_grad():
        logits = []
        for xb, _ in va_dl:
            xb = xb.to(device)
            logits.append(model(xb).cpu().numpy())
        logits = np.concatenate(logits)
        probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    acc = float((preds == yva).mean())
    print(f"Validation accuracy (horizon +{HORIZON} bars): {acc:.3f}")

    # save artifacts
    torch.save({"in_dim": X.shape[1], "state_dict": model.state_dict()}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEAT_JSON, "w") as f:
        json.dump({"features": feat_names, "horizon": HORIZON}, f)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved feature meta -> {FEAT_JSON}")

if __name__ == "__main__":
    main()
