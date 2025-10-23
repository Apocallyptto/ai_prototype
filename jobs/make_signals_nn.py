# jobs/make_signals_nn.py
from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import psycopg2

import torch
import torch.nn as nn

# === Model artifact paths ===
OUTDIR      = os.getenv("NN_OUTDIR", "models")
MODEL_PATH  = os.path.join(OUTDIR, "nn_5m.pt")
SCALER_PATH = os.path.join(OUTDIR, "nn_5m_scaler.pkl")
FEAT_JSON   = os.path.join(OUTDIR, "nn_5m_features.json")

TIMEFRAME   = os.getenv("NN_TIMEFRAME", "5Min")
SYMBOLS     = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MIN_STRENGTH= float(os.getenv("MIN_STRENGTH", "0.60"))
USE_YF      = os.getenv("USE_YFINANCE_TRAIN","1").lower() in {"1","true","yes","y"}  # reuse same source by default
DATABASE_URL= os.getenv("DATABASE_URL")
PORTFOLIO_ID= os.getenv("PORTFOLIO_ID")  # optional

# --- import the exact same feature builder used in training
from jobs.train_nn import make_features  # IMPORTANT: stay in sync with training features

# ------------- Data fetch helpers -------------
def _bars_alpaca(symbol: str, lookback_days: int = 10, timeframe: str = "5Min") -> pd.DataFrame | None:
    try:
        from lib.broker_alpaca import get_bars
    except Exception:
        return None
    from datetime import datetime, timedelta, timezone
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    try:
        rows = get_bars(symbol, timeframe, start, end, limit=10000)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df.set_index("ts", inplace=True)
        df = df[["open","high","low","close","volume"]]
        df = df[~df.index.duplicated(keep="last")]
        return df.astype("float64")
    except Exception:
        return None

def _bars_yf(symbol: str, interval: str = "5m") -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        # yfinance caps 5m to ~60 days; we only need enough for features
        df = yf.download(symbol, period="30d", interval=interval, progress=False, auto_adjust=False)
        if df is None or len(df) == 0:
            return None
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
        df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        df = df[["open","high","low","close","volume"]]
        df = df[~df.index.duplicated(keep="last")]
        return df.astype("float64")
    except Exception:
        return None

def fetch_recent_bars(symbol: str) -> pd.DataFrame:
    df = None
    if not USE_YF:
        df = _bars_alpaca(symbol, 10, "5Min")
    if df is None:
        df = _bars_yf(symbol, "5m")
    if df is None or df.empty:
        raise RuntimeError(f"No bars for {symbol}")
    return df.tail(400)  # enough history for EMAs/ATR features

# ------------- Torch model -------------
class MLP(nn.Module):
    """
    Matches the trainer's architecture (nn.Sequential assigned to self.net).
    """
    def __init__(self, in_dim: int, hidden: List[int] = [192, 96, 48], dropout: float = 0.25):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def _load_artifacts() -> Tuple[MLP, object, List[str], int]:
    # load feature meta
    with open(FEAT_JSON, "r") as f:
        meta = json_load_safely(f.read())
    feat_names: List[str] = meta["features"]
    horizon: int = int(meta.get("horizon", 3))

    # scaler
    scaler = joblib.load(SCALER_PATH)

    # model
    import torch
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    in_dim_ckpt = int(ckpt.get("in_dim", len(feat_names)))
    model = MLP(in_dim=in_dim_ckpt, hidden=[192,96,48], dropout=_env_float("NN_DROPOUT", 0.25))

    sd = ckpt.get("state_dict", ckpt)  # support raw state_dict saves
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        # Try to remap old-style keys (fc1/fc2/out) -> new net indices
        remapped = {}
        for k, v in sd.items():
            nk = k
            if k.startswith("fc1."):
                nk = k.replace("fc1", "net.0")
            elif k.startswith("fc2."):
                nk = k.replace("fc2", "net.3")
            elif k.startswith("out."):
                nk = k.replace("out", "net.6")
            remapped[nk] = v
        try:
            model.load_state_dict(remapped, strict=False)
            print(f"[WARN] Loaded NN with remapped keys due to mismatch: {e}")
        except Exception as e2:
            # Or inverse mapping (new -> old)
            remapped2 = {}
            for k, v in sd.items():
                nk = k.replace("net.0", "fc1").replace("net.3", "fc2").replace("net.6", "out")
                remapped2[nk] = v
            model = MLP(in_dim=in_dim_ckpt, hidden=[192,96,48], dropout=_env_float("NN_DROPOUT", 0.25))
            model.load_state_dict(remapped2, strict=False)
            print(f"[WARN] Loaded NN with inverse-remapped keys due to mismatch: {e2}")

    model.eval()
    return model, scaler, feat_names, horizon

# ------------- Utilities -------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def json_load_safely(s: str):
    import json
    return json.loads(s)

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

# ------------- Core prediction -------------
def predict_for_symbols(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Returns: { "AAPL": {"side":"buy","strength":0.62}, ... }
    (no DB writes)
    """
    model, scaler, feat_names, _ = _load_artifacts()

    results: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        df = fetch_recent_bars(sym)
        feats = make_features(df)
        # make sure we only use the features the model expects
        feats = feats.reindex(columns=feat_names).dropna()
        if feats.empty:
            continue
        x = feats.iloc[[-1]].values.astype("float32")
        x_s = scaler.transform(x)
        with torch.no_grad():
            xb = torch.from_numpy(x_s)
            logit = float(model(xb).item())
            prob_up = 1.0 / (1.0 + np.exp(-logit))

        side = "buy" if prob_up >= 0.5 else "sell"
        strength = float(prob_up if side == "buy" else 1.0 - prob_up)
        results[sym] = {"side": side, "strength": strength}
    return results

# Backwards-compat alias used by ensemble job
nn_predict = predict_for_symbols

# ------------- CLI entrypoint -------------
def main():
    import sys
    now = datetime.now(timezone.utc)
    syms = SYMBOLS
    preds = predict_for_symbols(syms)
    for sym in syms:
        r = preds.get(sym)
        if not r:
            print(f"{sym}: no prediction")
            continue
        side = r["side"]
        strength = r["strength"]
        if strength < MIN_STRENGTH:
            print(f"{sym}: below MIN_STRENGTH ({strength:.2f} < {MIN_STRENGTH}) (nn)")
            continue
        # print to console
        print(f"{sym}: {side} strength={strength:.2f} at {now.isoformat()} (nn)")
        # insert into DB if configured
        try:
            _insert_signal(sym, side, strength, now, PORTFOLIO_ID)
        except Exception as e:
            print(f"[WARN] failed to insert signal for {sym}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
