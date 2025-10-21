# jobs/make_signals_nn.py
from __future__ import annotations
import os, json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import joblib

# DB
try:
    import psycopg  # type: ignore
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2  # type: ignore

from lib.broker_alpaca import get_bars
from ml.nn_train import make_features  # reuse exact same feature recipe

# ---------- Config ----------
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
TIMEFRAME = os.getenv("NN_TIMEFRAME", "5Min")
OUTDIR = os.getenv("NN_OUTDIR", "models")
MODEL_PATH = os.path.join(OUTDIR, "nn_5m.pt")
SCALER_PATH = os.path.join(OUTDIR, "nn_5m_scaler.pkl")
FEAT_JSON = os.path.join(OUTDIR, "nn_5m_features.json")

PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.30"))

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

def insert_signal(symbol: str, side: str, strength: float, ts: datetime):
    sql = """
    INSERT INTO signals (symbol, side, strength, ts, portfolio_id)
    VALUES (%s, %s, %s, %s, %s)
    """
    with _pg_conn() as c:
        with c.cursor() as cur:
            cur.execute(sql, (symbol, side, float(strength), ts, PORTFOLIO_ID))
        c.commit()

def latest_features(symbol: str) -> Optional[np.ndarray]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)  # enough for indicators
    bars = get_bars(symbol, TIMEFRAME, start, end, limit=2000)
    df = pd.DataFrame(bars)
    if df.empty:
        return None
    df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[["open","high","low","close","volume"]]
    feats = make_features(df)
    if feats.empty:
        return None
    return feats.iloc[-1:].values.astype("float32")  # shape (1, F)

def main():
    # load artifacts
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = torch.jit.trace(torch.nn.Sequential(), torch.randn(1))  # dummy to appease types
    from ml.nn_model import MLP
    model = MLP(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    with open(FEAT_JSON, "r") as f:
        meta = json.load(f)
    # meta["feature_names"] available if you need alignment later

    for sym in SYMBOLS:
        x = latest_features(sym)
        if x is None:
            print(f"{sym}: no features")
            continue
        xs = scaler.transform(x).astype("float32")
        with torch.no_grad():
            p_up = float(torch.sigmoid(model(torch.from_numpy(xs))).view(-1).item())
        # convert to side + strength
        if p_up >= 0.5:
            side = "buy"
            strength = p_up  # 0.5..1
        else:
            side = "sell"
            strength = 1.0 - p_up  # 0.5..1
        # threshold to avoid spam
        if strength >= MIN_STRENGTH:
            ts = datetime.now(timezone.utc)
            insert_signal(sym, side, strength, ts)
            print(f"{sym}: {side} strength={strength:.2f} at {ts.isoformat()}")
        else:
            print(f"{sym}: below MIN_STRENGTH ({strength:.2f} < {MIN_STRENGTH})")

if __name__ == "__main__":
    main()
