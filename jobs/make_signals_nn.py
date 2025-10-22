# jobs/make_signals_nn.py
from __future__ import annotations
import os, json
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from jobs.train_nn import make_features

import numpy as np
import pandas as pd
import torch
import joblib

# DB drivers (psycopg3 preferred, fall back to psycopg2)
try:
    import psycopg  # psycopg3
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2

from lib.broker_alpaca import get_bars
from ml.nn_train import make_features  # keep the exact same feature recipe

# ---- Config ----
TIMEFRAME   = os.getenv("NN_TIMEFRAME", "5Min")
OUTDIR      = os.getenv("NN_OUTDIR", "models")
MODEL_PATH  = os.path.join(OUTDIR, "nn_5m.pt")
SCALER_PATH = os.path.join(OUTDIR, "nn_5m_scaler.pkl")
FEAT_JSON   = os.path.join(OUTDIR, "nn_5m_features.json")
SYMBOLS     = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
PORTFOLIO_ID= int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH= float(os.getenv("MIN_STRENGTH", "0.30"))

# Allow yfinance fallback for inference too (optional)
USE_YF = os.getenv("USE_YFINANCE_TRAIN", "0").lower() in {"1","true","yes","y"} or \
         os.getenv("USE_YFINANCE_INFER", "0").lower() in {"1","true","yes","y"}

# ---- DB helpers ----
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

def _signal_columns() -> set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name='signals'
    """
    with _pg_conn() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return {r[0] for r in rows}

def _insert_signal(symbol: str, side: str, strength: float, ts: datetime):
    cols = _signal_columns()
    col_names: list[str] = ["symbol", "side", "strength"]
    values: list[object] = [symbol, side, float(strength)]

    ts_col = "ts" if "ts" in cols else ("created_at" if "created_at" in cols else None)
    if ts_col:
        col_names.append(ts_col)
        values.append(ts)

    if "portfolio_id" in cols:
        col_names.append("portfolio_id")
        values.append(PORTFOLIO_ID)

    placeholders = ",".join(["%s"]*len(values))
    col_sql = ",".join(col_names)
    sql = f"INSERT INTO signals ({col_sql}) VALUES ({placeholders})"
    with _pg_conn() as c:
        with c.cursor() as cur:
            cur.execute(sql, tuple(values))
        c.commit()

# ---- Market data ----
def _bars_yf(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period="7d", interval="5m", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance empty for {symbol}")
    df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")
    df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def _latest_features(symbol: str) -> Optional[np.ndarray]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    try:
        bars = get_bars(symbol, TIMEFRAME, start, end, limit=2000)
        df = pd.DataFrame(bars)
        df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[["open","high","low","close","volume"]]
    except Exception as e:
        if USE_YF:
            print(f"[WARN] Alpaca bars failed for {symbol}: {e}. Falling back to yfinance.")
            df = _bars_yf(symbol)
        else:
            raise

    feats = make_features(df)
    if feats.empty:
        return None
    return feats.iloc[-1:].values.astype("float32")

# ---- Reusable predictor (NEW) ----
def predict_for_symbols(symbols: Sequence[str]) -> dict[str, dict]:
    """
    Returns a dict like:
       {'AAPL': {'side': 'buy', 'strength': 0.62}, ...}
    Does NOT write to DB. Safe to import from other modules.
    """
    from ml.nn_model import MLP  # local import to keep module load light

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = MLP(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    if os.path.exists(FEAT_JSON):
        try:
            with open(FEAT_JSON, "r") as f:
                _ = json.load(f)  # reserved for future validation
        except Exception:
            pass

    out: dict[str, dict] = {}
    for sym in [s.strip().upper() for s in symbols if s and s.strip()]:
        x = _latest_features(sym)
        if x is None:
            continue
        xs = scaler.transform(x).astype("float32")
        with torch.no_grad():
            p_up = float(torch.sigmoid(model(torch.from_numpy(xs))).view(-1).item())
        if p_up >= 0.5:
            side, strength = "buy", p_up
        else:
            side, strength = "sell", 1.0 - p_up
        out[sym] = {"side": side, "strength": float(strength)}
    return out

# ---- CLI main (kept as before) ----
def main():
    preds = predict_for_symbols(SYMBOLS)
    for sym in SYMBOLS:
        p = preds.get(sym)
        if not p:
            print(f"{sym}: no features")
            continue
        if p["strength"] >= MIN_STRENGTH:
            ts = datetime.now(timezone.utc)
            _insert_signal(sym, p["side"], p["strength"], ts)
            print(f"{sym}: {p['side']} strength={p['strength']:.2f} at {ts.isoformat()}")
        else:
            print(f"{sym}: below MIN_STRENGTH ({p['strength']:.2f} < {MIN_STRENGTH})")

if __name__ == "__main__":
    main()
