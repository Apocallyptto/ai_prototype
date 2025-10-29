# jobs/make_signals_nn.py
import os, logging, joblib, psycopg2
import numpy as np
import pandas as pd
from typing import Dict

# Yahoo fallback
import yfinance as yf

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_nn")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")

# Alpaca
try:
    from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
    ALPACA_OK = True
except Exception:
    ALPACA_OK = False

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

def _conn():
    return psycopg2.connect(DB_URL)

def _active_model_path() -> str | None:
    sql = "SELECT path FROM public.models_meta WHERE is_active=true ORDER BY created_at DESC LIMIT 1"
    with _conn() as c, c.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return row[0] if row else None

def _fetch_latest(sym: str, bars: int = 60) -> pd.DataFrame:
    # Try Alpaca intraday first
    if ALPACA_OK and ALPACA_API_KEY and ALPACA_API_SECRET:
        try:
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(5, TimeFrame.Minute),
                limit=bars
            )
            resp = client.get_stock_bars(req)
            rows = resp.data.get(sym, [])
            if rows:
                df = pd.DataFrame([{
                    "ts": r.timestamp, "open": float(r.open), "high": float(r.high),
                    "low": float(r.low), "close": float(r.close), "volume": int(r.volume)
                } for r in rows]).sort_values("ts").reset_index(drop=True)
                return df
        except Exception as e:
            log.warning(f"Alpaca latest failed for {sym}: {e}")

    # Fallback Yahoo
    df = yf.download(sym, period="5d", interval="5m", progress=False, auto_adjust=False, prepost=False)
    if df is None or df.empty:
        raise RuntimeError(f"no latest data for {sym}")
    df = df.rename(columns=str.lower).reset_index().rename(columns={"datetime": "ts"})
    return df[["ts","open","high","low","close","volume"]].dropna().reset_index(drop=True)

def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["ret5"] = df["close"].pct_change(5)
    df["ma_fast"] = df["close"].rolling(10).mean()
    df["ma_slow"] = df["close"].rolling(50).mean()
    df["ma_diff"] = (df["ma_fast"] - df["ma_slow"]) / df["close"]
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    return df.dropna().reset_index(drop=True)

def nn_predict(symbols_csv: str | None = None) -> Dict[str, Dict[str, float]]:
    syms = (symbols_csv or DEFAULT_SYMBOLS).split(",")
    model_path = _active_model_path()
    if not model_path:
        log.warning("No ACTIVE model found in models_meta.")
        return {s: {"side": "hold", "strength": 0.0} for s in syms}

    blob = joblib.load(model_path)
    model = blob["model"]
    feats = blob["feats"]

    out = {}
    for s in syms:
        try:
            df = _fetch_latest(s, bars=120)
            Xdf = _features(df)
            if Xdf.empty:
                raise RuntimeError("no feature rows")
            x = Xdf.iloc[-1][feats].values.reshape(1, -1)
            proba = model.predict_proba(x)[0][1]  # P(up)
            # Convert to side/strength
            side = "buy" if proba > 0.55 else ("sell" if proba < 0.45 else "hold")
            strength = abs(proba - 0.5) * 2  # 0..1
            out[s] = {"side": side, "strength": float(round(strength, 3))}
        except Exception as e:
            log.warning(f"{s}: NN inference failed: {e}; HOLD")
            out[s] = {"side": "hold", "strength": 0.0}
    return out

if __name__ == "__main__":
    import sys as _sys
    arg = _sys.argv[1] if len(_sys.argv) > 1 else None
    print(nn_predict(arg))
