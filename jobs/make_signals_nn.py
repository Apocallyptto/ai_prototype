# jobs/make_signals_nn.py
import os, logging, joblib, psycopg2
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from datetime import timedelta
import yfinance as yf

# Alpaca creds (optional, falls back to Yahoo)
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET", "")
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")
USE_ALPACA = bool(ALPACA_KEY and ALPACA_SEC)

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_nn")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")

# ---------- features ----------
def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def _atr(h, l, c, n=14):
    hl = (h-l).abs()
    hc = (h-c.shift(1)).abs()
    lc = (l-c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _make_features(bars: pd.DataFrame) -> Optional[pd.DataFrame]:
    if bars is None or bars.empty or len(bars) < 30:
        return None
    df = bars.copy()
    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret6"] = df["close"].pct_change(6)
    df["rsi14"] = _rsi(df["close"], 14)
    df["atr14"] = _atr(df["high"], df["low"], df["close"], 14)
    out = df[["ret1","ret3","ret6","rsi14","atr14"]].iloc[-1:].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return out

# ---------- data ----------
def _bars_alpaca(symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.requests import StockBarsRequest
        cli = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SEC)
        req = StockBarsRequest(symbol_or_symbols=symbol,
                               timeframe=TimeFrame(5, TimeFrame.Unit.Minute),
                               limit=limit,
                               feed=ALPACA_FEED)
        out = cli.get_stock_bars(req)
        rows = out.data.get(symbol, [])
        if not rows: return None
        df = pd.DataFrame([{"t": r.timestamp, "open": r.open, "high": r.high,
                            "low": r.low, "close": r.close, "volume": r.volume}
                           for r in rows]).set_index("t").sort_index()
        return df
    except Exception as e:
        log.debug(f"Alpaca bars failed for {symbol}: {e}")
        return None

def _bars_yahoo(symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, interval="5m", period="2d", progress=False)
        if df is None or df.empty: return None
        df = df.rename(columns=str.lower)
        df.index.name = "t"
        return df.tail(limit)
    except Exception as e:
        log.debug(f"Yahoo bars failed for {symbol}: {e}")
        return None

def _recent_features(symbol: str) -> Optional[pd.DataFrame]:
    bars = _bars_alpaca(symbol) if USE_ALPACA else None
    if bars is None:
        bars = _bars_yahoo(symbol)
    return _make_features(bars)

# ---------- model registry ----------
def _active_model_path() -> str:
    sql = """SELECT path
             FROM public.models_meta
             WHERE is_active=true
             ORDER BY created_at DESC
             LIMIT 1"""
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError("No ACTIVE model found in models_meta.")
    return row[0]

def _load_model_and_features(path: str):
    obj = joblib.load(path)
    # Handle both plain estimator and dict bundle
    if isinstance(obj, dict):
        model = obj.get("model", obj.get("estimator", obj))
        features: List[str] = obj.get("features", ["ret1","ret3","ret6","rsi14","atr14"])
    else:
        model = obj
        features = ["ret1","ret3","ret6","rsi14","atr14"]
    return model, features

# ---------- public ----------
def nn_predict(symbols_csv: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    symbols = [s.strip().upper() for s in (symbols_csv or DEFAULT_SYMBOLS).split(",") if s.strip()]
    model_path = _active_model_path()
    if not os.path.isabs(model_path):
        model_path = os.path.join(MODEL_DIR, os.path.basename(model_path))
    model, feature_list = _load_model_and_features(model_path)

    out: Dict[str, Dict[str, float]] = {}
    for s in symbols:
        feats = _recent_features(s)
        if feats is None:
            log.warning(f"{s}: NN inference failed: no bars/feats; HOLD")
            out[s] = {"side":"hold","strength":0.0}
            continue
        # ensure all expected features are present/order fixed
        for col in feature_list:
            if col not in feats.columns:
                feats[col] = 0.0
        X = feats[feature_list].to_numpy()
        try:
            if hasattr(model, "predict_proba"):
                proba_up = float(model.predict_proba(X)[0][1])
            else:
                # fallback to decision_function or predict
                if hasattr(model, "decision_function"):
                    s_val = float(model.decision_function(X)[0])
                    proba_up = 1/(1+np.exp(-s_val))
                else:
                    pred = int(model.predict(X)[0])
                    proba_up = 0.75 if pred == 1 else 0.25
            side = "buy" if proba_up > 0.55 else ("sell" if proba_up < 0.45 else "hold")
            strength = round(abs(proba_up-0.5)*2, 3)
            out[s] = {"side": side, "strength": float(strength)}
        except Exception as e:
            log.warning(f"{s}: NN inference failed: {e}; HOLD")
            out[s] = {"side":"hold","strength":0.0}
    return out

if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv)>1 else None
    print(nn_predict(arg))
