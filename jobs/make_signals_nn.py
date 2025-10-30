# jobs/make_signals_nn.py
import os, logging, joblib, psycopg2
import numpy as np
import pandas as pd
from typing import Dict, Optional

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_nn")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")

# Alpaca (optional, IEX feed for free tier)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    ALPACA_OK = True
except Exception:
    ALPACA_OK = False

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

def _conn():
    return psycopg2.connect(DB_URL)

def _active_model_path() -> Optional[str]:
    sql = "SELECT path FROM public.models_meta WHERE is_active=true ORDER BY created_at DESC LIMIT 1"
    with _conn() as c, c.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return row[0] if row else None

def _fetch_latest(sym: str, bars: int = 200) -> pd.DataFrame:
    # Try Alpaca first (IEX feed)
    if ALPACA_OK and ALPACA_API_KEY and ALPACA_API_SECRET:
        try:
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
            # minute bars over ~1–2 days to make 5m features
            end = pd.Timestamp.utcnow().tz_localize("UTC")
            start = end - pd.Timedelta(days=2)
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Minute,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                feed=DataFeed.IEX,
                adjustment=None,
            )
            df = client.get_stock_bars(req).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level=0).copy()
            if not df.empty:
                df = df.tz_convert("UTC")
                # resample to 5m
                o = df["open"].resample("5min").first()
                h = df["high"].resample("5min").max()
                l = df["low"].resample("5min").min()
                c = df["close"].resample("5min").last()
                v = df["volume"].resample("5min").sum()
                out = pd.DataFrame({"open": o,"high": h,"low": l,"close": c,"volume": v}).dropna()
                return out.tail(bars)
        except Exception as e:
            log.warning(f"Alpaca latest failed for {sym}: {e}")

    # Yahoo fallback
    import yfinance as yf
    df = yf.download(sym, period="5d", interval="5m", progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"no latest data for {sym}")
    df = df.rename(columns=str.lower)
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df[["open","high","low","close","volume"]].dropna().tail(bars)

def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_u = up.ewm(alpha=1/period, min_periods=period).mean()
    ma_d = down.ewm(alpha=1/period, min_periods=period).mean()
    rs = ma_u / (ma_d + 1e-9)
    return 100 - (100 / (1 + rs))

def _atr(h, l, c, period):
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

# IMPORTANT: features must match training
_FEATS = ["ret1","ret5","ret10","vol_z","rsi14","atr14"]

def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["vol_z"] = (df["volume"] - df["volume"].rolling(50).mean()) / (df["volume"].rolling(50).std()+1e-9)
    df["rsi14"] = _rsi(df["close"], 14)
    df["atr14"] = _atr(df["high"], df["low"], df["close"], 14) / (df["close"].rolling(14).mean()+1e-9)
    return df.dropna().reset_index(drop=True)

def nn_predict(symbols_csv: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    syms = [s.strip().upper() for s in (symbols_csv or DEFAULT_SYMBOLS).split(",") if s.strip()]
    model_path = _active_model_path()
    if not model_path:
        log.warning("No ACTIVE model found in models_meta.")
        return {s: {"side": "hold", "strength": 0.0} for s in syms}

    blob = joblib.load(model_path)
    model = blob["model"]
    feats = blob.get("features") or blob.get("feats") or _FEATS  # tolerate old keys

    out = {}
    for s in syms:
        try:
            df = _fetch_latest(s, bars=200)
            Xdf = _features(df)
            if Xdf.empty:
                raise RuntimeError("no feature rows")
            x = Xdf.iloc[-1][feats].values.reshape(1, -1)
            proba = model.predict_proba(x)[0][1]  # P(up)
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
