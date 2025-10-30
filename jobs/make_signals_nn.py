# jobs/make_signals_nn.py
import os, logging, joblib, psycopg2
from typing import Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# Alpaca (bars via IEX feed)
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET", "")
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")  # 'iex' works on free/paper
USE_ALPACA = bool(ALPACA_KEY and ALPACA_SEC)

# Yahoo fallback
import yfinance as yf

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("make_signals_nn")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")

# -------------------------------
# Feature engineering (minimal, consistent)
# -------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = (high - low).abs()
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _make_features(bars: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Expect columns: ['open','high','low','close','volume'] indexed by time
    Returns a single-row DataFrame with latest features.
    """
    if len(bars) < 30:
        return None

    df = bars.copy()
    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret6"] = df["close"].pct_change(6)
    df["rsi14"] = _rsi(df["close"], 14)
    df["atr14"] = _atr(df["high"], df["low"], df["close"], 14)

    feats = df[["ret1", "ret3", "ret6", "rsi14", "atr14"]].iloc[-1:]
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# -------------------------------
# Data readers
# -------------------------------
def _bars_alpaca(symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch recent 5m bars from Alpaca (IEX).
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SEC)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(5, TimeFrame.Unit.Minute),
            limit=limit,
            feed=ALPACA_FEED,
        )
        out = client.get_stock_bars(req)
        if symbol not in out.data or len(out.data[symbol]) == 0:
            return None
        recs = out.data[symbol]
        df = pd.DataFrame([{
            "t": r.timestamp, "open": r.open, "high": r.high, "low": r.low,
            "close": r.close, "volume": r.volume
        } for r in recs])
        df = df.set_index("t").sort_index()
        return df
    except Exception as e:
        log.debug(f"Alpaca bars failed for {symbol}: {e}")
        return None

def _bars_yahoo(symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch recent 5m bars from Yahoo as fallback.
    """
    try:
        # ~ 60*5m bars ≈ last trading day intraday
        df = yf.download(tickers=symbol, interval="5m", period="2d", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        df.index.name = "t"
        return df.tail(limit)
    except Exception as e:
        log.debug(f"Yahoo bars failed for {symbol}: {e}")
        return None

def _recent_features(symbol: str) -> Optional[pd.DataFrame]:
    bars = None
    if USE_ALPACA:
        bars = _bars_alpaca(symbol)
    if bars is None:
        bars = _bars_yahoo(symbol)
    if bars is None or bars.empty:
        return None
    return _make_features(bars)

# -------------------------------
# Model registry
# -------------------------------
def _active_model_path() -> str:
    """
    Read the active model path from models_meta (created by tools.db_migrate_models & train job).
    """
    sql = """
        SELECT path
        FROM public.models_meta
        WHERE is_active = true
        ORDER BY created_at DESC
        LIMIT 1
    """
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError("No ACTIVE model found in models_meta.")
        return row[0]

# -------------------------------
# Public API
# -------------------------------
def nn_predict(symbols_csv: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    symbols = [s.strip().upper() for s in (symbols_csv or DEFAULT_SYMBOLS).split(",") if s.strip()]
    model_path = _active_model_path()
    model_file = model_path if os.path.isabs(model_path) else os.path.join(MODEL_DIR, os.path.basename(model_path))
    model = joblib.load(model_file)

    out: Dict[str, Dict[str, float]] = {}
    for s in symbols:
        feats = _recent_features(s)
        if feats is None:
            log.warning(f"{s}: NN inference failed: no bars/feats; HOLD")
            out[s] = {"side": "hold", "strength": 0.0}
            continue

        try:
            # Ensure column order stable; add missing with zeros
            needed = ["ret1", "ret3", "ret6", "rsi14", "atr14"]
            for col in needed:
                if col not in feats.columns:
                    feats[col] = 0.0
            x = feats[needed].to_numpy()

            proba = model.predict_proba(x)[0][1]  # probability of "up"
            side = "buy" if proba > 0.55 else ("sell" if proba < 0.45 else "hold")
            strength = float(round(abs(proba - 0.5) * 2, 3))
            out[s] = {"side": side, "strength": strength}
        except Exception as e:
            log.warning(f"{s}: NN inference failed: {e}; HOLD")
            out[s] = {"side": "hold", "strength": 0.0}
    return out

if __name__ == "__main__":
    import sys as _sys
    arg = _sys.argv[1] if len(_sys.argv) > 1 else None
    print(nn_predict(arg))
