# jobs/train_model_gbc.py
import os, sys, json, time, math, logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import psycopg2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "30"))
BAR_INTERVAL = os.getenv("ML_BAR_INTERVAL", "5m")  # only '5m' supported here
HORIZON = int(os.getenv("ML_TARGET_HORIZON_BARS", "6"))  # 6 * 5m = 30m

def _dsn():
    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if dsn: return dsn
    host = os.getenv("DB_HOST","postgres")
    user = os.getenv("DB_USER","postgres")
    pw   = os.getenv("DB_PASSWORD","postgres")
    db   = os.getenv("DB_NAME","trader")
    port = os.getenv("DB_PORT","5432")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"

def _have_alpaca():
    return (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET"))

def _fetch_alpaca(sym: str) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    api = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    tf = TimeFrame.Minute
    if BAR_INTERVAL != "5m":
        logger.warning("Only 5m supported in this example; defaulting to 5m.")
    req = StockBarsRequest(symbol_or_symbols=sym, timeframe=tf, start=start, end=end)
    bars = api.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(sym, level=0).copy()
    if bars.empty:
        raise RuntimeError(f"Alpaca returned no bars for {sym}")
    # Resample to 5m
    bars = bars.tz_convert("UTC")
    o = bars["open"].resample("5min").first()
    h = bars["high"].resample("5min").max()
    l = bars["low"].resample("5min").min()
    c = bars["close"].resample("5min").last()
    v = bars["volume"].resample("5min").sum()
    df = pd.DataFrame({"open":o, "high":h, "low":l, "close":c, "volume":v}).dropna()
    return df

def _fetch_yahoo(sym: str) -> pd.DataFrame:
    import yfinance as yf
    # conservative: we fetch 1d 1m and resample or 60d 5m; use 60d 5m to get longer history
    for attempt in range(5):
        df = yf.download(sym, period="60d", interval="5m", progress=False, auto_adjust=False, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.tz_localize("UTC") if df.tz is None else df.tz_convert("UTC")
            return df.rename(columns=str.lower)[["open","high","low","close","volume"]]
        sleep = 2*(attempt+1)
        logger.warning("Yahoo empty/blocked; retrying in %ss …", sleep)
        time.sleep(sleep)
    raise RuntimeError(f"Yahoo returned no bars for {sym}")

def _download(sym: str) -> pd.DataFrame:
    if _have_alpaca():
        try:
            logger.info("download %s (Alpaca) …", sym)
            return _fetch_alpaca(sym)
        except Exception as e:
            logger.warning("Alpaca failed for %s: %s. Trying Yahoo fallback …", sym, e)
    logger.info("download %s (Yahoo) …", sym)
    return _fetch_yahoo(sym)

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["vol_z"] = (df["volume"] - df["volume"].rolling(50).mean()) / (df["volume"].rolling(50).std()+1e-9)
    df["rsi14"] = _rsi(df["close"], 14)
    df["atr14"] = _atr(df["high"], df["low"], df["close"], 14) / (df["close"].rolling(14).mean()+1e-9)
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    df = df.dropna()
    return df

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

def _aggregate_training(symbols: List[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = _download(sym.strip().upper())
        f = _make_features(df)
        f["symbol"] = sym.strip().upper()
        frames.append(f)
    allx = pd.concat(frames).sort_index()
    return allx

def _save_and_register(model, features: List[str], symbols: List[str], params: Dict, metrics: Dict) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "gbc_5m.pkl")
    joblib.dump({"model": model, "features": features, "params": params}, path)

    dsn = _dsn()
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE public.models_meta SET is_active=false WHERE is_active=true;")
            cur.execute("""
                INSERT INTO public.models_meta (model_type, path, features, symbols, params, metrics, is_active)
                VALUES (%s,%s,%s,%s,%s,%s,true)
            """, ("gbc_5m", path, features, symbols, json.dumps(params), json.dumps(metrics)))
    return path

def main():
    data = _aggregate_training(SYMBOLS)
    feats = ["ret1","ret5","ret10","vol_z","rsi14","atr14"]
    X = data[feats].values
    y = data["target"].values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, shuffle=False)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtrain, ytrain)
    pred = clf.predict(Xtest)
    acc = float(accuracy_score(ytest, pred))
    f1  = float(f1_score(ytest, pred))
    logger.info("trained GBC | acc=%.4f f1=%.4f | n_train=%d n_test=%d", acc, f1, len(Xtrain), len(Xtest))
    params = {"lookback_days":LOOKBACK_DAYS,"bar_interval":BAR_INTERVAL,"horizon":HORIZON}
    metrics = {"acc":acc,"f1":f1}
    path = _save_and_register(clf, feats, [s.strip().upper() for s in SYMBOLS], params, metrics)
    logger.info("saved & activated model at %s", path)

if __name__ == "__main__":
    main()
