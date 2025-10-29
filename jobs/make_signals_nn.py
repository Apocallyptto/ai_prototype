# jobs/make_signals_nn.py
"""
nn_predict() returns a dict:
{
  "AAPL": {"side": "buy"/"sell"/"hold", "strength": 0.00-1.00},
  ...
}
Currently loads the ACTIVE GradientBoosting model from models_meta.
"""
import os, logging, joblib, psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
BAR_INTERVAL = os.getenv("ML_BAR_INTERVAL", "5m")
LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "10"))
P_BUY = float(os.getenv("ML_BUY_THRESHOLD", "0.53"))   # small alpha threshold
P_SELL = float(os.getenv("ML_SELL_THRESHOLD", "0.47"))
MAX_HOLD_BIAS = float(os.getenv("ML_HOLD_BAND", "0.02"))  # deadzone around 0.5

def _active_artifact_path() -> str | None:
    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        logging.warning("DB_URL/DATABASE_URL not set; cannot query models_meta.")
        return None
    sql = """
      SELECT artifact_path FROM models_meta
      WHERE model_name = 'gbc_5m' AND is_active = TRUE
      ORDER BY created_at DESC LIMIT 1
    """
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return row[0] if row else None

def _download(sym: str) -> pd.DataFrame:
    start = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    df = yf.download(sym, interval=BAR_INTERVAL, start=start, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"no data for {sym}")
    df = df.rename(columns=str.title)
    return df[["Open","High","Low","Close","Volume"]].dropna()

def _features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["ret1"] = df["Close"].pct_change(1)
    f["ret3"] = df["Close"].pct_change(3)
    f["ret6"] = df["Close"].pct_change(6)
    f["hl_spread"] = (df["High"] - df["Low"]) / df["Close"].shift(1)
    f["oc_spread"] = (df["Close"] - df["Open"]) / df["Open"]
    f["vchg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    f["ma5"] = df["Close"].rolling(5).mean() / df["Close"] - 1
    f["ma12"] = df["Close"].rolling(12).mean() / df["Close"] - 1
    f["rsi14"] = _rsi(df["Close"], 14)
    return f.fillna(0)

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = -np.where(delta < 0, delta, 0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def nn_predict() -> dict[str, dict]:
    art = _active_artifact_path()
    if not art or not os.path.exists(art):
        logging.warning("No ACTIVE model found; returning HOLD for all symbols.")
        return {s: {"side":"hold","strength":0.0} for s in SYMBOLS}

    bundle = joblib.load(art)
    model = bundle["model"]

    out = {}
    for sym in SYMBOLS:
        try:
            df = _download(sym)
            X = _features(df).values
            if X.shape[0] == 0:
                out[sym] = {"side":"hold","strength":0.0}
                continue
            p = float(model.predict_proba(X[-1:])[:,1][0])  # prob up
            side = "hold"
            strength = abs(p - 0.5) * 2.0  # map [0,1] prob to [0,1] confidence
            if p >= P_BUY and (p - 0.5) > MAX_HOLD_BIAS:
                side = "buy"
            elif p <= P_SELL and (0.5 - p) > MAX_HOLD_BIAS:
                side = "sell"
            out[sym] = {"side": side, "strength": round(min(1.0, max(0.0, strength)), 4)}
        except Exception as e:
            logging.exception(f"{sym}: nn_predict failed: {e}")
            out[sym] = {"side":"hold","strength":0.0}
    return out

if __name__ == "__main__":
    import pprint
    pprint.pp(nn_predict())
