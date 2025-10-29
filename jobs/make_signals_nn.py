# jobs/make_signals_nn.py
import os, logging, json
from datetime import datetime, timedelta, timezone
from typing import Dict
import numpy as np
import pandas as pd
import psycopg2
import joblib
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BAR_INTERVAL = os.getenv("ML_BAR_INTERVAL", "5m")
LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "30"))
TARGET_HORIZON = int(os.getenv("ML_TARGET_HORIZON_BARS", "6"))
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = -np.where(delta < 0, delta, 0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

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

def _download(sym: str) -> pd.DataFrame:
    start = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    df = yf.download(sym, interval=BAR_INTERVAL, start=start, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"no data for {sym}")
    df = df.rename(columns=str.title)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

def _load_active_model():
    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        logging.warning("No DB_URL set; cannot look up models_meta")
        return None

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT model_name, version, artifact_path, notes
            FROM models_meta
            WHERE is_active=TRUE
            ORDER BY created_at DESC NULLS LAST, version DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            logging.warning("No ACTIVE model found in models_meta.")
            return None
        model_name, version, artifact_path, notes = row
        path = artifact_path
        try:
            payload = joblib.load(path)
            logging.info(f"Loaded ACTIVE model {model_name} v{version} from {path}")
            return payload
        except Exception as e:
            logging.exception(f"Failed to load model artifact {path}: {e}")
            return None

def nn_predict(symbols_csv: str | None = None) -> Dict[str, Dict[str, float]]:
    """
    Returns: {SYM: {"side": "buy|sell|hold", "strength": float}}
    """
    # Allow turning off via env
    if os.getenv("DISABLE_NN", "0") == "1":
        logging.info("NN path is DISABLED via DISABLE_NN=1")
        return _holds(symbols_csv)

    symbols = (symbols_csv or os.getenv("SYMBOLS", "AAPL,MSFT,SPY")).split(",")
    model_payload = _load_active_model()
    if not model_payload:
        logging.warning("No ACTIVE model found; returning HOLD for all symbols.")
        return _holds(",".join(symbols))

    model = model_payload["model"]

    out: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        try:
            df = _download(sym)
            feats = _features(df).iloc[:-TARGET_HORIZON] if len(df) > TARGET_HORIZON else _features(df)
            if feats.empty:
                raise RuntimeError("no features")
            x = feats.iloc[-1:].values  # latest row
            prob_up = float(model.predict_proba(x)[0, 1])
            # map to side/strength
            if prob_up > 0.55:
                out[sym] = {"side": "buy", "strength": round(prob_up, 3)}
            elif prob_up < 0.45:
                out[sym] = {"side": "sell", "strength": round(1 - prob_up, 3)}
            else:
                out[sym] = {"side": "hold", "strength": round(abs(prob_up - 0.5), 3)}
        except Exception as e:
            logging.warning(f"{sym}: NN predict failed: {e}; HOLD.")
            out[sym] = {"side": "hold", "strength": 0.0}

    return out

def _holds(symbols_csv: str | None) -> Dict[str, Dict[str, float]]:
    symbols = (symbols_csv or os.getenv("SYMBOLS", "AAPL,MSFT,SPY")).split(",")
    return {s: {"side": "hold", "strength": 0.0} for s in symbols}

if __name__ == "__main__":
    print(json.dumps(nn_predict(), indent=2))
