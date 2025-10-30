# jobs/make_signals_nn.py
import os, logging, json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import psycopg2
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_nn")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "30"))

def _dsn():
    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    host = os.getenv("DB_HOST","postgres"); user = os.getenv("DB_USER","postgres")
    pw = os.getenv("DB_PASSWORD","postgres"); db = os.getenv("DB_NAME","trader")
    port = os.getenv("DB_PORT","5432")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"

def _active_model_path() -> str:
    with psycopg2.connect(_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT path FROM public.models_meta WHERE is_active=true ORDER BY created_at DESC LIMIT 1;")
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError("No ACTIVE model found in models_meta.")
        return row[0]

def _latest_features_for(symbol: str) -> pd.DataFrame:
    """
    Build a single-row feature vector from most recent signals/bars.
    Simple approach: reuse your technical features from jobs.make_signals
    by querying last N bars from your bars table if you have one; if not,
    derive from last signals row. For now, derive from last 50 signals rows.
    """
    with psycopg2.connect(_dsn()) as conn, conn.cursor() as cur:
        # expect you already insert rule/technical signals into 'signals'
        cur.execute("""
            SELECT ts, price, rsi14, atr14, ret1, ret5, ret10, vol_z
            FROM signals
            WHERE symbol=%s
            ORDER BY ts DESC
            LIMIT 50;
        """, (symbol,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"No recent signals for {symbol} to build features.")

    df = pd.DataFrame(rows, columns=["ts","price","rsi14","atr14","ret1","ret5","ret10","vol_z"]).sort_values("ts")
    # Use last row as current feature snapshot
    f = df.iloc[-1][["ret1","ret5","ret10","vol_z","rsi14","atr14"]].to_frame().T
    f.index = [pd.Timestamp.utcnow()]
    return f

def nn_predict(symbols_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Return {SYM: {"side": "buy|sell|hold", "strength": float}}
    """
    try:
        model_art = joblib.load(_active_model_path())
        model = model_art["model"]
        feats: List[str] = model_art["features"]
    except Exception as e:
        log.warning("No ACTIVE model found; returning HOLD for all symbols.")
        syms = [s.strip().upper() for s in symbols_csv.split(",")] if symbols_csv else ["AAPL","MSFT","SPY"]
        return {s: {"side":"hold","strength":0.0} for s in syms}

    results: Dict[str, Dict[str, Any]] = {}
    syms = [s.strip().upper() for s in symbols_csv.split(",")] if symbols_csv else ["AAPL","MSFT","SPY"]
    for sym in syms:
        try:
            X = _latest_features_for(sym)[feats].values
            prob_up = float(model.predict_proba(X)[0][1])  # class 1 == up
            side = "buy" if prob_up >= 0.55 else ("sell" if prob_up <= 0.45 else "hold")
            strength = abs(prob_up - 0.5) * 2  # 0..1
            results[sym] = {"side": side, "strength": round(strength, 3)}
        except Exception as e:
            log.warning("NN failed for %s: %s; default HOLD.", sym, e)
            results[sym] = {"side":"hold","strength":0.0}
    return results

if __name__ == "__main__":
    print(json.dumps(nn_predict(os.getenv("SYMBOLS","AAPL,MSFT,SPY")), indent=2))
