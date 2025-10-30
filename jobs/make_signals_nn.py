# jobs/make_signals_nn.py
import os, logging, json
import pandas as pd, numpy as np, psycopg2, joblib
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_nn")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

def _dsn():
    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    host = os.getenv("DB_HOST","postgres")
    user = os.getenv("DB_USER","postgres")
    pw   = os.getenv("DB_PASSWORD","postgres")
    db   = os.getenv("DB_NAME","trader")
    port = os.getenv("DB_PORT","5432")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"

def _active_model_path() -> str:
    with psycopg2.connect(_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT path FROM public.models_meta WHERE is_active=true ORDER BY created_at DESC LIMIT 1;")
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError("No ACTIVE model found in models_meta.")
        return row[0]

# compute features directly from price history in signals table
def _latest_features_for(symbol: str) -> pd.DataFrame:
    with psycopg2.connect(_dsn()) as conn, conn.cursor() as cur:
        # try to detect which price column your signals table uses
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='signals';
        """)
        cols = [r[0] for r in cur.fetchall()]
        if "price" in cols:
            price_col = "price"
        elif "px" in cols:
            price_col = "px"
        elif "price_close" in cols:
            price_col = "price_close"
        else:
            raise RuntimeError(f"No known price column found in 'signals' table. Columns={cols}")

        query = f"""
            SELECT created_at, {price_col}
            FROM public.signals
            WHERE symbol=%s
            ORDER BY created_at DESC
            LIMIT 200;
        """
        cur.execute(query, (symbol,))
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"No recent signals for {symbol}")

    df = pd.DataFrame(rows, columns=["ts", "close"]).sort_values("ts").reset_index(drop=True)
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["vol_z"] = (df["ret1"].rolling(50).std())
    df["rsi14"] = _rsi(df["close"], 14)
    df["atr14"] = df["ret1"].rolling(14).std()
    return df.dropna().iloc[-1:][["ret1", "ret5", "ret10", "vol_z", "rsi14", "atr14"]]

def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_u = up.ewm(alpha=1/period, min_periods=period).mean()
    ma_d = down.ewm(alpha=1/period, min_periods=period).mean()
    rs = ma_u / (ma_d + 1e-9)
    return 100 - (100 / (1 + rs))

def nn_predict(symbols_csv: str) -> Dict[str, Dict[str, Any]]:
    syms = [s.strip().upper() for s in symbols_csv.split(",")]
    try:
        blob = joblib.load(_active_model_path())
        model = blob["model"]
        feats = blob["features"]
    except Exception as e:
        log.warning(f"No ACTIVE model found: {e}")
        return {s: {"side": "hold", "strength": 0.0} for s in syms}

    results = {}
    for sym in syms:
        try:
            X = _latest_features_for(sym)[feats].values
            proba = float(model.predict_proba(X)[0][1])
            side = "buy" if proba > 0.55 else ("sell" if proba < 0.45 else "hold")
            strength = abs(proba - 0.5) * 2
            results[sym] = {"side": side, "strength": round(strength, 3)}
        except Exception as e:
            log.warning(f"{sym}: NN inference failed: {e}; HOLD")
            results[sym] = {"side": "hold", "strength": 0.0}
    return results

if __name__ == "__main__":
    print(json.dumps(nn_predict(os.getenv("SYMBOLS","AAPL,MSFT,SPY")), indent=2))
