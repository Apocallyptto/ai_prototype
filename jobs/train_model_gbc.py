# jobs/train_model_gbc.py
import os, logging, joblib, json, time, io
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "30"))
BAR_INTERVAL = os.getenv("ML_BAR_INTERVAL", "5m")  # 1m/2m/5m/15m/30m/60m
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_NAME = "gbc_5m"
VERSION = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
TARGET_HORIZON = int(os.getenv("ML_TARGET_HORIZON_BARS", "6"))  # e.g. 6*5m = 30m ahead

os.makedirs(MODEL_DIR, exist_ok=True)

def _features(df: pd.DataFrame) -> pd.DataFrame:
    # Basic price/volatility features (keep it minimal & fast)
    f = pd.DataFrame(index=df.index)
    f["ret1"] = df["Close"].pct_change(1)
    f["ret3"] = df["Close"].pct_change(3)
    f["ret6"] = df["Close"].pct_change(6)
    f["hl_spread"] = (df["High"] - df["Low"]) / df["Close"].shift(1)
    f["oc_spread"] = (df["Close"] - df["Open"]) / df["Open"]
    f["vchg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    # small MAs
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

def _make_labels(df: pd.DataFrame) -> pd.Series:
    # Up/Down label after TARGET_HORIZON bars
    fut = df["Close"].shift(-TARGET_HORIZON)
    return (fut > df["Close"]).astype(int)

def _download(sym: str) -> pd.DataFrame:
    start = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    df = yf.download(sym, interval=BAR_INTERVAL, start=start, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"no data for {sym}")
    df = df.rename(columns=str.title)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

def main():
    X_list, y_list = [], []
    for sym in SYMBOLS:
        logging.info(f"download {sym} …")
        df = _download(sym)
        feats = _features(df)
        y = _make_labels(df)
        # align
        mask = y.index.intersection(feats.index)
        feats = feats.loc[mask]
        y = y.loc[mask]
        # cut last TARGET_HORIZON rows (no future)
        feats = feats.iloc[:-TARGET_HORIZON]
        y = y.iloc[:-TARGET_HORIZON]
        X_list.append(feats.values)
        y_list.append(y.values)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    logging.info(f"fit GBC on {X.shape[0]} rows, {X.shape[1]} features")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    # quick AUC on train (ok for now)
    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    logging.info(f"train AUC ~ {auc:.3f}")

    artifact_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{VERSION}.pkl")
    joblib.dump({
        "model": model,
        "symbols": SYMBOLS,
        "bar_interval": BAR_INTERVAL,
        "lookback_days": LOOKBACK_DAYS,
        "target_horizon": TARGET_HORIZON,
        "features": list(pd.DataFrame(X, columns=None).columns)  # placeholder
    }, artifact_path)
    logging.info(f"saved: {artifact_path}")

    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        logging.warning("DB_URL not set; skipping models_meta insert")
        return

    notes = json.dumps({"train_auc": float(auc)})
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        # deactivate previous actives for this model_name
        cur.execute("UPDATE models_meta SET is_active=FALSE WHERE model_name=%s", (MODEL_NAME,))
        # insert this one as active
        cur.execute("""
            INSERT INTO models_meta (model_name, version, artifact_path, notes, is_active)
            VALUES (%s,%s,%s,%s,TRUE)
        """, (MODEL_NAME, VERSION, artifact_path, notes))
        conn.commit()
    logging.info("models_meta updated (this version is ACTIVE).")

if __name__ == "__main__":
    main()
