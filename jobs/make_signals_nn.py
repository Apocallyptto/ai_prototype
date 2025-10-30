# jobs/make_signals_nn.py
import os, logging, joblib, yfinance as yf
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_nn")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
BAR_INTERVAL = os.getenv("ML_BAR_INTERVAL", "5m")

FEATURES = ["return", "vol", "macd", "rsi", "ema_fast", "ema_slow"]  # 6 features expected by model

def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # df has columns: Open, High, Low, Close, Adj Close, Volume
    df["return"]    = df["Close"].pct_change()
    df["vol"]       = df["return"].rolling(10).std()
    df["ema_fast"]  = df["Close"].ewm(span=12).mean()
    df["ema_slow"]  = df["Close"].ewm(span=26).mean()
    df["macd"]      = df["ema_fast"] - df["ema_slow"]
    df["rsi"]       = _rsi(df["Close"])
    df = df.dropna()
    return df

def _load_model():
    path = os.path.join(MODEL_DIR, "gbc_5m.pkl")
    obj = joblib.load(path)
    log.info("Loaded model bundle: %s", path)
    # allow dict bundles: {"model": clf, "scaler": scaler} or just the estimator
    if isinstance(obj, dict):
        model  = obj.get("model", obj.get("estimator", obj))
        scaler = obj.get("scaler")  # may be None
    else:
        model, scaler = obj, None
    return model, scaler

def nn_predict(symbols_csv: str):
    model, bundle_scaler = _load_model()
    results = {}

    for symbol in symbols_csv.split(","):
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        try:
            df = yf.download(symbol, interval=BAR_INTERVAL, period="2d", progress=False, auto_adjust=False)
            if df is None or df.empty:
                continue
            df = _prepare_features(df)
            if df.empty:
                continue

            X = df[FEATURES].astype(float).values
            if bundle_scaler is not None:
                Xs = bundle_scaler.transform(X)
            else:
                # Fit a throwaway scaler on recent window (ok for inference-only if model was trained with scaling in-bundle)
                Xs = StandardScaler().fit_transform(X)

            # Probability or decision score → probability
            if hasattr(model, "predict_proba"):
                p_up = float(model.predict_proba(Xs)[-1, 1])
            else:
                # robust fallback: sigmoid(decision_function)
                if hasattr(model, "decision_function"):
                    dfv = float(model.decision_function(Xs)[-1])
                else:
                    # worst-case fallback: map predict() class to pseudo-prob
                    cls = float(model.predict(Xs)[-1])
                    dfv = 2 * (cls - 0.5)  # rough mapping {-1..+1} -> prob-ish
                p_up = 1.0 / (1.0 + np.exp(-dfv))

            strength = float(abs(p_up - 0.5) * 2.0)  # 0..1
            side = "buy" if p_up > 0.55 else "sell" if p_up < 0.45 else "hold"
            results[symbol] = {"side": side, "strength": round(strength, 3)}
        except Exception as e:
            log.warning("%s failed: %s", symbol, e)

    return results

if __name__ == "__main__":
    print(nn_predict(",".join(SYMBOLS)))
