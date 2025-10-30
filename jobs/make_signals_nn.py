# jobs/make_signals_nn.py
import os, logging, joblib, yfinance as yf
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_nn")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
LOOKBACK_DAYS = int(os.getenv("ML_LOOKBACK_DAYS", "30"))
BAR_INTERVAL = os.getenv("ML_BAR_INTERVAL", "5m")
TARGET_HORIZON_BARS = int(os.getenv("ML_TARGET_HORIZON_BARS", "6"))

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df["return"] = df["Close"].pct_change()
    df["vol"] = df["return"].rolling(10).std()
    df["ema_fast"] = df["Close"].ewm(span=12).mean()
    df["ema_slow"] = df["Close"].ewm(span=26).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["rsi"] = _rsi(df["Close"])
    df = df.dropna()
    return df

def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def nn_predict(symbols_csv: str):
    results = {}
    model_path = os.path.join(MODEL_DIR, "gbc_5m.pkl")
    model = joblib.load(model_path)
    log.info("Loaded model: %s", model_path)

    for symbol in symbols_csv.split(","):
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        try:
            df = yf.download(symbol, interval=BAR_INTERVAL, period="2d", progress=False, auto_adjust=False)
            df = _prepare_features(df)
            if df.empty:
                continue
            X = df[["return", "vol", "macd", "rsi"]].values
            X_scaled = StandardScaler().fit_transform(X)
            probs = model.predict_proba(X_scaled)
            p_up = probs[-1, 1]
            strength = abs(p_up - 0.5) * 2
            side = "buy" if p_up > 0.55 else "sell" if p_up < 0.45 else "hold"
            results[symbol] = {"side": side, "strength": round(strength, 3)}
        except Exception as e:
            log.warning("%s failed: %s", symbol, e)

    return results

if __name__ == "__main__":
    print(nn_predict(",".join(SYMBOLS)))
