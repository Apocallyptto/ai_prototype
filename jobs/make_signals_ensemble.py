"""
make_signals_ensemble.py
---------------------------------------
Combines rule-based + ML ensemble to generate trading signals
and write them to the 'signals' table.
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from sqlalchemy import create_engine
from joblib import load
from sklearn.preprocessing import StandardScaler

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("make_signals_ensemble")

# ============================================================
# CONFIG
# ============================================================
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
ENGINE = create_engine(DB_URL, pool_pre_ping=True)

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")]
MODEL_PATH = os.getenv("MODEL_DIR", "/app/models/gbc_5m.pkl")

MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.50"))
RULE_WEIGHT = float(os.getenv("RULE_WEIGHT", "0.4"))
NN_WEIGHT = float(os.getenv("NN_WEIGHT", "0.6"))
LOOKBACK_BARS = int(os.getenv("ENSEMBLE_LOOKBACK_BARS", "200"))

# ============================================================
# HELPERS
# ============================================================

def _fetch_bars(sym: str, interval: str = "5m", period: str = "2d") -> pd.DataFrame:
    """
    Robust yfinance fetch that always returns lower-case columns
    (open, high, low, close, volume).
    Handles MultiIndex, mixed-case, and empty returns.
    """
    df = yf.download(
        tickers=sym,
        interval=interval,
        period=period,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",
        prepost=False,
    )

    if df is None or len(df) == 0:
        raise ValueError(f"no bars for {sym}")

    # Handle MultiIndex format ('Open','AAPL') or ('open','aapl')
    if isinstance(df.columns, pd.MultiIndex):
        sub = None
        for key in (sym.lower(), sym.upper()):
            try:
                sub = df.xs(key, level=1, axis=1)
                break
            except Exception:
                continue
        if sub is None:
            try:
                sub = df.swaplevel(0, 1, axis=1).xs(sym.lower(), level=0, axis=1)
            except Exception:
                raise ValueError(f"multiindex bars missing {sym} slice")

        sub.columns = [str(c).lower() for c in sub.columns]
        need = ["open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in sub.columns]
        if missing:
            raise ValueError(f"bars missing columns for {sym}: have {list(sub.columns)} need {need}")
        out = sub[need].dropna().copy()

    else:
        df.columns = [str(c).lower() for c in df.columns]
        need = ["open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"bars missing columns for {sym}: have {list(df.columns)} need {need}")
        out = df[need].dropna().copy()

    try:
        out.index = out.index.tz_localize(None)
    except Exception:
        pass

    if out.empty:
        raise ValueError(f"empty bars after cleaning for {sym}")

    return out


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI and EMA-based features."""
    close = df["close"]
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["ema_fast"] = close.ewm(span=12, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=26, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
    df["ema_signal"] = np.where(df["ema_diff"] > 0, 1, -1)
    return df.dropna()


def _rule_signal(df: pd.DataFrame) -> float:
    """Rule-based signal using RSI and EMA crossover."""
    if len(df) < 2:
        return 0.0
    last = df.iloc[-1]
    rsi = last["rsi"]
    ema_signal = last["ema_signal"]
    if rsi < 35 and ema_signal > 0:
        return 1.0
    elif rsi > 65 and ema_signal < 0:
        return -1.0
    return 0.0


def _nn_signal(df: pd.DataFrame, model) -> float:
    """Feed the most recent features to the ML model."""
    try:
        feat_cols = ["close", "ema_fast", "ema_slow", "ema_diff", "rsi"]
        X = df[feat_cols].tail(LOOKBACK_BARS).values
        X_scaled = StandardScaler().fit_transform(X)
        probs = model.predict_proba(X_scaled)[-1]
        # prob[1] = buy probability, prob[0] = sell probability
        return float(probs[1] - probs[0])
    except Exception:
        return 0.0


def _write_signal(symbol: str, side: str, strength: float, px: float):
    """Insert signal into DB."""
    ts = datetime.now(timezone.utc)
    data = pd.DataFrame(
        [{"created_at": ts, "symbol": symbol, "side": side, "strength": strength, "px": px}]
    )
    data.to_sql("signals", ENGINE, if_exists="append", index=False)


# ============================================================
# MAIN
# ============================================================
def main():
    log.info("make_signals_ensemble | symbols=%s", ",".join(SYMBOLS))

    try:
        model = load(MODEL_PATH)
        log.info("Loaded model bundle: %s", MODEL_PATH)
    except Exception as e:
        log.warning("No model loaded: %s", e)
        model = None

    inserted = 0
    for sym in SYMBOLS:
        try:
            df = _fetch_bars(sym)
            df = _calc_indicators(df)

            rule_val = _rule_signal(df)
            nn_val = _nn_signal(df, model) if model else 0.0
            ensemble = RULE_WEIGHT * rule_val + NN_WEIGHT * nn_val

            if ensemble > MIN_STRENGTH:
                side = "buy"
            elif ensemble < -MIN_STRENGTH:
                side = "sell"
            else:
                log.info("%s neutral (%.3f)", sym, ensemble)
                continue

            px = float(df["close"].iloc[-1])
            log.info("Ensemble -> %s: %s (%.3f)", sym, side, ensemble)
            _write_signal(sym, side, ensemble, px)
            inserted += 1

        except Exception as e:
            log.warning("%s failed: %s", sym, e)
            continue

    log.info("✔ inserted %d signals", inserted)


if __name__ == "__main__":
    main()
