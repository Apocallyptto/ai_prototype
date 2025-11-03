"""
make_signals_ensemble.py
---------------------------------------
Combines rule-based + ML ensemble to generate trading signals
and write them to the 'signals' table.
"""

import os
import glob
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import load
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

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

# Accept either a file path or a directory containing *.pkl
MODEL_PATH_ENV = os.getenv("MODEL_DIR", "/app/models/gbc_5m.pkl")

MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.50"))
RULE_WEIGHT = float(os.getenv("RULE_WEIGHT", "0.4"))
NN_WEIGHT = float(os.getenv("NN_WEIGHT", "0.6"))
LOOKBACK_BARS = int(os.getenv("ENSEMBLE_LOOKBACK_BARS", "200"))

# ============================================================
# HELPERS
# ============================================================

def _resolve_model_path(path_env: str) -> str | None:
    """
    If path_env is a file, use it. If it's a directory, search for *.pkl
    and return the newest. If nothing found, return None.
    """
    path_env = os.path.abspath(path_env)
    if os.path.isfile(path_env):
        return path_env
    if os.path.isdir(path_env):
        candidates = sorted(glob.glob(os.path.join(path_env, "*.pkl")))
        if candidates:
            return candidates[-1]  # newest by name sort (works if timestamped) or last added
    return None


def _fetch_bars(sym: str, interval: str = "5m", period: str = "2d") -> pd.DataFrame:
    """
    Robust yfinance fetch that always yields columns:
    open, high, low, close, volume (all lower-case).
    Handles MultiIndex in either ordering:
      ('close','aapl') or ('aapl','close'), and single-index.
    """
    df = yf.download(
        tickers=sym,
        interval=interval,
        period=period,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",  # may still give single index for single ticker; that's fine.
        prepost=False,
    )

    if df is None or len(df) == 0:
        raise ValueError(f"no bars for {sym}")

    need = ["open", "high", "low", "close", "volume"]

    if isinstance(df.columns, pd.MultiIndex):
        # Figure out which level holds price names by checking level values
        lvl0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        lvl1 = [str(x).lower() for x in df.columns.get_level_values(1)]

        # Case A: price first level -> ('open','aapl')
        price_first = set(["open", "high", "low", "close", "volume"]).issubset(set(lvl0))
        # Case B: ticker first level -> ('aapl','open')
        price_second = set(["open", "high", "low", "close", "volume"]).issubset(set(lvl1))

        if price_first:
            # select all (price, any_ticker) and then drop the ticker level
            sub = df.loc[:, df.columns.get_level_values(0).str.lower().isin(need)]
            sub.columns = [str(c[0]).lower() for c in sub.columns]
            out = sub[need].copy()
        elif price_second:
            sub = df.loc[:, df.columns.get_level_values(1).str.lower().isin(need)]
            sub.columns = [str(c[1]).lower() for c in sub.columns]
            out = sub[need].copy()
        else:
            # Fallback: try slicing by sym explicitly (both cases)
            grabbed = None
            for ticker_level in (0, 1):
                try:
                    tmp = df.xs(sym, level=ticker_level, axis=1)
                    tmp.columns = [str(c).lower() for c in tmp.columns]
                    if set(need).issubset(set(tmp.columns)):
                        grabbed = tmp[need].copy()
                        break
                except Exception:
                    continue
            if grabbed is None:
                raise ValueError(f"bars missing required columns for {sym}: MultiIndex={list(map(str, df.columns))}")
            out = grabbed
    else:
        # Single index: normalize names
        df.columns = [str(c).lower() for c in df.columns]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"bars missing {missing} for {sym}: have {list(df.columns)}")
        out = df[need].copy()

    # Clean
    out = out.dropna()
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
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(14, min_periods=14).mean()
    roll_down = down.rolling(14, min_periods=14).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    df["rsi"] = 100 - (100 / (1 + rs))
    df["ema_fast"] = close.ewm(span=12, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=26, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
    df["ema_signal"] = np.where(df["ema_diff"] > 0, 1, -1)
    return df.dropna()


def _rule_signal(df: pd.DataFrame) -> float:
    """Simple RSI + EMA rule."""
    if len(df) < 2:
        return 0.0
    last = df.iloc[-1]
    rsi = float(last["rsi"])
    ema_sig = int(last["ema_signal"])
    if rsi < 35 and ema_sig > 0:
        return 1.0
    if rsi > 65 and ema_sig < 0:
        return -1.0
    return 0.0


def _nn_signal(df: pd.DataFrame, model) -> float:
    """Feed recent features to the ML model; return buy_prob - sell_prob."""
    if model is None or df.empty:
        return 0.0
    try:
        feat_cols = ["close", "ema_fast", "ema_slow", "ema_diff", "rsi"]
        X = df[feat_cols].tail(LOOKBACK_BARS).values
        if X.shape[0] < 5:  # not enough context
            return 0.0
        X_scaled = StandardScaler().fit_transform(X)
        probs = model.predict_proba(X_scaled)
        # use the last row's probs
        p_buy = float(probs[-1][1])
        p_sell = float(probs[-1][0])
        return p_buy - p_sell
    except Exception as e:
        log.warning("NN path failed: %s", e)
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

    model_path = _resolve_model_path(MODEL_PATH_ENV)
    model = None
    if model_path:
        try:
            model = load(model_path)
            log.info("Loaded model: %s", model_path)
        except Exception as e:
            log.warning("Model load failed (%s): %s", model_path, e)
    else:
        log.warning("No model file found in: %s", MODEL_PATH_ENV)

    inserted = 0
    for sym in SYMBOLS:
        try:
            df = _fetch_bars(sym, interval="5m", period="2d")
            df = _calc_indicators(df)

            rule_val = _rule_signal(df)
            nn_val = _nn_signal(df, model)
            ensemble = RULE_WEIGHT * rule_val + NN_WEIGHT * nn_val

            side = "buy" if ensemble > MIN_STRENGTH else "sell" if ensemble < -MIN_STRENGTH else "hold"
            if side == "hold":
                log.info("%s neutral (%.3f)", sym, ensemble)
                continue

            px = float(df["close"].iloc[-1])
            log.info("Ensemble -> %s: %s (%.3f)", sym, side, ensemble)
            _write_signal(sym, side, ensemble, px)
            inserted += 1

        except Exception as e:
            log.warning("%s failed: %s", sym, e)

    log.info("✔ inserted %d signals", inserted)


if __name__ == "__main__":
    main()
