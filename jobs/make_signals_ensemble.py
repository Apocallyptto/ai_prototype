"""
jobs/make_signals_ensemble.py
---------------------------------------
Combines rule-based + ML ensemble to generate trading signals
and writes them to the 'signals' table.
"""

import os
import glob
import logging
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import load
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

# MODEL_DIR may be a file or a directory
MODEL_PATH_ENV = os.getenv("MODEL_DIR", "/app/models")

MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.50"))
RULE_WEIGHT = float(os.getenv("RULE_WEIGHT", "0.4"))
NN_WEIGHT = float(os.getenv("NN_WEIGHT", "0.6"))
LOOKBACK_BARS = int(os.getenv("ENSEMBLE_LOOKBACK_BARS", "200"))

# ============================================================
# HELPERS: model discovery
# ============================================================

def _is_scaler(obj: Any) -> bool:
    cls = obj.__class__.__name__.lower()
    return "scaler" in cls or "standardscaler" in cls or "minmaxscaler" in cls


def _resolve_model_artifacts(path_env: str) -> Tuple[Optional[Any], Optional[Any], Dict[str, str]]:
    """
    Returns (model, scaler, meta) where meta contains chosen file paths.
    Accepts:
      - a file pointing to a model (or a dict bundle with {'model','scaler'})
      - a file pointing to a scaler (then we search sibling model)
      - a directory containing *.pkl (we pick the best combo)
    Preference order in a directory:
      1) A dict-bundle pickle with keys {'model', 'scaler'} or {'model'}
      2) A non-scaler *.pkl as model + optional *_scaler.pkl as scaler
    """
    meta = {}
    path_env = os.path.abspath(path_env)

    def _try_load(path: str):
        try:
            return load(path)
        except Exception as e:
            log.warning("Failed loading %s: %s", path, e)
            return None

    # Case: explicit file
    if os.path.isfile(path_env):
        obj = _try_load(path_env)
        meta["loaded"] = path_env
        if obj is None:
            return None, None, meta
        if isinstance(obj, dict):
            model = obj.get("model")
            scaler = obj.get("scaler")
            if model is None:
                # maybe the dict itself is the model?
                model = obj
            if model is not None:
                meta["model_file"] = path_env
            if scaler is not None:
                meta["scaler_file"] = path_env
            return model, scaler, meta
        # single object
        if _is_scaler(obj):
            # look for a sibling model
            cand = _find_sibling_model(os.path.dirname(path_env))
            if cand:
                model_obj = _try_load(cand)
                if model_obj is not None:
                    meta["model_file"] = cand
                    meta["scaler_file"] = path_env
                    return model_obj, obj, meta
            # scaler alone is not useful for predictions
            log.warning("Loaded a scaler from %s but no model found next to it.", path_env)
            return None, obj, meta
        else:
            meta["model_file"] = path_env
            # try find sibling scaler
            scaler_path = _find_sibling_scaler(os.path.dirname(path_env))
            scaler_obj = _try_load(scaler_path) if scaler_path else None
            if scaler_obj:
                meta["scaler_file"] = scaler_path
            return obj, scaler_obj, meta

    # Case: directory
    if os.path.isdir(path_env):
        pkls = sorted(glob.glob(os.path.join(path_env, "*.pkl")))
        # First pass: dict bundles
        for p in pkls:
            obj = _try_load(p)
            if isinstance(obj, dict) and ("model" in obj or "scaler" in obj):
                model = obj.get("model")
                scaler = obj.get("scaler")
                if model is not None:
                    meta["model_file"] = p
                    if scaler is not None:
                        meta["scaler_file"] = p
                    return model, scaler, meta
        # Second pass: pick a non-scaler as model, nearest *_scaler as scaler
        model_path = None
        for p in pkls:
            obj = _try_load(p)
            if obj is not None and not _is_scaler(obj):
                model_path = p
        if model_path:
            model = _try_load(model_path)
            scaler_path = _find_sibling_scaler(path_env)
            scaler = _try_load(scaler_path) if scaler_path else None
            meta["model_file"] = model_path
            if scaler_path:
                meta["scaler_file"] = scaler_path
            return model, scaler, meta

        # Third pass: only scaler files present
        for p in pkls:
            obj = _try_load(p)
            if obj is not None and _is_scaler(obj):
                # try find *another* non-scaler
                cand = _find_sibling_model(path_env)
                if cand:
                    model_obj = _try_load(cand)
                    if model_obj is not None:
                        meta["model_file"] = cand
                        meta["scaler_file"] = p
                        return model_obj, obj, meta
                log.warning("Only scaler found in %s; no model.", path_env)
                return None, obj, meta

        log.warning("No usable pickle files found in %s", path_env)
        return None, None, meta

    # Otherwise nothing
    log.warning("MODEL_DIR path not found: %s", path_env)
    return None, None, meta


def _find_sibling_scaler(dirpath: str) -> Optional[str]:
    # look for explicit *_scaler.pkl first
    cands = sorted(glob.glob(os.path.join(dirpath, "*scaler*.pkl")))
    return cands[-1] if cands else None


def _find_sibling_model(dirpath: str) -> Optional[str]:
    # choose any *.pkl that is not obviously a scaler
    cands = [p for p in sorted(glob.glob(os.path.join(dirpath, "*.pkl"))) if "scaler" not in os.path.basename(p).lower()]
    return cands[-1] if cands else None

# ============================================================
# HELPERS: market data
# ============================================================

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
        group_by="ticker",
        prepost=False,
    )

    if df is None or len(df) == 0:
        raise ValueError(f"no bars for {sym}")

    need = ["open", "high", "low", "close", "volume"]

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        lvl1 = [str(x).lower() for x in df.columns.get_level_values(1)]
        price_first = set(need).issubset(set(lvl0))
        price_second = set(need).issubset(set(lvl1))

        if price_first:
            sub = df.loc[:, df.columns.get_level_values(0).str.lower().isin(need)]
            sub.columns = [str(c[0]).lower() for c in sub.columns]
            out = sub[need].copy()
        elif price_second:
            sub = df.loc[:, df.columns.get_level_values(1).str.lower().isin(need)]
            sub.columns = [str(c[1]).lower() for c in sub.columns]
            out = sub[need].copy()
        else:
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
        df.columns = [str(c).lower() for c in df.columns]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"bars missing {missing} for {sym}: have {list(df.columns)}")
        out = df[need].copy()

    out = out.dropna()
    try:
        out.index = out.index.tz_localize(None)
    except Exception:
        pass
    if out.empty:
        raise ValueError(f"empty bars after cleaning for {sym}")
    return out

# ============================================================
# INDICATORS & SIGNALS
# ============================================================

def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def _safe_predict_score(model, X: np.ndarray) -> float:
    """
    Try predict_proba; if missing, try decision_function; else use predict (0/1)
    Return a value in [-1, 1] where >0 favors BUY, <0 favors SELL.
    We assume class order [0,1] = [SELL, BUY] when using proba.
    """
    # predict_proba path
    try:
        probs = model.predict_proba(X)
        # use last row
        p_buy = float(probs[-1][-1])
        p_sell = 1.0 - p_buy
        return p_buy - p_sell
    except Exception:
        pass

    # decision_function path (SVM/linear models)
    try:
        dec = model.decision_function(X)
        score = float(dec[-1])
        # scale to [-1,1] if needed
        # many models already output approximate margins; tanh squashes large margins
        return float(np.tanh(score))
    except Exception:
        pass

    # predict (class label) fallback
    try:
        y = model.predict(X)
        cls = int(y[-1])
        return 1.0 if cls == 1 else -1.0
    except Exception:
        return 0.0


def _nn_signal(df: pd.DataFrame, model, scaler) -> float:
    """Use model (+ optional scaler) on recent features; return buy_prob - sell_prob in [-1,1]."""
    if model is None or df.empty:
        return 0.0
    feat_cols = ["close", "ema_fast", "ema_slow", "ema_diff", "rsi"]
    if not all(c in df.columns for c in feat_cols):
        return 0.0
    X = df[feat_cols].tail(LOOKBACK_BARS).values
    if X.shape[0] < 5:
        return 0.0

    # apply scaler if provided
    if scaler is not None and _is_scaler(scaler):
        try:
            X = scaler.transform(X)
        except Exception as e:
            log.warning("Provided scaler.transform failed: %s", e)
    else:
        # lightweight fit on the fly (kept only for compatibility)
        try:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
        except Exception:
            pass

    return _safe_predict_score(model, X)

# ============================================================
# DB WRITE
# ============================================================

def _write_signal(symbol: str, side: str, strength: float, px: float):
    ts = datetime.now(timezone.utc)
    data = pd.DataFrame(
        [{"created_at": ts, "symbol": symbol, "side": side, "strength": float(strength), "px": float(px)}]
    )
    data.to_sql("signals", ENGINE, if_exists="append", index=False)

# ============================================================
# MAIN
# ============================================================

def main():
    log.info("make_signals_ensemble | symbols=%s", ",".join(SYMBOLS))

    model, scaler, meta = _resolve_model_artifacts(MODEL_PATH_ENV)
    if model is None and scaler is None:
        log.warning("No model loaded: %s", MODEL_PATH_ENV)
    else:
        msg = "Loaded"
        if "model_file" in meta:
            msg += f" model: {meta['model_file']}"
        if "scaler_file" in meta:
            msg += f" | scaler: {meta['scaler_file']}"
        log.info(msg)

    inserted = 0
    for sym in SYMBOLS:
        try:
            df = _fetch_bars(sym, interval="5m", period="2d")
            df = _calc_indicators(df)

            rule_val = _rule_signal(df)                   # in {-1,0,1}
            nn_val = _nn_signal(df, model, scaler)        # in [-1,1]
            # blend to [-1,1]
            ensemble_raw = RULE_WEIGHT * rule_val + NN_WEIGHT * nn_val

            # convert to side + absolute strength [0,1]
            if ensemble_raw > MIN_STRENGTH:
                side = "buy"
                strength = float(min(1.0, abs(ensemble_raw)))
            elif ensemble_raw < -MIN_STRENGTH:
                side = "sell"
                strength = float(min(1.0, abs(ensemble_raw)))
            else:
                log.info("%s neutral (%.3f)", sym, ensemble_raw)
                continue

            px = float(df["close"].iloc[-1])
            log.info("Ensemble -> %s: %s (%.3f)", sym, side, ensemble_raw)
            _write_signal(sym, side, strength, px)
            inserted += 1

        except Exception as e:
            log.warning("%s failed: %s", sym, e)

    log.info("✔ inserted %d signals", inserted)


if __name__ == "__main__":
    main()
