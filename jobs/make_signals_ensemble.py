# jobs/make_signals_ensemble.py
# ---------------------------------------------------------------------
# Ensemble signal writer with robust model loader + ATR features.
# Produces continuous rule score to avoid 0.000 outputs.
# ---------------------------------------------------------------------

import os
import glob
import logging
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import load
from sqlalchemy import create_engine

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("make_signals_ensemble")

# --- ENV / CONFIG -----------------------------------------------------
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
ENGINE = create_engine(DB_URL, pool_pre_ping=True)
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")]

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

# If your model was trained on 6 features, include atr:
# e.g. "close,ema_fast,ema_slow,ema_diff,rsi,atr"
FEAT_COLS = [c.strip() for c in os.getenv(
    "FEAT_COLS",
    "close,ema_fast,ema_slow,ema_diff,rsi,atr"
).split(",")]

LOOKBACK_BARS = int(os.getenv("ENSEMBLE_LOOKBACK_BARS", "200"))
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.50"))
RULE_WEIGHT = float(os.getenv("RULE_WEIGHT", "0.40"))
NN_WEIGHT = float(os.getenv("NN_WEIGHT", "0.60"))

# --- MODEL LOADING ----------------------------------------------------
def _is_scaler(obj: Any) -> bool:
    name = obj.__class__.__name__.lower()
    return "scaler" in name or "standardscaler" in name or "minmaxscaler" in name

def _try_load(path: str):
    try:
        return load(path)
    except Exception as e:
        log.warning("Failed loading %s: %s", path, e)
        return None

def _find_sibling_scaler(dirpath: str) -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(dirpath, "*scaler*.pkl")))
    return cands[-1] if cands else None

def _find_sibling_model(dirpath: str) -> Optional[str]:
    cands = [p for p in sorted(glob.glob(os.path.join(dirpath, "*.pkl")))
             if "scaler" not in os.path.basename(p).lower()]
    return cands[-1] if cands else None

def _resolve_model_artifacts(path_env: str) -> Tuple[Optional[Any], Optional[Any], Dict[str, str]]:
    meta: Dict[str, str] = {}
    path_env = os.path.abspath(path_env)
    if os.path.isfile(path_env):
        obj = _try_load(path_env)
        meta["loaded"] = path_env
        if obj is None:
            return None, None, meta
        if isinstance(obj, dict) and ("model" in obj or "scaler" in obj):
            model = obj.get("model")
            scaler = obj.get("scaler")
            if model is None:
                model = obj  # rare: dict IS the model-like object
            if model is not None:
                meta["model_file"] = path_env
            if scaler is not None:
                meta["scaler_file"] = path_env
            return model, scaler, meta
        if _is_scaler(obj):
            # scaler-only: try sibling model
            cand = _find_sibling_model(os.path.dirname(path_env))
            if cand:
                model_obj = _try_load(cand)
                if model_obj is not None:
                    meta["model_file"] = cand
                    meta["scaler_file"] = path_env
                    return model_obj, obj, meta
            log.warning("Scaler loaded but no sibling model found next to %s", path_env)
            return None, obj, meta
        # single model
        meta["model_file"] = path_env
        scaler_path = _find_sibling_scaler(os.path.dirname(path_env))
        scaler_obj = _try_load(scaler_path) if scaler_path else None
        if scaler_obj:
            meta["scaler_file"] = scaler_path
        return obj, scaler_obj, meta

    if os.path.isdir(path_env):
        pkls = sorted(glob.glob(os.path.join(path_env, "*.pkl")))
        # Prefer bundle dict {'model','scaler'}
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
        # Then plain model + optional *_scaler
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
        # Only scaler present
        for p in pkls:
            obj = _try_load(p)
            if obj is not None and _is_scaler(obj):
                cand = _find_sibling_model(path_env)
                if cand:
                    model_obj = _try_load(cand)
                    if model_obj is not None:
                        meta["model_file"] = cand
                        meta["scaler_file"] = p
                        return model_obj, obj, meta
                log.warning("Only scaler found under %s; no model", path_env)
                return None, obj, meta
        log.warning("No usable pickle files in %s", path_env)
        return None, None, meta

    log.warning("MODEL_DIR not found: %s", path_env)
    return None, None, meta

# --- MARKET DATA ------------------------------------------------------
def _fetch_bars(sym: str, interval: str = "5m", period: str = "2d") -> pd.DataFrame:
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
        # Try both MultiIndex orientations
        def _extract(mi: pd.MultiIndex, want: List[str]) -> Optional[pd.DataFrame]:
            lvl0 = [str(x).lower() for x in mi.get_level_values(0)]
            lvl1 = [str(x).lower() for x in mi.get_level_values(1)]
            if set(want).issubset(set(lvl0)):
                sub = df.loc[:, mi.get_level_values(0).str.lower().isin(want)]
                sub.columns = [str(c[0]).lower() for c in sub.columns]
                return sub[want].copy()
            if set(want).issubset(set(lvl1)):
                sub = df.loc[:, mi.get_level_values(1).str.lower().isin(want)]
                sub.columns = [str(c[1]).lower() for c in sub.columns]
                return sub[want].copy()
            # Try xs by ticker in either level
            for level in (0, 1):
                try:
                    tmp = df.xs(sym, level=level, axis=1)
                    tmp.columns = [str(c).lower() for c in tmp.columns]
                    if set(want).issubset(tmp.columns):
                        return tmp[want].copy()
                except Exception:
                    pass
            return None

        out = _extract(df.columns, need)
        if out is None:
            raise ValueError(f"bars missing required columns for {sym}: {list(map(str, df.columns))}")
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

# --- INDICATORS -------------------------------------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = _ema(df["close"], 12)
    df["ema_slow"] = _ema(df["close"], 26)
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

    # RSI(14)
    delta = df["close"].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(14, min_periods=14).mean()
    roll_down = down.rolling(14, min_periods=14).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR(14)
    df["atr"] = _atr(df, 14)

    return df.dropna()

# --- SIGNALS ----------------------------------------------------------
def _rule_score(df: pd.DataFrame) -> float:
    """
    Continuous rule score in [-1, 1]:
      - RSI channel pressure (distance from 50, clipped)
      - EMA slope/position
    """
    last = df.iloc[-1]
    rsi = float(last["rsi"])
    ema_diff = float(last["ema_diff"])
    ema_slope = float(df["ema_fast"].diff().iloc[-1])

    # Normalize components
    rsi_term = (rsi - 50.0) / 50.0          # ~[-1,1]
    ema_pos = np.tanh(ema_diff / max(1e-6, last["close"] * 0.002))   # ~[-1,1]
    ema_slp = np.tanh(ema_slope / max(1e-6, last["close"] * 0.001))  # ~[-1,1]

    score = 0.5 * rsi_term + 0.3 * ema_pos + 0.2 * ema_slp
    return float(np.clip(score, -1.0, 1.0))

def _safe_predict_score(model, X: np.ndarray) -> float:
    # Try predict_proba
    try:
        probs = model.predict_proba(X)
        p_buy = float(probs[-1][-1])
        return float(2 * p_buy - 1)  # map [0,1] -> [-1,1]
    except Exception:
        pass
    # Try decision_function
    try:
        dec = model.decision_function(X)
        return float(np.tanh(float(dec[-1])))
    except Exception:
        pass
    # Try predict -> {-1,1}
    try:
        y = model.predict(X)
        return 1.0 if int(y[-1]) == 1 else -1.0
    except Exception:
        return 0.0

def _nn_score(df: pd.DataFrame, model, scaler, feat_cols: List[str]) -> float:
    if model is None or df.empty:
        return 0.0
    if not all(c in df.columns for c in feat_cols):
        missing = [c for c in feat_cols if c not in df.columns]
        log.warning("NN missing features %s; returning 0.0", missing)
        return 0.0
    X = df[feat_cols].tail(LOOKBACK_BARS).values
    if X.shape[0] < 10:
        return 0.0
    # Apply scaler if provided
    if scaler is not None and _is_scaler(scaler):
        try:
            X = scaler.transform(X)
        except Exception as e:
            log.warning("scaler.transform failed: %s (proceeding unscaled)", e)
    else:
        # OPTIONAL: light standardize on the fly to stabilize models trained on standardized data
        try:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
        except Exception:
            pass
    return _safe_predict_score(model, X)

# --- DB WRITE ---------------------------------------------------------
def _write_signal(symbol: str, side: str, strength: float, px: float):
    ts = datetime.now(timezone.utc)
    row = pd.DataFrame([{
        "created_at": ts,
        "symbol": symbol,
        "side": side,
        "strength": float(strength),
        "px": float(px),
    }])
    row.to_sql("signals", ENGINE, if_exists="append", index=False)

# --- MAIN -------------------------------------------------------------
def main():
    log.info("make_signals_ensemble | symbols=%s", ",".join(SYMBOLS))

    model, scaler, meta = _resolve_model_artifacts(MODEL_DIR)
    if model is None and scaler is None:
        log.warning("No model loaded from %s", MODEL_DIR)
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
            bars = _fetch_bars(sym, interval="5m", period="2d")
            df = _calc_indicators(bars)

            # Component scores in [-1,1]
            rule = _rule_score(df)
            nn = _nn_score(df, model, scaler, FEAT_COLS)

            ensemble_raw = RULE_WEIGHT * rule + NN_WEIGHT * nn
            px = float(df["close"].iloc[-1])

            log.info(
                "%s components | rule=%.3f nn=%.3f -> ensemble=%.3f (px=%.2f)",
                sym, rule, nn, ensemble_raw, px
            )

            if ensemble_raw > MIN_STRENGTH:
                _write_signal(sym, "buy", min(1.0, abs(ensemble_raw)), px)
                inserted += 1
            elif ensemble_raw < -MIN_STRENGTH:
                _write_signal(sym, "sell", min(1.0, abs(ensemble_raw)), px)
                inserted += 1
            else:
                log.info("%s neutral (%.3f)", sym, ensemble_raw)

        except Exception as e:
            log.warning("%s failed: %s", sym, e)

    log.info("✔ inserted %d signals", inserted)

if __name__ == "__main__":
    main()
