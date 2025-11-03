# jobs/make_signals_ensemble.py
import os
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
import joblib

# --- env / config ---
DB_URL = os.getenv("DB_URL", os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/trader"))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/gbc_5m.pkl")
DISABLE_NN = os.getenv("DISABLE_NN", "0") == "1"
LONG_ONLY = os.getenv("LONG_ONLY", "0") == "1"  # <- NEW: filter out shorts at the writer level

# rules weights / thresholds (can tune later)
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
RULE_W = float(os.getenv("RULE_WEIGHT", "0.6"))
NN_W = float(os.getenv("NN_WEIGHT", "0.4"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_ensemble")


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Classic RSI."""
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(method="bfill").fillna(50.0)


def _load_model():
    if DISABLE_NN:
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        log.info("Loaded model bundle: %s", MODEL_PATH)
        # Accept plain estimator or dict bundle like {"model": clf, "scaler": scaler}
        if hasattr(bundle, "predict_proba"):
            return {"model": bundle, "scaler": None}
        if isinstance(bundle, dict):
            return {"model": bundle.get("model"), "scaler": bundle.get("scaler")}
        return None
    except Exception as e:
        log.warning("NN unavailable (%s) -> proceeding rule-only", e)
        return None


def _download_5m(sym: str) -> pd.DataFrame:
    df = yf.download(sym, interval="5m", period="2d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"no bars for {sym}")
    df = df.rename(columns=str.lower)
    # For some yfinance versions, multiindex can appear — flatten if needed:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    # Ensure required cols
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise RuntimeError(f"bars missing '{col}' for {sym}")
    return df


def _rule_strength(sym: str, df: pd.DataFrame):
    c = df["close"]
    ema20 = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema50 = c.ewm(span=EMA_SLOW, adjust=False).mean()
    rsi14 = rsi(c, RSI_LEN)

    px = _safe_float(c.iloc[-1])
    ema20v = _safe_float(ema20.iloc[-1])
    ema50v = _safe_float(ema50.iloc[-1])
    rsi_v = _safe_float(rsi14.iloc[-1])

    # trend bias by ema20 vs ema50 and current px vs ema20
    up_trend = px > ema50v
    down_trend = px < ema50v
    dist_ema = px - ema20v

    # simple normalized score: rsi tilt + distance to ema20
    # map RSI 30..70 -> 0..1, clamp
    rsi_score = np.clip((rsi_v - 30.0) / 40.0, 0.0, 1.0)
    # distance scaled by recent ATR-ish proxy
    vol = (df["high"] - df["low"]).rolling(20).mean().iloc[-1]
    vol = max(_safe_float(vol, 1e-3), 1e-3)
    dist_score = np.clip((dist_ema / vol) * 0.2 + 0.5, 0.0, 1.0)

    buy_score = 0.0
    sell_score = 0.0
    if up_trend:
        buy_score = 0.6 * rsi_score + 0.4 * dist_score
        sell_score = 1.0 - buy_score * 0.7
    elif down_trend:
        sell_score = 0.6 * (1.0 - rsi_score) + 0.4 * (1.0 - dist_score)
        buy_score = 1.0 - sell_score * 0.7
    else:
        # neutral: let RSI dominate a bit
        buy_score = 0.55 * rsi_score + 0.45 * dist_score
        sell_score = 1.0 - buy_score

    # pick side by higher rule score
    if buy_score >= sell_score:
        return "buy", float(np.clip(buy_score, 0.0, 1.0)), px
    else:
        return "sell", float(np.clip(sell_score, 0.0, 1.0)), px


def _nn_prob_buy(sym: str, df: pd.DataFrame, model_bundle):
    """Very light feature stub; catches shape mismatch gracefully."""
    if not model_bundle:
        return None
    try:
        model = model_bundle["model"]
        scaler = model_bundle["scaler"]
        if not hasattr(model, "predict_proba"):
            return None

        c = df["close"]
        ret1 = c.pct_change().iloc[-10:].fillna(0.0).to_numpy()
        vol = (df["high"] - df["low"]).pct_change().iloc[-10:].fillna(0.0).to_numpy()
        feats = np.concatenate([ret1[-5:], vol[-5:]])  # 10 features
        X = feats.reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        proba = model.predict_proba(X)
        # assume proba[:,1] is "up"
        return float(np.clip(proba[0, 1], 0.0, 1.0))
    except Exception as e:
        log.info("%s NN skip: %s", sym, e)
        return None


def main():
    log.info("make_signals_ensemble | symbols=%s", ",".join(SYMBOLS))
    eng = create_engine(DB_URL, pool_pre_ping=True)
    model_bundle = _load_model()

    inserted = 0
    for sym in SYMBOLS:
        try:
            df = _download_5m(sym)
            side_r, s_rule, px = _rule_strength(sym, df)
            s_nn = _nn_prob_buy(sym, df, model_bundle)
            # Build an ensemble strength on the chosen side
            # If side is buy, NN contributes its buy-prob; if sell, 1 - buyProb
            if s_nn is None:
                s_ens_buy = s_rule if side_r == "buy" else 1.0 - s_rule
            else:
                s_ens_buy = (RULE_W * (s_rule if side_r == "buy" else 1.0 - s_rule)) + (NN_W * s_nn)

            # final side & strength
            if s_ens_buy >= 0.5:
                side = "buy"
                strength = float(np.clip(s_ens_buy, 0.0, 1.0))
            else:
                side = "sell"
                strength = float(np.clip(1.0 - s_ens_buy, 0.0, 1.0))

            # --- LONG_ONLY writer guard (NEW) ---
            if LONG_ONLY and side == "sell":
                log.info("LONG_ONLY=1 -> skip writing short for %s", sym)
                continue

            with eng.begin() as conn:
                conn.execute(
                    text("INSERT INTO signals (symbol, side, strength, px) VALUES (:sym, :side, :str, :px)"),
                    {"sym": sym, "side": side, "str": strength, "px": px},
                )
            inserted += 1
            log.info("Ensemble -> %s: %s (%.3f)", sym, side, strength)
        except Exception as e:
            log.warning("%s failed: %s", sym, e)

    log.info("✔ inserted %d signals", inserted)


if __name__ == "__main__":
    main()
