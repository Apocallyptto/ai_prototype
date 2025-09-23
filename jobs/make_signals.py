# jobs/make_signals.py
import os
import pandas as pd
import numpy as np
import sqlalchemy as sa
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from lib.db import make_engine  # you already have DB helpers

TICKERS = os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY").split(",")
TIMEFRAME = "1d"
MODEL_NAME = "lr_v1"
PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))

def _get_bars(ticker: str) -> pd.DataFrame:
    # use yfinance for demo; you can swap in Alpaca data
    import yfinance as yf
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
    df = df.rename(columns={"Open":"o","High":"h","Low":"l","Close":"c","Volume":"v"})
    df.index.name = "ts"
    df = df.reset_index()
    df["ticker"] = ticker
    return df

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = SMAIndicator(df["c"], window=10).sma_indicator()
    df["sma_slow"] = SMAIndicator(df["c"], window=30).sma_indicator()
    df["rsi"] = RSIIndicator(df["c"], window=14).rsi()
    macd = MACD(df["c"])
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["ret_1d"] = df["c"].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df

def _train_quick_model(df: pd.DataFrame):
    # label: next-day up/down
    y = (df["c"].shift(-1) > df["c"]).astype(int)[:-1]
    X = df[["sma_fast","sma_slow","rsi","macd","macd_sig","ret_1d"]][:-1]
    if len(X) < 50:
        return None
    m = LogisticRegression(max_iter=1000)
    m.fit(X, y)
    return m

def _predict_latest(m, df: pd.DataFrame):
    row = df.iloc[[-1]][["sma_fast","sma_slow","rsi","macd","macd_sig","ret_1d"]]
    proba_up = float(m.predict_proba(row)[0,1])
    side = "buy" if proba_up >= 0.55 else ("sell" if proba_up <= 0.45 else "hold")
    strength = abs(proba_up - 0.5) * 2  # 0..1
    return side, strength, proba_up

def run():
    eng = make_engine()
    rows = []
    now = datetime.now(timezone.utc)
    for t in TICKERS:
        bars = _get_bars(t)
        feat = _build_features(bars)
        model = _train_quick_model(feat)
        if model is None:
            continue
        side, strength, _ = _predict_latest(model, feat)
        if side == "hold":
            continue
        rows.append({
            "ts": now,
            "ticker": t,
            "timeframe": TIMEFRAME,
            "model": MODEL_NAME,
            "side": side,
            "strength": round(float(strength), 4),
        })
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    with eng.begin() as c:
        df.to_sql("signals", c, if_exists="append", index=False)
    return len(rows)

if __name__ == "__main__":
    n = run()
    print(f"wrote {n} signals")
