# jobs/make_signals_ml.py
import os, joblib, datetime as dt
import numpy as np
import pandas as pd
import sqlalchemy as sa

from ml.features import make_features
from jobs.download_bars import fetch_5m  # reuse the downloader
from lib.db import make_engine            # you already have this

def decide(prob: float, buy_thr=0.55, sell_thr=0.45):
    if prob >= buy_thr:  return "buy",  float(prob)         # strength ~ prob
    if prob <= sell_thr: return "sell", float(1.0 - prob)
    return None, None

def main():
    model_path = os.environ.get("ML_MODEL_PATH","models/gbc_5m.pkl")
    obj = joblib.load(model_path)
    clf, feats = obj["model"], obj["features"]

    symbols = [s.strip() for s in os.environ.get("ML_SYMBOLS","AAPL,MSFT,SPY").split(",")]
    port_id  = int(os.environ.get("PORTFOLIO_ID","1"))
    qty_default = int(os.environ.get("ML_QTY","1"))

    rows = []
    for sym in symbols:
        df = fetch_5m(sym, days=5)
        if df.empty or len(df) < 40:
            continue
        f = make_features(df)
        x = f.iloc[-1:]  # last bar
        X = x[feats]
        prob = float(clf.predict_proba(X)[:,1][0])
        side, strength = decide(prob)
        if side:
            rows.append({
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "portfolio_id": port_id,
                "ticker": sym,
                "side": side,
                "strength": strength,
                "qty": qty_default
            })

    if not rows:
        print("no signals")
        return

    eng = make_engine()
    with eng.begin() as c:
        for r in rows:
            c.execute(sa.text("""
                insert into signals (ts, portfolio_id, ticker, side, strength, qty)
                values (:ts, :portfolio_id, :ticker, :side, :strength, :qty)
            """), r)
    print(f"inserted {len(rows)} ML signals")

if __name__ == "__main__":
    main()
