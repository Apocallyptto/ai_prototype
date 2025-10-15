# jobs/make_signals_ml.py
import os, joblib, datetime as dt
import numpy as np
import pandas as pd
import sqlalchemy as sa

from ml.features import make_features
from jobs.download_bars import fetch_5m   # reuse the downloader
from lib.db import make_engine            # you already have this

def decide(prob: float, buy_thr=0.55, sell_thr=0.45):
    if prob >= buy_thr:  return "buy",  float(prob)
    if prob <= sell_thr: return "sell", float(1.0 - prob)
    return None, None

def get_signal_columns(conn) -> set:
    rows = conn.execute(sa.text("""
        select column_name
        from information_schema.columns
        where table_name='signals'
    """)).fetchall()
    return {r[0] for r in rows}

def main():
    model_path = os.environ.get("ML_MODEL_PATH","models/gbc_5m.pkl")
    obj = joblib.load(model_path)
    clf, feat_cols = obj["model"], obj["features"]

    symbols   = [s.strip() for s in os.environ.get("ML_SYMBOLS","AAPL,MSFT,SPY").split(",")]
    port_id   = int(os.environ.get("PORTFOLIO_ID","1"))
    qty_def   = int(os.environ.get("ML_QTY","1"))
    buy_thr   = float(os.environ.get("ML_BUY_THR","0.55"))
    sell_thr  = float(os.environ.get("ML_SELL_THR","0.45"))
    src_value = os.environ.get("ML_SOURCE","ml")

    eng = make_engine()
    with eng.begin() as c:
        cols = get_signal_columns(c)

        # build the dynamic INSERT (only include columns your table has)
        allowed = ["ts","ticker","side","strength","qty","portfolio_id","source"]
        use_cols = [col for col in allowed if col in cols]
        placeholders = ",".join([f":{col}" for col in use_cols])
        collist = ",".join(use_cols)

        if not use_cols:
            raise RuntimeError("signals table has no expected columns; need at least: ts,ticker,side,strength,qty")

        stmt = sa.text(f"insert into signals ({collist}) values ({placeholders})")

        inserted = 0
        for sym in symbols:
            df = fetch_5m(sym, days=5)
            if df.empty or len(df) < 40:
                continue
            f = make_features(df)
            x = f.iloc[-1:]
            prob = float(clf.predict_proba(x[feat_cols])[:,1][0])
            side, strength = decide(prob, buy_thr, sell_thr)
            if not side:
                continue

            payload = {
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "ticker": sym,
                "side": side,
                "strength": strength,
                "qty": qty_def,
                "portfolio_id": port_id,
                "source": src_value,
            }
            # trim to only the columns that exist
            payload = {k: v for k, v in payload.items() if k in cols}
            c.execute(stmt, payload)
            inserted += 1

    print(f"inserted {inserted} ML signals")

if __name__ == "__main__":
    main()
