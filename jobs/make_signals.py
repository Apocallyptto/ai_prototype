# jobs/make_signals.py
from __future__ import annotations
import os, json
import sqlalchemy as sa
from datetime import datetime, timezone
from strategies.ma_cross import ma_cross_signal

DB_URL = (
    f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT','5432')}/{os.environ['DB_NAME']}"
    f"?sslmode=require&channel_binding=require"
)
ENGINE = sa.create_engine(DB_URL, pool_pre_ping=True)

MODEL = "ma_cross_v1"
TIMEFRAME = "1d"

def main():
    with ENGINE.begin() as conn:
        # get tickers
        syms = conn.execute(sa.text("select id, ticker from symbols order by ticker")).fetchall()
        if not syms:
            print("No symbols found. Insert into symbols table first.")
            return

        ins = sa.text("""
            insert into signals(symbol_id, ts, timeframe, model, signal)
            values (:sid, :ts, :tf, :model, :payload::jsonb)
        """)

        made = 0
        for sid, ticker in syms:
            sig = ma_cross_signal(ticker)
            if not sig:
                continue
            conn.execute(ins, {
                "sid": sid,
                "ts": datetime.now(timezone.utc),
                "tf": TIMEFRAME,
                "model": MODEL,
                "payload": json.dumps(sig),
            })
            made += 1
        print(f"Inserted {made} signals.")

if __name__ == "__main__":
    main()
