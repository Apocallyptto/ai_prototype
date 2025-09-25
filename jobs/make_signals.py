# jobs/make_signals.py
import os, random, pandas as pd, sqlalchemy as sa
from datetime import datetime, timedelta, timezone
from lib.db import make_engine

TICKERS = os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY").split(",")

def main():
    eng = make_engine()
    with eng.begin() as conn:
        # map ticker -> id
        m = pd.read_sql(sa.text("select id, ticker from symbols"), conn).set_index("ticker")["id"].to_dict()

        rows = []
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        for t in TICKERS:
            sid = m.get(t.strip().upper())
            if not sid:
                continue
            for d in range(12):
                ts = now - timedelta(days=11 - d)
                side = random.choice(["buy", "sell"])
                strength = round(random.uniform(0.2, 0.9), 2)
                rows.append(dict(
                    ts=ts, symbol_id=sid, timeframe="1d", model="baseline",
                    side=side, strength=strength,
                ))
        df = pd.DataFrame(rows)
        if not df.empty:
            # optional: clear recent demo rows
            conn.execute(sa.text("delete from signals where model='baseline'"))
            df.to_sql("signals", conn, if_exists="append", index=False)
            print(f"Inserted {len(df)} signals")

if __name__ == "__main__":
    main()
