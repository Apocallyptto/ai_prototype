from lib.db import make_engine
import pandas as pd
import sqlalchemy as sa

# === Edit this if your timestamp differs ===
TS = "2025-09-28 21:29:42.999221+00:00"
TICKERS = ("AAPL","SPY")

eng = make_engine()

with eng.begin() as c:
    print("Preview (will delete these rows):")
    q_preview = sa.text("""
        select id, ts, ticker, side, qty, order_type, limit_price, status
          from orders
         where ts = :ts and ticker = any(:tks)
         order by id
    """)
    df = pd.read_sql(q_preview, c, params={"ts": TS, "tks": list(TICKERS)})
    print(df.to_string(index=False) if not df.empty else "(no rows matched)")

    # delete the rows
    del_stmt = sa.text("""
        delete from orders
         where ts = :ts and ticker = any(:tks)
    """)
    result = c.execute(del_stmt, {"ts": TS, "tks": list(TICKERS)})
    print(f"\nDeleted {result.rowcount or 0} row(s).")

# show the latest rows for sanity
with eng.connect() as c:
    q_check = """
        select ts, ticker, side, qty, order_type, limit_price, status
          from orders
         order by ts desc
         limit 20
    """
    print("\nAfter delete:")
    print(pd.read_sql(q_check, c).to_string(index=False))