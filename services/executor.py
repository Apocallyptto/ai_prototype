import os, math, pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

def eng():
    url = (f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
           f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT','5432')}/{os.environ['DB_NAME']}"
           "?sslmode=require&channel_binding=require")
    return create_engine(url, pool_pre_ping=True)

def fetch_new_signals(e):
    q = text("""
      SELECT s.id, sy.ticker, s.ts, s.signal
      FROM signals s
      JOIN symbols sy ON sy.id = s.symbol_id
      WHERE s.ts >= now() - interval '2 days'
      ORDER BY s.ts DESC
    """)
    with e.connect() as c:
        return pd.read_sql(q, c)

def decide_qty(equity, price, risk_frac=0.05):
    notional = equity * risk_frac
    return max(0, math.floor(notional / max(price, 1e-6)))

def last_equity(e):
    q = text("""SELECT (realized+unrealized-fees) AS eq
                FROM daily_pnl WHERE portfolio_id=1
                ORDER BY date DESC LIMIT 1""")
    with e.connect() as c:
        row = c.execute(q).fetchone()
    return float(row[0]) if row else 100_000.0

def place_order(e, ticker, side, qty, price, meta):
    ins = text("""
      INSERT INTO orders(ts, symbol_id, side, qty, type, limit_price, status, meta)
      SELECT now(), id, :side, :qty, 'market', NULL, 'new', :meta::jsonb
      FROM symbols WHERE ticker=:t
      RETURNING id
    """)
    with e.begin() as c:
        oid = c.execute(ins, {"t": ticker, "side": side, "qty": qty, "meta": meta}).scalar()
    return oid

def run():
    e = eng()
    eq = last_equity(e)
    sigs = fetch_new_signals(e)
    if sigs.empty:
        return
    # very naive: buy when signal["action"] == "buy", sell when "sell"
    for _, r in sigs.iterrows():
        s = r["signal"]
        # assume your signal json looks like {"action":"buy","price":123.45}
        action = s.get("action")
        px = float(s.get("price", 0.0))
        if action not in ("buy", "sell") or px <= 0:
            continue
        qty = decide_qty(eq, px, risk_frac=0.05)
        if qty == 0: 
            continue
        place_order(e, r["ticker"], action, qty, px, meta=s)

if __name__ == "__main__":
    run()
