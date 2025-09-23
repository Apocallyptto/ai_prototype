# services/executor.py
import os
import math
import pandas as pd
import sqlalchemy as sa
from datetime import datetime, timezone, timedelta
from alpaca_trade_api import REST
from lib.db import make_engine

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% of equity
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.20"))        # 20% per name
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "-0.03"))  # -3%
PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.2"))

def _alpaca():
    return REST(
        key_id=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_API_SECRET"],
        base_url=os.environ.get("ALPACA_BASE_URL","https://paper-api.alpaca.markets"),
    )

def _equity(api: REST) -> float:
    a = api.get_account()
    return float(a.equity)

def _price(api: REST, ticker: str) -> float:
    q = api.get_latest_quote(ticker)
    px = float(q.ap) if q.ap and q.ap > 0 else float(q.bp)
    return px

def _get_open_positions(api: REST) -> dict:
    pos = {}
    for p in api.list_positions():
        pos[p.symbol] = float(p.qty) * (1 if p.side=="long" else -1)
    return pos

def _read_latest_signals(eng) -> pd.DataFrame:
    sql = """
      select distinct on (ticker) *
      from signals
      where ts >= now() - interval '2 days'
      order by ticker, ts desc
    """
    with eng.connect() as c:
        return pd.read_sql(sa.text(sql), c)

def _insert_order(c, row):
    df = pd.DataFrame([row])
    df.to_sql("orders", c, if_exists="append", index=False)

def run_once():
    eng = make_engine()
    api = _alpaca()

    sig = _read_latest_signals(eng)
    if sig.empty:
        print("no signals")
        return 0

    eq = _equity(api)
    positions = _get_open_positions(api)

    placed = 0
    with eng.begin() as c:
        for _, s in sig.iterrows():
            tkr = s["ticker"]
            side = s["side"]
            strength = float(s["strength"])
            if strength < MIN_STRENGTH:
                continue

            px = _price(api, tkr)
            # position sizing
            max_pos_notional = eq * MAX_POS_PCT
            risk_notional = eq * RISK_PER_TRADE * max(0.5, strength)  # scale by strength
            notional = min(max_pos_notional, risk_notional)
            qty = max(1, math.floor(notional / px))

            # simple direction rule: avoid flipping rapidly
            cur = positions.get(tkr, 0)
            if side == "buy" and cur > 0:
                continue
            if side == "sell" and cur < 0:
                continue

            # place market order (simplest first)
            try:
                api.submit_order(symbol=tkr,
                                 qty=qty,
                                 side=side,
                                 type="market",
                                 time_in_force="day")
                order_row = {
                    "ts": datetime.now(timezone.utc),
                    "ticker": tkr,
                    "side": side,
                    "qty": qty,
                    "order_type": "market",
                    "limit_price": None,
                    "status": "new",
                    "filled_at": None,
                }
                _insert_order(c, order_row)
                placed += 1
                print(f"placed {side} {qty} {tkr} @ ~{px}")
            except Exception as e:
                print("order error:", e)
                continue
    return placed

if __name__ == "__main__":
    n = run_once()
    print("orders placed:", n)
