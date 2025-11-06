import os
import uuid
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import LimitOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

SYMBOL = os.getenv("TEST_SYMBOL", "AAPL")
QTY = float(os.getenv("TEST_QTY", "0.05"))
BIAS = float(os.getenv("TEST_LIMIT_BIAS", "-0.50"))  # BUY below ask so it stays unfilled for a while

def qt(x, p=2): return round(float(x)+1e-9, p)

def latest_mid():
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=SYMBOL))[SYMBOL]
    bid = float(q.bid_price) if q.bid_price is not None else None
    ask = float(q.ask_price) if q.ask_price is not None else None
    if bid is not None and ask is not None:
        return (bid+ask)/2.0, bid, ask
    return None, bid, ask

def main():
    cli = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    mid, bid, ask = latest_mid()
    if mid is None:
        raise SystemExit("no quote available right now")

    limit_px = qt((ask if ask is not None else mid) + BIAS, 2)
    coid = f"test-stale-{uuid.uuid4().hex[:6]}"
    req = LimitOrderRequest(
        symbol=SYMBOL,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        qty=QTY,
        limit_price=limit_px,
        extended_hours=True,
        client_order_id=coid,
    )
    o = cli.submit_order(req)
    print(f"submitted STALE-ish LIMIT -> id={o.id} symbol={o.symbol} qty={QTY} limit={limit_px} coid={coid}")

if __name__ == "__main__":
    main()
