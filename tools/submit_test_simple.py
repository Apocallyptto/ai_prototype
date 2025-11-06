import os, uuid
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

SYMBOL = os.getenv("TEST_SYMBOL", "AAPL")
QTY = float(os.getenv("TEST_QTY", "0.05"))
BIAS = float(os.getenv("TEST_LIMIT_BIAS", "0.10"))  # add to ask after-hours so it fills

def _qt(x, p=2): return round(float(x)+1e-9, p)

def _latest_bid_ask(symbol: str):
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
    bid = float(q.bid_price) if q.bid_price is not None else None
    ask = float(q.ask_price) if q.ask_price is not None else None
    return bid, ask

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    clk = tc.get_clock()
    is_open = bool(getattr(clk, "is_open", False))

    if is_open:
        # RTH: use MARKET, extended_hours must be False/omitted
        req = MarketOrderRequest(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            qty=QTY,
            client_order_id=f"test-simple-{uuid.uuid4().hex[:8]}",
        )
        o = tc.submit_order(req)
        print(f"RTH MARKET BUY -> id={o.id} symbol={o.symbol} qty={QTY}")
    else:
        # After-hours: MARKET not allowed; use DAY LIMIT with extended_hours=True
        bid, ask = _latest_bid_ask(SYMBOL)
        if ask is None and bid is None:
            raise SystemExit("No quotes available to price a limit after-hours.")
        px = _qt((ask if ask is not None else bid) + abs(BIAS), 2)
        req = LimitOrderRequest(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            qty=QTY,
            limit_price=px,
            extended_hours=True,
            client_order_id=f"test-ah-limit-{uuid.uuid4().hex[:8]}",
        )
        o = tc.submit_order(req)
        print(f"AH LIMIT BUY -> id={o.id} symbol={o.symbol} qty={QTY} limit={px} ext=True")

if __name__ == "__main__":
    main()
