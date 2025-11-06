import os, sys, uuid
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"
SYMBOL = os.getenv("NUDGE_SYMBOL", "AAPL")

def qt(x, p=2): return round(float(x)+1e-9, p)

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # find our synthetic TP limit (client_order_id endswith -tp)
    od = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[SYMBOL]))
    tps = [o for o in od if (o.client_order_id or "").endswith("-tp")]
    if not tps:
        print("no TP leg found"); sys.exit(0)
    tp = tps[0]

    clk = tc.get_clock()
    is_open = bool(getattr(clk, "is_open", False))

    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=SYMBOL))[SYMBOL]
    bid = float(q.bid_price) if q.bid_price is not None else None
    ask = float(q.ask_price) if q.ask_price is not None else None
    if bid is None and ask is None:
        print("no quotes available"); sys.exit(0)

    # Target a price that should execute:
    # - During RTH: set at current bid (sell TP will hit)
    # - After-hours: set a tiny tick under bid and make it extended-hours (simple limit supports AH)
    target = bid if bid is not None else ask
    if not is_open:
        target = (bid if bid is not None else ask)  # fall back to ask if bid missing
        target = target - 0.01  # one cent under bid to encourage immediate match AH
    new_px = qt(target, 2)

    # Alpaca requires a NEW client_order_id on replace
    new_coid = f"{(tp.client_order_id or 'tp')}-n{uuid.uuid4().hex[:4]}"

    # Build replace request. Most fields are immutable; for simple LIMIT you can tweak price and client_order_id.
    # Some SDK versions ignore extended_hours on replace; if so, we leave it as-is and let it fill at next open.
    req = ReplaceOrderRequest(limit_price=new_px, client_order_id=new_coid)

    print(f"nudging {SYMBOL} TP {tp.limit_price} -> {new_px} (RTH={is_open}) coid={new_coid}")
    tc.replace_order_by_id(tp.id, order_data=req)
    print("done")

if __name__ == "__main__":
    main()
