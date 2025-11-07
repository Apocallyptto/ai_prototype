import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderType

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"
SYMBOL = os.getenv("FLAT_SYMBOL", "AAPL")
AH_LIMIT_OFFSET = float(os.getenv("AH_LIMIT_OFFSET", "0.02"))

def _is_rth(tc: TradingClient) -> bool:
    try:
        clk = tc.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return True

def _qt(x, p=2): return round(float(x)+1e-9, p)

def cancel_all_sells(tc, sym):
    od = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[sym]))
    for o in od:
        try:
            if str(getattr(o, "side", "")).lower().endswith("sell"):
                tc.cancel_order_by_id(o.id)
                print(f"cancelled {o.client_order_id}")
        except Exception as e:
            print(f"cancel fail {o.client_order_id}: {e}")

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    pos = {p.symbol: p for p in tc.get_all_positions()}
    if SYMBOL not in pos:
        print("no position")
        return
    p = pos[SYMBOL]
    qty = abs(float(p.qty))
    side = OrderSide.SELL if float(p.qty) > 0 else OrderSide.BUY
    print(f"flatten {SYMBOL} qty={qty} via {'RTH MARKET' if _is_rth(tc) else 'AH LIMIT'} {side}")

    # Cancel existing sells to avoid wash trade
    cancel_all_sells(tc, SYMBOL)

    if _is_rth(tc):
        req = MarketOrderRequest(symbol=SYMBOL, side=side, type=OrderType.MARKET,
                                 time_in_force=TimeInForce.DAY, qty=qty)
    else:
        # simple LIMIT with extended_hours so it can execute AH
        ref = float(p.avg_entry_price)
        px = _qt(ref - AH_LIMIT_OFFSET if side == OrderSide.SELL else ref + AH_LIMIT_OFFSET, 2)
        req = LimitOrderRequest(symbol=SYMBOL, side=side, type=OrderType.LIMIT,
                                time_in_force=TimeInForce.DAY, qty=qty,
                                limit_price=px, extended_hours=True)
    tc.submit_order(req)
    print("submitted flatten")

if __name__ == "__main__":
    main()
