import os
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "no", "NO")

def main():
    if len(sys.argv) < 2:
        print("usage: python -m tools.cancel_tp <SYMBOL>")
        sys.exit(1)
    sym = sys.argv[1].upper()

    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    paper = env_bool("ALPACA_PAPER", True)

    tc = TradingClient(key, sec, paper=paper)

    # Find all open SELL LIMIT orders that look like TP legs
    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        nested=True,
        symbols=[sym],
    )
    ods = tc.get_orders(filter=req)

    cancelled = 0
    for o in ods:
        if o.symbol != sym:
            continue
        if o.side != OrderSide.SELL:
            continue
        if o.type != OrderType.LIMIT:
            continue
        coid = (o.client_order_id or "")
        # Heuristic: treat any "-tp" suffix as TP, but also allow plain SELL LIMITs
        if coid.endswith("-tp") or True:
            try:
                tc.cancel_order_by_id(o.id)
                print(f"cancelled TP {o.id} coid={coid} px={o.limit_price} qty={o.qty}")
                cancelled += 1
            except Exception as e:
                print(f"cancel failed {o.id} coid={coid}: {e}")

    if cancelled == 0:
        print("no open TP orders found")
    else:
        print(f"done, cancelled {cancelled} TP(s)")

if __name__ == "__main__":
    main()
