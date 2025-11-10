import os
import sys
import uuid
from decimal import Decimal, ROUND_DOWN

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, AssetClass, OrderType, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "no", "NO")

def d2(x):  # price to 2 decimals
    return Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

def dq(x):  # qty to 3 decimals (fractional stocks)
    return Decimal(x).quantize(Decimal("0.001"), rounding=ROUND_DOWN)

def get_free_qty(tc: TradingClient, symbol: str) -> Decimal:
    """
    Approximate 'free' qty = position qty - sum(open SELL qty)
    Good enough for paper + our simple flow.
    """
    pos_qty = Decimal("0")
    try:
        p = tc.get_open_position(symbol)
        pos_qty = Decimal(str(p.qty))
    except Exception:
        pos_qty = Decimal("0")

    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        nested=True,
        symbols=[symbol],
    )
    open_orders = tc.get_orders(filter=req)

    sell_held = Decimal("0")
    for o in open_orders:
        if o.symbol == symbol and o.side == OrderSide.SELL:
            # alpaca-py exposes o.qty as str/Decimal-like
            sell_held += Decimal(str(o.qty))

    free = pos_qty - sell_held
    if free < 0:
        free = Decimal("0")
    return free

def main():
    if len(sys.argv) < 3:
        print("usage: python -m tools.place_tp <SYMBOL> <LIMIT_PRICE>")
        sys.exit(1)

    sym = sys.argv[1].upper()
    limit_px = d2(sys.argv[2])

    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    paper = env_bool("ALPACA_PAPER", True)

    tc = TradingClient(key, sec, paper=paper)

    free_qty = get_free_qty(tc, sym)
    if free_qty <= 0:
        print(f"no free qty available on {sym} (nothing to place)")
        return

    # Alpaca min notional is typically $1.00; ensure price*qty >= 1.00
    min_notional = Decimal("1.00")
    notional = limit_px * free_qty
    if notional < min_notional:
        # bump qty up to meet notional (capped by position)
        target_qty = (min_notional / limit_px).quantize(Decimal("0.001"), rounding=ROUND_DOWN)
        # Don't exceed free qty
        if target_qty > free_qty:
            target_qty = free_qty
        qty = dq(target_qty)
    else:
        qty = dq(free_qty)

    if qty <= 0:
        print(f"free qty too small to place TP at {limit_px}")
        return

    coid = f"manualtp-{sym}-{uuid.uuid4().hex[:6]}"

    # SELL LIMIT (simple), DAY, RTH only
    req = LimitOrderRequest(
        symbol=sym,
        side=OrderSide.SELL,
        qty=float(qty),
        limit_price=float(limit_px),
        time_in_force=TimeInForce.DAY,
        extended_hours=False,         # keep RTH-only so it won't violate stop eligibility rules
        order_class=OrderClass.SIMPLE # simple fractional limit
    )

    o = tc.submit_order(req)
    print(f"TP placed: id={o.id} coid={coid} sym={sym} qty={qty} @ {limit_px}")

if __name__ == "__main__":
    main()
