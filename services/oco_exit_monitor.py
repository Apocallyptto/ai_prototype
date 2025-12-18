import os, time
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest

def place_oco_exit(trading_client, symbol: str, qty: float, tp_price: float, sl_stop: float):
    prefix = os.getenv("EXIT_ORDER_PREFIX", "EXIT-OCO-")
    tif = os.getenv("OCO_TIF", "gtc").lower()
    tif_enum = TimeInForce.GTC if tif == "gtc" else TimeInForce.DAY

    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    close_qty = abs(float(qty))

    client_id = f"{prefix}{symbol}-{int(time.time())}"

    oco = LimitOrderRequest(
        symbol=symbol,
        qty=close_qty,
        side=side,
        time_in_force=tif_enum,
        # v OCO je "parent" take-profit limit; alpaca-py LimitOrderRequest zvyčajne vyžaduje limit_price,
        # preto ho nastavíme rovnako ako TP.
        limit_price=float(tp_price),
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=float(tp_price)),
        # stop_loss bez limit_price = stop (market) po triggri (robustnejšie než stop-limit)
        stop_loss=StopLossRequest(stop_price=float(sl_stop)),
        client_order_id=client_id,
    )

    return trading_client.submit_order(oco)
