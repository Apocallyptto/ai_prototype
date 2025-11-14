import os
import time
import logging
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.requests import StockLatestQuoteRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trailing_guard")


def get_midquote(data_client, symbol: str):
    """Return (bid, ask, mid)."""
    try:
        latest = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )
        bid = float(latest[symbol].bid_price) if latest[symbol].bid_price else None
        ask = float(latest[symbol].ask_price) if latest[symbol].ask_price else None
        if bid is None or ask is None:
            return None, None, None
        return bid, ask, (bid + ask) / 2.0
    except Exception as e:
        log.error(f"{symbol}: latest quote fetch failed: {e}")
        return None, None, None


def main():
    # === Alpaca clients ===
    trading = TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_API_SECRET"),
        paper=os.getenv("ALPACA_PAPER", "1") != "0"
    )

    data_client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_API_SECRET")
    )

    symbols = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")

    while True:
        try:
            # ===============================================================
            # NEW SDK-COMPATIBLE WAY TO FETCH OPEN ORDERS
            # ===============================================================
            req = GetOrdersRequest(status="open")
            open_orders = trading.get_orders(filter=req)
            # ===============================================================

            # Convert open orders into dict by symbol
            open_by_symbol = {sym: [] for sym in symbols}
            for o in open_orders:
                if o.symbol in open_by_symbol:
                    open_by_symbol[o.symbol].append(o)

            changed = False

            # ===============================================================
            # MAIN TRAILING LOGIC
            # ===============================================================
            for sym in symbols:
                pos_orders = open_by_symbol.get(sym, [])
                if not pos_orders:
                    continue

                # Fetch midquote
                bid, ask, mid = get_midquote(data_client, sym)
                if mid is None:
                    continue

                for o in pos_orders:
                    # Only consider STOP orders
                    if o.order_type.value != "stop":
                        continue

                    stop_px = float(o.stop_price)
                    trail_gap = 0.30   # you can tune this

                    # If price moved up enough, move stop upward
                    new_stop = mid - trail_gap
                    new_stop = round(new_stop, 2)

                    if new_stop > stop_px:
                        # Replace the stop order
                        new_coid = f"{o.client_order_id}-tg"

                        log.info(f"{sym}: trail {stop_px} -> {new_stop}  (mid={mid})")

                        try:
                            trading.replace_order_by_id(
                                order_id=o.id,
                                stop_price=new_stop,
                                client_order_id=new_coid
                            )
                            changed = True
                        except Exception as e:
                            log.error(f"{sym}: trailing replace error: {e}")

            if not changed:
                log.info("trailing_guard | no changes")

        except Exception as e:
            log.error(f"loop error: {e}")

        time.sleep(3)


if __name__ == "__main__":
    main()
