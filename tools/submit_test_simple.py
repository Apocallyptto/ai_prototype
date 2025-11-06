import os
import uuid
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import MarketOrderRequest

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

SYMBOL = os.getenv("TEST_SYMBOL", "AAPL")
QTY = float(os.getenv("TEST_QTY", "0.05"))  # fractional position
EXT = os.getenv("TEST_EXT_HOURS", "1") == "1"  # allow after-hours

def main():
    cli = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    req = MarketOrderRequest(
        symbol=SYMBOL,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        qty=QTY,
        extended_hours=EXT,
        client_order_id=f"test-simple-{uuid.uuid4().hex[:8]}",
    )
    o = cli.submit_order(req)
    print(f"submitted simple BUY -> id={o.id} symbol={o.symbol} qty={QTY} ext={EXT}")

if __name__ == "__main__":
    main()
