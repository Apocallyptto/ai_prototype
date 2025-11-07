import os
from alpaca.trading.client import TradingClient
from services.order_router import place_entry

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)

    # 1) Fractional test -> will route to simple (synthetic exits attach)
    o1 = place_entry(tc, symbol="AAPL", side="buy", qty=0.05, use_limit=False)
    print("fractional simple ->", o1.id)

    # 2) Whole-share test -> will route to native bracket (RTH only)
    # o2 = place_entry(tc, symbol="AAPL", side="buy", qty=1, use_limit=False)
    # print("whole-share bracket ->", o2.id)

if __name__ == "__main__":
    main()
