import os
from alpaca.trading.client import TradingClient
from services.order_router import place_entry

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)

    # Fractional test -> AH-safe LIMIT using Alpaca quotes; RTH -> MARKET unless you set use_limit=True
    o1 = place_entry(tc, symbol="AAPL", side="buy", qty=0.05, use_limit=False)
    print("fractional entry ->", o1.id)

    # Whole-share test -> bracket in RTH, LIMIT AH simple if outside hours
    # o2 = place_entry(tc, symbol="AAPL", side="buy", qty=1, use_limit=False)
    # print("whole-share entry ->", o2.id)

if __name__ == "__main__":
    main()
