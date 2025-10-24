# tools/show_size.py
import os
from services.bracket_helper import _compute_dynamic_qty, _latest_trade_price

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

def main():
    for sym in SYMBOLS:
        lp = _latest_trade_price(sym)
        buy_qty = _compute_dynamic_qty(sym, "buy", lp)
        sell_qty = _compute_dynamic_qty(sym, "sell", lp)
        print(f"{sym}: last={lp:.2f}  qty(buy)={buy_qty}  qty(sell)={sell_qty}")

if __name__ == "__main__":
    main()
