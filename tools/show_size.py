# tools/show_size.py
from __future__ import annotations
import os
from services.bracket_helper import get_last_price, _compute_dynamic_qty

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

def main():
    for sym in SYMBOLS:
        try:
            lp = get_last_price(sym)
            buy_q  = _compute_dynamic_qty(sym, "buy", lp)
            sell_q = _compute_dynamic_qty(sym, "sell", lp)
            print(f"{sym}: last={lp:.2f}  qty(buy)={buy_q}  qty(sell)={sell_q}")
        except Exception as e:
            print(f"{sym}: ERROR {e}")

if __name__ == "__main__":
    main()
