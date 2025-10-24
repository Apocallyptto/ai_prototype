# tools/show_size.py
from __future__ import annotations
import os, traceback
from services.bracket_helper import _compute_dynamic_qty, _get_last_price

SYMS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

def main():
    use_dyn = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
    if not use_dyn:
        print("USE_DYNAMIC_SIZE=0 (dynamic sizing off). Set it to 1 to enable.")
        return
    for sym in SYMS:
        try:
            lp = _get_last_price(sym)
            buy_qty  = _compute_dynamic_qty(sym, "buy", lp)
            sell_qty = _compute_dynamic_qty(sym, "sell", lp)
            print(f"{sym}: last={lp:.2f}  qty(buy)={buy_qty}  qty(sell)={sell_qty}")
        except Exception as e:
            print(f"{sym}: ERROR computing size -> {e}")
            # uncomment to debug:
            # traceback.print_exc()

if __name__ == "__main__":
    main()
