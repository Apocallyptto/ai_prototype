# tools/show_size.py
from __future__ import annotations
import os
import argparse
from services.bracket_helper import _compute_dynamic_qty, _get_last_price

def main():
    ap = argparse.ArgumentParser(description="Preview dynamic (and strength-weighted) sizing per symbol.")
    ap.add_argument("--strength", type=float, default=None, help="Optional signal strength (0..1) to scale size")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

    for sym in symbols:
        lp = _get_last_price(sym)
        buy_qty = _compute_dynamic_qty(sym, "buy", lp, strength=args.strength)
        sell_qty = _compute_dynamic_qty(sym, "sell", lp, strength=args.strength)
        print(f"{sym}: last={lp:.2f}  qty(buy)={buy_qty}  qty(sell)={sell_qty}")

if __name__ == "__main__":
    main()
