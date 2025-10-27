# tools/show_size.py
from __future__ import annotations
import argparse, os

from services.bracket_helper import _compute_dynamic_qty as dyn_qty
from services.bracket_helper import _last_quote as last_px

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", type=float, default=None, help="Optional signal strength to scale size-by-strength.")
    args = ap.parse_args()

    for sym in SYMBOLS:
        px = last_px(sym)
        qb = dyn_qty(sym, "buy", px, strength=args.strength)
        qs = dyn_qty(sym, "sell", px, strength=args.strength)
        print(f"{sym}: last={px:.2f}  qty(buy)={qb}  qty(sell)={qs}")

if __name__ == "__main__":
    main()
