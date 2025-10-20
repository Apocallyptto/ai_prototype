# services/submit_bracket_cli.py
from __future__ import annotations
import os, sys
from services.bracket_helper import submit_bracket_entry

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Submit ATR-aware bracket entry (market RTH, limit ETH).")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["buy","sell"])
    ap.add_argument("--qty", type=int, default=0, help="Quantity (0 = auto-size)")
    args = ap.parse_args(argv)

    qty = args.qty if args.qty > 0 else None
    submit_bracket_entry(args.symbol.upper(), args.side, qty)

if __name__ == "__main__":
    main(sys.argv[1:])
