# services/submit_bracket_cli.py
from __future__ import annotations
import argparse

from services.bracket_helper import submit_bracket

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["buy","sell"])
    ap.add_argument("--qty", type=int, default=None, help="If omitted, dynamic sizing is used.")
    ap.add_argument("--strength", type=float, default=None, help="Optional signal strength (enables size-by-strength).")
    args = ap.parse_args()

    try:
        resp = submit_bracket(args.symbol, args.side, qty=args.qty, strength=args.strength)
        print(f"OK placed {args.symbol} {args.side} qty={'(dynamic)' if args.qty is None else args.qty} id={resp.get('id')}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
