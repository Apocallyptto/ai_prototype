# services/submit_bracket_cli.py
from __future__ import annotations
import argparse, json, sys

from services.bracket_helper import submit_bracket  # alias submit_bracket_entry also available

def main():
    ap = argparse.ArgumentParser(description="Submit a single ATR-aware bracket order.")
    ap.add_argument("--symbol", required=True, help="Ticker, e.g., AAPL")
    ap.add_argument("--side", required=True, choices=["buy","sell"], help="Entry side")
    ap.add_argument("--qty", type=int, default=1, help="Share quantity (default: 1)")
    ap.add_argument("--prefer-limit-when-closed", action="store_true",
                    help="If market is closed, use limit entry near last trade (default True)")
    args = ap.parse_args()

    resp = submit_bracket(
        args.symbol.upper(),
        args.side.lower(),
        args.qty,
        prefer_limit_when_closed=True if args.prefer_limit_when_closed else True,
    )
    print(json.dumps(resp, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
