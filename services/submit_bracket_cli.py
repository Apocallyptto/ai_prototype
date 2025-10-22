# services/submit_bracket_cli.py
from __future__ import annotations
import os
import argparse
from services.bracket_helper import submit_bracket_entry

def main():
    ap = argparse.ArgumentParser(description="Submit a single bracket order (ATR-aware).")
    ap.add_argument("--symbol", required=True, type=str)
    ap.add_argument("--side", required=True, choices=["buy", "sell"])
    ap.add_argument("--qty", type=int, default=int(os.getenv("QTY_PER_TRADE", "1")))
    ap.add_argument("--limit", type=float, default=None, help="Parent limit if BRACKET_ENTRY_TYPE=limit")
    ap.add_argument("--tp", type=float, default=None, help="Optional TP (ignored when USE_ATR_ENTRY=1)")
    ap.add_argument("--sl", type=float, default=None, help="Optional SL (ignored when USE_ATR_ENTRY=1)")
    args = ap.parse_args()

    submit_bracket_entry(
        symbol=args.symbol.upper(),
        side=args.side,
        qty=args.qty,
        tp_price=args.tp,
        sl_price=args.sl,
        limit_price=args.limit,
        client_id=None,
    )

if __name__ == "__main__":
    main()
