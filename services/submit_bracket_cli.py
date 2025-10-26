# services/submit_bracket_cli.py
from __future__ import annotations
import argparse
import os
import sys
from services.bracket_helper import submit_bracket

def build_parser():
    p = argparse.ArgumentParser(
        description="Submit a single ATR-aware bracket order via Alpaca."
    )
    p.add_argument("--symbol", required=True, help="Ticker, e.g., MSFT")
    p.add_argument("--side", required=True, choices=["buy", "sell"], help="buy or sell")
    p.add_argument("--qty", type=int, default=None, help="Quantity (omit to use dynamic sizing if enabled)")
    p.add_argument("--time-in-force", default="day", choices=["day", "gtc"], help="Time in force (brackets are day-only on Alpaca RTH)")
    p.add_argument("--type", dest="order_type", default="market", choices=["market", "limit"], help="Entry order type")
    p.add_argument("--client-id", default=None, help="Optional client_order_id")
    return p

def main():
    args = build_parser().parse_args()
    try:
        resp = submit_bracket(
            symbol=args.symbol.upper(),
            side=args.side.lower(),
            qty=args.qty,
            time_in_force=args.time_in_force,
            order_type=args.order_type,
            client_id=args.client_id,
        )
        oid = resp.get("id") or resp.get("order", {}).get("id")
        print(f"OK placed {args.symbol} {args.side} qty={args.qty or '(dynamic)'} id={oid}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
