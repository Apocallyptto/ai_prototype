# services/submit_bracket_cli.py
from __future__ import annotations

import argparse
import json
from services.bracket_helper import submit_bracket_entry

def main():
    p = argparse.ArgumentParser(
        description="Submit an Alpaca bracket order (ATR-aware and constraint-safe)."
    )
    p.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    p.add_argument("--side", required=True, choices=["buy", "sell"], help="Order side")
    p.add_argument("--qty", type=int, default=1, help="Quantity (default 1)")
    p.add_argument("--parent-type", choices=["market", "limit"],
                   help="Parent type override. Defaults to PARENT_TYPE env or 'market'.")
    p.add_argument("--limit-price", type=float, help="Parent limit price (needed if parent-type=limit)")
    p.add_argument("--tp", type=float, dest="tp_price", help="Take profit limit price override")
    p.add_argument("--sl", type=float, dest="sl_price", help="Stop loss price override")
    p.add_argument("--client-id", help="Optional client_order_id")

    args = p.parse_args()

    res = submit_bracket_entry(
        symbol=args.symbol.upper(),
        side=args.side.lower(),
        qty=args.qty,
        tp_price=args.tp_price,
        sl_price=args.sl_price,
        limit_price=args.limit_price,
        client_id=args.client_id,
        parent_type=args.parent_type,
    )

    # Pretty print the server response so you see order/legs ids right away
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
