import argparse
import os
from alpaca.trading.client import TradingClient

from bracket_helper import (
    submit_bracket,
    submit_simple_entry,
)

def main():
    ap = argparse.ArgumentParser(description="Submit a bracket (or simple) order with quote guard + penny rounding.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["buy", "sell"])
    ap.add_argument("--qty", required=True, type=float, help="Shares. If FRACTIONAL=1, decimals allowed.")
    ap.add_argument("--allow-after-hours", action="store_true", help="Only applies to simple orders. Brackets are RTH only.")
    ap.add_argument("--tp-mult", type=float, default=None, help="Override TP ATR multiplier.")
    ap.add_argument("--sl-mult", type=float, default=None, help="Override SL ATR multiplier.")
    ap.add_argument("--simple", action="store_true", help="If set, submit a simple LIMIT instead of bracket.")
    args = ap.parse_args()

    cli = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

    if args.simple:
        cid = submit_simple_entry(
            cli, args.symbol, args.side, args.qty,
            px_hint=None,
            allow_after_hours=args.allow_after_hours
        )
    else:
        # Brackets: extended hours are not supported by Alpaca
        cid = submit_bracket(
            cli, args.symbol, args.side, args.qty,
            px_hint=None,
            allow_after_hours=False,
            tp_mult=args.tp_mult,
            sl_mult=args.sl_mult,
        )

    print("submitted client_order_id:", cid)

if __name__ == "__main__":
    main()
