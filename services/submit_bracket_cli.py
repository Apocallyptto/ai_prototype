import argparse
import os
from alpaca.trading.client import TradingClient
from services.bracket_helper import (
    submit_bracket,
    submit_simple_entry,
)

def main():
    p = argparse.ArgumentParser(description="Submit bracket (RTH) or simple entry (AH).")
    p.add_argument("--symbol", required=True)
    p.add_argument("--side", required=True, choices=["buy", "sell"])
    p.add_argument("--qty", required=True, type=float)
    p.add_argument("--tp-mult", type=float, default=float(os.getenv("TP_ATR_MULT", "1.5")))
    p.add_argument("--sl-mult", type=float, default=float(os.getenv("SL_ATR_MULT", "1.0")))
    p.add_argument("--after-hours", action="store_true",
                   help="If set, place simple LIMIT entry (no TP/SL) with extended-hours.")
    args = p.parse_args()

    # Forward custom multipliers via ENV (bracket_helper reads from ENV)
    os.environ["TP_ATR_MULT"] = str(args.tp_mult)
    os.environ["SL_ATR_MULT"] = str(args.sl_mult)

    cli = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

    if args.after_hours:
        cid = submit_simple_entry(cli, args.symbol, args.side, args.qty, px_hint=None, extended_hours=True)
        print("SUBMITTED SIMPLE ENTRY (AH) cid=", cid)
    else:
        cid = submit_bracket(cli, args.symbol, args.side, args.qty, px_hint=None, allow_after_hours=False)
        print("SUBMITTED BRACKET (RTH) cid=", cid)

if __name__ == "__main__":
    main()
