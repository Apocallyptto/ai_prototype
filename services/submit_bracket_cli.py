import argparse
import os
from alpaca.trading.client import TradingClient
from services.bracket_helper import submit_bracket

def _client() -> TradingClient:
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=True,
    )

def main():
    p = argparse.ArgumentParser("submit_bracket_cli")
    p.add_argument("--symbol", required=True)
    p.add_argument("--side", choices=["buy", "sell"], required=True)
    p.add_argument("--qty", type=float, default=None)
    p.add_argument("--notional", type=float, default=None)
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--atr", type=float, default=None)
    p.add_argument("--extended-hours", action="store_true")
    args = p.parse_args()

    cid = submit_bracket(
        _client(),
        args.symbol,
        args.side,
        limit_price=args.limit,
        qty=args.qty,
        notional=args.notional,
        atr=args.atr,
        extended_hours=args.extended_hours,
    )
    print("submitted order_id:", cid)

if __name__ == "__main__":
    main()
