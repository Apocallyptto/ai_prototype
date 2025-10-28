# jobs/submit_bracket.py
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from services.executor_bracket import Signal, place_bracket_for_signal

# Simple logger setup
logger = logging.getLogger("submit_bracket")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description="Submit one bracket order with guards.")
    p.add_argument("--symbol", required=True, help="Ticker, e.g., MSFT")
    p.add_argument("--side", required=True, choices=["buy", "sell"])
    p.add_argument("--qty", type=int, help="Optional fixed quantity (else dynamic sizing)")
    p.add_argument("--price", type=float, help="Optional entry price (else last price)")
    p.add_argument("--strength", type=float, help="Optional model strength 0..1")
    p.add_argument("--source", default="manual_cli", help="Signal source tag")
    args = p.parse_args()

    sig = Signal(
        symbol=args.symbol,
        side=args.side,
        qty=args.qty,
        price=args.price,
        strength=args.strength,
        source=args.source,
    )

    logger.info(f"[{_utcnow()}] submitting {sig.symbol} {sig.side} qty={sig.qty or 'auto'}")
    resp = place_bracket_for_signal(sig, logger)
    if resp is None:
        logger.info(f"[{_utcnow()}] no order placed (skipped or blocked)")
    else:
        logger.info(f"[{_utcnow()}] done. broker_response={resp}")


if __name__ == "__main__":
    main()
