# tools/why_size.py
from __future__ import annotations
import os
import argparse
from services.bracket_helper import (
    _get_last_price, _account_equity, _risk_per_share, _compute_dynamic_qty, _strength_scale,
    RISK_PCT_PER_TRADE, MIN_QTY, MAX_QTY
)

def main():
    ap = argparse.ArgumentParser(description="Explain dynamic sizing math for each symbol.")
    ap.add_argument("--strength", type=float, default=None, help="Optional 0..1 strength to weight size")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
    equity = _account_equity()
    per_trade_risk = max(equity * RISK_PCT_PER_TRADE, 1.0)
    mult = _strength_scale(args.strength)

    print(f"equity={equity:.2f}  RISK_PCT_PER_TRADE={RISK_PCT_PER_TRADE:.4f} -> per_trade_risk={per_trade_risk:.2f}")
    if args.strength is not None:
        print(f"strength={args.strength:.2f} -> strength_mult={mult:.3f}")
    print(f"MIN_QTY={MIN_QTY}  MAX_QTY={MAX_QTY}")
    print("")

    for sym in symbols:
        lp = _get_last_price(sym)
        rps_buy = _risk_per_share(sym, "buy", lp)
        rps_sell = _risk_per_share(sym, "sell", lp)

        raw_buy = (per_trade_risk / rps_buy) * (mult if args.strength is not None else 1.0)
        raw_sell = (per_trade_risk / rps_sell) * (mult if args.strength is not None else 1.0)

        qty_buy = _compute_dynamic_qty(sym, "buy", lp, strength=args.strength)
        qty_sell = _compute_dynamic_qty(sym, "sell", lp, strength=args.strength)

        print(f"{sym}: last={lp:.2f}")
        print(f"  BUY : rps={rps_buy:.4f}  raw_qty={raw_buy:.2f}  -> clamped={qty_buy}")
        print(f"  SELL: rps={rps_sell:.4f}  raw_qty={raw_sell:.2f} -> clamped={qty_sell}")
        print("")

if __name__ == "__main__":
    main()
