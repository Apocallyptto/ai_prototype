# tools/show_size_verbose.py
import os
import math
from datetime import datetime, timezone

from services.bracket_public import (
    get_last_price,
    atr_value,
    risk_per_share,
    compute_dynamic_qty,
)

def _f(env_name: str, default: str) -> float:
    try:
        return float(os.getenv(env_name, default))
    except Exception:
        return float(default)

def main():
    symbols      = [s.strip() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
    strength     = max(0.0, min(1.0, _f("STRENGTH", "0.65")))
    atr_period   = int(_f("ATR_PERIOD", "14"))
    atr_lookback = int(_f("ATR_LOOKBACK_DAYS", "30"))
    equity       = _f("EQUITY", "100000")
    risk_pct     = _f("RISK_PCT_PER_TRADE", "0.0025")
    min_qty      = int(_f('MIN_QTY', '1'))
    max_qty      = int(_f('MAX_QTY', '10'))

    print(f"Verbose sizing diagnostic (strength={strength:.2f})\n")

    for sym in symbols:
        lp  = get_last_price(sym)
        atr = atr_value(sym, atr_period, atr_lookback)
        rps = risk_per_share(sym, "buy", lp)

        risk_budget = equity * risk_pct
        raw_qty = 0 if rps <= 0 else math.floor(risk_budget / rps)
        scaled_qty = int(round(raw_qty * strength))
        clamped_qty = max(min_qty, min(max_qty, max(0, scaled_qty)))

        # What the executor uses (via helper). Good to verify they match:
        helper_qty = compute_dynamic_qty(sym, "buy", lp)

        print(f"=== {sym} ===")
        print(f"last={lp:.2f}  ATR={atr:.4f}  risk/share={rps:.4f}")
        print(
            "equity={:.2f}  risk%={:.2f}%  risk_amount={:.2f}".format(
                equity, risk_pct * 100.0, risk_budget
            )
        )
        print(f"raw_qty={raw_qty}  × strength={strength:.2f} → scaled={scaled_qty}  → clamp[{min_qty},{max_qty}] → final={clamped_qty}")
        print(f"executor_helper_qty={helper_qty}  (should match final above if logic is aligned)")
        print(f"time={datetime.now(timezone.utc).isoformat()}")
        print("-")

if __name__ == "__main__":
    main()
