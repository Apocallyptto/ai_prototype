# tools/show_size_verbose.py
import os
from datetime import datetime, timezone

# Import ONLY from the stable public shim:
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
    strength     = _f("STRENGTH", "0.65")
    atr_period   = int(_f("ATR_PERIOD", "14"))
    atr_lookback = int(_f("ATR_LOOKBACK_DAYS", "30"))
    equity       = _f("EQUITY", "100000")
    risk_pct     = _f("RISK_PCT_PER_TRADE", "0.0025")

    print(f"Verbose sizing diagnostic (strength={strength:.2f})\n")

    for sym in symbols:
        # last price
        lp = get_last_price(sym)

        # ATR and risk/share (uses your existing ATR exits logic under the hood)
        atr = atr_value(sym, atr_period, atr_lookback)
        rps = risk_per_share(sym, "buy", lp)

        # dynamic quantity (respects USE_DYNAMIC_SIZE, MIN_QTY, MAX_QTY,
        # size-by-strength if your helper supports it, etc.)
        qty = compute_dynamic_qty(sym, "buy", lp)

        # Pretty print
        print(f"=== {sym} ===")
        print(f"last={lp:.2f}  ATR={atr:.4f}  risk/share={rps:.4f}")
        print(
            "equity={:.2f}  risk%={:.2f}%  risk_amount={:.2f}".format(
                equity, risk_pct * 100.0, equity * risk_pct
            )
        )
        print(f"base_qty={qty}  (min={os.getenv('MIN_QTY')}  max={os.getenv('MAX_QTY')})")
        print(f"time={datetime.now(timezone.utc).isoformat()}")
        print("-")

if __name__ == "__main__":
    main()
