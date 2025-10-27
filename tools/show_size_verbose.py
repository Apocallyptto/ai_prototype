# tools/show_size_verbose.py
import os
from datetime import datetime, timezone

# Import the *public* helpers (no leading underscores)
from services.bracket_helper import (
    compute_dynamic_qty,
    get_last_price,
    atr_value,
    risk_per_share,
)

def main():
    symbols = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
    strength = float(os.getenv("STRENGTH", "0.65"))
    print(f"Verbose sizing diagnostic (strength={strength})\n")

    atr_period = int(os.getenv("ATR_PERIOD", "14"))
    atr_lookback = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
    eq = float(os.getenv("EQUITY", "100000"))
    risk_pct = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))

    for sym in symbols:
        lp = get_last_price(sym)
        atr = atr_value(sym, atr_period, atr_lookback)
        rps = risk_per_share(sym, "buy", lp)
        qty = compute_dynamic_qty(sym, "buy", lp)
        risk_amt = eq * risk_pct

        print(f"=== {sym} ===")
        print(f"last={lp:.2f}  ATR={atr:.4f}  risk/share={rps:.4f}")
        print(f"equity={eq:.2f}  risk%={risk_pct*100:.2f}%  risk_amount={risk_amt:.2f}")
        print(f"base_qty={qty}  (min={os.getenv('MIN_QTY')} max={os.getenv('MAX_QTY')})")
        print(f"time={datetime.now(timezone.utc).isoformat()}")
        print("-")

if __name__ == "__main__":
    main()
