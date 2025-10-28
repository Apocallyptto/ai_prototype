# services/trailing_stop.py
"""
Trailing Profit Protector
--------------------------
Monitors open positions and automatically raises stop-loss
levels as profit increases. Works with Alpaca bracket legs.

Usage (manual test):
    python -m services.trailing_stop
"""

import os, time, requests, math
from datetime import datetime, timezone
from services.bracket_helper import _alpaca_headers, _atr, ALPACA_BASE_URL

INTERVAL_SECONDS = int(os.getenv("TRAIL_INTERVAL", "120"))  # every 2 min
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))

def _get_positions():
    url = f"{ALPACA_BASE_URL}/v2/positions"
    r = requests.get(url, headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def _update_order_stop(order_id: str, new_stop: float):
    url = f"{ALPACA_BASE_URL}/v2/orders/{order_id}"
    payload = {"stop_price": f"{new_stop:.2f}"}
    r = requests.patch(url, headers=_alpaca_headers(), json=payload, timeout=10)
    if r.ok:
        print(f"[{datetime.now(timezone.utc).isoformat()}] ✅ stop updated -> {new_stop:.2f}")
    else:
        print(f"[WARN] failed to update stop: {r.status_code} {r.text}")

def _check_leg_for_symbol(symbol: str) -> str | None:
    """Find the stop leg ID for symbol’s active bracket order."""
    url = f"{ALPACA_BASE_URL}/v2/orders"
    r = requests.get(url, headers=_alpaca_headers(), timeout=10, params={"status": "open"})
    r.raise_for_status()
    for o in r.json():
        if o.get("symbol") == symbol and o.get("order_class") == "bracket":
            for leg in o.get("legs", []) or []:
                if leg.get("order_type") == "stop":
                    return leg["id"]
    return None

def trailing_logic():
    print(f"[{datetime.now(timezone.utc).isoformat()}] 🔍 checking positions...")
    for pos in _get_positions():
        symbol = pos["symbol"]
        side = pos["side"]
        qty = float(pos["qty"])
        entry = float(pos["avg_entry_price"])
        current = float(pos["current_price"])
        atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK_DAYS)

        gain = (current - entry) if side == "long" else (entry - current)
        if gain <= 0:
            continue

        if gain > atr and gain < 2 * atr:
            new_sl = entry
        elif gain >= 2 * atr:
            new_sl = entry + atr if side == "long" else entry - atr
        else:
            continue

        stop_id = _check_leg_for_symbol(symbol)
        if stop_id:
            _update_order_stop(stop_id, new_sl)
        else:
            print(f"[WARN] no stop leg found for {symbol}")

def main():
    while True:
        try:
            trailing_logic()
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
