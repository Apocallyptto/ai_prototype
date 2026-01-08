import os
import time
import requests

def _base_url():
    return (os.getenv("ALPACA_TRADING_URL") or os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")

def _headers():
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
    }

def alpaca_clock(timeout=10):
    base = _base_url()
    r = requests.get(f"{base}/v2/clock", headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()

def should_trade_now(stop_new_entries_minutes_before_close=0):
    """
    Returns (ok: bool, reason: str, clock: dict)
    - ok=False if market closed
    - ok=False if within N minutes before close (optional)
    """
    clock = alpaca_clock()
    if not clock.get("is_open"):
        return False, "market_closed", clock

    mins = int(stop_new_entries_minutes_before_close or 0)
    if mins > 0:
        # parse timestamps (ISO string) -> epoch seconds using time module only (simple approach)
        # We'll approximate by string slicing if needed, but easiest is: rely on Alpaca returning RFC3339 with offset.
        # We'll just skip close-window check if parsing fails.
        try:
            # Use python's datetime with fromisoformat
            from datetime import datetime
            ts = datetime.fromisoformat(clock["timestamp"])
            nc = datetime.fromisoformat(clock["next_close"])
            seconds_left = (nc - ts).total_seconds()
            if seconds_left <= mins * 60:
                return False, f"near_close_{mins}m", clock
        except Exception:
            pass

    return True, "ok", clock
