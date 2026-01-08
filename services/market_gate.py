import os
from datetime import datetime
from typing import Any, Dict, Tuple

import requests


def _alpaca_base_url() -> str:
    base = (
        os.getenv("ALPACA_TRADING_URL")
        or os.getenv("ALPACA_BASE_URL")
        or "https://paper-api.alpaca.markets"
    )
    return base.rstrip("/")


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
    }


def get_clock(timeout: float = 10.0) -> Dict[str, Any]:
    """Return Alpaca v2 clock JSON."""
    base = _alpaca_base_url()
    h = _alpaca_headers()

    r = requests.get(f"{base}/v2/clock", headers=h, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # Handles offsets like -05:00 and also Z
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def should_trade_now(stop_new_entries_min_before_close: int = 10) -> Tuple[bool, str, Dict[str, Any]]:
    """Gate for *NEW ENTRIES*.

    Returns (ok, reason, clock_json).
    - ok=False when market closed OR too close to next_close.
    """
    try:
        clock = get_clock()
    except Exception as e:
        # Safer to stop entries if we can't verify the clock
        return False, f"clock_error:{e.__class__.__name__}", {}

    is_open = bool(clock.get("is_open"))
    if not is_open:
        return False, "market_closed", clock

    ts = _parse_iso(clock.get("timestamp"))
    next_close = _parse_iso(clock.get("next_close"))

    if ts and next_close and stop_new_entries_min_before_close is not None:
        mins_left = (next_close - ts).total_seconds() / 60.0
        if mins_left <= float(stop_new_entries_min_before_close):
            return False, f"near_close_{mins_left:.1f}m", clock

    return True, "ok", clock
