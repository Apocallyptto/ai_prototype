# services/market_gate.py
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# jednoduchá cache, aby sme nebúchali /v2/clock pri každom loope
_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "clock": None,  # type: Optional[Dict[str, Any]]
}

DEFAULT_PAPER_URL = "https://paper-api.alpaca.markets"


def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        if v is None or str(v).strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse Alpaca ISO time to aware datetime (UTC)."""
    if not ts:
        return None
    s = ts.strip()
    # Alpaca často vracia 'Z'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _alpaca_base_url() -> str:
    base = (os.getenv("ALPACA_TRADING_URL") or os.getenv("ALPACA_BASE_URL") or DEFAULT_PAPER_URL).strip()
    return base.rstrip("/")


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
    }


def _make_session(retries: int) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.6,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _fetch_clock(session: requests.Session, timeout_s: float) -> Dict[str, Any]:
    base = _alpaca_base_url()
    url = f"{base}/v2/clock"

    r = session.get(url, headers=_alpaca_headers(), timeout=timeout_s)
    # keď Alpaca vráti ne-JSON alebo 5xx, Retry už skúsil; tu to len ošetri
    if r.status_code >= 400:
        # skúsiť vytiahnuť detail
        try:
            j = r.json()
        except Exception:
            j = {"text": r.text[:200]}
        raise RuntimeError(f"clock_http_{r.status_code}:{j}")

    j = r.json()
    # normalize
    clock = {
        "timestamp": j.get("timestamp"),
        "is_open": j.get("is_open"),
        "next_open": j.get("next_open"),
        "next_close": j.get("next_close"),
    }
    return clock


def should_trade_now(stop_new_entries_min_before_close: int = 10) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns:
      (ok_to_trade, reason, clock_dict)

    clock_dict keys:
      timestamp, is_open, next_open, next_close
    """
    cache_seconds = _env_int("ALPACA_CLOCK_CACHE_SECONDS", 20)
    timeout_s = _env_float("ALPACA_CLOCK_TIMEOUT_SECONDS", 12.0)
    retries = _env_int("ALPACA_CLOCK_RETRIES", 3)

    now = datetime.now(timezone.utc)

    # 1) cache hit
    cached = _CACHE.get("clock")
    fetched_at = float(_CACHE.get("fetched_at") or 0.0)
    age = time.time() - fetched_at
    if cached and age <= cache_seconds:
        clock = cached
    else:
        # 2) fetch with retry
        session = _make_session(retries=retries)
        try:
            clock = _fetch_clock(session, timeout_s=timeout_s)
            _CACHE["clock"] = clock
            _CACHE["fetched_at"] = time.time()
        except Exception as e:
            # 3) fallback: ak máme aspoň nejakú cache (aj staršiu), použi ju,
            # aby si neblokoval trading len kvôli dočasnému timeoutu
            if cached:
                clock = cached
                # ak je cache príliš stará, radšej skip (bezpečnejšie)
                if age > max(120, cache_seconds * 6):
                    return False, f"clock_error:{type(e).__name__}:stale_cache", _safe_clock(clock)
            else:
                return False, f"clock_error:{type(e).__name__}", _safe_clock({})

    clock = _safe_clock(clock)

    is_open = clock.get("is_open")
    ts = _parse_iso(clock.get("timestamp")) or now
    next_close = _parse_iso(clock.get("next_close"))

    if is_open is not True:
        return False, "market_closed", clock

    # stop entries pred close
    if next_close and stop_new_entries_min_before_close and stop_new_entries_min_before_close > 0:
        mins = (next_close - ts).total_seconds() / 60.0
        if mins <= float(stop_new_entries_min_before_close):
            return False, f"near_close:{mins:.1f}m", clock

    return True, "ok", clock


def _safe_clock(clock: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": clock.get("timestamp"),
        "is_open": clock.get("is_open"),
        "next_open": clock.get("next_open"),
        "next_close": clock.get("next_close"),
    }
