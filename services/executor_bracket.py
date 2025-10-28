# services/executor_bracket.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, List

# Stable helper API (shim + fallbacks)
from services.bracket_public import (
    get_last_price,
    compute_dynamic_qty,
    submit_bracket,
)

# Guards
from services.position_guard import has_same_side_position
from services.risk_budget import can_open

# Notifications
from services.notify import (
    notify_trade_opened,
    notify_trade_blocked,
    notify_trade_skipped,
)


@dataclass
class Signal:
    """
    Minimal signal model used by this executor.
    Extend if your pipeline adds fields.
    """
    symbol: str
    side: str                   # 'buy' or 'sell'
    strength: Optional[float] = None
    price: Optional[float] = None    # preferred entry/limit if present
    qty: Optional[int] = None
    source: Optional[str] = None     # e.g., 'ml_ensemble', 'rule', etc.


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def place_bracket_for_signal(signal: Signal, logger) -> Optional[dict]:
    """
    Execute a single signal with:
      1) same-side position guard
      2) portfolio risk-budget guard
      3) submit bracket order
      4) send notifications
    Returns broker response dict (if any), or None when skipped/blocked.
    """
    symbol = signal.symbol.upper().strip()
    side = str(signal.side).lower().strip()
    if side not in ("buy", "sell"):
        logger.error(f"[{_utcnow()}] invalid side '{signal.side}' for {symbol}; skipping")
        return None

    last_price = signal.price if signal.price is not None else get_last_price(symbol)
    qty = signal.qty if signal.qty is not None else compute_dynamic_qty(symbol, side, last_price)

    # ---- Guard 1: prevent same-side duplicates ----
    if has_same_side_position(symbol, side):
        why = "same-side position already open"
        logger.info(f"[{_utcnow()}] skip {symbol} {side}: {why}")
        notify_trade_skipped(symbol, side, why)
        return None

    # ---- Guard 2: portfolio risk-budget cap ----
    ok, reason = can_open(symbol, side, qty)
    logger.info(f"[{_utcnow()}] risk-budget check for {symbol} {side} qty={qty}: {reason}")
    if not ok:
        notify_trade_blocked(symbol, side, qty, reason)
        return None

    # ---- Submit the bracket order ----
    resp = submit_bracket(
        symbol=symbol,
        side=side,               # 'buy' or 'sell'
        qty=qty,
        last_price=last_price,   # mapped by shim to your helper's expected kw
    )
    logger.info(
        f"[{_utcnow()}] alpaca order -> {symbol} {side} qty={qty} "
        f"last={last_price:.4f} | response={resp}"
    )

    # Extract TP/SL best-effort for the notify
    tp = None
    sl = None
    try:
        legs = resp.get("legs") or []
        for leg in legs:
            if str(leg.get("type")) == "limit" and str(leg.get("side")) == "sell":
                tp = float(leg.get("limit_price"))
            if str(leg.get("type")) == "stop" and str(leg.get("side")) == "sell":
                sl = float(leg.get("stop_price"))
    except Exception:
        pass

    reason_txt = (signal.source or "executor")
    if signal.strength is not None:
        try:
            reason_txt += f" (strength {float(signal.strength):.2f})"
        except Exception:
            pass

    notify_trade_opened(symbol, side, int(qty), last_price, tp, sl, reason_txt)
    return resp


def place_brackets_for_signals(signals: Iterable[Signal], logger) -> List[Optional[dict]]:
    """Batch runner."""
    results: List[Optional[dict]] = []
    for sig in signals:
        try:
            results.append(place_bracket_for_signal(sig, logger))
        except Exception as e:
            logger.exception(f"[{_utcnow()}] error placing order for {sig.symbol} {sig.side}: {e}")
            results.append(None)
    return results


__all__ = [
    "Signal",
    "place_bracket_for_signal",
    "place_brackets_for_signals",
]
