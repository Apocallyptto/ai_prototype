# services/bracket_helper.py
from __future__ import annotations
import os
from typing import Dict, Any, Tuple

# ===== Exit params (static; can be tuned by env) =====
BASE_TP_PCT = float(os.getenv("BASE_TP_PCT", "0.006"))  # 0.6%
BASE_SL_PCT = float(os.getenv("BASE_SL_PCT", "0.004"))  # 0.4%

# Extended hours MUST be False for bracket orders on Alpaca
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "0").lower() in {"1","true","yes","y"}

def compute_exits(last_price: float, side: str) -> Tuple[float, float]:
    """
    Returns (tp_price, sl_price) rounded to cents.
    """
    if side == "buy":
        tp = last_price * (1.0 + BASE_TP_PCT)
        sl = last_price * (1.0 - BASE_SL_PCT)
    else:
        tp = last_price * (1.0 - BASE_TP_PCT)
        sl = last_price * (1.0 + BASE_SL_PCT)
    return float(f"{tp:.2f}"), float(f"{sl:.2f}")

def build_bracket_order(symbol: str, side: str, qty: int | float, last_price: float, client_id: str) -> Dict[str, Any]:
    """
    Construct a bracket order dict compatible with Alpaca's API.
    NOTE: extended_hours must be False for bracket orders (Alpaca requirement).
    """
    tp_price, sl_price = compute_exits(last_price, side)
    order: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": "limit",
        "qty": str(int(qty)),
        "time_in_force": "day",
        "order_class": "bracket",
        "limit_price": f"{last_price:.2f}",
        "take_profit": {"limit_price": f"{tp_price:.2f}"},
        "stop_loss": {"stop_price": f"{sl_price:.2f}"},
        "extended_hours": False if EXTENDED_HOURS else False,  # force False for safety
        "client_order_id": client_id,
    }
    return order

def submit_bracket(api, symbol: str, side: str, qty: int | float, last_price: float, client_id: str):
    """
    Submit the bracket order with Alpaca client `api`.
    Keeps your previous logging style.
    """
    order = build_bracket_order(symbol, side, qty, last_price, client_id)
    print(f"INFO bracket_helper | submit bracket: {order}")
    try:
        resp = api.submit_order(**order)
        return resp
    except Exception as e:
        # Try to print Alpaca error body if available
        try:
            msg = getattr(e, "response", None)
            if msg is not None:
                print(f"INFO bracket_helper | submit failed: {msg.text}")
        except Exception:
            pass
        print(f"ERROR bracket_helper | submit failed: {e}")
        raise
