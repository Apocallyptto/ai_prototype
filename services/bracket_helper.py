# services/bracket_helper.py
from __future__ import annotations

import os
from math import floor
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

import requests

# --- Alpaca REST lightweight (account/bars) -----------------
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})

def _get_account_json() -> dict:
    r = S.get(f"{ALPACA_BASE_URL}/v2/account", timeout=20)
    r.raise_for_status()
    return r.json()

def _get_last_price(symbol: str) -> float:
    r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest", timeout=20)
    r.raise_for_status()
    j = r.json()
    # Fallbacks if best ask not present
    if "quote" in j and j["quote"] and "ap" in j["quote"]:
        return float(j["quote"]["ap"]) or float(j["quote"].get("bp", 0)) or float(j["quote"].get("ap", 0))
    # try last trade
    r2 = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest", timeout=20)
    r2.raise_for_status()
    return float(r2.json()["trade"]["p"])

# --- ATR utilities ------------------------------------------
def _fetch_bars(symbol: str, lookback_days: int, timeframe: str = "5Min"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000
    }
    r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars?timeframe={timeframe}", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("bars", [])

def _atr(symbol: str, period: int, lookback_days: int, timeframe: str = "5Min") -> float:
    import pandas as pd
    bars = _fetch_bars(symbol, lookback_days, timeframe)
    if not bars:
        return 0.0
    df = pd.DataFrame(bars)
    # expected columns: t,o,h,l,c,v
    df["h_l"] = df["h"] - df["l"]
    df["h_pc"] = (df["h"] - df["c"].shift(1)).abs()
    df["l_pc"] = (df["l"] - df["c"].shift(1)).abs()
    tr = df[["h_l", "h_pc", "l_pc"]].max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if atr and atr > 0 else 0.0

# --- Env config for exits and sizing ------------------------
ATR_ON         = os.getenv("ATR_EXITS", "0") == "1"
ATR_PERIOD     = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK   = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP    = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL    = float(os.getenv("ATR_MULT_SL", "1.0"))

MAX_TP_PCT     = float(os.getenv("MAX_TP_PCT", "0.015"))  # 1.5%
MAX_SL_PCT     = float(os.getenv("MAX_SL_PCT", "0.015"))  # 1.5%

# Dynamic position sizing
USE_DYNAMIC_SIZE      = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
RISK_PCT_PER_TRADE    = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))  # 0.25% of equity
MIN_QTY               = int(os.getenv("MIN_QTY", "1"))
MAX_QTY               = int(os.getenv("MAX_QTY", "5"))
MIN_SL_DOLLARS        = float(os.getenv("MIN_SL_DOLLARS", "0.05"))        # ensure non-zero risk/shr

def _cap_pct(base: float, pct: float, up: bool) -> float:
    return base * (1 + pct) if up else base * (1 - pct)

def _atr_exits(symbol: str, side: str, last_price: float) -> Tuple[float, float]:
    """
    Returns (tp_price, sl_price) using ATR or static caps.
    """
    if ATR_ON:
        atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK)
        if atr > 0:
            if side == "buy":
                tp = last_price + ATR_MULT_TP * atr
                sl = last_price - max(ATR_MULT_SL * atr, MIN_SL_DOLLARS)
            else:
                tp = last_price - ATR_MULT_TP * atr
                sl = last_price + max(ATR_MULT_SL * atr, MIN_SL_DOLLARS)
            # extra safety caps by pct
            tp_cap_up   = _cap_pct(last_price, MAX_TP_PCT, up=True)
            tp_cap_down = _cap_pct(last_price, MAX_TP_PCT, up=False)
            sl_cap_up   = _cap_pct(last_price, MAX_SL_PCT, up=True)
            sl_cap_down = _cap_pct(last_price, MAX_SL_PCT, up=False)

            if side == "buy":
                tp = min(tp, tp_cap_up)
                sl = max(sl, sl_cap_down)
            else:
                tp = max(tp, tp_cap_down)
                sl = min(sl, sl_cap_up)
            return (round(tp, 2), round(sl, 2))

    # Fallback: only pct caps
    if side == "buy":
        return (round(_cap_pct(last_price, MAX_TP_PCT, True), 2),
                round(_cap_pct(last_price, MAX_SL_PCT, False), 2))
    else:
        return (round(_cap_pct(last_price, MAX_TP_PCT, False), 2),
                round(_cap_pct(last_price, MAX_SL_PCT, True), 2))

def _risk_per_share(symbol: str, side: str, last_price: float) -> float:
    """
    Estimate risk per share using the SL distance we would place.
    """
    tp, sl = _atr_exits(symbol, side, last_price)
    return max(abs(last_price - sl), MIN_SL_DOLLARS)

def _compute_dynamic_qty(symbol: str, side: str, last_price: float) -> int:
    """
    Qty = floor( risk_budget_usd / risk_per_share ), clamped by buying power and min/max.
    """
    acct = _get_account_json()
    equity = float(acct.get("equity", 0))
    buying_power = float(acct.get("buying_power", 0))

    risk_budget = equity * RISK_PCT_PER_TRADE
    rps = _risk_per_share(symbol, side, last_price)
    if rps <= 0:
        return MIN_QTY

    qty = int(floor(risk_budget / rps))
    if qty < MIN_QTY:
        qty = MIN_QTY
    if qty > MAX_QTY:
        qty = MAX_QTY

    # Do not exceed buying power
    max_by_bp = int(floor(buying_power / max(last_price, 0.01)))
    if max_by_bp < 1:
        return 0
    qty = min(qty, max_by_bp)
    return qty

# --- Order submit -------------------------------------------
def submit_bracket(
    symbol: str,
    side: str,                 # "buy" | "sell"
    strength: float,
    qty: Optional[int] = None, # if dynamic sizing on, this may be ignored
    api: Optional[object] = None,  # kept for compatibility (unused here)
    last_price: Optional[float] = None,
    client_id: Optional[str] = None,
    time_in_force: str = "day",
    order_type: str = "market",
    extended_hours: bool = False,
) -> Dict:
    """
    Build and submit a bracket order with ATR exits and (optional) dynamic sizing.
    Returns the Alpaca order JSON (or raises for HTTP errors).
    """
    if last_price is None:
        last_price = _get_last_price(symbol)

    # choose qty
    if USE_DYNAMIC_SIZE:
        dyn_qty = _compute_dynamic_qty(symbol, side, last_price)
        if dyn_qty <= 0:
            raise RuntimeError(f"dynamic sizing -> qty=0 for {symbol}, check buying power / params")
        use_qty = dyn_qty
    else:
        use_qty = int(qty or 1)

    tp, sl = _atr_exits(symbol, side, last_price)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(use_qty),
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss":   {"stop_price": f"{sl:.2f}"},
        "extended_hours": bool(extended_hours),
        "type": order_type,
    }
    if client_id:
        payload["client_order_id"] = client_id

    print(
        f"submit bracket {'(atr(period=%s,tp×%s,sl×%s))' % (ATR_PERIOD, ATR_MULT_TP, ATR_MULT_SL) if ATR_ON else ''}: "
        f"{payload}"
    )

    r = S.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=20)
    # Raise for 4xx/5xx with readable message in logs
    if r.status_code >= 400:
        try:
            print("submit failed:", r.text)
        finally:
            r.raise_for_status()
    return r.json()
