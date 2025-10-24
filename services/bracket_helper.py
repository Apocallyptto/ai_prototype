# services/bracket_helper.py
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import requests
import pandas as pd

# ---------- ENV ----------
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

ATR_EXITS = os.getenv("ATR_EXITS", "1") == "1"
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))
MAX_TP_PCT = float(os.getenv("MAX_TP_PCT", "0.015"))  # 1.5%
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "0.015"))

USE_DYNAMIC_SIZE = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))  # 0.25%
MIN_QTY = int(os.getenv("MIN_QTY", "1"))
MAX_QTY = int(os.getenv("MAX_QTY", "10"))

# price increment; Alpaca equity min tick is $0.01 for most large caps
TICK = 0.01

S = requests.Session()
if API_KEY and API_SECRET:
    S.headers.update(
        {
            "APCA-API-KEY-ID": API_KEY,
            "APCA-API-SECRET-KEY": API_SECRET,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _round_price(x: float) -> float:
    # round to 2 decimals; could be smarter per-symbol tick
    return float(f"{x:.2f}")

# ---------- Alpaca data helpers ----------

def _latest_trade_price(symbol: str) -> float:
    """
    Use Alpaca's latest trade as the 'base_price' that Alpaca validates against.
    """
    try:
        r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest", timeout=5)
        r.raise_for_status()
        j = r.json()
        px = j.get("trade", {}).get("p")
        if px:
            return float(px)
    except Exception:
        pass
    # fallback to latest quote mid if trade not available
    try:
        r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest", timeout=5)
        r.raise_for_status()
        j = r.json()
        bp = j.get("quote", {}).get("bp")
        ap = j.get("quote", {}).get("ap")
        if bp and ap:
            return (float(bp) + float(ap)) / 2.0
    except Exception:
        pass
    # last resort: account a tiny default
    raise RuntimeError(f"Could not fetch latest price for {symbol}")

def _fetch_bars(symbol: str, lookback_days: int, timeframe: str = "5Min") -> pd.DataFrame:
    """
    Get historical bars from Alpaca data.
    """
    end = _utcnow()
    start = end - timedelta(days=lookback_days)
    params = {
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "adjustment": "all",
    }
    r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars", params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    bars = j.get("bars", [])
    if not bars:
        raise RuntimeError(f"No bars from Alpaca for {symbol}")
    df = pd.DataFrame(bars)
    # standardize column names
    df.rename(
        columns={"t": "time", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"},
        inplace=True,
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    df = df.sort_index()
    return df

# ---------- ATR ----------
def _atr(symbol: str, period: int, lookback_days: int, timeframe: str = "5Min") -> float:
    df = _fetch_bars(symbol, lookback_days, timeframe)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
    return float(atr)

# ---------- ATR-based exits anchored to Alpaca base ----------
def _atr_exits(symbol: str, side: str, base_price: float) -> Tuple[float, float]:
    """
    Return (take_profit_price, stop_loss_price) with ATR scaling and hard Alpaca constraints,
    computed AROUND the Alpaca latest base price.
    """
    if not ATR_EXITS:
        # simple fixed offsets if ATR disabled: 1% each way as a fallback
        if side == "buy":
            return _round_price(base_price * (1 + 0.01)), _round_price(base_price * (1 - 0.01))
        else:
            return _round_price(base_price * (1 - 0.01)), _round_price(base_price * (1 + 0.01))

    atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK_DAYS)
    # ATR deltas
    tp_delta = ATR_MULT_TP * atr
    sl_delta = ATR_MULT_SL * atr

    # Also cap deltas by MAX_TP_PCT / MAX_SL_PCT
    tp_cap = base_price * MAX_TP_PCT
    sl_cap = base_price * MAX_SL_PCT
    tp_delta = min(tp_delta, tp_cap)
    sl_delta = min(sl_delta, sl_cap)

    if side == "buy":
        tp = base_price + tp_delta
        sl = base_price - sl_delta
        # Enforce Alpaca constraints (buy)
        tp = max(tp, base_price + TICK)
        sl = min(sl, base_price - TICK)
    else:  # sell
        tp = base_price - tp_delta
        sl = base_price + sl_delta
        # Enforce Alpaca constraints (sell)
        tp = min(tp, base_price - TICK)
        sl = max(sl, base_price + TICK)

    return _round_price(tp), _round_price(sl)

# ---------- Dynamic sizing ----------
def _account_equity() -> float:
    r = S.get(f"{ALPACA_BASE_URL}/v2/account", timeout=5)
    r.raise_for_status()
    j = r.json()
    eq = j.get("equity") or j.get("cash") or "0"
    return float(eq)

def _risk_per_share(symbol: str, side: str, base_price: float) -> float:
    """
    Approximate R per share = distance to stop.
    """
    tp, sl = _atr_exits(symbol, side, base_price)
    if side == "buy":
        rps = base_price - sl
    else:
        rps = sl - base_price
    return max(rps, TICK)

def _compute_dynamic_qty(symbol: str, side: str, base_price: float) -> int:
    """
    Van Tharp-style: position size so that (Qty * R_per_share) ~= risk_budget
    risk_budget = equity * RISK_PCT_PER_TRADE
    """
    try:
        equity = _account_equity()
    except Exception:
        equity = 10000.0  # safe fallback
    risk_budget = equity * RISK_PCT_PER_TRADE
    rps = _risk_per_share(symbol, side, base_price)
    if rps <= 0:
        return MIN_QTY
    qty = math.floor(risk_budget / rps)
    qty = max(qty, MIN_QTY)
    qty = min(qty, MAX_QTY)
    return int(qty)

# ---------- Public: submit bracket ----------
def submit_bracket(
    symbol: str,
    side: str,
    strength: float,
    qty: Optional[int] = None,
    client_id: Optional[str] = None,
    time_in_force: str = "day",
    order_type: str = "market",
    extended_hours: bool = False,
) -> Dict:
    """
    Build and submit a bracket order with ATR exits anchored on Alpaca's latest trade price.
    """
    side = side.lower().strip()
    if side not in ("buy", "sell"):
        raise ValueError(f"submit_bracket: invalid side={side}")

    # 1) Anchor to Alpaca base price (this is what the API validates against)
    base_price = _latest_trade_price(symbol)

    # 2) Size
    if qty is None and USE_DYNAMIC_SIZE:
        qty = _compute_dynamic_qty(symbol, side, base_price)
        dyn = True
    else:
        qty = qty or int(os.getenv("QTY_PER_TRADE", "1"))
        dyn = False

    # 3) Exits (enforce constraints)
    tp, sl = _atr_exits(symbol, side, base_price)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": False,  # Alpaca brackets are RTH-only
        "client_order_id": client_id or f"BRK-{symbol}-{int(time.time())}",
        "type": order_type,       # "market" recommended with these constraints
    }

    log_atr = f"atr(period={ATR_PERIOD},tp×{ATR_MULT_TP},sl×{ATR_MULT_SL})"
    log_dyn = f"dyn={str(dyn)}"
    print(f"INFO bracket_helper | submit bracket ({log_atr}, {log_dyn}) -> {payload}")

    r = S.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=10)
    # Raise with useful body if error
    try:
        r.raise_for_status()
    except requests.HTTPError as he:
        body = r.text
        print(f"INFO bracket_helper | submit failed: {body}")
        raise
    return r.json()
