# services/bracket_helper.py
from __future__ import annotations

import os, math, time, uuid, logging
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional, Dict, Any, List

import requests
import pandas as pd

log = logging.getLogger("bracket_helper")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# --- ENV / knobs -------------------------------------------------------------

TRADING_MODE           = os.getenv("TRADING_MODE", "paper")
ALPACA_BASE_URL        = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL        = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_API_KEY         = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET      = os.getenv("ALPACA_API_SECRET", "")

TIMEFRAME              = os.getenv("NN_TIMEFRAME", "5Min")

# ATR exits
USE_ATR_EXITS          = os.getenv("ATR_EXITS", "1") == "1"
ATR_PERIOD             = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS      = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP            = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL            = float(os.getenv("ATR_MULT_SL", "1.0"))
MAX_TP_PCT             = float(os.getenv("MAX_TP_PCT", "0.015"))  # absolute safety caps
MAX_SL_PCT             = float(os.getenv("MAX_SL_PCT", "0.015"))

# Dynamic sizing (risk-per-trade)
USE_DYNAMIC_SIZE       = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
RISK_PCT_PER_TRADE     = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))  # 0.25%
MIN_QTY                = int(os.getenv("MIN_QTY", "1"))
MAX_QTY                = int(os.getenv("MAX_QTY", "10"))

# Optional: size curve by strength
USE_SIZE_BY_STRENGTH   = os.getenv("USE_SIZE_BY_STRENGTH", "0") == "1"
SIZE_MIN_STRENGTH      = float(os.getenv("SIZE_MIN_STRENGTH", "0.58"))
SIZE_MAX_STRENGTH      = float(os.getenv("SIZE_MAX_STRENGTH", "0.75"))
SIZE_EXP               = float(os.getenv("SIZE_EXP", "2.0"))
SIZE_MIN_MULT          = float(os.getenv("SIZE_MIN_MULT", "0.50"))
SIZE_MAX_MULT          = float(os.getenv("SIZE_MAX_MULT", "1.00"))

# --- helpers -----------------------------------------------------------------

def _alpaca_headers() -> Dict[str, str]:
    return {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}


def _fetch_bars(symbol: str, lookback_days: int, timeframe: str) -> pd.DataFrame:
    """Fetch bars from Alpaca or Yahoo (if USE_YF_DATA=1 or SIP restricted)."""
    import yfinance as yf
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    use_yf = os.getenv("USE_YF_DATA", "0") == "1"

    if not use_yf:
        try:
            url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
            params = {
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 10000,
                "adjustment": "all",
            }
            r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=20)
            if r.status_code == 403 or r.status_code == 429:
                use_yf = True
            else:
                r.raise_for_status()
                js = r.json()
                bars = js.get("bars", [])
                if not bars:
                    raise RuntimeError("No bars from Alpaca")
                df = pd.DataFrame(bars)
                df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
                return df
        except Exception as e:
            print(f"WARN: Alpaca data failed ({e}), using Yahoo fallback.")
            use_yf = True

    # === Yahoo fallback ===
    max_days = 55 if timeframe.lower() in ("5min", "5m") else 365
    safe_start = max(end - timedelta(days=max_days), start)
    df = yf.download(symbol, start=safe_start.date(), end=end.date(), interval="5m", progress=False)
    if df.empty:
        raise RuntimeError("No bars from Yahoo fallback")
    df = df.reset_index().rename(columns={"Datetime": "t"})
    return df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})


def _atr(symbol: str, period: int, lookback_days: int) -> float:
    df = _fetch_bars(symbol, lookback_days, TIMEFRAME)
    highs = df["High"].astype(float)
    lows = df["Low"].astype(float)
    closes = df["Close"].astype(float)
    prev_close = closes.shift(1)
    tr = pd.concat([(highs - lows), (highs - prev_close).abs(), (lows - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


def _atr_exits(symbol: str, side: str, base_price: float) -> Tuple[float, float]:
    """
    Compute ATR-based exits with adaptive regime support.
    """
    # --- Adaptive ATR regime ---
    try:
        from services.atr_regime import get_dynamic_tp_sl
        tp_mult, sl_mult = get_dynamic_tp_sl(symbol)
    except Exception as e:
        print(f"[ATR_REGIME] fallback static ({e})")
        tp_mult = ATR_MULT_TP
        sl_mult = ATR_MULT_SL

    atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK_DAYS)
    if side == "buy":
        tp = base_price + tp_mult * atr
        sl = base_price - sl_mult * atr
    else:
        tp = base_price - tp_mult * atr
        sl = base_price + sl_mult * atr

    # apply absolute safety caps
    tp_cap = base_price * (1 + (MAX_TP_PCT if side == "buy" else -MAX_TP_PCT))
    sl_cap = base_price * (1 - (MAX_SL_PCT if side == "buy" else -MAX_SL_PCT))
    if side == "buy":
        tp = min(tp, tp_cap)
        sl = max(sl, sl_cap)
    else:
        tp = max(tp, tp_cap)
        sl = min(sl, sl_cap)
    return tp, sl


def _get_equity() -> float:
    url = f"{ALPACA_BASE_URL}/v2/account"
    r = requests.get(url, headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    js = r.json()
    return float(js["equity"])


def _scale_by_strength(strength: Optional[float]) -> float:
    """Return multiplicative factor for base qty based on strength."""
    if not (USE_SIZE_BY_STRENGTH and strength is not None):
        return 1.0
    lo, hi = SIZE_MIN_STRENGTH, SIZE_MAX_STRENGTH
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (strength - lo) / (hi - lo)))
    if SIZE_EXP != 1.0:
        t = t ** SIZE_EXP
    mult = SIZE_MIN_MULT + (SIZE_MAX_MULT - SIZE_MIN_MULT) * t
    return float(max(0.0, mult))


def _risk_per_share(symbol: str, side: str, base_price: float) -> float:
    tp, sl = _atr_exits(symbol, side, base_price)
    return float(max(0.01, (base_price - sl) if side == "buy" else (sl - base_price)))


def _compute_dynamic_qty(symbol: str, side: str, base_price: float, strength: Optional[float] = None) -> int:
    if not USE_DYNAMIC_SIZE:
        return max(MIN_QTY, 1)
    equity = _get_equity()
    risk_dollars = max(1.0, equity * RISK_PCT_PER_TRADE)
    rps = _risk_per_share(symbol, side, base_price)
    base_qty = int(max(1, math.floor(risk_dollars / rps)))
    mult = _scale_by_strength(strength)
    qty = int(max(1, math.floor(base_qty * mult)))
    return int(max(MIN_QTY, min(MAX_QTY, qty)))


def _last_quote(symbol: str) -> float:
    """Lightweight last price via Alpaca /v2/stocks/quotes/latest (IEX)."""
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest"
    r = requests.get(url, headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    q = r.json().get("quote", {})
    bid, ask = float(q.get("bp", 0) or 0), float(q.get("ap", 0) or 0)
    if bid and ask:
        return round((bid + ask) / 2.0, 2)
    import yfinance as yf
    px = yf.Ticker(symbol).fast_info.last_price
    return float(px)


# --- main submit -------------------------------------------------------------

def submit_bracket(symbol: str, side: str, qty: Optional[int] = None, *, strength: Optional[float] = None) -> Dict[str, Any]:
    base_price = _last_quote(symbol)

    if USE_ATR_EXITS:
        tp, sl = _atr_exits(symbol, side, base_price)
    else:
        if side == "buy":
            tp, sl = base_price * 1.01, base_price * 0.99
        else:
            tp, sl = base_price * 0.99, base_price * 1.01

    if qty is None:
        qty = _compute_dynamic_qty(symbol, side, base_price, strength)
    qty = max(1, int(qty))

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": False,
        "client_order_id": f"BRK-{symbol}-{int(time.time())}",
        "type": "limit",
        "limit_price": f"{base_price:.2f}",
    }

    log.info("submit bracket (atr(period=%s,tp×dynamic,sl×dynamic), dyn=%s) -> %s",
             ATR_PERIOD, "True" if USE_DYNAMIC_SIZE else "False", payload)

    url = f"{ALPACA_BASE_URL}/v2/orders"
    r = requests.post(url, headers=_alpaca_headers(), json=payload, timeout=20)
    if not r.ok:
        log.info("submit failed: %s", r.text)
        r.raise_for_status()
    return r.json()
