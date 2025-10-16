"""
jobs/manage_exits.py

Post-fill exit management for ai_prototype (Alpaca paper trading bot).

Purpose
- For each open stock position (AAPL/MSFT/SPY), ensure there is an attached exit: an OCO order
  comprised of a take-profit (limit) and a stop-loss (stop/stop-limit).
- Prices are ATR(14)-aware using 5-minute bars by default.
- Idempotent: if suitable TP/SL already exist, do nothing.

Usage
    $env:PYTHONPATH = "$PWD"
    python -m jobs.manage_exits --symbols AAPL,MSFT,SPY

Optional args (env or CLI)
- ATR_MULT_TP (float): default 1.2  -> TP distance = +1.2 * ATR
- ATR_MULT_SL (float): default 1.0  -> SL distance = -1.0 * ATR
- ATR_TIMEFRAME (str): default "5Min" (Alpaca v2 bars: 1Min/5Min/15Min/1Hour/1Day)
- ATR_LENGTH (int): default 14
- MAX_EXIT_AGE_MIN (int): if existing exits are older than this and far from market, we can rearm (future)
- EXTENDED_HOURS (bool): default False (RTH only)

Notes
- Uses Alpaca's REST v2 API via requests (keeps consistency with your retrying session approach).
- Avoids storing secrets in code; reads from env.
- Creates client_order_id tags for traceability: f"EXIT-{symbol}-{side}-{ts}"
- If paper account doesn't support OCO for crypto, we skip non-stock assets.

Dependencies: pandas, numpy, python-dateutil
"""
from __future__ import annotations
import os
import sys
import time
import math
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparser

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' works on free/paper; 'sip' needs subscription

# Strategy knobs (env -> defaults)
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.2"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))
ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))
ATR_TIMEFRAME = os.getenv("ATR_TIMEFRAME", "5Min")
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in {"1","true","yes","y"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

# Basic retry/backoff wrapper (simple, since your code already has a robust version)
def http(method: str, url: str, **kwargs):
    for attempt in range(5):
        try:
            resp = SESSION.request(method, url, timeout=15, **kwargs)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} {resp.text}")
            return resp
        except Exception as e:
            wait = min(2 ** attempt, 8)
            log(f"HTTP error {e} -> retrying in {wait}s", level="WARN")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")


def log(msg: str, level: str = "INFO"):
    levels = ["DEBUG","INFO","WARN","ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} manage_exits | {msg}")


@dataclass
class Position:
    symbol: str
    qty: int
    side: str  # 'long' or 'short'
    avg_entry: float


@dataclass
class ExitPlan:
    tp: float
    sl_stop: float
    sl_limit: Optional[float]  # allow stop-limit; can be None for plain stop


# ---------------- Alpaca helpers ---------------- #

def get_open_positions() -> List[Position]:
    url = f"{ALPACA_BASE_URL}/v2/positions"
    r = http("GET", url)
    r.raise_for_status()
    out = []
    for p in r.json():
        # Only stocks/etfs handled (asset_class == 'us_equity')
        if p.get("asset_class") not in ("us_equity", None):
            continue
        qty = int(float(p["qty"]))
        side = "long" if float(p["avg_entry_price"]) >= 0 and qty > 0 else "short"
        out.append(Position(
            symbol=p["symbol"],
            qty=qty,
            side=side,
            avg_entry=float(p["avg_entry_price"]),
        ))
    return out


def get_open_orders(symbol: Optional[str] = None) -> List[dict]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbol:
        url += f"&symbols={symbol}"
    r = http("GET", url)
    r.raise_for_status()
    return r.json()


def _quantize_price(price: float, tick: Decimal = Decimal('0.01')) -> float:
    """Quantize to the nearest tick (default 1 cent) using bankers-safe decimal rounding."""
    return float(Decimal(str(price)).quantize(tick, rounding=ROUND_HALF_UP))

def submit_oco_exit(symbol: str, side_exit: str, qty: int, tp_price: float, sl_stop: float, sl_limit: Optional[float]) -> dict:
    # Enforce US equity tick size (no sub-pennies)
    tp_q = _quantize_price(tp_price)
    sl_q = _quantize_price(sl_stop)
    sl_lim_q = _quantize_price(sl_limit) if sl_limit is not None else None

    url = f"{ALPACA_BASE_URL}/v2/orders"
    client_id = f"EXIT-{symbol}-{int(time.time())}"
    payload = {
        "symbol": symbol,
        "side": side_exit,
        "type": "limit",
        "qty": str(qty),
        "time_in_force": "gtc",
        "order_class": "oco",
        "take_profit": {"limit_price": f"{tp_q:.2f}"},
        "client_order_id": client_id,
        "extended_hours": EXTENDED_HOURS,
    }
    stop = {"stop_price": f"{sl_q:.2f}"}
    if sl_lim_q is not None:
        stop["limit_price"] = f"{sl_lim_q:.2f}"
    payload["stop_loss"] = stop

    log(f"Submitting OCO exit: {payload}")
    r = http("POST", url, data=json.dumps(payload))
    if r.status_code >= 300:
        log(f"OCO submit failed: {r.status_code} {r.text}", level="ERROR")
        r.raise_for_status()
    return r.json()


def recent_bars(symbol: str, timeframe: str = ATR_TIMEFRAME, lookback: int = 400) -> pd.DataFrame:
    """Fetch recent bars. Default feed is 'iex' (paper/free). Fallback to 'iex' on 403 if user tried 'sip'."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=10)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"

    def _get(feed: str):
        params = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": now.isoformat(),
            "limit": lookback,
            "adjustment": "split",
            "feed": feed,
        }
        r = http("GET", url, params=params)
        return r

    # Try configured feed first, then fallback
    feeds_to_try = [ALPACA_DATA_FEED]
    if ALPACA_DATA_FEED.lower() != "iex":
        feeds_to_try.append("iex")

    last_err = None
    for feed in feeds_to_try:
        try:
            r = _get(feed)
            if r.status_code == 403:
                last_err = RuntimeError(f"403 on feed '{feed}' — likely not enabled for your plan")
                continue
            r.raise_for_status()
            js = r.json()
            bars = js.get("bars", [])
            if not bars:
                raise RuntimeError(f"No bars for {symbol} (feed={feed})")
            df = pd.DataFrame(bars)
            df.rename(columns={"t": "timestamp", "o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception as e:
            last_err = e
            log(f"Bar fetch failed for {symbol} on feed={feed}: {e}", level="WARN")
            continue
    raise RuntimeError(f"Bar fetch failed for {symbol} on all feeds tried: {feeds_to_try}. Last error: {last_err}")


def compute_atr(df: pd.DataFrame, length: int = ATR_LENGTH) -> float:
    # Classic Wilder's ATR
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    val = float(atr.dropna().iloc[-1])
    return val


def has_active_exit(orders: List[dict], position: Position) -> bool:
    sym = position.symbol
    # Look for order_class 'oco' or child orders with client id prefix EXIT-
    for o in orders:
        if o.get("symbol") != sym:
            continue
        if o.get("order_class") == "oco":
            return True
        cid = o.get("client_order_id", "")
        if cid.startswith("EXIT-"):
            return True
        # Bracket child orders contain 'parent_order_id' and 'type' combo; treat as exit present
        if o.get("parent_order_id") and o.get("side") in ("sell","buy"):
            # naive, but sufficient
            return True
    return False


def build_exit_plan(pos: Position, atr: float) -> ExitPlan:
    if pos.side == "long":
        tp = pos.avg_entry + ATR_MULT_TP * atr
        sl_stop = pos.avg_entry - ATR_MULT_SL * atr
    else:  # short
        tp = pos.avg_entry - ATR_MULT_TP * atr
        sl_stop = pos.avg_entry + ATR_MULT_SL * atr
    # Optional stop-limit offset (a small extra cushion)
    sl_limit = None  # or e.g., sl_stop -/+ 0.02 depending on side
    return ExitPlan(tp=round(tp, 4), sl_stop=round(sl_stop, 4), sl_limit=sl_limit)


def _summarize_orders(orders: List[dict]) -> str:
    if not orders:
        return "(no open orders)"
    by_sym = {}
    for o in orders:
        sym = o.get("symbol")
        by_sym.setdefault(sym, 0)
        by_sym[sym] += 1
    parts = [f"{sym}:{cnt}" for sym, cnt in sorted(by_sym.items())]
    return ", ".join(parts)

def print_state(symbols: Optional[List[str]] = None):
    positions = get_open_positions()
    orders = get_open_orders()
    if symbols:
        syms = set(s.strip().upper() for s in symbols)
        positions = [p for p in positions if p.symbol in syms]
        orders = [o for o in orders if o.get("symbol") in syms]
    if positions:
        log("Open positions:")
        for p in positions:
            log(f"- {p.symbol} qty={p.qty} side={p.side} avg={p.avg_entry}")
    else:
        log("No open positions found.")
    log(f"Open orders summary: {_summarize_orders(orders)}")


def ensure_exits(symbols: Optional[List[str]] = None):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    positions = get_open_positions()
    if symbols:
        symbols_set = set(s.strip().upper() for s in symbols)
        positions = [p for p in positions if p.symbol in symbols_set]

    if not positions:
        log("No open positions to manage exits for.")
        # Extra context to help debugging
        try:
            oo = get_open_orders()
            log(f"Open orders summary: {_summarize_orders(oo)}")
        except Exception as e:
            log(f"Failed to query open orders: {e}", level="WARN")
        return

    all_open_orders = get_open_orders()

    for p in positions:
        if has_active_exit(all_open_orders, p):
            log(f"{p.symbol}: exit already present; skipping.")
            continue
        try:
            df = recent_bars(p.symbol, timeframe=ATR_TIMEFRAME)
            atr = compute_atr(df, length=ATR_LENGTH)
            plan = build_exit_plan(p, atr)
            side_exit = "sell" if p.side == "long" else "buy"
            submit_oco_exit(p.symbol, side_exit, p.qty, plan.tp, plan.sl_stop, plan.sl_limit)
            log(f"{p.symbol}: OCO exits placed (TP={plan.tp}, SL={plan.sl_stop}).")
        except Exception as e:
            log(f"{p.symbol}: failed to place exits -> {e}", level="ERROR")


def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Ensure OCO exits exist for open positions.")
    ap.add_argument("--symbols", type=str, default="", help="Comma-separated whitelist, e.g., AAPL,MSFT,SPY")
    ap.add_argument("--debug", action="store_true", help="Print current positions and open orders, then exit.")
    args = ap.parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    return args, symbols


if __name__ == "__main__":
    args, syms = parse_cli(sys.argv[1:])
    if args.debug:
        print_state(syms)
        sys.exit(0)
    ensure_exits(syms)
