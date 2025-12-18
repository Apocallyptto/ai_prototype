# services/oco_exit_monitor.py
from __future__ import annotations

import os
import time
import math
import logging
import requests
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("oco_exit_monitor")

# ---------------- ENV ----------------
POLL_SECONDS = int(os.getenv("EXIT_MONITOR_POLL_SECONDS", "15"))

# Percent-based "ATR" like your ensure_exits_now logs (1% default)
ATR_PCT = float(os.getenv("ATR_PCT", "0.01"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))

MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "1.00"))
MIN_QTY = float(os.getenv("MIN_QTY", "0.001"))

COOLDOWN_SECONDS = int(os.getenv("EXIT_COOLDOWN_SECONDS", "45"))
FORCE_FIX_EXITS = os.getenv("FORCE_FIX_EXITS", "0") == "1"

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
# prefer TRADING_MODE like signal_executor.py
ALPACA_PAPER = os.getenv("TRADING_MODE", "paper") == "paper"

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
})

_last_action_ts = 0.0


# ---------------- helpers ----------------
def _qt(x: float, p: int = 2) -> float:
    return round(float(x) + 1e-9, p)

def _qq(q: float) -> float:
    q = float(q)
    if q <= 0:
        return 0.0
    return math.floor(q * 1000.0 + 1e-9) / 1000.0

def _min_notional_price(px: float, qty: float) -> float:
    if qty <= 0:
        return px
    need = MIN_NOTIONAL / qty
    return max(px, _qt(need, 2))

def _latest_mid(dc: StockHistoricalDataClient, sym: str) -> Optional[float]:
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
        bid = float(q.bid_price) if q.bid_price is not None else None
        ask = float(q.ask_price) if q.ask_price is not None else None
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    return None

def _get_open_orders_nested(symbol: str):
    # nested=true lets us see oco legs too
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&direction=desc&nested=true&symbols={symbol}"
    r = S.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def _cancel_order(order_id: str):
    url = f"{ALPACA_BASE_URL}/v2/orders/{order_id}"
    r = S.delete(url, timeout=15)
    # Alpaca returns 204 on success
    if r.status_code not in (200, 204):
        r.raise_for_status()

def _submit_oco_exit(symbol: str, exit_side: str, qty: float, tp: float, sl: float):
    qty = _qq(qty)
    if qty < MIN_QTY:
        return None

    tp = _min_notional_price(_qt(tp, 2), qty)
    sl = _min_notional_price(_qt(sl, 2), qty)

    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": exit_side,          # "sell" for long exits, "buy" for short exits
        "type": "limit",            # TP is limit
        "time_in_force": "day",
        "order_class": "oco",
        "limit_price": str(tp),     # TP price
        "stop_loss": {
            "stop_price": str(sl)   # SL price
        },
        # DO NOT set extended_hours here (stop-loss doesn't support it)
    }

    r = S.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json()

def _compute_tp_sl(anchor: float, is_long: bool) -> tuple[float, float]:
    # percent-based ATR proxy
    atr = max(0.01, anchor * ATR_PCT)
    if is_long:
        tp = anchor + TP_ATR_MULT * atr
        sl = max(0.01, anchor - SL_ATR_MULT * atr)
        if tp <= sl:
            tp = sl + 0.02
    else:
        tp = max(0.01, anchor - TP_ATR_MULT * atr)
        sl = anchor + SL_ATR_MULT * atr
        if tp >= sl:
            sl = tp + 0.02
    return _qt(tp, 2), _qt(sl, 2)


# ---------------- loop ----------------
def run_once(tr: TradingClient, dc: StockHistoricalDataClient):
    global _last_action_ts
    now = time.time()
    if now - _last_action_ts < COOLDOWN_SECONDS:
        return

    try:
        positions = tr.get_all_positions()
    except Exception:
        positions = []

    if not positions:
        log.info("no positions")
        return

    for p in positions:
        sym = str(getattr(p, "symbol", "")).upper()
        if SYMBOLS and sym not in SYMBOLS:
            continue

        qty = float(getattr(p, "qty", 0.0) or 0.0)
        if abs(qty) < MIN_QTY:
            continue

        is_long = qty > 0
        abs_qty = abs(qty)
        exit_side = "sell" if is_long else "buy"

        # open orders for this symbol
        try:
            oo = _get_open_orders_nested(sym)
        except Exception as e:
            log.warning("%s: failed to fetch open orders: %s", sym, e)
            continue

        # detect existing OCO exit (same side)
        exit_orders = [o for o in oo if (o.get("symbol") == sym and (o.get("side") == exit_side))]
        has_oco = any(o.get("order_class") == "oco" for o in exit_orders)

        if has_oco and not FORCE_FIX_EXITS:
            log.info("%s: exits already present (OCO)", sym)
            continue

        if FORCE_FIX_EXITS and exit_orders:
            # cancel any old exit orders so we can recreate clean OCO
            for o in exit_orders:
                try:
                    _cancel_order(o["id"])
                except Exception as e:
                    log.warning("%s: cancel failed for %s: %s", sym, o.get("id"), e)

        # recompute free qty (in case partial exits exist)
        held = 0.0
        for o in exit_orders:
            try:
                held += float(o.get("qty") or 0.0)
            except Exception:
                pass
        free = _qq(max(0.0, abs_qty - held))
        if free < MIN_QTY:
            log.info("%s: no free qty to protect", sym)
            continue

        anchor = _latest_mid(dc, sym) or float(getattr(p, "avg_entry_price", 0.0) or 0.0)
        if anchor <= 0:
            log.info("%s: no anchor price; skip", sym)
            continue

        tp, sl = _compute_tp_sl(anchor, is_long=is_long)
        # For SHORT: tp is below, sl is above (still correct)

        try:
            res = _submit_oco_exit(sym, exit_side, free, tp, sl)
            log.info("%s: OCO exit submitted side=%s qty=%.3f tp=%.2f sl=%.2f id=%s",
                     sym, exit_side, free, tp, sl, res.get("id"))
            _last_action_ts = time.time()
        except Exception as e:
            log.warning("%s: OCO submit failed: %s", sym, e)


def main():
    tr = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    while True:
        try:
            run_once(tr, dc)
        except Exception as e:
            log.exception("oco_exit_monitor fatal: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
