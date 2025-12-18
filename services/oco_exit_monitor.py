# services/oco_exit_monitor.py
from __future__ import annotations

import os
import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

log = logging.getLogger("oco_exit_monitor")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))

ATR_PCT = float(os.getenv("ATR_PCT", "0.0100"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))

# keď je 1, monitor “opraví” TP-only / SL-only stavy tým, že zruší exits a pošle nový OCO
FORCE_FIX_EXITS = os.getenv("FORCE_FIX_EXITS", "0") == "1"

# bezpečnostný prepínač (keď 0, monitor nič neposiela)
EXIT_MONITOR_ENABLED = os.getenv("EXIT_MONITOR_ENABLED", "1") != "0"

CLIENT_ID_PREFIX = os.getenv("EXIT_CLIENT_ID_PREFIX", "EXIT-OCO")

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})

dc = StockHistoricalDataClient(API_KEY, API_SECRET)

EPS = 1e-6


def _round2(x: float) -> float:
    return round(float(x) + 1e-12, 2)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _get_positions() -> List[Dict[str, Any]]:
    r = S.get(f"{ALPACA_BASE_URL}/v2/positions", timeout=15)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json()


def _get_open_orders(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&direction=desc&nested=true"
    if symbol:
        url += f"&symbols={symbol}"
    r = S.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def _cancel_order(order_id: str) -> None:
    r = S.delete(f"{ALPACA_BASE_URL}/v2/orders/{order_id}", timeout=15)
    # Alpaca vracia 204 pri OK
    if r.status_code not in (200, 204):
        r.raise_for_status()


def _cancel_exit_orders_for_symbol(symbol: str) -> int:
    """
    Zruší len OCO exit objednávky pre symbol (parent).
    Zrušenie parentu zruší aj leg.
    """
    count = 0
    for o in _get_open_orders(symbol):
        if (o.get("symbol", "").upper() == symbol.upper()) and (o.get("order_class") == "oco"):
            _cancel_order(o["id"])
            count += 1
    return count


def _latest_mid(symbol: str, fallback: float) -> float:
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        bid = getattr(q, "bid_price", None)
        ask = getattr(q, "ask_price", None)
        if bid and ask:
            return (float(bid) + float(ask)) / 2.0
    except Exception:
        pass
    return float(fallback)


def _submit_oco_exit(symbol: str, side: str, qty: float, tp: float, sl_stop: float) -> Dict[str, Any]:
    """
    OCO payload podľa Alpaca docs: type=limit, order_class=oco, take_profit + stop_loss. :contentReference[oaicite:1]{index=1}
    """
    payload = {
        "symbol": symbol,
        "side": side,                  # "sell" (exit long) alebo "buy" (exit short)
        "type": "limit",               # OCO vyžaduje limit ako TP leg
        "time_in_force": "day",
        "qty": str(_safe_float(qty)),
        "order_class": "oco",
        "client_order_id": f"{CLIENT_ID_PREFIX}-{symbol}-{int(time.time())}",
        "take_profit": {
            "limit_price": f"{_round2(tp):.2f}",
        },
        "stop_loss": {
            "stop_price": f"{_round2(sl_stop):.2f}",
        }
    }
    r = S.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json()


def _has_valid_oco(orders: List[Dict[str, Any]], abs_qty: float) -> bool:
    """
    Validný stav = existuje OCO parent a má aspoň 1 leg (stop_loss).
    Qty berieme z parenta (TP leg). Ak qty >= abs_qty => považujeme za OK.
    """
    tp_qty = 0.0
    sl_qty = 0.0
    for o in orders:
        if o.get("order_class") != "oco":
            continue
        tp_qty += _safe_float(o.get("qty", 0))
        legs = o.get("legs") or []
        for l in legs:
            # stop_loss leg
            if (l.get("type") == "stop") or (l.get("type") == "stop_limit"):
                sl_qty += _safe_float(l.get("qty", 0))
    return (tp_qty + EPS >= abs_qty) and (sl_qty + EPS >= abs_qty)


def ensure_exits_for_position(pos: Dict[str, Any]) -> None:
    symbol = pos.get("symbol", "").upper()
    qty = _safe_float(pos.get("qty"))
    if not symbol or abs(qty) < EPS:
        return
    if SYMBOLS and symbol not in SYMBOLS:
        return

    is_long = qty > 0
    abs_qty = abs(qty)

    # ak exits už existujú (OCO), nič nerob
    open_orders = _get_open_orders(symbol)
    if _has_valid_oco(open_orders, abs_qty):
        log.info("%s: exits already present (valid OCO)", symbol)
        return

    # ak je tam “rozbitý” stav (napr. TP-only), a force je zapnutý, zruš OCO/exit a pošli nové
    if FORCE_FIX_EXITS:
        canceled = _cancel_exit_orders_for_symbol(symbol)
        if canceled:
            log.warning("%s: canceled %d old OCO exit(s) to repair state", symbol, canceled)

    # anchor = mid (alebo avg_entry_price z pozície)
    avg_entry = _safe_float(pos.get("avg_entry_price"), 0.0)
    mid = _latest_mid(symbol, fallback=avg_entry if avg_entry > 0 else 1.0)

    # tp/sl z ATR_PCT
    if is_long:
        tp = _round2(mid * (1.0 + ATR_PCT * TP_ATR_MULT))
        sl = _round2(mid * (1.0 - ATR_PCT * SL_ATR_MULT))
        side = "sell"
        # ochrana: stop musí byť aspoň o 0.01 nižšie než TP base price
        if sl >= tp:
            sl = _round2(tp - 0.01)
    else:
        tp = _round2(mid * (1.0 - ATR_PCT * TP_ATR_MULT))
        sl = _round2(mid * (1.0 + ATR_PCT * SL_ATR_MULT))
        side = "buy"
        if sl <= tp:
            sl = _round2(tp + 0.01)

    log.info("%s: submit OCO exit side=%s qty=%.4f tp=%.2f sl=%.2f (mid=%.4f)", symbol, side, abs_qty, tp, sl, mid)
    res = _submit_oco_exit(symbol, side=side, qty=abs_qty, tp=tp, sl_stop=sl)
    log.info("%s: OCO submitted id=%s status=%s", symbol, res.get("id"), res.get("status"))


def main() -> None:
    if not EXIT_MONITOR_ENABLED:
        log.warning("EXIT_MONITOR_ENABLED=0 -> monitor disabled")
        while True:
            time.sleep(60)

    log.info("oco_exit_monitor start | SYMBOLS=%s | POLL=%ss | ATR_PCT=%.4f | FORCE_FIX_EXITS=%s",
             SYMBOLS, POLL_SECONDS, ATR_PCT, FORCE_FIX_EXITS)

    while True:
        try:
            positions = _get_positions()
            if not positions:
                log.info("no positions")
            else:
                for p in positions:
                    ensure_exits_for_position(p)
        except Exception as e:
            log.exception("loop error: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
