# tools/diagnose_atr_brackets.py
from __future__ import annotations
import os, sys, math, requests
from typing import Optional, Dict, Any
from lib.atr_utils import last_atr

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex")

# ATR config (must match services/bracket_helper.py)
ATR_PERIOD         = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS  = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP        = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL        = float(os.getenv("ATR_MULT_SL", "1.0"))
MAX_TP_PCT         = float(os.getenv("MAX_TP_PCT", "0.015"))
MAX_SL_PCT         = float(os.getenv("MAX_SL_PCT", "0.015"))
TOL                = float(os.getenv("ATR_TOL", "0.10"))  # 10% tolerance vs latest ATR

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})

def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "None"
    return f"{x:.4f}"

def _fetch_open_parents(symbols: Optional[str]) -> list[Dict[str, Any]]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&direction=desc&nested=true"
    if symbols:
        url += "&symbols=" + ",".join(s.strip().upper() for s in symbols.split(",") if s.strip())
    r = S.get(url, timeout=20)
    r.raise_for_status()
    orders = r.json() or []
    # parent = bracket class and no parent_order_id
    parents = [o for o in orders if o.get("order_class") == "bracket" and not o.get("parent_order_id")]
    return parents

def _almost(a: float, b: float, eps: float = 0.01) -> bool:
    return abs(a - b) <= eps

def _diagnose_one(o: Dict[str, Any]) -> None:
    sym  = o.get("symbol")
    side = (o.get("side") or "").lower()
    tp   = o.get("take_profit", {}) or {}
    sl   = o.get("stop_loss", {}) or {}
    tp_px = tp.get("limit_price")
    sl_px = sl.get("stop_price")
    typ   = o.get("type")
    cid   = o.get("client_order_id")
    created = o.get("created_at")

    if tp_px is None or sl_px is None:
        print(f"- {sym} {side} cid={cid}: missing TP/SL fields; cannot diagnose.")
        return

    try:
        tp_px = float(tp_px)
        sl_px = float(sl_px)
    except Exception:
        print(f"- {sym} {side} cid={cid}: TP/SL not numeric; cannot diagnose.")
        return

    # Infer ATR and base price from the placed TP/SL (assuming not capped)
    if side == "buy":
        # tp = b + a*mTP  ;  sl = b - a*mSL
        a_implied = (tp_px - sl_px) / (ATR_MULT_TP + ATR_MULT_SL)
        b_from_tp = tp_px - a_implied * ATR_MULT_TP
        b_from_sl = sl_px + a_implied * ATR_MULT_SL
        b_implied = 0.5 * (b_from_tp + b_from_sl)
    else:
        # tp = b - a*mTP  ;  sl = b + a*mSL
        a_implied = (sl_px - tp_px) / (ATR_MULT_TP + ATR_MULT_SL)
        b_from_tp = tp_px + a_implied * ATR_MULT_TP
        b_from_sl = sl_px - a_implied * ATR_MULT_SL
        b_implied = 0.5 * (b_from_tp + b_from_sl)

    # Pull latest ATR for comparison
    try:
        atr_now = last_atr(sym, period=ATR_PERIOD, lookback_days=ATR_LOOKBACK_DAYS)
    except Exception as e:
        atr_now = None

    # Check if caps likely applied
    capped = False
    if side == "buy":
        tp_cap = b_implied * (1.0 + MAX_TP_PCT)
        sl_cap = b_implied * (1.0 - MAX_SL_PCT)
        if _almost(tp_px, tp_cap, eps=0.02) or _almost(sl_px, sl_cap, eps=0.02):
            capped = True
    else:
        tp_cap = b_implied * (1.0 - MAX_TP_PCT)
        sl_cap = b_implied * (1.0 + MAX_SL_PCT)
        if _almost(tp_px, tp_cap, eps=0.02) or _almost(sl_px, sl_cap, eps=0.02):
            capped = True

    verdict = "OK"
    note = ""

    if atr_now is not None and a_implied > 0:
        rel_diff = abs(a_implied - atr_now) / atr_now
        if rel_diff <= TOL:
            verdict = "OK (ATR match)"
            note = f"implied_ATR={_fmt(a_implied)} ~ latest_ATR={_fmt(atr_now)} (Δ={rel_diff*100:.1f}%)"
        else:
            if capped:
                verdict = "OK (caps)"
                note = f"caps likely in effect; implied_ATR={_fmt(a_implied)} vs latest_ATR={_fmt(atr_now)}"
            else:
                verdict = "CHECK"
                note = f"implied_ATR={_fmt(a_implied)} vs latest_ATR={_fmt(atr_now)} (Δ={rel_diff*100:.1f}%)"
    else:
        verdict = "INFO"
        note = f"could not compare to latest ATR (implied={_fmt(a_implied)}, latest={_fmt(atr_now)})"

    print(f"- {created}  {sym} {side}  parent cid={cid} type={typ}")
    print(f"  TP={tp_px:.2f}  SL={sl_px:.2f}  |  base≈{b_implied:.2f}  implied_ATR={_fmt(a_implied)}")
    print(f"  verdict: {verdict}  {note}")

def main():
    symbols = os.environ.get("SYMBOLS", "")
    try:
        parents = _fetch_open_parents(symbols)
    except Exception as e:
        print(f"ERROR: failed to fetch open orders: {e}", file=sys.stderr)
        sys.exit(2)

    if not parents:
        print("No open parent bracket orders found.")
        return

    print(f"Diagnosing {len(parents)} open parent brackets "
          f"(ATR period={ATR_PERIOD}, tp×{ATR_MULT_TP}, sl×{ATR_MULT_SL}, caps={MAX_TP_PCT*100:.2f}%/{MAX_SL_PCT*100:.2f}%, tol={TOL*100:.0f}%).")

    for p in parents:
        _diagnose_one(p)

if __name__ == "__main__":
    main()
