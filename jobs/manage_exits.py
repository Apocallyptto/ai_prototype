# jobs/manage_exits.py
from __future__ import annotations
import os, json, time
from datetime import datetime, timezone
import requests

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
QTY_PER_TRADE = int(os.getenv("QTY_PER_TRADE", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL","INFO").upper()

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

def log(msg: str, level: str = "INFO"):
    order = ["DEBUG","INFO","WARN","ERROR"]
    if order.index(level) >= order.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} manage_exits | {msg}", flush=True)

def http(method: str, url: str, **kwargs) -> requests.Response:
    for i in range(4):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500: raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            time.sleep(min(2**i, 8))
            log(f"HTTP retry {i+1}: {e}", "WARN")
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

def list_positions() -> dict[str, dict]:
    r = http("GET", f"{ALPACA_BASE_URL}/v2/positions"); r.raise_for_status()
    pos = {}
    for p in r.json():
        pos[p["symbol"]] = p
    return pos

def list_open_orders(symbol: str) -> list[dict]:
    r = http("GET", f"{ALPACA_BASE_URL}/v2/orders", params={"status":"open","symbols":symbol,"nested":"true","limit":200})
    r.raise_for_status()
    return r.json()

def has_oco_exits(symbol: str) -> bool:
    for o in list_open_orders(symbol):
        if o.get("symbol") != symbol: continue
        if o.get("order_class") == "oco": return True
        if o.get("parent_order_id") and o.get("order_class") == "bracket": return True
    return False

def submit_oco_exit(symbol: str, side: str, qty: int, tp_price: float, sl_stop: float):
    payload = {
        "symbol": symbol,
        "side": side,  # 'sell' for long exits, 'buy' for short exits
        "type": "limit",
        "qty": str(qty),
        "time_in_force": "gtc",
        "order_class": "oco",
        "take_profit": {"limit_price": f"{tp_price:.2f}"},
        "stop_loss":   {"stop_price": f"{sl_stop:.2f}"},
        "extended_hours": False,
        "client_order_id": f"EXIT-{symbol}-{int(time.time())}",
    }
    r = http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        log(f"OCO submit failed: {r.status_code} {r.text}", "ERROR"); r.raise_for_status()
    return r.json()

def compute_tp_sl_for_position(p: dict) -> tuple[float,float]:
    # Simple ATR-less fallback: ±0.6% of market price (you can replace with your ATR helper)
    px = float(p.get("market_value", "0") or 0) / max(float(p.get("qty","1")), 1.0)
    if px <= 0:  # fallback to avg entry
        px = float(p.get("avg_entry_price","0") or 0)
    bump = 0.006 * px
    if float(p.get("qty", "0")) > 0:  # long
        return (px + bump, px - bump)
    else:  # short
        return (px - bump, px + bump)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated tickers")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    pos = list_positions()
    for sym in symbols:
        p = pos.get(sym)
        if not p:
            log(f"{sym}: no position; skipping.")
            continue

        # Only place exits when free quantity is available
        avail = float(p.get("qty_available", "0") or 0)
        if avail < float(QTY_PER_TRADE):
            log(f"{sym}: skip OCO (available={avail}, need>={QTY_PER_TRADE}).", "WARN")
            continue

        # Skip if exits already present (bracket children or OCO)
        if has_oco_exits(sym):
            log(f"{sym}: exit already present; skipping.")
            continue

        # Build TP/SL (placeholder; your bracket_helper has ATR logic if you want to reuse)
        tp, sl = compute_tp_sl_for_position(p)
        side = "sell" if float(p.get("qty","0")) > 0 else "buy"
        log(f"Submitting OCO exit: {{sym:{sym}, side:{side}, qty:{int(QTY_PER_TRADE)}, TP:{tp:.2f}, SL:{sl:.2f}}}")
        try:
            submit_oco_exit(sym, side, int(QTY_PER_TRADE), tp, sl)
            log(f"{sym}: OCO exits placed (TP={tp:.4f}, SL={sl:.4f}).")
        except requests.HTTPError as he:
            log(f"{sym}: failed to place exits -> {he.response.text if he.response else he}", "ERROR")

if __name__ == "__main__":
    main()
