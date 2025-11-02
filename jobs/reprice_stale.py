# jobs/reprice_stale.py
import os, logging, time
from decimal import Decimal, ROUND_DOWN

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from tools.quotes import get_bid_ask_mid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reprice_stale")

def _cli():
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

def _round_px(p: float) -> float:
    return float(Decimal(p).quantize(Decimal("0.01"), rounding=ROUND_DOWN))

def main():
    cli = _cli()
    max_age = int(os.getenv("REPRICE_AFTER_SEC", "180"))
    max_tries = int(os.getenv("REPRICE_MAX_TRIES", "2"))
    slip = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

    try:
        orders = cli.get_orders()
    except Exception as e:
        log.warning("get_orders failed: %s", e)
        return

    now = time.time()
    n_repriced = 0

    for o in orders:
        status = str(getattr(o, "status", "")).lower()
        if status not in ("new", "open", "accepted", "pending_new"):
            continue

        # detect prior reprice attempts using client_order_id prefix
        coid = str(getattr(o, "client_order_id", "") or "")
        if coid.startswith("repr-"):
            try_num = int(coid.split("-")[1])
            if try_num >= max_tries:
                continue
        else:
            try_num = 0

        created_at = getattr(o, "created_at", None)
        if not created_at:
            continue
        try:
            ts = getattr(created_at, "timestamp", None)
            age = now - (ts() if callable(ts) else created_at.timestamp())
        except Exception:
            age = 0

        if age < max_age:
            continue

        sym = o.symbol
        side = str(getattr(o, "side", "buy")).lower()
        qty = float(getattr(o, "qty", 0) or 0) - float(getattr(o, "filled_qty", 0) or 0)
        if qty <= 0:
            continue

        quote = get_bid_ask_mid(sym)
        if not quote:
            # no fresh quote → skip this cycle
            continue
        bid, ask, mid = quote

        # Build a new more-marketable limit around the quote
        if side == "buy":
            new_px = max(ask, mid) + slip
        else:
            new_px = min(bid, mid) - slip
        new_px = _round_px(new_px)

        try:
            cli.cancel_order(o.id)
        except Exception as e:
            log.warning("cancel failed %s: %s", o.id, e)
            continue

        try:
            req = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                limit_price=new_px,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"repr-{try_num+1}-{int(now)}"
            )
            new_o = cli.submit_order(req)
            n_repriced += 1
            log.info("repriced %s %s qty=%.4f → limit=%.2f (try %d) id=%s",
                     sym, side, qty, new_px, try_num+1, new_o.id)
        except Exception as e:
            log.warning("reprice submit failed %s %s: %s", sym, side, e)

    log.info("reprice_stale | repriced=%d", n_repriced)

if __name__ == "__main__":
    main()
