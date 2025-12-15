# tools/sync_open_orders.py
import os
import requests
import sqlalchemy as sa
from lib.db import make_engine

BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
K = os.environ["ALPACA_API_KEY"]
S = os.environ["ALPACA_API_SECRET"]

HDR = {
    "APCA-API-KEY-ID": K,
    "APCA-API-SECRET-KEY": S,
    "accept": "application/json",
}


def main():
    eng = make_engine()
    r = requests.get(f"{BASE}/v2/orders?status=open&limit=200", headers=HDR, timeout=20)
    r.raise_for_status()
    opens = r.json() if isinstance(r.json(), list) else []

    updated = 0
    with eng.begin() as c:
        for o in opens:
            sym = o.get("symbol")
            st = o.get("status", "open")
            lim = float(o["limit_price"]) if o.get("limit_price") else None

            res = c.execute(
                sa.text(
                    """
                    UPDATE orders
                    SET status=%(st)s, limit_price=%(lim)s
                    WHERE symbol=%(t)s
                    AND status IN ('pending_new','submitted')
                    AND abs(extract(epoch from (now()-ts))) < 86400
                    """
                ),
                {"t": sym, "st": st, "lim": lim},
            )
            updated += res.rowcount or 0

    print(f"updated {updated} rows from {len(opens)} open Alpaca orders")


if __name__ == "__main__":
    main()
