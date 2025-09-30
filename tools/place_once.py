# tools/place_once.py
import os, argparse, sqlalchemy as sa, datetime as dt
from datetime import timezone
from lib.db import make_engine
from lib.broker_alpaca import place_marketable_limit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=os.environ.get("TEST_TICKER","AAPL"))
    ap.add_argument("--side", default=os.environ.get("TEST_SIDE","buy"))
    ap.add_argument("--qty", type=int, default=int(os.environ.get("TEST_QTY","1")))
    ap.add_argument("--pad-up", type=float, default=float(os.environ.get("PAD_UP","1.05")))
    ap.add_argument("--pad-down", type=float, default=float(os.environ.get("PAD_DOWN","0.95")))
    args = ap.parse_args()

    res = place_marketable_limit(
        args.ticker, args.side, args.qty,
        pad_up=args.pad_up, pad_down=args.pad_down, extended_hours=True
    )

    status_text = "submitted"
    if res["http_status"] == 200 and res["json"]:
        status_text = res["json"].get("status","submitted")

    ts_iso = dt.datetime.now(timezone.utc).isoformat()

    eng = make_engine()
    with eng.begin() as c:
        c.execute(sa.text("""
            INSERT INTO orders
                (ts, symbol_id, ticker, side, qty, order_type, limit_price, status)
            VALUES
                (:ts, (SELECT id FROM symbols WHERE ticker=:t), :t, :s, :q, 'limit', :lim, :st)
        """), {"ts": ts_iso, "t": args.ticker, "s": args.side, "q": args.qty,
               "lim": res["limit_price"], "st": status_text})

    print(f"Placed {args.side} {args.qty} {args.ticker} @ limit {res['limit_price']} | status={status_text}")

if __name__ == "__main__":
    main()
