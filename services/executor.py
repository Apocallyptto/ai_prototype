# services/executor.py
from __future__ import annotations
import os, json
import sqlalchemy as sa
from datetime import datetime, timezone

DB_URL = (
    f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT','5432')}/{os.environ['DB_NAME']}"
    f"?sslmode=require&channel_binding=require"
)
ENGINE = sa.create_engine(DB_URL, pool_pre_ping=True)

MODEL = "ma_cross_v1"
TIMEFRAME = "1d"

MAX_OPEN_POSITIONS = 10   # naive portfolio cap
QTY_PER_TRADE = 1         # 1 share per trade for now

def _net_position(conn, symbol_id: int) -> int:
    # +qty for buys, -qty for sells on filled orders
    q = sa.text("""
        select coalesce(sum(case when side='buy' then qty else -qty end),0) as pos
        from orders where symbol_id=:sid and status='filled'
    """)
    return int(conn.execute(q, {"sid": symbol_id}).scalar_one())

def _open_positions_count(conn) -> int:
    # count symbols with net position > 0
    q = sa.text("""
        with p as (
          select symbol_id,
                 sum(case when side='buy' then qty else -qty end) as pos
          from orders where status='filled'
          group by symbol_id
        )
        select count(*) from p where pos <> 0
    """)
    return int(conn.execute(q).scalar_one())

def main():
    with ENGINE.begin() as conn:
        # latest signal per symbol for our model/timeframe
        latest = conn.execute(sa.text(f"""
            select distinct on (s.symbol_id)
                   s.symbol_id, sym.ticker, s.ts, s.signal
            from signals s
            join symbols sym on sym.id = s.symbol_id
            where s.model = :model and s.timeframe = :tf
            order by s.symbol_id, s.ts desc
        """), {"model": MODEL, "tf": TIMEFRAME}).fetchall()

        open_count = _open_positions_count(conn)

        ins_order = sa.text("""
            insert into orders(ts, symbol_id, side, qty, type, status, meta, filled_at)
            values (:ts, :sid, :side, :qty, 'market', 'filled', :meta::jsonb, :filled_at)
        """)

        placed = 0
        for sid, ticker, ts, payload in latest:
            sig = payload if isinstance(payload, dict) else json.loads(payload)
            side = sig.get("side")
            if side not in ("buy", "sell"):
                continue

            pos = _net_position(conn, sid)

            # very simple policy:
            # - buy only if pos <= 0 and we have capacity
            # - sell only if pos >= 0 (flip or flatten)
            do_buy = side == "buy" and pos <= 0 and open_count < MAX_OPEN_POSITIONS
            do_sell = side == "sell" and pos >= 0 and open_count > -MAX_OPEN_POSITIONS  # symmetric

            if not (do_buy or do_sell):
                continue

            qty = QTY_PER_TRADE
            meta = {"source": "paper", "model": MODEL, "signal": sig}
            conn.execute(ins_order, {
                "ts": datetime.now(timezone.utc),
                "sid": sid,
                "side": "buy" if do_buy else "sell",
                "qty": qty,
                "meta": json.dumps(meta),
                "filled_at": datetime.now(timezone.utc),
            })
            # update open_count if we opened a new long
            if do_buy and pos == 0:
                open_count += 1
            if do_sell and pos > 0 and pos - qty <= 0:
                open_count -= 1
            placed += 1

        print(f"Placed {placed} paper orders.")

if __name__ == "__main__":
    main()
d