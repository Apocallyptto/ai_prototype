#!/usr/bin/env python3
"""
tools/sync_orders.py

Reconcile local DB 'signals' table with Alpaca order states.

What it does:
- Repeats every SYNC_POLL seconds (default 30).
- Finds signals where processed_status='submitted'.
- Fetches Alpaca open+closed orders (limit 500 each).
- If Alpaca order is terminal (filled/canceled/expired/rejected/replaced),
  updates processed_status/processed_note/processed_at.

Env:
  DB_URL or DATABASE_URL:
    Examples:
      postgresql://postgres:postgres@postgres:5432/trader
      postgresql+psycopg://postgres:postgres@postgres:5432/trader
      postgresql+psycopg2://postgres:postgres@postgres:5432/trader
    (we normalize these for psycopg2.connect)

  ALPACA_API_KEY / ALPACA_API_SECRET (required)

  TRADING_MODE: "paper" (default) or "live"
  ALPACA_PAPER: optional legacy override ("1"/"0", "true"/"false") if TRADING_MODE not set

  LOOKBACK_DAYS (default 30)  # only used as a soft filter / caching boundary
  SYNC_POLL (default 30)
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import psycopg2

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _resolve_trading_mode() -> str:
    """
    Primary: TRADING_MODE = live|paper
    Fallback: ALPACA_PAPER (legacy)
    Default: paper
    """
    m = (os.getenv("TRADING_MODE") or "").strip().lower()
    if m in ("live", "paper"):
        return m

    # legacy fallback
    if os.getenv("ALPACA_PAPER") is not None:
        return "paper" if _env_bool("ALPACA_PAPER", True) else "live"

    return "paper"


def _normalize_db_url_for_psycopg2() -> str:
    """
    psycopg2.connect accepts postgresql://... but NOT postgresql+psycopg://...
    Normalize common SQLAlchemy-style URLs to plain postgresql://...
    """
    raw = (
        os.getenv("DATABASE_URL")
        or os.getenv("DB_URL")
        or "postgresql://postgres:postgres@postgres:5432/trader"
    ).strip()

    # Normalize SQLAlchemy driver selectors -> plain postgresql:// for psycopg2.connect
    if raw.startswith("postgresql+psycopg2://"):
        raw = raw.replace("postgresql+psycopg2://", "postgresql://", 1)
    if raw.startswith("postgresql+psycopg://"):
        raw = raw.replace("postgresql+psycopg://", "postgresql://", 1)

    return raw


def _sval(x) -> str:
    """
    Alpaca SDK often returns enums for fields like status/order_class.
    Convert them to their string 'value' (e.g. OrderStatus.FILLED -> 'filled').
    """
    if x is None:
        return ""
    v = getattr(x, "value", None)
    if v is not None:
        return str(v)
    return str(x)


def _load_columns(cur) -> set[str]:
    cur.execute(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_name='signals';
        """
    )
    return {r[0] for r in cur.fetchall()}


def _alpaca_to_local(status: str) -> tuple[str | None, str | None]:
    s = (status or "").strip().lower()

    # Defensive fallback if enum string looks like "orderstatus.filled"
    if "." in s:
        s = s.split(".")[-1]

    if s in ("new", "accepted", "partially_filled", "held"):
        return (None, None)  # keep submitted
    if s == "filled":
        return ("filled", "order filled")
    if s in ("canceled", "expired"):
        return ("skipped", f"broker {s}")
    if s in ("rejected", "replaced"):
        return ("error", f"broker {s}")
    return (None, None)


def main_loop() -> None:
    lookback_days = _env_int("LOOKBACK_DAYS", 30)
    poll = _env_int("SYNC_POLL", 30)

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET")
        sys.exit(2)

    mode = _resolve_trading_mode()
    paper = (mode != "live")

    tc = TradingClient(api_key, api_secret, paper=paper)
    db_url = _normalize_db_url_for_psycopg2()

    print(
        f"sync_orders starting | poll={poll}s | lookback_days={lookback_days} | mode={mode} | paper={paper}"
    )

    while True:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            open_orders = tc.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
            ) or []
            closed_orders = tc.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500)
            ) or []
            all_orders = list(open_orders) + list(closed_orders)

            # Index orders by id
            orders_by_id: dict[str, object] = {}
            for o in all_orders:
                oid = _sval(getattr(o, "id", None)).strip()
                if not oid:
                    continue

                # Soft recency check (do not exclude—just defensive)
                sub_at = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
                if sub_at is not None:
                    try:
                        if getattr(sub_at, "tzinfo", None) is None:
                            sub_at = sub_at.replace(tzinfo=timezone.utc)
                        sub_at = sub_at.astimezone(timezone.utc)
                        _ = (sub_at >= cutoff)  # no-op; keep for future filtering if needed
                    except Exception:
                        pass

                orders_by_id[oid] = o

            checked = 0
            updated = 0
            not_found = 0

            with psycopg2.connect(db_url) as conn:
                conn.autocommit = False
                with conn.cursor() as cur:
                    cols = _load_columns(cur)
                    required = {
                        "id",
                        "created_at",
                        "symbol",
                        "side",
                        "processed_status",
                        "alpaca_order_id",
                        "processed_at",
                        "processed_note",
                    }
                    missing = required - cols
                    if missing:
                        print(f"ERROR: signals schema missing columns: {sorted(list(missing))}")
                        conn.rollback()
                        time.sleep(poll)
                        continue

                    cur.execute(
                        """
                        SELECT id, created_at, symbol, side, processed_status, alpaca_order_id, processed_note
                          FROM signals
                         WHERE processed_status='submitted'
                      ORDER BY id DESC;
                        """
                    )
                    rows = cur.fetchall()
                    checked = len(rows)

                    for sid, created_at, sym, side, st, oid, note in rows:
                        oid = (oid or "").strip()
                        if not oid:
                            not_found += 1
                            continue

                        o = orders_by_id.get(oid)
                        if o is None:
                            not_found += 1
                            continue

                        alp_status = _sval(getattr(o, "status", None)).strip().lower()
                        new_status, reason = _alpaca_to_local(alp_status)
                        if not new_status:
                            continue

                        cur.execute(
                            """
                            UPDATE signals
                               SET processed_status=%s,
                                   processed_at=NOW(),
                                   processed_note=%s
                             WHERE id=%s;
                            """,
                            (new_status, reason, sid),
                        )
                        updated += 1

                    conn.commit()

            print(
                f"checked={checked} submitted | updated={updated} | not_found_in_cache={not_found}"
            )

        except Exception as e:
            print("sync_orders ERROR:", repr(e))

        time.sleep(poll)


if __name__ == "__main__":
    main_loop()
