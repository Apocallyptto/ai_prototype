#!/usr/bin/env python3
"""
sync_orders.py - keep DB orders table in sync with Alpaca orders.

Runs in a loop:
  - Pulls recent orders from Alpaca (last N days).
  - Upserts into DB.

Important:
  - `paper` endpoint selection is resolved primarily from TRADING_MODE=live|paper.
  - ALPACA_PAPER can override (backwards compatibility / emergency).
"""
import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

LOG = logging.getLogger("sync_orders")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _normalize_db_url() -> str:
    raw_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or ""
    raw_url = raw_url.strip()
    if not raw_url:
        raise RuntimeError("DB_URL / DATABASE_URL not set")

    # normalize SQLAlchemy-style URLs to plain postgresql:// for psycopg2
    if raw_url.startswith("postgresql+psycopg2://"):
        raw_url = raw_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    if raw_url.startswith("postgresql+psycopg://"):
        raw_url = raw_url.replace("postgresql+psycopg://", "postgresql://", 1)

    return raw_url


def _resolve_mode() -> str:
    """Resolve trading mode (live|paper).

    Priority:
      1) TRADING_MODE
      2) infer from ALPACA_BASE_URL (contains 'paper' -> paper, else live)
      3) default -> paper (safer)
    """
    mode = (os.getenv("TRADING_MODE") or "").strip().lower()
    if mode in ("live", "paper"):
        return mode

    base = (os.getenv("ALPACA_BASE_URL") or "").strip().lower()
    if "paper" in base:
        return "paper"
    if base:
        return "live"

    return "paper"


def _resolve_paper(mode: str) -> bool:
    """Resolve whether to use Alpaca paper endpoint, with ALPACA_PAPER override."""
    if os.getenv("ALPACA_PAPER") is not None:
        v = (os.getenv("ALPACA_PAPER") or "").strip().lower()
        return v not in ("0", "false", "no", "off")
    return mode != "live"


def _sval(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _j(v: Any) -> str:
    try:
        return json.dumps(v, default=str)
    except Exception:
        return json.dumps({"_repr": repr(v)})


def _connect(db_url: str):
    return psycopg2.connect(db_url)


def _ensure_table(con):
    with con.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
              id TEXT PRIMARY KEY,
              created_at TIMESTAMPTZ,
              updated_at TIMESTAMPTZ,
              submitted_at TIMESTAMPTZ,
              filled_at TIMESTAMPTZ,
              expired_at TIMESTAMPTZ,
              canceled_at TIMESTAMPTZ,
              failed_at TIMESTAMPTZ,
              replaced_at TIMESTAMPTZ,
              replaced_by TEXT,
              replaces TEXT,
              asset_id TEXT,
              symbol TEXT,
              asset_class TEXT,
              qty TEXT,
              filled_qty TEXT,
              side TEXT,
              type TEXT,
              time_in_force TEXT,
              limit_price TEXT,
              stop_price TEXT,
              status TEXT,
              extended_hours BOOLEAN,
              client_order_id TEXT,
              order_class TEXT,
              raw_json JSONB
            );
            """
        )
    con.commit()


def _upsert_order(con, o: Any) -> None:
    # Alpaca SDK objects are usually dataclasses / have __dict__-like access via vars()
    d = getattr(o, "__dict__", None) or {}
    # Some versions wrap in ._raw or ._raw_data
    raw = getattr(o, "_raw", None) or getattr(o, "_raw_data", None) or d
    if not isinstance(raw, dict):
        raw = d

    oid = _sval(getattr(o, "id", None) or raw.get("id"))
    if not oid:
        return

    def ts(name: str) -> Optional[str]:
        v = getattr(o, name, None)
        if v is None and isinstance(raw, dict):
            v = raw.get(name)
        if v is None:
            return None
        return str(v)

    cols = {
        "id": oid,
        "created_at": ts("created_at"),
        "updated_at": ts("updated_at"),
        "submitted_at": ts("submitted_at"),
        "filled_at": ts("filled_at"),
        "expired_at": ts("expired_at"),
        "canceled_at": ts("canceled_at"),
        "failed_at": ts("failed_at"),
        "replaced_at": ts("replaced_at"),
        "replaced_by": _sval(getattr(o, "replaced_by", None) or raw.get("replaced_by")),
        "replaces": _sval(getattr(o, "replaces", None) or raw.get("replaces")),
        "asset_id": _sval(getattr(o, "asset_id", None) or raw.get("asset_id")),
        "symbol": _sval(getattr(o, "symbol", None) or raw.get("symbol")),
        "asset_class": _sval(getattr(o, "asset_class", None) or raw.get("asset_class")),
        "qty": _sval(getattr(o, "qty", None) or raw.get("qty")),
        "filled_qty": _sval(getattr(o, "filled_qty", None) or raw.get("filled_qty")),
        "side": _sval(getattr(o, "side", None) or raw.get("side")),
        "type": _sval(getattr(o, "type", None) or raw.get("type")),
        "time_in_force": _sval(getattr(o, "time_in_force", None) or raw.get("time_in_force")),
        "limit_price": _sval(getattr(o, "limit_price", None) or raw.get("limit_price")),
        "stop_price": _sval(getattr(o, "stop_price", None) or raw.get("stop_price")),
        "status": _sval(getattr(o, "status", None) or raw.get("status")),
        "extended_hours": bool(getattr(o, "extended_hours", None) or raw.get("extended_hours") or False),
        "client_order_id": _sval(getattr(o, "client_order_id", None) or raw.get("client_order_id")),
        "order_class": _sval(getattr(o, "order_class", None) or raw.get("order_class")),
        "raw_json": _j(raw),
    }

    with con.cursor() as cur:
        cur.execute(
            """
            INSERT INTO orders (
              id, created_at, updated_at, submitted_at, filled_at, expired_at, canceled_at, failed_at,
              replaced_at, replaced_by, replaces, asset_id, symbol, asset_class, qty, filled_qty,
              side, type, time_in_force, limit_price, stop_price, status, extended_hours,
              client_order_id, order_class, raw_json
            )
            VALUES (
              %(id)s, %(created_at)s, %(updated_at)s, %(submitted_at)s, %(filled_at)s, %(expired_at)s, %(canceled_at)s, %(failed_at)s,
              %(replaced_at)s, %(replaced_by)s, %(replaces)s, %(asset_id)s, %(symbol)s, %(asset_class)s, %(qty)s, %(filled_qty)s,
              %(side)s, %(type)s, %(time_in_force)s, %(limit_price)s, %(stop_price)s, %(status)s, %(extended_hours)s,
              %(client_order_id)s, %(order_class)s, %(raw_json)s::jsonb
            )
            ON CONFLICT (id) DO UPDATE SET
              created_at=EXCLUDED.created_at,
              updated_at=EXCLUDED.updated_at,
              submitted_at=EXCLUDED.submitted_at,
              filled_at=EXCLUDED.filled_at,
              expired_at=EXCLUDED.expired_at,
              canceled_at=EXCLUDED.canceled_at,
              failed_at=EXCLUDED.failed_at,
              replaced_at=EXCLUDED.replaced_at,
              replaced_by=EXCLUDED.replaced_by,
              replaces=EXCLUDED.replaces,
              asset_id=EXCLUDED.asset_id,
              symbol=EXCLUDED.symbol,
              asset_class=EXCLUDED.asset_class,
              qty=EXCLUDED.qty,
              filled_qty=EXCLUDED.filled_qty,
              side=EXCLUDED.side,
              type=EXCLUDED.type,
              time_in_force=EXCLUDED.time_in_force,
              limit_price=EXCLUDED.limit_price,
              stop_price=EXCLUDED.stop_price,
              status=EXCLUDED.status,
              extended_hours=EXCLUDED.extended_hours,
              client_order_id=EXCLUDED.client_order_id,
              order_class=EXCLUDED.order_class,
              raw_json=EXCLUDED.raw_json
            ;
            """,
            cols,
        )
    con.commit()


def main_loop() -> None:
    poll = int(os.getenv("SYNC_POLL_SECONDS", "30"))
    lookback_days = int(os.getenv("SYNC_LOOKBACK_DAYS", "30"))

    mode = _resolve_mode()
    paper = _resolve_paper(mode)

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    tc = TradingClient(api_key, api_secret, paper=paper)

    db_url = _normalize_db_url()
    con = _connect(db_url)
    _ensure_table(con)

    print(f"sync_orders starting | poll={poll}s | lookback_days={lookback_days} | mode={mode} | paper={paper}")

    while True:
        try:
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            # Pull orders (OPEN + CLOSED) since lookback
            req_open = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, after=since)
            req_closed = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500, after=since)

            open_orders = tc.get_orders(filter=req_open) or []
            closed_orders = tc.get_orders(filter=req_closed) or []

            orders = list(open_orders) + list(closed_orders)

            checked = 0
            submitted = 0

            for o in orders:
                checked += 1
                _upsert_order(con, o)
                submitted += 1

            print(f"checked={checked} submitted | updated=0 | not_found_in_cache={checked}")

        except Exception as e:
            print(f"sync_orders ERROR: {e!r}")

        time.sleep(poll)


if __name__ == "__main__":
    main_loop()
