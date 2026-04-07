#!/usr/bin/env python3
"""
sync_orders.py - keep DB orders table in sync with Alpaca orders.

Runs in a loop:
  - Pulls recent orders from Alpaca (last N days).
  - Upserts into DB.

Important:
  - `paper` endpoint selection is resolved primarily from TRADING_MODE=live|paper.
  - ALPACA_PAPER can override (backwards compatibility / emergency).

This version is hardened against startup DNS / DB races:
  - waits for DB readiness instead of crashing the container
  - reconnects if the DB connection drops mid-loop
  - rolls back failed transactions cleanly
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import psycopg2
from psycopg2 import InterfaceError, OperationalError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

LOG = logging.getLogger("sync_orders")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


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


def _connect(db_url: str, connect_timeout_seconds: int):
    return psycopg2.connect(db_url, connect_timeout=max(1, int(connect_timeout_seconds)))


def _db_ping(con) -> None:
    with con.cursor() as cur:
        cur.execute("SELECT 1")


def _close_quietly(con) -> None:
    if con is None:
        return
    try:
        con.close()
    except Exception:
        pass


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


def _ensure_db_connection(
    con,
    db_url: str,
    connect_timeout_seconds: int,
    warmup_max_attempts: int,
    warmup_sleep_seconds: float,
):
    """
    Ensure we have a live psycopg2 connection.

    warmup_max_attempts:
      - 0 or negative => retry forever
      - positive => log bounded attempt count, but continue retrying in-process
    """
    attempt = 0
    attempt_label = "∞" if warmup_max_attempts <= 0 else str(warmup_max_attempts)

    while True:
        attempt += 1
        try:
            if con is None or getattr(con, "closed", 1):
                con = _connect(db_url, connect_timeout_seconds)
                _ensure_table(con)
            else:
                _db_ping(con)
            if attempt > 1:
                LOG.info("db_ready | attempts=%s", attempt)
            return con
        except Exception as e:
            _close_quietly(con)
            con = None
            if attempt == 1 or attempt % 5 == 0:
                LOG.warning(
                    "db_wait_retry | attempt=%s/%s | sleep=%.1fs | err=%r",
                    attempt,
                    attempt_label,
                    warmup_sleep_seconds,
                    e,
                )
            # never crash the container here; keep retrying in-process
            time.sleep(max(0.1, float(warmup_sleep_seconds)))


def _upsert_order(cur, o: Any) -> bool:
    d = getattr(o, "__dict__", None) or {}
    raw = getattr(o, "_raw", None) or getattr(o, "_raw_data", None) or d
    if not isinstance(raw, dict):
        raw = d

    oid = _sval(getattr(o, "id", None) or raw.get("id"))
    if not oid:
        return False

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
    return True


def _dedupe_orders(open_orders, closed_orders) -> Dict[str, Any]:
    """
    Merge open + closed order lists by order id.
    If an id appears twice, keep the later object encountered.
    """
    out: Dict[str, Any] = {}
    for o in list(open_orders or []) + list(closed_orders or []):
        oid = _sval(getattr(o, "id", None))
        if not oid:
            continue
        out[oid] = o
    return out


def main_loop() -> None:
    poll = _env_int("SYNC_POLL_SECONDS", 30)
    lookback_days = _env_int("SYNC_LOOKBACK_DAYS", 30)
    fetch_limit = _env_int("SYNC_FETCH_LIMIT", 500)

    db_warmup_max_attempts = _env_int("SYNC_DB_WARMUP_MAX_ATTEMPTS", 0)
    db_warmup_sleep_seconds = _env_float("SYNC_DB_WARMUP_SLEEP_SECONDS", 2.0)
    connect_timeout_seconds = _env_int("SYNC_CONNECT_TIMEOUT_SECONDS", 5)

    mode = _resolve_mode()
    paper = _resolve_paper(mode)

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    tc = TradingClient(api_key, api_secret, paper=paper)

    db_url = _normalize_db_url()
    con = None

    LOG.info(
        "sync_orders starting | poll=%ss | lookback_days=%s | mode=%s | paper=%s | "
        "fetch_limit=%s | db_warmup_max_attempts=%s | db_warmup_sleep_seconds=%.1f | connect_timeout=%ss",
        poll,
        lookback_days,
        mode,
        paper,
        fetch_limit,
        db_warmup_max_attempts,
        db_warmup_sleep_seconds,
        connect_timeout_seconds,
    )

    while True:
        try:
            con = _ensure_db_connection(
                con=con,
                db_url=db_url,
                connect_timeout_seconds=connect_timeout_seconds,
                warmup_max_attempts=db_warmup_max_attempts,
                warmup_sleep_seconds=db_warmup_sleep_seconds,
            )

            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            req_open = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=fetch_limit, after=since, nested=True)
            req_closed = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=fetch_limit, after=since, nested=True)

            open_orders = tc.get_orders(filter=req_open) or []
            closed_orders = tc.get_orders(filter=req_closed) or []

            merged = _dedupe_orders(open_orders, closed_orders)

            checked = 0
            upserted = 0

            with con.cursor() as cur:
                for o in merged.values():
                    checked += 1
                    if _upsert_order(cur, o):
                        upserted += 1
            con.commit()

            LOG.info(
                "checked=%s | open=%s | closed=%s | unique=%s | upserted=%s",
                checked,
                len(open_orders),
                len(closed_orders),
                len(merged),
                upserted,
            )

        except (OperationalError, InterfaceError) as e:
            LOG.warning("sync_orders db_error | err=%r", e)
            try:
                if con is not None and not getattr(con, "closed", 1):
                    con.rollback()
            except Exception:
                pass
            _close_quietly(con)
            con = None

        except Exception as e:
            LOG.warning("sync_orders ERROR: %r", e)
            try:
                if con is not None and not getattr(con, "closed", 1):
                    con.rollback()
            except Exception:
                pass

        time.sleep(poll)


if __name__ == "__main__":
    main_loop()