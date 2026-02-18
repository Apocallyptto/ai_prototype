#!/usr/bin/env python3
"""sync_orders: sync Alpaca order states into Postgres.

- Reads signals where processed_status='submitted'
- Pulls recent Alpaca orders (after LOOKBACK_DAYS) and matches by alpaca_order_id
- When an Alpaca order reaches terminal status, updates processed_status/processed_note/processed_at

Env:
  DB_URL or DATABASE_URL (optional; defaults to local docker postgres)
  ALPACA_API_KEY / ALPACA_API_SECRET (required)
  TRADING_MODE ('paper' or 'live'; default 'paper')          # preferred
  ALPACA_PAPER (legacy bool; 0/false/no => live; else paper) # fallback
  ALPACA_BASE_URL (optional hint; if contains 'paper' => paper; else live)
  LOOKBACK_DAYS (default 30)
  SYNC_POLL_SECONDS (default 30)
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

import psycopg2

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def _normalize_db_url() -> str:
    raw_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@postgres:5432/trader"
    # Allow SQLAlchemy-style URLs as well; psycopg2 expects plain postgresql://
    if raw_url.startswith("postgresql+psycopg://"):
        raw_url = raw_url.replace("postgresql+psycopg://", "postgresql://", 1)
    if raw_url.startswith("postgresql+psycopg2://"):
        raw_url = raw_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    # legacy
    if raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql://", 1)
    return raw_url


def _sval(x) -> str:
    if x is None:
        return ""
    return str(x)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    v = (v or "").strip().lower()
    return v not in ("0", "false", "no")


def _infer_mode_and_paper() -> tuple[str, bool]:
    """Resolve mode/paper with safe precedence.

    Priority:
      1) TRADING_MODE: 'live' or 'paper'
      2) ALPACA_PAPER: 0/false/no => live, otherwise paper
      3) ALPACA_BASE_URL: contains 'paper' => paper, otherwise live
      4) default => paper (fail-safe)
    """
    mode = (os.getenv("TRADING_MODE") or "").strip().lower()
    if mode in ("live", "paper"):
        return mode, (mode != "live")

    if os.getenv("ALPACA_PAPER") not in (None, ""):
        paper = _env_bool("ALPACA_PAPER", True)
        return ("paper" if paper else "live"), paper

    base = (os.getenv("ALPACA_BASE_URL") or "").strip().lower()
    if base:
        paper = "paper" in base
        return ("paper" if paper else "live"), paper

    return "paper", True


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_terminal(status: str) -> bool:
    s = (status or "").lower()
    return s in ("filled", "canceled", "expired", "rejected", "replaced")


def main_loop() -> None:
    poll = int(os.getenv("SYNC_POLL_SECONDS", "30"))
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "30"))

    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    mode, paper = _infer_mode_and_paper()

    print(f"sync_orders starting | poll={poll}s | lookback_days={lookback_days} | mode={mode} | paper={paper}")

    tc = TradingClient(api_key, api_secret, paper=paper)

    db_url = _normalize_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = True

    while True:
        try:
            # Find submitted signals we need to finalize
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, alpaca_order_id
                    FROM signals
                    WHERE processed_status='submitted'
                    ORDER BY id DESC
                    LIMIT 500
                    """
                )
                rows = cur.fetchall()

            if not rows:
                time.sleep(poll)
                continue

            # Fetch recent orders from Alpaca and index by id
            after = (_utcnow() - timedelta(days=lookback_days)).isoformat()
            req = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=500,
                nested=True,
                after=after,
            )
            orders = tc.get_orders(req) or []
            orders_by_id: Dict[str, Any] = {str(getattr(o, "id", "")): o for o in orders if getattr(o, "id", None)}

            checked = 0
            updated = 0
            not_found = 0

            for sid, alpaca_order_id in rows:
                checked += 1
                oid = _sval(alpaca_order_id)
                if not oid:
                    continue

                o = orders_by_id.get(oid)
                if not o:
                    not_found += 1
                    continue

                status = _sval(getattr(o, "status", ""))
                if not _is_terminal(status):
                    continue

                note = f"alpaca_status={status}"

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE signals
                        SET processed_status=%s,
                            processed_note=%s,
                            processed_at=NOW()
                        WHERE id=%s
                        """,
                        (status, note, sid),
                    )
                updated += 1

            print(f"checked={checked} submitted | updated={updated} | not_found_in_cache={not_found}")

        except Exception as e:
            print(f"sync_orders ERROR: {e!r}")

        time.sleep(poll)


if __name__ == "__main__":
    main_loop()
