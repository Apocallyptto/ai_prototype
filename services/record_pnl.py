import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, List

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from tools.db import get_engine

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
LOG = logging.getLogger("pnl_recorder")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _dt_utc_naive(dt: Any) -> Optional[datetime]:
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return None
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt
    try:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return dt.replace(tzinfo=None)


def _now_utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_dumps_safe(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            return None


def _make_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")

    mode = (os.getenv("TRADING_MODE") or "").strip().lower()
    if mode in ("paper", "live"):
        paper = (mode == "paper")
    else:
        paper = _env_bool("ALPACA_PAPER", True)

    return TradingClient(key, sec, paper=paper)


def _ensure_tables(engine) -> None:
    with engine.begin() as con:
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL,
            equity DOUBLE PRECISION,
            cash DOUBLE PRECISION,
            buying_power DOUBLE PRECISION,
            portfolio_value DOUBLE PRECISION,
            long_market_value DOUBLE PRECISION,
            short_market_value DOUBLE PRECISION,
            daytrade_count INTEGER,
            daytrading_buying_power DOUBLE PRECISION,
            account_status TEXT,
            raw JSONB
        );
        """))

        con.execute(text("""
        CREATE TABLE IF NOT EXISTS position_snapshots (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL,
            symbol TEXT NOT NULL,
            qty DOUBLE PRECISION NOT NULL,
            avg_entry_price DOUBLE PRECISION,
            current_price DOUBLE PRECISION,
            market_value DOUBLE PRECISION,
            unrealized_pl DOUBLE PRECISION,
            unrealized_plpc DOUBLE PRECISION
        );
        """))

        con.execute(text("""
        CREATE TABLE IF NOT EXISTS alpaca_orders (
            alpaca_order_id TEXT PRIMARY KEY,
            client_order_id TEXT,
            symbol TEXT,
            side TEXT,
            type TEXT,
            status TEXT,
            time_in_force TEXT,
            qty DOUBLE PRECISION,
            notional DOUBLE PRECISION,
            filled_qty DOUBLE PRECISION,
            filled_avg_price DOUBLE PRECISION,
            submitted_at TIMESTAMP,
            filled_at TIMESTAMP,
            updated_at TIMESTAMP,
            raw JSONB,
            recorded_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """))


def _insert_equity(engine, account: Any) -> None:
    ts = _now_utc_naive()

    raw_obj = None
    try:
        if hasattr(account, "model_dump"):
            raw_obj = account.model_dump()
        elif hasattr(account, "dict"):
            raw_obj = account.dict()
    except Exception:
        raw_obj = None

    raw_json = _json_dumps_safe(raw_obj)

    with engine.begin() as con:
        con.execute(
            text("""
            INSERT INTO equity_snapshots (
                ts, equity, cash, buying_power, portfolio_value,
                long_market_value, short_market_value,
                daytrade_count, daytrading_buying_power, account_status, raw
            )
            VALUES (
                :ts, :equity, :cash, :buying_power, :portfolio_value,
                :long_market_value, :short_market_value,
                :daytrade_count, :daytrading_buying_power, :account_status, CAST(:raw AS JSONB)
            );
            """),
            {
                "ts": ts,
                "equity": _to_float(getattr(account, "equity", None), default=None),
                "cash": _to_float(getattr(account, "cash", None), default=None),
                "buying_power": _to_float(getattr(account, "buying_power", None), default=None),
                "portfolio_value": _to_float(getattr(account, "portfolio_value", None), default=None),
                "long_market_value": _to_float(getattr(account, "long_market_value", None), default=None),
                "short_market_value": _to_float(getattr(account, "short_market_value", None), default=None),
                "daytrade_count": int(getattr(account, "daytrade_count", 0) or 0),
                "daytrading_buying_power": _to_float(getattr(account, "daytrading_buying_power", None), default=None),
                "account_status": str(getattr(account, "status", "") or ""),
                "raw": raw_json,
            },
        )


def _insert_positions(engine, positions: List[Any]) -> int:
    ts = _now_utc_naive()
    rows = []

    for p in positions or []:
        qty = _to_float(getattr(p, "qty", 0.0), default=0.0)
        if not qty:
            continue
        rows.append({
            "ts": ts,
            "symbol": str(getattr(p, "symbol", "") or "").upper(),
            "qty": qty,
            "avg_entry_price": _to_float(getattr(p, "avg_entry_price", None), default=None),
            "current_price": _to_float(getattr(p, "current_price", None), default=None),
            "market_value": _to_float(getattr(p, "market_value", None), default=None),
            "unrealized_pl": _to_float(getattr(p, "unrealized_pl", None), default=None),
            "unrealized_plpc": _to_float(getattr(p, "unrealized_plpc", None), default=None),
        })

    if not rows:
        return 0

    with engine.begin() as con:
        con.execute(
            text("""
            INSERT INTO position_snapshots (
                ts, symbol, qty, avg_entry_price, current_price, market_value, unrealized_pl, unrealized_plpc
            )
            VALUES (
                :ts, :symbol, :qty, :avg_entry_price, :current_price, :market_value, :unrealized_pl, :unrealized_plpc
            );
            """),
            rows,
        )
    return len(rows)


def _upsert_alpaca_orders(engine, orders: List[Any]) -> int:
    if not orders:
        return 0

    q = text("""
    INSERT INTO alpaca_orders (
        alpaca_order_id, client_order_id, symbol, side, type, status, time_in_force,
        qty, notional, filled_qty, filled_avg_price,
        submitted_at, filled_at, updated_at, raw
    )
    VALUES (
        :alpaca_order_id, :client_order_id, :symbol, :side, :type, :status, :time_in_force,
        :qty, :notional, :filled_qty, :filled_avg_price,
        :submitted_at, :filled_at, :updated_at, CAST(:raw AS JSONB)
    )
    ON CONFLICT (alpaca_order_id) DO UPDATE SET
        client_order_id = EXCLUDED.client_order_id,
        symbol = EXCLUDED.symbol,
        side = EXCLUDED.side,
        type = EXCLUDED.type,
        status = EXCLUDED.status,
        time_in_force = EXCLUDED.time_in_force,
        qty = EXCLUDED.qty,
        notional = EXCLUDED.notional,
        filled_qty = EXCLUDED.filled_qty,
        filled_avg_price = EXCLUDED.filled_avg_price,
        submitted_at = EXCLUDED.submitted_at,
        filled_at = EXCLUDED.filled_at,
        updated_at = EXCLUDED.updated_at,
        raw = EXCLUDED.raw;
    """)

    rows = []
    for o in orders:
        oid = str(getattr(o, "id", "") or "")
        if not oid:
            continue

        raw_obj = None
        try:
            if hasattr(o, "model_dump"):
                raw_obj = o.model_dump()
            elif hasattr(o, "dict"):
                raw_obj = o.dict()
        except Exception:
            raw_obj = None

        rows.append({
            "alpaca_order_id": oid,
            "client_order_id": str(getattr(o, "client_order_id", "") or "") or None,
            "symbol": str(getattr(o, "symbol", "") or "") or None,
            "side": str(getattr(o, "side", "") or "") or None,
            "type": str(getattr(o, "type", "") or "") or None,
            "status": str(getattr(o, "status", "") or "") or None,
            "time_in_force": str(getattr(o, "time_in_force", "") or "") or None,
            "qty": _to_float(getattr(o, "qty", None), default=None),
            "notional": _to_float(getattr(o, "notional", None), default=None),
            "filled_qty": _to_float(getattr(o, "filled_qty", None), default=None),
            "filled_avg_price": _to_float(getattr(o, "filled_avg_price", None), default=None),
            "submitted_at": _dt_utc_naive(getattr(o, "submitted_at", None)),
            "filled_at": _dt_utc_naive(getattr(o, "filled_at", None)),
            "updated_at": _dt_utc_naive(getattr(o, "updated_at", None)),
            "raw": _json_dumps_safe(raw_obj),
        })

    if not rows:
        return 0

    with engine.begin() as con:
        con.execute(q, rows)

    return len(rows)


def _wait_for_db(engine, sleep_seconds: int) -> None:
    while True:
        try:
            with engine.connect() as con:
                con.execute(text("SELECT 1"))
            return
        except OperationalError as e:
            LOG.warning("db_not_ready (will retry) | err=%r", e)
            time.sleep(sleep_seconds)
        except Exception as e:
            # includes DNS resolution errors surfaced via SQLAlchemy/psycopg
            LOG.warning("db_not_ready (will retry) | err=%r", e)
            time.sleep(sleep_seconds)


def main() -> None:
    poll = _env_int("PNL_POLL_SECONDS", 300)
    orders_lookback_hours = _env_int("PNL_ORDERS_LOOKBACK_HOURS", 48)
    record_positions = _env_bool("PNL_RECORD_POSITIONS", True)
    db_retry_seconds = _env_int("DB_RETRY_SECONDS", 5)

    tc = _make_trading_client()
    engine = get_engine()

    # NEW: wait for DB/DNS to be ready instead of crashing
    _wait_for_db(engine, db_retry_seconds)
    _ensure_tables(engine)

    LOG.info(
        "pnl_recorder starting | poll=%ss | orders_lookback_hours=%s | record_positions=%s | db_retry=%ss",
        poll, orders_lookback_hours, record_positions, db_retry_seconds
    )

    while True:
        try:
            # if DB becomes temporarily unavailable mid-run, just retry next cycle
            acct = tc.get_account()
            _insert_equity(engine, acct)

            npos = 0
            if record_positions:
                pos = tc.get_all_positions() or []
                npos = _insert_positions(engine, pos)

            after_dt = datetime.now(timezone.utc) - timedelta(hours=orders_lookback_hours)
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500,
                nested=True,
                after=after_dt,
            )
            closed = tc.get_orders(filter=req) or []
            norders = _upsert_alpaca_orders(engine, list(closed))

            LOG.info(
                "recorded | equity=%.2f cash=%.2f daytrade_count=%s | positions_rows=%s | upsert_alpaca_orders=%s",
                (_to_float(getattr(acct, "equity", 0.0), default=0.0) or 0.0),
                (_to_float(getattr(acct, "cash", 0.0), default=0.0) or 0.0),
                getattr(acct, "daytrade_count", None),
                npos,
                norders,
            )

        except Exception as e:
            LOG.exception("loop_error: %r", e)

        time.sleep(poll)


if __name__ == "__main__":
    main()