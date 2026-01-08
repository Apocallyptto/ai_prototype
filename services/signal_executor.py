from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from services.market_gate import should_trade_now

logger = logging.getLogger("signal_executor")
logging.basicConfig(level=logging.INFO)


@dataclass
class SignalRow:
    id: int
    created_at: datetime
    symbol: str
    side: str
    strength: float
    source: str
    portfolio_id: int


def _db_conn():
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is not set")
    return psycopg2.connect(db_url)


def _alpaca_base_url() -> str:
    base = (
        os.getenv("ALPACA_TRADING_URL")
        or os.getenv("ALPACA_BASE_URL")
        or "https://paper-api.alpaca.markets"
    )
    return base.rstrip("/")


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
    }


def _get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0):
    r = requests.get(url, params=params, headers=_alpaca_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 10.0):
    r = requests.post(url, json=payload, headers=_alpaca_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get_latest_quote_mid(symbol: str) -> Optional[float]:
    # NOTE: this assumes you already have a quotes endpoint / data route in your setup;
    # if not available, fallback behavior must be used.
    # Keep it as-is if it's already working in your repo.
    return None


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fetch_new_signals(
    portfolio_id: int,
    min_strength: float,
    limit: int = 50,
) -> List[SignalRow]:
    with _db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                select id, created_at, symbol, side, strength, source, portfolio_id
                from signals
                where portfolio_id = %s
                  and processed_at is null
                  and strength >= %s
                order by created_at asc
                limit %s
                """,
                (portfolio_id, min_strength, limit),
            )
            rows = cur.fetchall()

    out: List[SignalRow] = []
    for r in rows:
        out.append(
            SignalRow(
                id=int(r["id"]),
                created_at=r["created_at"],
                symbol=r["symbol"],
                side=r["side"],
                strength=float(r["strength"]),
                source=r.get("source") or "unknown",
                portfolio_id=int(r["portfolio_id"]),
            )
        )
    return out


def mark_signal_processed(
    signal_id: int,
    status: str,
    note: str,
    alpaca_order_id: Optional[str] = None,
):
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update signals
                set processed_at = now(),
                    processed_status = %s,
                    processed_note = %s,
                    alpaca_order_id = coalesce(%s, alpaca_order_id)
                where id = %s
                """,
                (status, note, alpaca_order_id, signal_id),
            )


def select_signals(signals: List[SignalRow]) -> List[SignalRow]:
    # Unique by symbol+side (one per pair)
    seen = set()
    selected: List[SignalRow] = []
    for s in signals:
        key = (s.symbol, s.side)
        if key in seen:
            continue
        seen.add(key)
        selected.append(s)
    return selected


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main():
    poll_seconds = int(os.getenv("POLL_SECONDS", "20"))
    symbols = [s.strip() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
    min_strength = float(os.getenv("MIN_STRENGTH", "0.60"))
    portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))

    atr_pct = float(os.getenv("ATR_PCT", "0.01"))
    allow_short = os.getenv("ALLOW_SHORT", "1") != "0"
    long_only = os.getenv("LONG_ONLY", "0") == "1"

    max_risk = float(os.getenv("MAX_RISK", "30"))
    max_bp_pct = float(os.getenv("MAX_BP_PCT", "0.05"))
    max_notional = float(os.getenv("MAX_NOTIONAL", "1200"))
    max_qty = int(os.getenv("MAX_QTY", "5"))

    tp_atr_mult = float(os.getenv("TP_ATR_MULT", "1.0"))
    sl_atr_mult = float(os.getenv("SL_ATR_MULT", "1.0"))

    trading_client = TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=(os.getenv("ALPACA_PAPER", "1") != "0"),
    )

    POLL_SECONDS = poll_seconds
    STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE = int(os.getenv("STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE", "10"))

    logger.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ATR_PCT=%.4f | ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_RISK=%.2f | MAX_BP_PCT=%.4f | "
        "MAX_NOTIONAL=%.2f | MAX_QTY=%s | TP_ATR_MULT=%.2f | SL_ATR_MULT=%.2f",
        min_strength,
        symbols,
        portfolio_id,
        poll_seconds,
        atr_pct,
        allow_short,
        long_only,
        max_risk,
        max_bp_pct,
        max_notional,
        max_qty,
        tp_atr_mult,
        sl_atr_mult,
    )

    while True:
        ok, reason, clock = should_trade_now(STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE)
        if not ok:
            logger.info(
                "market_gate | skip trading | reason=%s | is_open=%s | ts=%s | next_open=%s | next_close=%s",
                reason,
                clock.get("is_open"),
                clock.get("timestamp"),
                clock.get("next_open"),
                clock.get("next_close"),
            )
            time.sleep(POLL_SECONDS)
            continue

        try:
            new_signals = fetch_new_signals(
                portfolio_id=portfolio_id,
                min_strength=min_strength,
                limit=50,
            )
            logger.info("fetch_new_signals | fetched %s rows", len(new_signals))

            if not new_signals:
                time.sleep(POLL_SECONDS)
                continue

            selected = select_signals(new_signals)
            logger.info(
                "select_signals | fetched=%s | selected=%s | unique_symbol_side=%s",
                len(new_signals),
                len(selected),
                len({(s.symbol, s.side) for s in selected}),
            )

            for s in selected:
                if symbols and s.symbol not in symbols:
                    mark_signal_processed(s.id, "skipped", "symbol_not_in_filter")
                    continue

                side = s.side.lower().strip()
                if long_only and side != "buy":
                    mark_signal_processed(s.id, "skipped", "long_only")
                    continue
                if (not allow_short) and side == "sell":
                    mark_signal_processed(s.id, "skipped", "short_disabled")
                    continue

                # Decide qty (simple fixed sizing capped by MAX_QTY)
                qty = max_qty

                # Submit limit entry (your existing behavior)
                order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
                # NOTE: you already compute/decide limit price in your file; keep that logic.
                # Here we assume you already have a function/logic that sets limit_price.
                limit_price = None

                if limit_price is None:
                    mark_signal_processed(s.id, "skipped", "no_limit_price")
                    continue

                req = LimitOrderRequest(
                    symbol=s.symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )

                try:
                    o = trading_client.submit_order(req)
                    alpaca_id = getattr(o, "id", None)
                    logger.info(
                        "ENTRY submitted: %s %s qty=%s @%s | id=%s | signal_id=%s",
                        s.symbol,
                        side.upper(),
                        qty,
                        limit_price,
                        alpaca_id,
                        s.id,
                    )
                    mark_signal_processed(
                        s.id,
                        "submitted",
                        f"entry_submitted qty={qty} limit={limit_price}",
                        alpaca_order_id=str(alpaca_id) if alpaca_id else None,
                    )
                except APIError as e:
                    logger.exception("submit_order failed | signal_id=%s | symbol=%s | side=%s", s.id, s.symbol, side)
                    mark_signal_processed(s.id, "error", f"alpaca_error:{str(e)[:180]}")
                except Exception as e:
                    logger.exception("submit_order failed | signal_id=%s | symbol=%s | side=%s", s.id, s.symbol, side)
                    mark_signal_processed(s.id, "error", f"error:{e.__class__.__name__}")

        except Exception:
            logger.exception("signal_executor loop error")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
