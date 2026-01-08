import os
import time
import math
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import requests
from sqlalchemy import create_engine, text

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
from alpaca.trading.models import Order

from services.market_gate import should_trade_now


logger = logging.getLogger("signal_executor")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip()


def env_int(name: str, default: int = 0) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return int(float(str(v).strip()))


def env_float(name: str, default: float = 0.0) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return float(str(v).strip())


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_latest_price(symbol: str) -> Optional[float]:
    """
    Pull latest price from Alpaca Data API v2.
    Uses ALPACA_DATA_URL + ALPACA_DATA_FEED (iex/sip).
    Falls back to quote mid if trade isn't available.
    """
    data_url = env_str("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
    data_feed = env_str("ALPACA_DATA_FEED", "").strip() or None

    api_key = env_str("ALPACA_API_KEY")
    api_secret = env_str("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        logger.warning("get_latest_price | missing ALPACA_API_KEY/SECRET")
        return None

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params = {"feed": data_feed} if data_feed else None

    # 1) latest trade
    try:
        r = requests.get(
            f"{data_url}/v2/stocks/{symbol}/trades/latest",
            headers=headers,
            params=params,
            timeout=10,
        )
        if r.ok:
            j = r.json() or {}
            t = j.get("trade") or {}
            p = t.get("p")
            if p is not None:
                return float(p)
        else:
            logger.debug("get_latest_price | trade latest failed | %s | %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.debug("get_latest_price | trade latest exception | %s | %s", symbol, e)

    # 2) latest quote mid
    try:
        r = requests.get(
            f"{data_url}/v2/stocks/{symbol}/quotes/latest",
            headers=headers,
            params=params,
            timeout=10,
        )
        if r.ok:
            j = r.json() or {}
            q = j.get("quote") or {}
            bp = q.get("bp")
            ap = q.get("ap")
            if bp is not None and ap is not None:
                return (float(bp) + float(ap)) / 2.0
            if ap is not None:
                return float(ap)
            if bp is not None:
                return float(bp)
        else:
            logger.debug("get_latest_price | quote latest failed | %s | %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.debug("get_latest_price | quote latest exception | %s | %s", symbol, e)

    return None


def fetch_unprocessed_signals(engine, portfolio_id: int, symbols: List[str], min_strength: float, limit: int = 50):
    sql = text(
        """
        SELECT id, created_at, symbol, side, strength
        FROM signals
        WHERE processed_status IS NULL
          AND portfolio_id = :pid
          AND symbol = ANY(:symbols)
          AND strength >= :min_strength
        ORDER BY created_at ASC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {
                "pid": portfolio_id,
                "symbols": symbols,
                "min_strength": min_strength,
                "limit": limit,
            },
        ).mappings().all()
    logger.info("fetch_new_signals | fetched %d rows", len(rows))
    return rows


def mark_signal(engine, signal_id: int, status: str, note: str, alpaca_order_id=None) -> None:
    # Alpaca SDK may return UUID objects; DB column is TEXT in our schema.
    if alpaca_order_id is not None:
        alpaca_order_id = str(alpaca_order_id)

    # Support both styles: some older code used param name 'oid' in SQL params
    sql = text("""
        UPDATE signals
        SET processed_status = :status,
            processed_note = :note,
            processed_at = NOW(),
            alpaca_order_id = COALESCE(CAST(:oid AS text), alpaca_order_id)
        WHERE id = :id
    """)
    with engine.begin() as conn:
        conn.execute(sql, {"status": status, "note": note, "oid": alpaca_order_id, "id": signal_id})


def select_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep at most one signal per (symbol, side) in this batch. (If duplicates, keep the strongest.)
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in signals:
        key = (s["symbol"], s["side"])
        if key not in best or float(s["strength"]) > float(best[key]["strength"]):
            best[key] = s
    selected = list(best.values())
    logger.info(
        "select_signals | fetched=%d | selected=%d | unique_symbol_side=%d",
        len(signals),
        len(selected),
        len(best),
    )
    return selected


def main():
    # ---- config ----
    api_key = env_str("ALPACA_API_KEY")
    api_secret = env_str("ALPACA_API_SECRET")
    paper = env_str("ALPACA_PAPER", "1") != "0"

    symbols = [s.strip().upper() for s in env_str("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
    portfolio_id = env_int("PORTFOLIO_ID", 1)
    min_strength = env_float("MIN_STRENGTH", 0.60)

    poll_seconds = env_int("POLL_SECONDS", 20)
    stop_new_entries_min_before_close = env_int("STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE", 10)

    max_qty = env_int("MAX_QTY", 5)
    max_notional = env_float("MAX_NOTIONAL", 1200.0)

    allow_short = env_str("ALLOW_SHORT", "1") != "0"
    long_only = env_str("LONG_ONLY", "0") != "0"

    tp_atr_mult = env_float("TP_ATR_MULT", 1.0)
    sl_atr_mult = env_float("SL_ATR_MULT", 1.0)

    # DB URL + engine
    db_url = env_str("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
    # tolerate SQLAlchemy-style URLs in env (postgresql+psycopg2://...) as well as plain psycopg2 DSNs
    db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")

    engine = create_engine(db_url, pool_pre_ping=True)

    # Alpaca client
    tc = TradingClient(api_key, api_secret, paper=paper)

    logger.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | TP_ATR_MULT=%.2f | SL_ATR_MULT=%.2f",
        min_strength,
        symbols,
        portfolio_id,
        poll_seconds,
        allow_short,
        long_only,
        max_notional,
        max_qty,
        tp_atr_mult,
        sl_atr_mult,
    )

    # ---- loop ----
    while True:
        try:
            # ✅ MARKET GATE (no trades when closed / too close to close)
            ok, reason, clock = should_trade_now(stop_new_entries_min_before_close)
            if not ok:
                logger.info(
                    "market_gate | skip trading | reason=%s | is_open=%s | ts=%s | next_open=%s | next_close=%s",
                    reason,
                    clock.get("is_open"),
                    clock.get("timestamp"),
                    clock.get("next_open"),
                    clock.get("next_close"),
                )
                time.sleep(poll_seconds)
                continue

            # 1) fetch signals
            signals = fetch_unprocessed_signals(engine, portfolio_id, symbols, min_strength, limit=50)
            if not signals:
                time.sleep(poll_seconds)
                continue

            # 2) dedupe batch
            signals = select_signals(signals)

            # 3) execute each
            for s in signals:
                sid = int(s["id"])
                symbol = str(s["symbol"]).upper()
                side = str(s["side"]).lower().strip()

                if long_only and side == "sell":
                    mark_signal(engine, sid, "skipped", "long_only")
                    continue
                if (not allow_short) and side == "sell":
                    mark_signal(engine, sid, "skipped", "short_disabled")
                    continue

                # latest price for limit calc / sanity
                px = get_latest_price(symbol)
                if px is None or not math.isfinite(px) or px <= 0:
                    mark_signal(engine, sid, "skipped", "no_limit_price")
                    continue

                # sizing (simple fixed caps)
                qty = max(1, min(max_qty, int(max_notional // px)))
                if qty <= 0:
                    mark_signal(engine, sid, "skipped", "qty_zero")
                    continue

                alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

                # entry order: limit near current (simple; you can later improve with ATR)
                limit_price = round(px, 2)

                try:
                    req = LimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=limit_price,
                    )
                    o: Order = tc.submit_order(req)
                    mark_signal(engine, sid, "submitted", f"limit={limit_price}", alpaca_order_id=o.id)

                except Exception as e:
                    logger.exception("submit_order failed | %s %s qty=%s px=%s", symbol, side, qty, limit_price)
                    mark_signal(engine, sid, "error", f"submit_failed:{type(e).__name__}")

            time.sleep(poll_seconds)

        except Exception:
            logger.exception("signal_executor loop error")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
