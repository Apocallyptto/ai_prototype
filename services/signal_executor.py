import os
import time
import math
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
from sqlalchemy import text

from tools.db import get_engine

LOG = logging.getLogger("signal_executor")
logging.basicConfig(level=logging.INFO)


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    v = (v or "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _resolve_mode_and_paper() -> tuple[str, bool]:
    """Resolve trading mode and whether to use Alpaca paper endpoint.

    Priority:
      1) TRADING_MODE=live|paper
      2) explicit ALPACA_PAPER override (backwards compatibility)
      3) default -> paper (fail-safe)
    """
    mode = (os.getenv("TRADING_MODE") or "paper").strip().lower()
    if mode not in ("live", "paper"):
        mode = "paper"

    paper = (mode != "live")

    # Back-compat / emergency override
    if os.getenv("ALPACA_PAPER") is not None:
        paper = _env_bool("ALPACA_PAPER", paper)
        mode = "paper" if paper else "live"

    return mode, paper


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _round_price(p: float) -> float:
    return round(p + 1e-9, 2)


@dataclass
class Cfg:
    poll_seconds: int
    min_strength: float
    symbols: list[str]
    portfolio_id: int
    allow_short: bool
    long_only: bool
    max_notional: float
    max_qty: float
    max_position_qty: float
    allow_add_to_position: bool
    alpaca_dedupe_minutes: int
    cancel_opposite_open_orders: bool
    max_open_positions: int
    max_open_orders: int
    daily_loss_stop_pct: float
    max_daily_loss_usd: float
    enable_daily_risk_guard: bool
    symbol_cooldown_seconds: int
    pick_ttl_seconds: int
    trade_only_when_market_open: bool
    preopen_window_seconds: int
    allow_trade_on_clock_error: bool
    trading_paused: bool
    dry_run: bool


def _load_cfg() -> Cfg:
    symbols_raw = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    return Cfg(
        poll_seconds=int(os.getenv("POLL_SECONDS", "20")),
        min_strength=float(os.getenv("MIN_STRENGTH", "0.60")),
        symbols=symbols,
        portfolio_id=int(os.getenv("PORTFOLIO_ID", "1")),
        allow_short=_env_bool("ALLOW_SHORT", False),
        long_only=_env_bool("LONG_ONLY", True),
        max_notional=float(os.getenv("MAX_NOTIONAL", "100")),
        max_qty=float(os.getenv("MAX_QTY", "1")),
        max_position_qty=float(os.getenv("MAX_POSITION_QTY", "1")),
        allow_add_to_position=_env_bool("ALLOW_ADD_TO_POSITION", False),
        alpaca_dedupe_minutes=int(os.getenv("ALPACA_DEDUPE_MINUTES", "2")),
        cancel_opposite_open_orders=_env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "1")),
        max_open_orders=int(os.getenv("MAX_OPEN_ORDERS", "1")),
        daily_loss_stop_pct=float(os.getenv("DAILY_LOSS_STOP_PCT", "1.0")),
        max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS_USD", "200.0")),
        enable_daily_risk_guard=_env_bool("ENABLE_DAILY_RISK_GUARD", True),
        symbol_cooldown_seconds=int(os.getenv("SYMBOL_COOLDOWN_SECONDS", "60")),
        pick_ttl_seconds=int(os.getenv("PICK_TTL_SECONDS", "120")),
        trade_only_when_market_open=_env_bool("TRADE_ONLY_WHEN_MARKET_OPEN", True),
        preopen_window_seconds=int(os.getenv("PREOPEN_WINDOW_SECONDS", "0")),
        allow_trade_on_clock_error=_env_bool("ALLOW_TRADE_ON_CLOCK_ERROR", False),
        trading_paused=_env_bool("TRADING_PAUSED", True),
        dry_run=_env_bool("DRY_RUN", True),
    )


def make_trading_client() -> tuple[TradingClient, str, bool, str]:
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    mode, paper = _resolve_mode_and_paper()
    base = (os.getenv("ALPACA_BASE_URL") or "").strip() or (
        "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    )

    return TradingClient(key, sec, paper=paper), mode, paper, base


def _get_clock_is_open(tc: TradingClient) -> bool:
    clk = tc.get_clock()
    return bool(getattr(clk, "is_open", False))


def _within_preopen_window(tc: TradingClient, window_seconds: int) -> bool:
    clk = tc.get_clock()
    if not getattr(clk, "next_open", None):
        return False
    now = _now_utc()
    nxt = getattr(clk, "next_open")
    # next_open from alpaca is usually ISO string or datetime
    if isinstance(nxt, str):
        try:
            nxt_dt = datetime.fromisoformat(nxt.replace("Z", "+00:00"))
        except Exception:
            return False
    else:
        nxt_dt = nxt
    delta = (nxt_dt - now).total_seconds()
    return 0 <= delta <= window_seconds


def _get_open_orders(tc: TradingClient) -> list:
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    return list(tc.get_orders(filter=req) or [])


def _get_positions(tc: TradingClient) -> list:
    return list(tc.get_all_positions() or [])


def _safe_mid_from_quotes(symbol: str) -> Optional[float]:
    # Placeholder: your project likely has quote fetching elsewhere.
    # Return None to fall back on price from signal if present.
    return None


def _dedupe_ok(engine, symbol: str, side: str, minutes: int) -> bool:
    # returns False if same symbol+side signal executed within last N minutes (DB-driven dedupe)
    with engine.begin() as con:
        r = con.execute(
            text(
                """
                SELECT created_at
                FROM orders
                WHERE symbol=:symbol AND side=:side
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            {"symbol": symbol, "side": side},
        ).fetchone()
        if not r:
            return True
        ts = r[0]
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return True
        return (_now_utc() - ts).total_seconds() > minutes * 60


def _pick_signal(engine, cfg: Cfg) -> Optional[dict]:
    # pick strongest recent signal for allowed symbols
    with engine.begin() as con:
        rows = con.execute(
            text(
                """
                SELECT id, created_at, symbol, side, strength, price
                FROM signals
                WHERE portfolio_id=:pid
                  AND symbol = ANY(:symbols)
                  AND strength >= :min_strength
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            {"pid": cfg.portfolio_id, "symbols": cfg.symbols, "min_strength": cfg.min_strength},
        ).fetchall()

    if not rows:
        return None

    r = rows[0]
    return {
        "id": r[0],
        "created_at": r[1],
        "symbol": (r[2] or "").upper(),
        "side": (r[3] or "").lower(),
        "strength": float(r[4]),
        "price": float(r[5]) if r[5] is not None else None,
    }


def _calc_qty(cfg: Cfg, price: float) -> float:
    if price <= 0:
        return 0.0
    qty = cfg.max_notional / price
    qty = min(qty, cfg.max_qty)
    qty = max(0.0, qty)
    # round down to 3 decimals for fractional
    qty = math.floor(qty * 1000) / 1000.0
    return qty


def _place_limit(tc: TradingClient, symbol: str, side: str, qty: float, limit_price: float) -> str:
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=str(_round_price(limit_price)),
        order_class=None,
        client_order_id=f"ENTRY-{symbol}-{uuid.uuid4().hex[:10]}",
    )
    o = tc.submit_order(order_data=req)
    return str(getattr(o, "id", ""))


def main() -> None:
    cfg = _load_cfg()
    engine = get_engine()

    tc, mode, paper, base = make_trading_client()

    LOG.info(
        "signal_executor starting | MODE=%s | paper=%s | base=%s | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | MAX_POSITION_QTY=%s | "
        "ALLOW_ADD_TO_POSITION=%s | ALPACA_DEDUPE_MINUTES=%s | CANCEL_OPPOSITE_OPEN_ORDERS=%s | "
        "MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | DAILY_LOSS_STOP_PCT=%s | MAX_DAILY_LOSS_USD=%s | "
        "ENABLE_DAILY_RISK_GUARD=%s | SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s | TRADE_ONLY_WHEN_MARKET_OPEN=%s | "
        "PREOPEN_WINDOW_SECONDS=%s | ALLOW_TRADE_ON_CLOCK_ERROR=%s | TRADING_PAUSED=%s | DRY_RUN=%s",
        mode, paper, base,
        cfg.min_strength,
        cfg.symbols,
        cfg.portfolio_id,
        cfg.poll_seconds,
        cfg.allow_short,
        cfg.long_only,
        cfg.max_notional,
        cfg.max_qty,
        cfg.max_position_qty,
        cfg.allow_add_to_position,
        cfg.alpaca_dedupe_minutes,
        cfg.cancel_opposite_open_orders,
        cfg.max_open_positions,
        cfg.max_open_orders,
        cfg.daily_loss_stop_pct,
        cfg.max_daily_loss_usd,
        cfg.enable_daily_risk_guard,
        cfg.symbol_cooldown_seconds,
        cfg.pick_ttl_seconds,
        cfg.trade_only_when_market_open,
        cfg.preopen_window_seconds,
        cfg.allow_trade_on_clock_error,
        cfg.trading_paused,
        cfg.dry_run,
    )

    while True:
        try:
            if cfg.trading_paused:
                LOG.info("trading_paused | sleep=%ss", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.trade_only_when_market_open:
                try:
                    is_open = _get_clock_is_open(tc)
                    if not is_open:
                        if cfg.preopen_window_seconds > 0 and _within_preopen_window(tc, cfg.preopen_window_seconds):
                            pass
                        else:
                            LOG.info("market_closed | sleep=%ss", cfg.poll_seconds)
                            time.sleep(cfg.poll_seconds)
                            continue
                except Exception as e:
                    if cfg.allow_trade_on_clock_error:
                        LOG.warning("clock_error allow_trade_on_clock_error=True | %r", e)
                    else:
                        LOG.warning("clock_error FAIL_CLOSED allow_trade_on_clock_error=False | %r", e)
                        time.sleep(cfg.poll_seconds)
                        continue

            sig = _pick_signal(engine, cfg)
            if not sig:
                time.sleep(cfg.poll_seconds)
                continue

            symbol = sig["symbol"]
            side = sig["side"]
            strength = sig["strength"]
            price = sig["price"] or _safe_mid_from_quotes(symbol)

            if side not in ("buy", "sell"):
                LOG.info("skip invalid side | %s", side)
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.long_only and side == "sell":
                LOG.info("skip short signal (long_only) | %s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            if not price or price <= 0:
                LOG.info("skip no_price | %s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            if not _dedupe_ok(engine, symbol, side, cfg.alpaca_dedupe_minutes):
                LOG.info("dedupe_skip | %s %s", symbol, side)
                time.sleep(cfg.poll_seconds)
                continue

            qty = _calc_qty(cfg, price)
            if qty <= 0:
                LOG.info("skip qty<=0 | %s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            limit_price = price
            if cfg.dry_run:
                LOG.info("DRY_RUN would_submit | %s %s qty=%s limit=%s strength=%.4f", symbol, side, qty, limit_price, strength)
            else:
                oid = _place_limit(tc, symbol, side, qty, limit_price)
                LOG.info("submitted | %s %s qty=%s limit=%s oid=%s strength=%.4f", symbol, side, qty, limit_price, oid, strength)

        except Exception as e:
            LOG.exception("loop_error: %r", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
