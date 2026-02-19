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


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_list(key: str, default: str) -> list[str]:
    s = os.getenv(key, default)
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]


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
    poll_seconds = _env_int("POLL_SECONDS", 20)
    min_strength = _env_float("MIN_STRENGTH", 0.6)
    symbols = _env_list("SYMBOLS", "AAPL,MSFT,SPY")
    portfolio_id = _env_int("PORTFOLIO_ID", 1)

    allow_short = _env_bool("ALLOW_SHORT", False)
    long_only = _env_bool("LONG_ONLY", True)

    max_notional = _env_float("MAX_NOTIONAL", 100.0)
    max_qty = _env_float("MAX_QTY", 1.0)
    max_position_qty = _env_float("MAX_POSITION_QTY", 1.0)
    allow_add_to_position = _env_bool("ALLOW_ADD_TO_POSITION", False)

    alpaca_dedupe_minutes = _env_int("ALPACA_DEDUPE_MINUTES", 2)
    cancel_opposite_open_orders = _env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True)

    max_open_positions = _env_int("MAX_OPEN_POSITIONS", 1)
    max_open_orders = _env_int("MAX_OPEN_ORDERS", 1)

    daily_loss_stop_pct = _env_float("DAILY_LOSS_STOP_PCT", 1.0)
    max_daily_loss_usd = _env_float("MAX_DAILY_LOSS_USD", 200.0)
    enable_daily_risk_guard = _env_bool("ENABLE_DAILY_RISK_GUARD", True)

    symbol_cooldown_seconds = _env_int("SYMBOL_COOLDOWN_SECONDS", 60)
    pick_ttl_seconds = _env_int("PICK_TTL_SECONDS", 120)

    trade_only_when_market_open = _env_bool("TRADE_ONLY_WHEN_MARKET_OPEN", True)
    preopen_window_seconds = _env_int("PREOPEN_WINDOW_SECONDS", 0)
    allow_trade_on_clock_error = _env_bool("ALLOW_TRADE_ON_CLOCK_ERROR", False)

    trading_paused = _env_bool("TRADING_PAUSED", True)
    dry_run = _env_bool("DRY_RUN", True)

    return Cfg(
        poll_seconds=poll_seconds,
        min_strength=min_strength,
        symbols=symbols,
        portfolio_id=portfolio_id,
        allow_short=allow_short,
        long_only=long_only,
        max_notional=max_notional,
        max_qty=max_qty,
        max_position_qty=max_position_qty,
        allow_add_to_position=allow_add_to_position,
        alpaca_dedupe_minutes=alpaca_dedupe_minutes,
        cancel_opposite_open_orders=cancel_opposite_open_orders,
        max_open_positions=max_open_positions,
        max_open_orders=max_open_orders,
        daily_loss_stop_pct=daily_loss_stop_pct,
        max_daily_loss_usd=max_daily_loss_usd,
        enable_daily_risk_guard=enable_daily_risk_guard,
        symbol_cooldown_seconds=symbol_cooldown_seconds,
        pick_ttl_seconds=pick_ttl_seconds,
        trade_only_when_market_open=trade_only_when_market_open,
        preopen_window_seconds=preopen_window_seconds,
        allow_trade_on_clock_error=allow_trade_on_clock_error,
        trading_paused=trading_paused,
        dry_run=dry_run,
    )


def _alpaca_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY") or ""
    sec = os.getenv("ALPACA_API_SECRET") or ""
    paper = _env_bool("ALPACA_PAPER", False)
    return TradingClient(key, sec, paper=paper)


def _pick_signal(engine, cfg: Cfg) -> Optional[dict]:
    # pick strongest recent signal for allowed symbols
    with engine.begin() as con:
        rows = con.execute(
            text(
                """
                SELECT id, created_at, symbol, side, strength, price
                FROM signals
                WHERE portfolio_id=:pid
                  AND symbol = ANY(CAST(:symbols AS text[]))
                  AND strength >= :min_strength
                  AND created_at >= (NOW() - (:ttl * INTERVAL '1 second'))
                ORDER BY strength DESC, created_at DESC
                LIMIT 1
                """
            ),
            {
                "pid": cfg.portfolio_id,
                "symbols": list(cfg.symbols),
                "min_strength": cfg.min_strength,
                "ttl": cfg.pick_ttl_seconds,
            },
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


def _normalize_side(side: str) -> Optional[OrderSide]:
    s = (side or "").strip().lower()
    if s in ("buy", "long"):
        return OrderSide.BUY
    if s in ("sell", "short"):
        return OrderSide.SELL
    return None


def _calc_qty(cfg: Cfg, price: float) -> float:
    if price <= 0:
        return 0.0
    qty_by_notional = cfg.max_notional / price
    qty = min(cfg.max_qty, qty_by_notional)
    qty = max(0.0, qty)
    # keep at least 1 share if rounding makes it zero, but only if qty>0
    if 0 < qty < 1.0:
        qty = 1.0
    return float(qty)


def _limit_price_from_signal(sig_price: Optional[float], side: OrderSide) -> Optional[float]:
    if sig_price is None:
        return None
    # keep it simple: use provided price as limit
    p = float(sig_price)
    if p <= 0:
        return None
    return p


def _safe_log_cfg(cfg: Cfg, mode: str, paper: bool, base: str) -> None:
    LOG.info(
        "signal_executor starting | MODE=%s | paper=%s | base=%s | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | "
        "POLL=%ss | ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%.1f | MAX_POSITION_QTY=%.1f | "
        "ALLOW_ADD_TO_POSITION=%s | ALPACA_DEDUPE_MINUTES=%s | CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | "
        "MAX_OPEN_ORDERS=%s | DAILY_LOSS_STOP_PCT=%.1f | MAX_DAILY_LOSS_USD=%.1f | ENABLE_DAILY_RISK_GUARD=%s | "
        "SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s | TRADE_ONLY_WHEN_MARKET_OPEN=%s | PREOPEN_WINDOW_SECONDS=%s | "
        "ALLOW_TRADE_ON_CLOCK_ERROR=%s | TRADING_PAUSED=%s | DRY_RUN=%s",
        mode,
        paper,
        base,
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


def main() -> None:
    cfg = _load_cfg()
    engine = get_engine()
    tc = _alpaca_client()

    mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    paper = _env_bool("ALPACA_PAPER", False)
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets")

    _safe_log_cfg(cfg, mode, paper, base)

    while True:
        try:
            if cfg.trading_paused:
                LOG.info("trading_paused | sleep=%ss", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            sig = _pick_signal(engine, cfg)
            if not sig:
                LOG.info("no_signal | sleep=%ss", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            symbol = sig["symbol"]
            side = _normalize_side(sig["side"])
            strength = sig["strength"]
            sig_price = sig.get("price")

            if symbol not in cfg.symbols:
                LOG.info("skip_signal_not_allowed_symbol | symbol=%s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            if side is None:
                LOG.info("skip_signal_bad_side | symbol=%s side=%s", symbol, sig.get("side"))
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.long_only and side == OrderSide.SELL:
                LOG.info("skip_signal_long_only | symbol=%s side=sell", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            limit_price = _limit_price_from_signal(sig_price, side)
            if limit_price is None:
                LOG.info("skip_signal_no_price | symbol=%s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            qty = _calc_qty(cfg, limit_price)
            if qty <= 0:
                LOG.info("skip_signal_qty_zero | symbol=%s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            oid = str(uuid.uuid4())

            if cfg.dry_run:
                LOG.info(
                    "DRY_RUN would_submit | symbol=%s side=%s qty=%.4f limit=%.4f strength=%.4f oid=%s",
                    symbol,
                    side.value,
                    qty,
                    limit_price,
                    strength,
                    oid,
                )
                time.sleep(cfg.poll_seconds)
                continue

            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                client_order_id=oid,
            )

            o = tc.submit_order(req)
            LOG.info("submitted | symbol=%s side=%s qty=%.4f limit=%.4f id=%s strength=%.4f", symbol, side.value, qty, limit_price, o.id, strength)

        except Exception as e:
            LOG.exception("loop_error: %r", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
