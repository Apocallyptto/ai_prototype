# SPAM_GUARD_V2
"""
services/signal_executor.py

Key fix for LIVE with small cash:
- With $116 cash you cannot buy 1 share of AAPL (~$260+) or MSFT (~$400+).
- Alpaca-py docs note fractional qty / notional are supported with *market* orders (not limit). So:
    * If computed qty < 1, we submit a MarketOrderRequest with `notional=MAX_NOTIONAL`.
    * If qty >= 1, we keep using a LimitOrderRequest at signal price.

Also includes:
- spam guard (symbol cooldown + seen signal ids)
- pick TTL window
- long-only side filtering in SQL (prevents repeated skip of sell signals)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import bindparam, text

from tools.db import get_engine

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

LOG = logging.getLogger("signal_executor")


def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _now_utc_naive() -> datetime:
    return datetime.utcnow()


def _parse_csv_symbols(s: str) -> List[str]:
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]


@dataclass(frozen=True)
class Cfg:
    mode: str
    paper: bool
    base_url: str

    poll_seconds: int
    symbols: List[str]
    portfolio_id: int
    min_strength: float

    allow_short: bool
    long_only: bool
    allow_add_to_position: bool

    max_notional: float
    max_qty: float
    max_position_qty: float

    cancel_opposite_open_orders: bool
    max_open_positions: int
    max_open_orders: int

    alpaca_dedupe_minutes: int
    symbol_cooldown_seconds: int
    pick_ttl_seconds: int

    enable_daily_risk_guard: bool
    daily_loss_stop_pct: float
    max_daily_loss_usd: float

    trade_only_when_market_open: bool
    preopen_window_seconds: int
    allow_trade_on_clock_error: bool

    trading_paused: bool
    dry_run: bool

    allow_fractional_market: bool  # NEW

    @staticmethod
    def from_env() -> "Cfg":
        mode = _env_str("TRADING_MODE", "paper").lower()
        paper = _env_bool("ALPACA_PAPER", mode != "live")
        base_url = _env_str(
            "ALPACA_BASE_URL",
            "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets",
        )

        symbols = _parse_csv_symbols(_env_str("SYMBOLS", "AAPL,MSFT"))
        if not symbols:
            symbols = ["AAPL", "MSFT"]

        return Cfg(
            mode=mode,
            paper=paper,
            base_url=base_url,
            poll_seconds=_env_int("POLL_SECONDS", _env_int("POLL", 20)),
            symbols=symbols,
            portfolio_id=_env_int("PORTFOLIO_ID", 1),
            min_strength=_env_float("MIN_STRENGTH", 0.6),
            allow_short=_env_bool("ALLOW_SHORT", False),
            long_only=_env_bool("LONG_ONLY", True),
            allow_add_to_position=_env_bool("ALLOW_ADD_TO_POSITION", False),
            max_notional=_env_float("MAX_NOTIONAL", 100.0),
            max_qty=_env_float("MAX_QTY", 1.0),
            max_position_qty=_env_float("MAX_POSITION_QTY", 1.0),
            cancel_opposite_open_orders=_env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True),
            max_open_positions=_env_int("MAX_OPEN_POSITIONS", 1),
            max_open_orders=_env_int("MAX_OPEN_ORDERS", 1),
            alpaca_dedupe_minutes=_env_int("ALPACA_DEDUPE_MINUTES", 2),
            symbol_cooldown_seconds=_env_int("SYMBOL_COOLDOWN_SECONDS", 60),
            pick_ttl_seconds=_env_int("PICK_TTL_SECONDS", 120),
            enable_daily_risk_guard=_env_bool("ENABLE_DAILY_RISK_GUARD", True),
            daily_loss_stop_pct=_env_float("DAILY_LOSS_STOP_PCT", 1.0),
            max_daily_loss_usd=_env_float("MAX_DAILY_LOSS_USD", 200.0),
            trade_only_when_market_open=_env_bool("TRADE_ONLY_WHEN_MARKET_OPEN", True),
            preopen_window_seconds=_env_int("PREOPEN_WINDOW_SECONDS", 0),
            allow_trade_on_clock_error=_env_bool("ALLOW_TRADE_ON_CLOCK_ERROR", False),
            trading_paused=_env_bool("TRADING_PAUSED", False),
            dry_run=_env_bool("DRY_RUN", True),
            allow_fractional_market=_env_bool("ALLOW_FRACTIONAL_MARKET", True),
        )

    def allowed_sides(self) -> List[str]:
        # If long_only OR short is disabled -> only buy entries.
        if self.long_only or (not self.allow_short):
            return ["buy"]
        return ["buy", "sell"]


def make_trading_client(cfg: Cfg) -> TradingClient:
    key = _env_str("ALPACA_API_KEY")
    sec = _env_str("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in environment")
    return TradingClient(key, sec, paper=cfg.paper, base_url=cfg.base_url)


def _clock_allows_trading(tc: TradingClient, cfg: Cfg) -> bool:
    if not cfg.trade_only_when_market_open:
        return True
    try:
        clock = tc.get_clock()
        if clock.is_open:
            return True
        if cfg.preopen_window_seconds > 0 and clock.next_open:
            now = datetime.utcnow()
            delta = (clock.next_open.replace(tzinfo=None) - now).total_seconds()
            return 0 <= delta <= cfg.preopen_window_seconds
        return False
    except Exception:
        LOG.exception("clock_error")
        return bool(cfg.allow_trade_on_clock_error)


def _daily_risk_allows_trading(tc: TradingClient, cfg: Cfg) -> bool:
    if not cfg.enable_daily_risk_guard:
        return True
    try:
        acc = tc.get_account()
        equity = float(acc.equity or 0)
        last_equity = float(getattr(acc, "last_equity", 0) or 0)
        if last_equity <= 0:
            return True
        pnl = equity - last_equity
        max_loss_pct = abs(cfg.daily_loss_stop_pct) / 100.0
        pct_threshold = -last_equity * max_loss_pct
        usd_threshold = -abs(cfg.max_daily_loss_usd)
        threshold = min(pct_threshold, usd_threshold)
        if pnl <= threshold:
            LOG.warning(
                "daily_risk_stop",
                extra={"equity": equity, "last_equity": last_equity, "pnl": pnl, "threshold": threshold},
            )
            return False
        return True
    except Exception:
        LOG.exception("daily_risk_guard_error")
        return False


def _round_price(px: float) -> float:
    return float(f"{px:.4f}")


def _calc_qty(cfg: Cfg, ref_price: float) -> float:
    """Sizing for LIMIT path (whole shares). For fractional we switch to market+notional."""
    if ref_price <= 0:
        return 0.0
    qty = min(cfg.max_qty, cfg.max_notional / ref_price)
    qty = min(qty, cfg.max_position_qty)
    if qty <= 0:
        return 0.0
    # floor to 3 decimals so logs stable; if < 1, we treat it as fractional candidate
    qty = float(f"{(int(qty * 1000) / 1000):.3f}")
    return qty


def _deterministic_oid(prefix: str, symbol: str, side: str, ref: str) -> str:
    h = hashlib.sha1(f"{prefix}|{symbol}|{side}|{ref}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{symbol}-{h}"


def _get_open_orders(tc: TradingClient) -> list:
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=False)
        return tc.get_orders(filter=req) or []
    except Exception:
        LOG.exception("open_orders_error")
        return []


def _count_open_orders_non_protective(tc: TradingClient) -> int:
    orders = _get_open_orders(tc)

    def is_protective(o) -> bool:
        cid = (getattr(o, "client_order_id", None) or "").lower()
        return cid.startswith("exit-") or cid.startswith("oco-") or cid.startswith("exit-oco-")

    return sum(1 for o in orders if not is_protective(o))


def _cancel_opposite_open_orders(tc: TradingClient, symbol: str, side: str) -> None:
    opp = "sell" if side == "buy" else "buy"
    for o in _get_open_orders(tc):
        if (getattr(o, "symbol", "") or "").upper() != symbol.upper():
            continue
        if (getattr(o, "side", "") or "").lower() != opp:
            continue
        oid = getattr(o, "id", None)
        if not oid:
            continue
        try:
            tc.cancel_order_by_id(oid)
            LOG.info("canceled_opposite_open_order", extra={"symbol": symbol, "order_id": oid, "side": opp})
        except Exception:
            LOG.exception("cancel_opposite_open_order_failed", extra={"symbol": symbol, "order_id": oid})


def _count_open_positions(tc: TradingClient) -> int:
    try:
        positions = tc.get_all_positions() or []
        cnt = 0
        for p in positions:
            try:
                q = float(getattr(p, "qty", 0) or 0)
            except Exception:
                continue
            if abs(q) > 0:
                cnt += 1
        return cnt
    except Exception:
        LOG.exception("open_positions_error")
        return 0


def _pick_signals(engine, cfg: Cfg, limit: int = 25) -> List[dict]:
    cutoff = _now_utc_naive() - timedelta(seconds=cfg.pick_ttl_seconds)

    stmt = (
        text(
            """
            SELECT id, created_at, symbol, side, strength, price
            FROM signals
            WHERE portfolio_id = :pid
              AND symbol IN :symbols
              AND side IN :sides
              AND strength >= :min_strength
              AND created_at >= :cutoff
            ORDER BY created_at DESC
            LIMIT :lim
            """
        )
        .bindparams(bindparam("symbols", expanding=True))
        .bindparams(bindparam("sides", expanding=True))
    )

    with engine.connect() as con:
        rows = (
            con.execute(
                stmt,
                {
                    "pid": cfg.portfolio_id,
                    "symbols": cfg.symbols,
                    "sides": cfg.allowed_sides(),
                    "min_strength": cfg.min_strength,
                    "cutoff": cutoff,
                    "lim": limit,
                },
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = Cfg.from_env()
    engine = get_engine()
    tc = make_trading_client(cfg)

    LOG.info(
        "signal_executor starting",
        extra={
            "MODE": cfg.mode,
            "paper": cfg.paper,
            "base": cfg.base_url,
            "MIN_STRENGTH": cfg.min_strength,
            "SYMBOLS": cfg.symbols,
            "PORTFOLIO_ID": cfg.portfolio_id,
            "POLL": f"{cfg.poll_seconds}s",
            "ALLOW_SHORT": cfg.allow_short,
            "LONG_ONLY": cfg.long_only,
            "MAX_NOTIONAL": cfg.max_notional,
            "MAX_QTY": cfg.max_qty,
            "MAX_POSITION_QTY": cfg.max_position_qty,
            "ALLOW_ADD_TO_POSITION": cfg.allow_add_to_position,
            "ALPACA_DEDUPE_MINUTES": cfg.alpaca_dedupe_minutes,
            "CANCEL_OPPOSITE_OPEN_ORDERS": cfg.cancel_opposite_open_orders,
            "MAX_OPEN_POSITIONS": cfg.max_open_positions,
            "MAX_OPEN_ORDERS": cfg.max_open_orders,
            "DAILY_LOSS_STOP_PCT": cfg.daily_loss_stop_pct,
            "MAX_DAILY_LOSS_USD": cfg.max_daily_loss_usd,
            "ENABLE_DAILY_RISK_GUARD": cfg.enable_daily_risk_guard,
            "SYMBOL_COOLDOWN_SECONDS": cfg.symbol_cooldown_seconds,
            "PICK_TTL_SECONDS": cfg.pick_ttl_seconds,
            "TRADE_ONLY_WHEN_MARKET_OPEN": cfg.trade_only_when_market_open,
            "PREOPEN_WINDOW_SECONDS": cfg.preopen_window_seconds,
            "ALLOW_TRADE_ON_CLOCK_ERROR": cfg.allow_trade_on_clock_error,
            "TRADING_PAUSED": cfg.trading_paused,
            "DRY_RUN": cfg.dry_run,
            "ALLOW_FRACTIONAL_MARKET": cfg.allow_fractional_market,
        },
    )

    # spam guard (resets on restart)
    last_trade_by_symbol: Dict[str, datetime] = {}
    seen_signal_ids: Dict[int, datetime] = {}

    while True:
        try:
            if cfg.trading_paused:
                LOG.info("trading_paused", extra={"sleep": cfg.poll_seconds})
                time.sleep(cfg.poll_seconds)
                continue

            if not _clock_allows_trading(tc, cfg):
                LOG.info("market_closed", extra={"sleep": cfg.poll_seconds})
                time.sleep(cfg.poll_seconds)
                continue

            if not _daily_risk_allows_trading(tc, cfg):
                LOG.warning("daily_risk_block", extra={"sleep": cfg.poll_seconds})
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.max_open_positions > 0:
                open_pos = _count_open_positions(tc)
                if open_pos >= cfg.max_open_positions:
                    LOG.info("max_open_positions_reached", extra={"open_positions": open_pos, "sleep": cfg.poll_seconds})
                    time.sleep(cfg.poll_seconds)
                    continue

            if cfg.max_open_orders > 0:
                open_orders = _count_open_orders_non_protective(tc)
                if open_orders >= cfg.max_open_orders:
                    LOG.info("max_open_orders_reached", extra={"open_orders": open_orders, "sleep": cfg.poll_seconds})
                    time.sleep(cfg.poll_seconds)
                    continue

            batch = _pick_signals(engine, cfg, limit=25)
            if not batch:
                LOG.info("no_signal", extra={"sleep": cfg.poll_seconds})
                time.sleep(cfg.poll_seconds)
                continue

            chosen: Optional[dict] = None
            now = _now_utc_naive()

            for sig in batch:
                sid = int(sig["id"])
                sym = str(sig["symbol"]).upper()

                # don't reuse same DB row for a short window
                if sid in seen_signal_ids and (now - seen_signal_ids[sid]).total_seconds() < cfg.pick_ttl_seconds:
                    continue

                # symbol cooldown
                if sym in last_trade_by_symbol and (now - last_trade_by_symbol[sym]).total_seconds() < cfg.symbol_cooldown_seconds:
                    continue

                chosen = sig
                break

            if not chosen:
                LOG.info("no_eligible_signal (seen/cooldown)", extra={"sleep": cfg.poll_seconds})
                time.sleep(cfg.poll_seconds)
                continue

            sid = int(chosen["id"])
            symbol = str(chosen["symbol"]).upper()
            side = str(chosen["side"]).lower()
            strength = float(chosen["strength"])
            ref_price = float(chosen["price"])
            limit_price = _round_price(ref_price)

            qty = _calc_qty(cfg, ref_price)
            if qty <= 0:
                LOG.info("skip_signal_bad_qty", extra={"symbol": symbol, "side": side, "ref_price": ref_price})
                seen_signal_ids[sid] = now
                time.sleep(1)
                continue

            if cfg.cancel_opposite_open_orders:
                _cancel_opposite_open_orders(tc, symbol, side)

            # Decide order type:
            use_fractional_market = cfg.allow_fractional_market and qty < 1.0

            if cfg.dry_run:
                oid = _deterministic_oid("DRY", symbol, side, f"{qty:.3f}|{limit_price:.4f}|{'MKT' if use_fractional_market else 'LMT'}")
                LOG.info(
                    "DRY_RUN would_submit",
                    extra={
                        "symbol": symbol,
                        "side": side,
                        "qty": f"{qty:.4f}",
                        "ref_price": f"{ref_price:.4f}",
                        "limit": f"{limit_price:.4f}",
                        "strength": f"{strength:.4f}",
                        "order_type": "market(notional)" if use_fractional_market else "limit(qty)",
                        "oid": oid,
                    },
                )
                last_trade_by_symbol[symbol] = now
                seen_signal_ids[sid] = now
                time.sleep(cfg.poll_seconds)
                continue

            # Submit live/paper order
            if use_fractional_market:
                # notional market order; max spend is MAX_NOTIONAL
                oid = _deterministic_oid("ENT", symbol, side, f"{cfg.max_notional:.2f}|MKT")
                req = MarketOrderRequest(
                    symbol=symbol,
                    notional=float(f"{cfg.max_notional:.2f}"),
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    type=OrderType.MARKET,
                    client_order_id=oid,
                )
            else:
                # whole-share / >=1 path: limit order with qty
                oid = _deterministic_oid("ENT", symbol, side, f"{qty:.3f}|{limit_price:.4f}|LMT")
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=float(f"{qty:.3f}"),
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    type=OrderType.LIMIT,
                    limit_price=limit_price,
                    client_order_id=oid,
                )

            o = tc.submit_order(req)
            LOG.info(
                "submitted",
                extra={
                    "symbol": symbol,
                    "side": side,
                    "order_type": "market(notional)" if use_fractional_market else "limit(qty)",
                    "qty": qty,
                    "notional": cfg.max_notional if use_fractional_market else None,
                    "limit": limit_price if not use_fractional_market else None,
                    "strength": strength,
                    "order_id": getattr(o, "id", None),
                    "client_order_id": oid,
                },
            )
            last_trade_by_symbol[symbol] = now
            seen_signal_ids[sid] = now
            time.sleep(cfg.poll_seconds)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            LOG.exception("loop_error", extra={"error": repr(e)})
            time.sleep(5)


if __name__ == "__main__":
    main()
