# SPAM_GUARD_V2
import os
import time
import uuid
import math
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Set

from sqlalchemy import text, bindparam

from tools.db import get_engine

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderSide,
    TimeInForce,
    OrderType,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

LOG = logging.getLogger("signal_executor")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


class Cfg:
    poll_seconds: int
    min_strength: float
    symbols: List[str]
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

    # optional risk guard knobs (kept for logging/compat)
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

    # NEW
    allow_fractional_market: bool


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_mode_and_paper() -> tuple[str, bool, str]:
    """
    TRADING_MODE=live|paper
    ALPACA_PAPER=1|0 (fallback)
    """
    mode = (os.getenv("TRADING_MODE") or "").strip().lower()
    if mode in {"paper", "live"}:
        paper = (mode == "paper")
    else:
        paper = _env_bool("ALPACA_PAPER", True)

    mode = "paper" if paper else "live"
    base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    return mode, paper, base


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_aware(dt: Any) -> Optional[datetime]:
    """
    Normalizuj datetime na UTC-aware:
    - ak je string ISO -> parse
    - ak je naive (bez tzinfo) -> predpokladaj UTC
    - ak je aware -> prehoď do UTC
    """
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
        return dt.replace(tzinfo=timezone.utc)

    try:
        return dt.astimezone(timezone.utc)
    except Exception:
        return dt


def _round_price(p: float) -> float:
    # bezpečne na 2 desatinné pre US equities (môžeš upraviť podľa potreby)
    return float(f"{p:.2f}")


def _load_cfg() -> Cfg:
    cfg = Cfg()
    cfg.poll_seconds = int(os.getenv("POLL_SECONDS", "20"))
    cfg.min_strength = float(os.getenv("MIN_STRENGTH", "0.6"))

    sym_raw = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
    cfg.symbols = [s.strip().upper() for s in sym_raw.split(",") if s.strip()]
    cfg.portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))

    cfg.allow_short = _env_bool("ALLOW_SHORT", False)
    cfg.long_only = _env_bool("LONG_ONLY", True)

    cfg.max_notional = float(os.getenv("MAX_NOTIONAL", "100"))
    cfg.max_qty = float(os.getenv("MAX_QTY", "1"))
    cfg.max_position_qty = float(os.getenv("MAX_POSITION_QTY", "1"))
    cfg.allow_add_to_position = _env_bool("ALLOW_ADD_TO_POSITION", False)

    cfg.alpaca_dedupe_minutes = int(os.getenv("ALPACA_DEDUPE_MINUTES", "2"))
    cfg.cancel_opposite_open_orders = _env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True)

    cfg.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
    cfg.max_open_orders = int(os.getenv("MAX_OPEN_ORDERS", "1"))

    cfg.daily_loss_stop_pct = float(os.getenv("DAILY_LOSS_STOP_PCT", "1.0"))
    cfg.max_daily_loss_usd = float(os.getenv("MAX_DAILY_LOSS_USD", "200.0"))
    cfg.enable_daily_risk_guard = _env_bool("ENABLE_DAILY_RISK_GUARD", True)

    cfg.symbol_cooldown_seconds = int(os.getenv("SYMBOL_COOLDOWN_SECONDS", "60"))
    cfg.pick_ttl_seconds = int(os.getenv("PICK_TTL_SECONDS", "120"))

    cfg.trade_only_when_market_open = _env_bool("TRADE_ONLY_WHEN_MARKET_OPEN", True)
    cfg.preopen_window_seconds = int(os.getenv("PREOPEN_WINDOW_SECONDS", "0"))
    cfg.allow_trade_on_clock_error = _env_bool("ALLOW_TRADE_ON_CLOCK_ERROR", False)

    cfg.trading_paused = _env_bool("TRADING_PAUSED", True)
    cfg.dry_run = _env_bool("DRY_RUN", True)

    # NEW
    cfg.allow_fractional_market = _env_bool("ALLOW_FRACTIONAL_MARKET", True)

    return cfg


def make_trading_client() -> tuple[TradingClient, str, bool, str]:
    key = os.getenv("ALPACA_API_KEY") or ""
    sec = os.getenv("ALPACA_API_SECRET") or ""
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    mode, paper, base = _resolve_mode_and_paper()

    # alpaca-py vyberá endpoint cez paper flag
    tc = TradingClient(key, sec, paper=paper)
    return tc, mode, paper, base


def _get_clock_is_open(tc: TradingClient) -> bool:
    clk = tc.get_clock()
    return bool(getattr(clk, "is_open", False))


def _within_preopen_window(tc: TradingClient, window_seconds: int) -> bool:
    clk = tc.get_clock()
    nxt = getattr(clk, "next_open", None)
    if not nxt:
        return False
    now = _now_utc()

    nxt_dt = _to_utc_aware(nxt)
    if not nxt_dt:
        return False

    delta = (nxt_dt - now).total_seconds()
    return 0 <= delta <= window_seconds


def _get_open_orders(tc: TradingClient) -> list:
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    return list(tc.get_orders(filter=req) or [])


def _get_positions(tc: TradingClient) -> list:
    return list(tc.get_all_positions() or [])


def _count_open_entry_orders(open_orders: list) -> int:
    # ignoruj exit/oco ordery, aby entry limit neblokovala ochranu
    n = 0
    for o in open_orders:
        cid = str(getattr(o, "client_order_id", "") or "").lower()
        if "exit-oco" in cid or cid.startswith("exit-") or cid.startswith("oco-"):
            continue
        n += 1
    return n


def _safe_mid_from_quotes(symbol: str) -> Optional[float]:
    # Placeholder: ak máš inde quote fetch, napoj to sem.
    return None


def _dedupe_ok(engine, symbol: str, side: str, minutes: int) -> bool:
    # DB-driven dedupe podľa orders table
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

        ts = _to_utc_aware(r[0])
        if not ts:
            return True

        now = _now_utc()
        age_sec = (now - ts).total_seconds()
        return age_sec >= minutes * 60


def _pick_signal(engine, cfg: Cfg, seen_ids: Set[int]) -> Optional[Dict[str, Any]]:
    # vezmi viac kandidátov a vyber prvý, ktorý nie je seen a nie je expirovaný
    q = (
        text(
            """
            SELECT id, created_at, symbol, side, strength, price
            FROM signals
            WHERE portfolio_id=:pid
              AND symbol IN :symbols
              AND strength >= :min_strength
            ORDER BY created_at DESC
            LIMIT 25
            """
        )
        .bindparams(bindparam("symbols", expanding=True))
    )

    now = _now_utc()
    with engine.connect() as con:
        rows = con.execute(
            q,
            {
                "pid": cfg.portfolio_id,
                "symbols": cfg.symbols,
                "min_strength": cfg.min_strength,
            },
        ).mappings().all()

    for r in rows:
        sid = int(r["id"])
        if sid in seen_ids:
            continue

        created_at = _to_utc_aware(r.get("created_at"))

        if created_at and cfg.pick_ttl_seconds > 0:
            age = (now - created_at).total_seconds()
            if age > cfg.pick_ttl_seconds:
                # príliš staré, preskoč
                continue

        return dict(r)

    return None


def _calc_qty(cfg: Cfg, price: float) -> float:
    # qty cap: max_qty, max_position_qty, max_notional/price
    if price <= 0:
        return 0.0
    q_notional = cfg.max_notional / price
    qty = min(cfg.max_qty, cfg.max_position_qty, q_notional)
    if qty <= 0:
        return 0.0
    # jemné zaokrúhlenie (fractional support)
    qty = float(f"{qty:.6f}")
    return qty


def _mk_dry_oid(symbol: str, side: str, sig_id: int, kind: str) -> str:
    seed = f"{symbol}|{side}|{sig_id}|{kind}"
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"DRY-{symbol}-{h}"


def _mk_client_order_id(symbol: str, side: str, sig_id: int, kind: str) -> str:
    # krátke a deterministické (pomáha proti duplicite pri reštarte)
    s = "B" if side == "buy" else "S"
    k = "N" if kind == "MARKET_NOTIONAL" else "L"
    return f"E-{symbol}-{s}-{k}-{sig_id}"


def _cancel_opposite(tc: TradingClient, symbol: str, side: str, open_orders: list) -> None:
    want_cancel_side = "sell" if side == "buy" else "buy"
    for o in open_orders:
        osym = str(getattr(o, "symbol", "") or "").upper()
        if osym != symbol:
            continue
        oside = str(getattr(o, "side", "") or "").lower()
        if oside != want_cancel_side:
            continue
        oid = str(getattr(o, "id", "") or "")
        if not oid:
            continue
        try:
            tc.cancel_order_by_id(oid)
            LOG.info("canceled_opposite_open_order | %s %s oid=%s", symbol, oside, oid)
        except Exception as e:
            LOG.warning("cancel_opposite_failed | %s oid=%s err=%r", symbol, oid, e)


def _place_limit(tc: TradingClient, symbol: str, side: str, qty: float, limit_price: float, client_order_id: str) -> str:
    req = LimitOrderRequest(
        symbol=symbol,
        qty=float(qty),
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        type=OrderType.LIMIT,
        limit_price=float(_round_price(limit_price)),
        client_order_id=client_order_id,
    )
    o = tc.submit_order(order_data=req)
    return str(getattr(o, "id", ""))


def _place_market_notional(tc: TradingClient, symbol: str, side: str, notional: float, client_order_id: str) -> str:
    # Alpaca notional order je relevantný hlavne pre BUY (SELL notional sa typicky nepoužíva)
    req = MarketOrderRequest(
        symbol=symbol,
        notional=float(f"{notional:.2f}"),
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        type=OrderType.MARKET,
        client_order_id=client_order_id,
    )
    o = tc.submit_order(order_data=req)
    return str(getattr(o, "id", ""))


def main() -> None:
    cfg = _load_cfg()
    engine = get_engine()
    tc, mode, paper, base = make_trading_client()

    LOG.info(
        "signal_executor starting | MODE=%s | paper=%s | base=%s | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%.1f | MAX_POSITION_QTY=%.1f | ALLOW_ADD_TO_POSITION=%s | "
        "ALPACA_DEDUPE_MINUTES=%s | CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | "
        "DAILY_LOSS_STOP_PCT=%.1f | MAX_DAILY_LOSS_USD=%.1f | ENABLE_DAILY_RISK_GUARD=%s | SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s | "
        "TRADE_ONLY_WHEN_MARKET_OPEN=%s | PREOPEN_WINDOW_SECONDS=%s | ALLOW_TRADE_ON_CLOCK_ERROR=%s | TRADING_PAUSED=%s | DRY_RUN=%s | "
        "ALLOW_FRACTIONAL_MARKET=%s",
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
        cfg.allow_fractional_market,
    )

    seen_signal_ids: Set[int] = set()
    last_submit_ts_by_symbol: Dict[str, float] = {}

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

            open_orders = _get_open_orders(tc)
            positions = _get_positions(tc)

            # max open positions gate
            if cfg.max_open_positions > 0:
                live_pos = []
                for p in positions:
                    qty = float(getattr(p, "qty", 0) or 0)
                    if qty == 0:
                        continue
                    sym = str(getattr(p, "symbol", "") or "").upper()
                    live_pos.append((sym, qty))

                if len(live_pos) >= cfg.max_open_positions:
                    LOG.info(
                        "gate_max_open_positions | have=%s limit=%s | positions=%s | sleep=%ss",
                        len(live_pos),
                        cfg.max_open_positions,
                        live_pos,
                        cfg.poll_seconds,
                    )
                    time.sleep(cfg.poll_seconds)
                    continue

            # max open orders gate (entry-only)
            if cfg.max_open_orders > 0:
                entry_open = _count_open_entry_orders(open_orders)
                if entry_open >= cfg.max_open_orders:
                    LOG.info("gate_max_open_orders | have=%s limit=%s | sleep=%ss", entry_open, cfg.max_open_orders, cfg.poll_seconds)
                    time.sleep(cfg.poll_seconds)
                    continue

            sig = _pick_signal(engine, cfg, seen_signal_ids)
            if not sig:
                LOG.info("no_signal | sleep=%ss", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            sig_id = int(sig["id"])
            symbol = str(sig["symbol"]).upper()
            side = str(sig["side"]).lower()
            strength = float(sig["strength"])
            price = float(sig["price"] or 0.0) or (_safe_mid_from_quotes(symbol) or 0.0)

            # spam guard: symbol cooldown
            now_ts = time.time()
            last_ts = last_submit_ts_by_symbol.get(symbol, 0.0)
            if cfg.symbol_cooldown_seconds > 0 and (now_ts - last_ts) < cfg.symbol_cooldown_seconds:
                LOG.info("no_eligible_signal (seen/cooldown) | symbol=%s cooldown=%ss", symbol, cfg.symbol_cooldown_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            if side not in ("buy", "sell"):
                LOG.info("skip invalid side | %s", side)
                seen_signal_ids.add(sig_id)
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.long_only and side == "sell":
                LOG.info("skip_signal_long_only | symbol=%s side=%s", symbol, side)
                seen_signal_ids.add(sig_id)
                time.sleep(cfg.poll_seconds)
                continue

            if not price or price <= 0:
                LOG.info("skip no_price | %s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            # do not add to position gate
            if not cfg.allow_add_to_position:
                cur_qty = 0.0
                for p in positions:
                    psym = str(getattr(p, "symbol", "") or "").upper()
                    if psym == symbol:
                        cur_qty = float(getattr(p, "qty", 0) or 0)
                        break

                if cur_qty != 0:
                    LOG.info("skip_add_to_position | %s qty=%s", symbol, cur_qty)
                    seen_signal_ids.add(sig_id)
                    last_submit_ts_by_symbol[symbol] = now_ts
                    time.sleep(cfg.poll_seconds)
                    continue

            # DB dedupe gate
            if not _dedupe_ok(engine, symbol, side, cfg.alpaca_dedupe_minutes):
                LOG.info("dedupe_skip | %s %s", symbol, side)
                time.sleep(cfg.poll_seconds)
                continue

            qty = _calc_qty(cfg, price)
            if qty <= 0:
                LOG.info("skip qty<=0 | %s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            # Decide order kind:
            kind = "LIMIT"
            notional = 0.0
            limit_price = price

            if cfg.allow_fractional_market and side == "buy" and qty < 1.0:
                # Market notional, cap by qty*price (<= max_notional)
                kind = "MARKET_NOTIONAL"
                notional = float(f"{min(cfg.max_notional, qty * price):.2f}")
                if notional <= 0:
                    LOG.info("skip notional<=0 | %s", symbol)
                    time.sleep(cfg.poll_seconds)
                    continue

            if cfg.cancel_opposite_open_orders:
                _cancel_opposite(tc, symbol, side, open_orders)

            client_order_id = _mk_client_order_id(symbol, side, sig_id, kind)

            if cfg.dry_run:
                dry_oid = _mk_dry_oid(symbol, side, sig_id, kind)
                if kind == "MARKET_NOTIONAL":
                    LOG.info(
                        "DRY_RUN would_submit | symbol=%s side=%s kind=%s notional=%.2f ref_price=%.4f strength=%.4f oid=%s cid=%s",
                        symbol, side, kind, notional, price, strength, dry_oid, client_order_id
                    )
                else:
                    LOG.info(
                        "DRY_RUN would_submit | symbol=%s side=%s kind=%s qty=%.6f limit=%.4f strength=%.4f oid=%s cid=%s",
                        symbol, side, kind, qty, limit_price, strength, dry_oid, client_order_id
                    )
                # mark as seen + cooldown (so to nestraľuje dookola)
                seen_signal_ids.add(sig_id)
                last_submit_ts_by_symbol[symbol] = now_ts
            else:
                if kind == "MARKET_NOTIONAL":
                    oid = _place_market_notional(tc, symbol, side, notional, client_order_id)
                    LOG.info(
                        "submitted | symbol=%s side=%s kind=%s notional=%.2f ref_price=%.4f oid=%s strength=%.4f cid=%s",
                        symbol, side, kind, notional, price, oid, strength, client_order_id
                    )
                else:
                    oid = _place_limit(tc, symbol, side, qty, limit_price, client_order_id)
                    LOG.info(
                        "submitted | symbol=%s side=%s kind=%s qty=%.6f limit=%.4f oid=%s strength=%.4f cid=%s",
                        symbol, side, kind, qty, limit_price, oid, strength, client_order_id
                    )
                seen_signal_ids.add(sig_id)
                last_submit_ts_by_symbol[symbol] = now_ts

        except Exception as e:
            LOG.exception("loop_error: %r", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()