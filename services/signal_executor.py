import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Set, Tuple

from sqlalchemy import text, bindparam

from tools.db import get_engine
from tools.system_flags import get_flag
from tools.execution_audit import log_blocked_signal, log_submitted_order

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderSide,
    TimeInForce,
    OrderType,
    PositionIntent,
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

    symbol_cooldown_seconds: int
    pick_ttl_seconds: int

    trade_only_when_market_open: bool
    preopen_window_seconds: int
    allow_trade_on_clock_error: bool

    trading_paused: bool
    dry_run: bool

    allow_fractional_market: bool

    # safety
    exit_prefix: str
    pause_on_unprotected_positions: bool
    pause_on_account_blocked: bool
    pause_on_pdt_flag: bool

    # swing / PDT safety
    exit_on_sell_signal: bool
    min_hold_minutes: int

    # confidence sizing
    confidence_sizing: bool
    notional_min: float
    notional_max: float
    confidence_full_strength: float


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_mode_and_paper() -> Tuple[str, bool, str]:
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
    return float(f"{p:.2f}")


def _safe_audit_block(sig: Optional[Dict[str, Any]], reason: str, detail: str) -> None:
    try:
        log_blocked_signal(
            symbol=(sig or {}).get("symbol"),
            side=(sig or {}).get("side"),
            strength=(sig or {}).get("strength"),
            source=(sig or {}).get("source"),
            portfolio_id=(sig or {}).get("portfolio_id"),
            signal_ts=(sig or {}).get("created_at"),
            reason=reason,
            detail=detail,
        )
    except Exception as e:
        LOG.warning("audit_block_failed | reason=%s err=%r", reason, e)


def _safe_audit_submit(sig: Optional[Dict[str, Any]], detail: str) -> None:
    try:
        log_submitted_order(
            symbol=(sig or {}).get("symbol"),
            side=(sig or {}).get("side"),
            strength=(sig or {}).get("strength"),
            source=(sig or {}).get("source"),
            portfolio_id=(sig or {}).get("portfolio_id"),
            signal_ts=(sig or {}).get("created_at"),
            detail=detail,
        )
    except Exception as e:
        LOG.warning("audit_submit_failed | err=%r", e)


def _load_cfg() -> Cfg:
    cfg = Cfg()
    cfg.poll_seconds = int(os.getenv("EXECUTOR_POLL_SECONDS", os.getenv("POLL_SECONDS", "20")))
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

    cfg.symbol_cooldown_seconds = int(os.getenv("SYMBOL_COOLDOWN_SECONDS", "60"))
    cfg.pick_ttl_seconds = int(os.getenv("PICK_TTL_SECONDS", "120"))

    cfg.trade_only_when_market_open = _env_bool("TRADE_ONLY_WHEN_MARKET_OPEN", True)
    cfg.preopen_window_seconds = int(os.getenv("PREOPEN_WINDOW_SECONDS", "0"))
    cfg.allow_trade_on_clock_error = _env_bool("ALLOW_TRADE_ON_CLOCK_ERROR", False)

    cfg.trading_paused = _env_bool("TRADING_PAUSED", True)
    cfg.dry_run = _env_bool("DRY_RUN", True)

    cfg.allow_fractional_market = _env_bool("ALLOW_FRACTIONAL_MARKET", True)

    cfg.exit_prefix = (os.getenv("EXIT_PREFIX") or "EXIT-OCO-").strip()
    cfg.pause_on_unprotected_positions = _env_bool("PAUSE_ON_UNPROTECTED_POSITIONS", True)
    cfg.pause_on_account_blocked = _env_bool("PAUSE_ON_ACCOUNT_BLOCKED", True)
    cfg.pause_on_pdt_flag = _env_bool("PAUSE_ON_PDT_FLAG", False)

    cfg.exit_on_sell_signal = _env_bool("EXIT_ON_SELL_SIGNAL", False)
    cfg.min_hold_minutes = int(os.getenv("MIN_HOLD_MINUTES", "0"))

    cfg.confidence_sizing = _env_bool("CONFIDENCE_SIZING", True)
    cfg.notional_min = float(os.getenv("NOTIONAL_MIN", "5"))
    cfg.notional_max = float(os.getenv("NOTIONAL_MAX", str(cfg.max_notional)))
    cfg.confidence_full_strength = float(os.getenv("CONFIDENCE_FULL_STRENGTH", "1.0"))

    if cfg.notional_min < 0:
        cfg.notional_min = 0.0
    if cfg.notional_max < 0:
        cfg.notional_max = 0.0
    if cfg.notional_min > cfg.notional_max:
        cfg.notional_min, cfg.notional_max = cfg.notional_max, cfg.notional_min

    cfg.notional_max = min(cfg.notional_max, cfg.max_notional)
    return cfg


def make_trading_client() -> Tuple[TradingClient, str, bool, str]:
    key = os.getenv("ALPACA_API_KEY") or ""
    sec = os.getenv("ALPACA_API_SECRET") or ""
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    mode, paper, base = _resolve_mode_and_paper()
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
    n = 0
    for o in open_orders:
        cid = str(getattr(o, "client_order_id", "") or "").lower()
        if cid.startswith("exit-") or "exit-oco" in cid or cid.startswith("oco-"):
            continue
        n += 1
    return n


def _has_exit_sl(open_orders: list, symbol: str, exit_prefix: str) -> bool:
    symu = (symbol or "").upper()
    pref = f"{(exit_prefix or '').strip()}{symu}-SL-".lower()
    for o in open_orders:
        osym = str(getattr(o, "symbol", "") or "").upper()
        if osym != symu:
            continue
        cid = str(getattr(o, "client_order_id", "") or "").lower()
        if cid.startswith(pref):
            return True
    return False


def _unprotected_symbols(positions: list, open_orders: list, exit_prefix: str) -> List[str]:
    unprot: List[str] = []
    for p in positions:
        qty = float(getattr(p, "qty", 0) or 0)
        if qty == 0:
            continue
        sym = str(getattr(p, "symbol", "") or "").upper()
        if not sym:
            continue
        if not _has_exit_sl(open_orders, sym, exit_prefix):
            unprot.append(sym)
    return sorted(set(unprot))


def _is_account_blocked(tc: TradingClient, pause_on_pdt_flag: bool):
    try:
        a = tc.get_account()
        status = str(getattr(a, "status", "") or "")
        trading_blocked = bool(getattr(a, "trading_blocked", False))
        account_blocked = bool(getattr(a, "account_blocked", False))
        pdt = bool(getattr(a, "pattern_day_trader", False))

        dtbp = getattr(a, "daytrading_buying_power", None)
        dtc = getattr(a, "daytrade_count", None)

        reason = f"status={status} trading_blocked={trading_blocked} account_blocked={account_blocked} pattern_day_trader={pdt}"
        if dtbp is not None:
            reason += f" daytrading_buying_power={dtbp}"
        if dtc is not None:
            reason += f" daytrade_count={dtc}"

        pause_on_daytrade_ge = int(os.getenv("PAUSE_ON_DAYTRADE_COUNT_GE", "999"))
        try:
            dtc_int = int(dtc) if dtc is not None else None
        except Exception:
            dtc_int = None

        if dtc_int is not None and dtc_int >= pause_on_daytrade_ge:
            return True, reason + f" PAUSE_ON_DAYTRADE_COUNT_GE={pause_on_daytrade_ge}", "pdt_gate"

        if trading_blocked or account_blocked:
            return True, reason, "account_blocked"
        if pause_on_pdt_flag and pdt:
            return True, reason, "pdt_flag"
        return False, reason, ""
    except Exception as e:
        return False, f"account_check_failed err={e!r}", ""


def _dedupe_ok(engine, symbol: str, side: str, minutes: int) -> bool:
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
    q = (
        text(
            """
            SELECT id, created_at, symbol, side, strength, price, source, portfolio_id
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
            {"pid": cfg.portfolio_id, "symbols": cfg.symbols, "min_strength": cfg.min_strength},
        ).mappings().all()

    for r in rows:
        sid = int(r["id"])
        if sid in seen_ids:
            continue

        created_at = _to_utc_aware(r.get("created_at"))
        if created_at and cfg.pick_ttl_seconds > 0:
            age = (now - created_at).total_seconds()
            if age > cfg.pick_ttl_seconds:
                continue

        return dict(r)

    return None


def _mk_client_order_id(symbol: str, side: str, sig_id: int, kind: str) -> str:
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


def _scaled_notional(cfg: Cfg, strength: float) -> float:
    cap = float(cfg.max_notional)
    if not getattr(cfg, "confidence_sizing", False):
        return float(f"{cap:.2f}")

    s = abs(float(strength))
    lo = float(cfg.min_strength)
    hi = float(getattr(cfg, "confidence_full_strength", 1.0) or 1.0)
    if hi <= lo:
        hi = lo + 1e-6

    t = (s - lo) / (hi - lo)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nmin = float(getattr(cfg, "notional_min", 0.0) or 0.0)
    nmax = float(getattr(cfg, "notional_max", cap) or cap)
    if nmin > nmax:
        nmin, nmax = nmax, nmin

    n = nmin + t * (nmax - nmin)
    if n > cap:
        n = cap
    if n < 0:
        n = 0
    return float(f"{n:.2f}")


def _qty_from_notional(cfg: Cfg, price: float, notional: float) -> float:
    if price <= 0 or notional <= 0:
        return 0.0
    q_notional = notional / price
    qty = min(cfg.max_qty, cfg.max_position_qty, q_notional)
    if qty <= 0:
        return 0.0
    return float(f"{qty:.6f}")


def main() -> None:
    cfg = _load_cfg()
    engine = get_engine()
    tc, mode, paper, base = make_trading_client()

    LOG.info(
        "signal_executor starting | MODE=%s | paper=%s | base=%s | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%.1f | MAX_POSITION_QTY=%.1f | ALLOW_ADD_TO_POSITION=%s | "
        "ALPACA_DEDUPE_MINUTES=%s | CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | "
        "SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s | "
        "TRADE_ONLY_WHEN_MARKET_OPEN=%s | PREOPEN_WINDOW_SECONDS=%s | ALLOW_TRADE_ON_CLOCK_ERROR=%s | TRADING_PAUSED=%s | DRY_RUN=%s | "
        "ALLOW_FRACTIONAL_MARKET=%s | EXIT_PREFIX=%s | PAUSE_ON_UNPROTECTED_POSITIONS=%s | PAUSE_ON_ACCOUNT_BLOCKED=%s | PAUSE_ON_PDT_FLAG=%s | "
        "EXIT_ON_SELL_SIGNAL=%s | MIN_HOLD_MINUTES=%s | CONFIDENCE_SIZING=%s | NOTIONAL_MIN=%.2f | NOTIONAL_MAX=%.2f | FULL_STRENGTH=%.2f",
        mode, paper, base,
        cfg.min_strength, cfg.symbols, cfg.portfolio_id, cfg.poll_seconds,
        cfg.allow_short, cfg.long_only,
        cfg.max_notional, cfg.max_qty, cfg.max_position_qty, cfg.allow_add_to_position,
        cfg.alpaca_dedupe_minutes, cfg.cancel_opposite_open_orders,
        cfg.max_open_positions, cfg.max_open_orders,
        cfg.symbol_cooldown_seconds, cfg.pick_ttl_seconds,
        cfg.trade_only_when_market_open, cfg.preopen_window_seconds, cfg.allow_trade_on_clock_error,
        cfg.trading_paused, cfg.dry_run,
        cfg.allow_fractional_market,
        cfg.exit_prefix, cfg.pause_on_unprotected_positions, cfg.pause_on_account_blocked, cfg.pause_on_pdt_flag,
        cfg.exit_on_sell_signal, cfg.min_hold_minutes,
        cfg.confidence_sizing, cfg.notional_min, cfg.notional_max, cfg.confidence_full_strength,
    )

    seen_signal_ids: Set[int] = set()
    last_submit_ts_by_symbol: Dict[str, float] = {}
    last_entry_ts_by_symbol: Dict[str, float] = {}

    while True:
        try:
            db_paused = get_flag("TRADING_PAUSED", "0") == "1"
            db_pause_reason = get_flag("TRADING_PAUSED_REASON", "")
            if db_paused:
                LOG.warning("trading_paused_by_db_flag | reason=%s", db_pause_reason)
                _safe_audit_block(None, "trading_paused_db", f"reason={db_pause_reason}")
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.trading_paused:
                LOG.info("trading_paused | sleep=%ss", cfg.poll_seconds)
                _safe_audit_block(None, "trading_paused_cfg", "TRADING_PAUSED config is enabled")
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
                            _safe_audit_block(None, "market_closed", "market is closed and preopen window not active")
                            time.sleep(cfg.poll_seconds)
                            continue
                except Exception as e:
                    if cfg.allow_trade_on_clock_error:
                        LOG.warning("clock_error allow_trade_on_clock_error=True | %r", e)
                    else:
                        LOG.warning("clock_error FAIL_CLOSED allow_trade_on_clock_error=False | %r", e)
                        _safe_audit_block(None, "clock_error_fail_closed", repr(e))
                        time.sleep(cfg.poll_seconds)
                        continue

            open_orders = _get_open_orders(tc)
            positions = _get_positions(tc)

            if cfg.pause_on_account_blocked:
                blocked, reason, reason_code = _is_account_blocked(tc, cfg.pause_on_pdt_flag)
                if blocked:
                    LOG.warning("gate_account_blocked | %s | sleep=%ss", reason, cfg.poll_seconds)
                    _safe_audit_block(None, reason_code or "account_blocked", reason)
                    time.sleep(cfg.poll_seconds)
                    continue

            if cfg.pause_on_unprotected_positions:
                unprot = _unprotected_symbols(positions, open_orders, cfg.exit_prefix)
                if unprot:
                    detail = f"exit_prefix={cfg.exit_prefix} unprotected={unprot}"
                    LOG.warning(
                        "gate_unprotected_positions | %s | sleep=%ss",
                        detail,
                        cfg.poll_seconds,
                    )
                    _safe_audit_block(None, "unprotected_positions", detail)
                    time.sleep(cfg.poll_seconds)
                    continue

            if cfg.max_open_positions > 0:
                live_pos = []
                for p in positions:
                    qty = float(getattr(p, "qty", 0) or 0)
                    if qty == 0:
                        continue
                    sym = str(getattr(p, "symbol", "") or "").upper()
                    live_pos.append((sym, qty))
                if len(live_pos) >= cfg.max_open_positions:
                    detail = f"have={len(live_pos)} limit={cfg.max_open_positions} positions={live_pos}"
                    LOG.info("gate_max_open_positions | %s | sleep=%ss", detail, cfg.poll_seconds)
                    _safe_audit_block(None, "max_open_positions", detail)
                    time.sleep(cfg.poll_seconds)
                    continue

            if cfg.max_open_orders > 0:
                entry_open = _count_open_entry_orders(open_orders)
                if entry_open >= cfg.max_open_orders:
                    detail = f"have={entry_open} limit={cfg.max_open_orders}"
                    LOG.info("gate_max_open_orders | %s | sleep=%ss", detail, cfg.poll_seconds)
                    _safe_audit_block(None, "max_open_orders", detail)
                    time.sleep(cfg.poll_seconds)
                    continue

            sig = _pick_signal(engine, cfg, seen_signal_ids)
            if not sig:
                LOG.info("no_signal | sleep=%ss", cfg.poll_seconds)
                _safe_audit_block(None, "no_signal", "no eligible fresh signal selected by picker")
                time.sleep(cfg.poll_seconds)
                continue

            sig_id = int(sig["id"])
            symbol = str(sig["symbol"]).upper()
            side = str(sig["side"]).lower()
            strength = float(sig["strength"])
            price = float(sig["price"] or 0.0)

            now_ts = time.time()
            last_ts = last_submit_ts_by_symbol.get(symbol, 0.0)
            if cfg.symbol_cooldown_seconds > 0 and (now_ts - last_ts) < cfg.symbol_cooldown_seconds:
                detail = f"symbol={symbol} cooldown={cfg.symbol_cooldown_seconds}s"
                LOG.info("cooldown_skip | %s", detail)
                _safe_audit_block(sig, "cooldown", detail)
                time.sleep(cfg.poll_seconds)
                continue

            if side not in ("buy", "sell"):
                LOG.info("skip invalid side | %s", side)
                _safe_audit_block(sig, "invalid_side", f"side={side}")
                seen_signal_ids.add(sig_id)
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.long_only and side == "sell":
                pos_qty = 0.0
                for p in positions:
                    psym = str(getattr(p, "symbol", "") or "").upper()
                    if psym == symbol:
                        pos_qty = float(getattr(p, "qty", 0) or 0)
                        break

                if pos_qty > 0:
                    if not cfg.exit_on_sell_signal:
                        detail = f"symbol={symbol} EXIT_ON_SELL_SIGNAL=0"
                        LOG.info("skip sell signal | %s", detail)
                        _safe_audit_block(sig, "sell_disabled_long_only", detail)
                        seen_signal_ids.add(sig_id)
                        time.sleep(cfg.poll_seconds)
                        continue

                    if cfg.min_hold_minutes > 0:
                        entry_ts = last_entry_ts_by_symbol.get(symbol)
                        if not entry_ts:
                            detail = f"symbol={symbol} MIN_HOLD_MINUTES={cfg.min_hold_minutes} entry_ts unknown"
                            LOG.warning("skip sell exit | %s", detail)
                            _safe_audit_block(sig, "min_hold_entry_ts_unknown", detail)
                            seen_signal_ids.add(sig_id)
                            time.sleep(cfg.poll_seconds)
                            continue

                        hold_sec = cfg.min_hold_minutes * 60
                        age_sec = now_ts - entry_ts
                        if age_sec < hold_sec:
                            detail = f"symbol={symbol} age_sec={age_sec:.1f} hold_sec={hold_sec}"
                            LOG.info("skip sell exit | %s", detail)
                            _safe_audit_block(sig, "min_hold_not_reached", detail)
                            seen_signal_ids.add(sig_id)
                            time.sleep(cfg.poll_seconds)
                            continue

                    cid = f"X-{symbol}-STC-{sig_id}"
                    if cfg.dry_run:
                        detail = f"DRY_RUN exit_long symbol={symbol} qty={pos_qty} cid={cid}"
                        LOG.info(detail)
                        _safe_audit_submit(sig, detail)
                    else:
                        req = MarketOrderRequest(
                            symbol=symbol,
                            qty=float(pos_qty),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            type=OrderType.MARKET,
                            client_order_id=cid,
                            position_intent=PositionIntent.SELL_TO_CLOSE,
                        )
                        o = tc.submit_order(order_data=req)
                        oid = str(getattr(o, "id", "") or "")
                        detail = f"exit_long_submitted symbol={symbol} qty={pos_qty} oid={oid} cid={cid}"
                        LOG.info(detail)
                        _safe_audit_submit(sig, detail)

                    seen_signal_ids.add(sig_id)
                    last_submit_ts_by_symbol[symbol] = now_ts
                    time.sleep(cfg.poll_seconds)
                    continue

                detail = f"symbol={symbol} long_only and no long to close"
                LOG.info("skip sell | %s", detail)
                _safe_audit_block(sig, "sell_no_position", detail)
                seen_signal_ids.add(sig_id)
                time.sleep(cfg.poll_seconds)
                continue

            if price <= 0:
                LOG.info("skip no_price | %s", symbol)
                _safe_audit_block(sig, "no_price", f"symbol={symbol} price={price}")
                time.sleep(cfg.poll_seconds)
                continue

            if not cfg.allow_add_to_position:
                cur_qty = 0.0
                for p in positions:
                    psym = str(getattr(p, "symbol", "") or "").upper()
                    if psym == symbol:
                        cur_qty = float(getattr(p, "qty", 0) or 0)
                        break
                if cur_qty != 0:
                    detail = f"symbol={symbol} current_qty={cur_qty}"
                    LOG.info("skip_add_to_position | %s", detail)
                    _safe_audit_block(sig, "add_to_position_blocked", detail)
                    seen_signal_ids.add(sig_id)
                    last_submit_ts_by_symbol[symbol] = now_ts
                    time.sleep(cfg.poll_seconds)
                    continue

            if not _dedupe_ok(engine, symbol, side, cfg.alpaca_dedupe_minutes):
                detail = f"symbol={symbol} side={side} dedupe_minutes={cfg.alpaca_dedupe_minutes}"
                LOG.info("dedupe_skip | %s", detail)
                _safe_audit_block(sig, "dedupe", detail)
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.cancel_opposite_open_orders:
                _cancel_opposite(tc, symbol, side, open_orders)

            trade_notional = cfg.max_notional
            if side == "buy":
                trade_notional = _scaled_notional(cfg, strength)

            qty = _qty_from_notional(cfg, price, trade_notional)
            if qty <= 0:
                detail = f"symbol={symbol} side={side} price={price} trade_notional={trade_notional}"
                LOG.info("skip qty<=0 | %s", detail)
                _safe_audit_block(sig, "qty_zero", detail)
                time.sleep(cfg.poll_seconds)
                continue

            kind = "LIMIT"
            notional = 0.0
            limit_price = price

            if cfg.allow_fractional_market and side == "buy" and qty < 1.0:
                kind = "MARKET_NOTIONAL"
                notional = float(f"{trade_notional:.2f}")
                if notional <= 0:
                    detail = f"symbol={symbol} side={side} notional={notional}"
                    LOG.info("skip notional<=0 | %s", detail)
                    _safe_audit_block(sig, "qty_zero", detail)
                    time.sleep(cfg.poll_seconds)
                    continue

            client_order_id = _mk_client_order_id(symbol, side, sig_id, kind)

            if cfg.dry_run:
                detail = (
                    f"DRY_RUN would_submit symbol={symbol} side={side} kind={kind} "
                    f"notional={float(trade_notional):.2f} qty={qty:.6f} price={price:.4f} "
                    f"strength={strength:.4f} cid={client_order_id}"
                )
                LOG.info(detail)
                _safe_audit_submit(sig, detail)
                if side == "buy":
                    last_entry_ts_by_symbol[symbol] = now_ts
            else:
                if kind == "MARKET_NOTIONAL":
                    oid = _place_market_notional(tc, symbol, side, notional, client_order_id)
                    detail = (
                        f"submitted symbol={symbol} side={side} kind={kind} "
                        f"notional={notional:.2f} oid={oid} strength={strength:.4f} cid={client_order_id}"
                    )
                    LOG.info(detail)
                    _safe_audit_submit(sig, detail)
                else:
                    oid = _place_limit(tc, symbol, side, qty, limit_price, client_order_id)
                    detail = (
                        f"submitted symbol={symbol} side={side} kind={kind} qty={qty:.6f} "
                        f"limit={limit_price:.4f} notional={float(trade_notional):.2f} "
                        f"oid={oid} strength={strength:.4f} cid={client_order_id}"
                    )
                    LOG.info(detail)
                    _safe_audit_submit(sig, detail)

                if side == "buy":
                    last_entry_ts_by_symbol[symbol] = now_ts

            seen_signal_ids.add(sig_id)
            last_submit_ts_by_symbol[symbol] = now_ts

        except Exception as e:
            LOG.exception("loop_error: %r", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()