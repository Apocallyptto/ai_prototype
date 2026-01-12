import os
import time
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import create_engine, text

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest


LOG = logging.getLogger("signal_executor")


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(float(v))
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


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_symbols(raw: str) -> List[str]:
    syms = [s.strip().upper() for s in raw.replace(";", ",").split(",") if s.strip()]
    out: List[str] = []
    seen: Set[str] = set()
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


@dataclass(frozen=True)
class Cfg:
    symbols: List[str]
    portfolio_id: int
    min_strength: float
    poll_seconds: int

    allow_short: bool
    long_only: bool
    max_notional: float
    max_qty: int

    alpaca_dedupe_minutes: int
    cancel_opposite_open_orders: bool

    max_open_positions: int
    max_open_orders: int

    enable_daily_risk_guard: bool
    daily_loss_stop_pct: float
    max_daily_loss_usd: float

    symbol_cooldown_seconds: int
    pick_ttl_seconds: int

    trading_paused: bool
    dry_run: bool


def load_cfg() -> Cfg:
    return Cfg(
        symbols=_parse_symbols(_env_str("SYMBOLS", "AAPL,MSFT,SPY,NVDA,AMD")),
        portfolio_id=_env_int("PORTFOLIO_ID", 1),
        min_strength=_env_float("MIN_STRENGTH", 0.75),
        poll_seconds=_env_int("POLL_SECONDS", _env_int("SIGNAL_POLL_SECONDS", 20)),
        allow_short=_env_bool("ALLOW_SHORT", False),
        long_only=_env_bool("LONG_ONLY", False),
        max_notional=_env_float("MAX_NOTIONAL", 200.0),
        max_qty=_env_int("MAX_QTY", 1),
        alpaca_dedupe_minutes=_env_int("ALPACA_DEDUPE_MINUTES", 2),
        cancel_opposite_open_orders=_env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True),
        max_open_positions=_env_int("MAX_OPEN_POSITIONS", 1),
        max_open_orders=_env_int("MAX_OPEN_ORDERS", 1),
        enable_daily_risk_guard=_env_bool("ENABLE_DAILY_RISK_GUARD", True),
        daily_loss_stop_pct=_env_float("DAILY_LOSS_STOP_PCT", 1.0),
        max_daily_loss_usd=_env_float("MAX_DAILY_LOSS_USD", 200.0),
        symbol_cooldown_seconds=_env_int("SYMBOL_COOLDOWN_SECONDS", 60),
        pick_ttl_seconds=_env_int("PICK_TTL_SECONDS", 120),
        trading_paused=_env_bool("TRADING_PAUSED", False),  # kill switch
        dry_run=_env_bool("DRY_RUN", False),
    )


def _setup_logging() -> None:
    level = _env_str("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def make_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        host = _env_str("POSTGRES_HOST", _env_str("PGHOST", "postgres"))
        port = _env_int("POSTGRES_PORT", _env_int("PGPORT", 5432))
        user = _env_str("POSTGRES_USER", _env_str("PGUSER", "postgres"))
        pwd = _env_str("POSTGRES_PASSWORD", _env_str("PGPASSWORD", "postgres"))
        db = _env_str("POSTGRES_DB", _env_str("PGDATABASE", "trader"))
        db_url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(db_url, pool_pre_ping=True)


def make_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")
    return TradingClient(key, sec, paper=_env_bool("ALPACA_PAPER", True))


def make_data_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")
    return StockHistoricalDataClient(key, sec)


def fetch_new_signals(engine, cfg: Cfg, limit: int = 300) -> List[dict]:
    q = text(
        """
        SELECT id, created_at, symbol, side, strength
        FROM signals
        WHERE processed_at IS NULL
          AND portfolio_id = :pid
          AND symbol = ANY(:symbols)
          AND strength >= :min_strength
        ORDER BY created_at ASC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            q,
            {"pid": cfg.portfolio_id, "symbols": cfg.symbols, "min_strength": cfg.min_strength, "limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def mark(engine, ids: Sequence[int], status: str, note: str) -> None:
    if not ids:
        return
    q = text(
        """
        UPDATE signals
        SET processed_at = now(),
            processed_status = :status,
            processed_note = :note
        WHERE id = ANY(:ids)
        """
    )
    with engine.begin() as conn:
        conn.execute(q, {"ids": list(ids), "status": status, "note": note})


def mark_one(engine, sid: int, status: str, note: str) -> None:
    mark(engine, [sid], status, note)


def pick(engine, ids: Sequence[int]) -> None:
    if not ids:
        return
    q = text(
        """
        UPDATE signals
        SET processed_at = now(),
            processed_status = 'picked',
            processed_note = 'picked'
        WHERE id = ANY(:ids)
          AND processed_at IS NULL
        """
    )
    with engine.begin() as conn:
        conn.execute(q, {"ids": list(ids)})


def unpick_stale_picks(engine, cfg: Cfg) -> int:
    q = text(
        """
        UPDATE signals
        SET processed_at = NULL,
            processed_status = NULL,
            processed_note = 'auto_unpick_pick_ttl'
        WHERE processed_status = 'picked'
          AND processed_at < (now() - (:ttl || ' seconds')::interval)
        """
    )
    with engine.begin() as conn:
        res = conn.execute(q, {"ttl": cfg.pick_ttl_seconds})
        return int(res.rowcount or 0)


def get_symbol_last_trade_ts(engine, symbols: Sequence[str]) -> Dict[str, datetime]:
    q = text(
        """
        SELECT symbol, max(processed_at) AS last_ts
        FROM signals
        WHERE symbol = ANY(:symbols)
          AND processed_at IS NOT NULL
          AND processed_status IN ('submitted','filled')
        GROUP BY symbol
        """
    )
    out: Dict[str, datetime] = {}
    with engine.connect() as conn:
        for r in conn.execute(q, {"symbols": list(symbols)}).mappings().all():
            if r["last_ts"] is not None:
                out[str(r["symbol"]).upper()] = r["last_ts"]
    return out


def select_signals(engine, cfg: Cfg, signals: List[dict]) -> Tuple[List[dict], int]:
    now = _now_utc()
    selected: List[dict] = []
    seen: Set[Tuple[str, str]] = set()
    stale_ids: List[int] = []

    for s in signals:
        sid = int(s["id"])
        sym = str(s["symbol"]).upper()
        side = str(s["side"]).lower()
        created_at = s["created_at"]

        if isinstance(created_at, datetime):
            ca = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
            if (now - ca).total_seconds() > cfg.pick_ttl_seconds:
                stale_ids.append(sid)
                continue

        k = (sym, side)
        if k in seen:
            continue
        seen.add(k)
        selected.append(s)

    if stale_ids:
        mark(engine, stale_ids, "skipped", f"stale_signal_ttl_{cfg.pick_ttl_seconds}s")

    return selected, len(stale_ids)


def risk_guard(tc: TradingClient, cfg: Cfg) -> Tuple[bool, str]:
    positions = tc.get_all_positions()
    open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))

    if len(positions) >= cfg.max_open_positions:
        return False, f"max_open_positions_reached:{len(positions)}/{cfg.max_open_positions}"
    if len(open_orders) >= cfg.max_open_orders:
        return False, f"max_open_orders_reached:{len(open_orders)}/{cfg.max_open_orders}"

    if cfg.enable_daily_risk_guard:
        try:
            acct = tc.get_account()
            equity = float(acct.equity)
            last_equity = getattr(acct, "last_equity", None)
            if last_equity is not None:
                last_equity_f = float(last_equity)
                if last_equity_f > 0:
                    dd_pct = max(0.0, (last_equity_f - equity) / last_equity_f * 100.0)
                    if dd_pct >= cfg.daily_loss_stop_pct:
                        return False, f"daily_loss_stop_pct_hit:{dd_pct:.2f}%"
                loss_usd = max(0.0, last_equity_f - equity)
                if loss_usd >= cfg.max_daily_loss_usd:
                    return False, f"max_daily_loss_usd_hit:{loss_usd:.2f}"
        except Exception as e:
            LOG.warning("risk_guard | daily guard calc failed: %s", e)

    return True, ""


def _recent_alpaca_dedupe(tc: TradingClient, symbol: str, side: OrderSide, window_minutes: int) -> bool:
    if window_minutes <= 0:
        return False
    cutoff = _now_utc() - timedelta(minutes=window_minutes)
    orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200))
    for o in orders:
        if str(o.symbol).upper() != symbol.upper():
            continue
        if o.side != side:
            continue
        if o.submitted_at and o.submitted_at >= cutoff:
            return True
    return False


def _cancel_opposite_open_orders(tc: TradingClient, symbol: str, side: OrderSide) -> int:
    opposite = OrderSide.BUY if side == OrderSide.SELL else OrderSide.SELL
    canceled = 0
    opens = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    for o in opens:
        if str(o.symbol).upper() == symbol.upper() and o.side == opposite:
            tc.cancel_order_by_id(o.id)
            canceled += 1
    return canceled


def _latest_trade_price(data: StockHistoricalDataClient, symbol: str) -> Optional[float]:
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=[symbol])
        resp = data.get_stock_latest_trade(req)
        t = resp.get(symbol)
        if t and t.price is not None:
            return float(t.price)
    except Exception as e:
        LOG.warning("price | latest trade failed for %s: %s", symbol, e)
    return None


def _calc_qty(limit_price: float, cfg: Cfg) -> int:
    if limit_price <= 0:
        return 0
    qty_by_notional = int(math.floor(cfg.max_notional / limit_price))
    return max(0, min(cfg.max_qty, qty_by_notional if qty_by_notional > 0 else 0))


def submit_limit(tc: TradingClient, symbol: str, side: OrderSide, qty: int, limit_price: float) -> str:
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        limit_price=round(float(limit_price), 2),
    )
    o = tc.submit_order(req)
    return str(o.id)


def main() -> None:
    _setup_logging()
    cfg = load_cfg()
    engine = make_engine()
    tc = make_trading_client()
    data = make_data_client()

    LOG.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | ALPACA_DEDUPE_MINUTES=%s | "
        "CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | DAILY_LOSS_STOP_PCT=%.1f | "
        "MAX_DAILY_LOSS_USD=%.1f | ENABLE_DAILY_RISK_GUARD=%s | SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s | "
        "TRADING_PAUSED=%s | DRY_RUN=%s",
        cfg.min_strength, cfg.symbols, cfg.portfolio_id, cfg.poll_seconds,
        cfg.allow_short, cfg.long_only, cfg.max_notional, cfg.max_qty, cfg.alpaca_dedupe_minutes,
        cfg.cancel_opposite_open_orders, cfg.max_open_positions, cfg.max_open_orders,
        cfg.daily_loss_stop_pct, cfg.max_daily_loss_usd, cfg.enable_daily_risk_guard,
        cfg.symbol_cooldown_seconds, cfg.pick_ttl_seconds, cfg.trading_paused, cfg.dry_run
    )

    while True:
        try:
            n = unpick_stale_picks(engine, cfg)
            if n:
                LOG.info("auto_unpick | count=%s", n)

            signals = fetch_new_signals(engine, cfg)
            LOG.info("fetch_new_signals | fetched %s rows", len(signals))
            if not signals:
                time.sleep(cfg.poll_seconds)
                continue

            selected, stale_cnt = select_signals(engine, cfg, signals)
            LOG.info(
                "select_signals | fetched=%s | selected=%s | unique_symbol_side=%s%s",
                len(signals), len(selected),
                len({(str(s['symbol']).upper(), str(s['side']).lower()) for s in selected}),
                f" | stale_skipped={stale_cnt}" if stale_cnt else ""
            )

            if not selected:
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.trading_paused or cfg.dry_run:
                note = "paused" if cfg.trading_paused else "dry_run"
                mark(engine, [int(s["id"]) for s in selected], "skipped", note)
                for s in selected:
                    LOG.info("skip | sid=%s %s %s | %s", s["id"], s["symbol"], s["side"], note)
                time.sleep(cfg.poll_seconds)
                continue

            last_ts = get_symbol_last_trade_ts(engine, cfg.symbols)
            now = _now_utc()

            pick(engine, [int(s["id"]) for s in selected])

            for idx, s in enumerate(selected):
                sid = int(s["id"])
                sym = str(s["symbol"]).upper()
                side_s = str(s["side"]).lower()

                if cfg.long_only and side_s != "buy":
                    mark_one(engine, sid, "skipped", "long_only")
                    LOG.info("skip | sid=%s %s %s | long_only", sid, sym, side_s)
                    continue

                if side_s == "sell" and not cfg.allow_short:
                    mark_one(engine, sid, "skipped", "short_disabled")
                    LOG.info("skip | sid=%s %s %s | short_disabled", sid, sym, side_s)
                    continue

                ts = last_ts.get(sym)
                if ts is not None:
                    t0 = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                    if (now - t0).total_seconds() < cfg.symbol_cooldown_seconds:
                        mark_one(engine, sid, "skipped", f"symbol_cooldown_{cfg.symbol_cooldown_seconds}s")
                        LOG.info("skip | sid=%s %s %s | symbol_cooldown_%ss", sid, sym, side_s, cfg.symbol_cooldown_seconds)
                        continue

                ok, reason = risk_guard(tc, cfg)
                if not ok:
                    remaining = len(selected) - idx
                    LOG.warning("risk_guard | blocked mid-batch | %s | unpicked_remaining=%s", reason, remaining)
                    ids = [int(x["id"]) for x in selected[idx:]]
                    q = text(
                        """
                        UPDATE signals
                        SET processed_at = NULL,
                            processed_status = NULL,
                            processed_note = :note
                        WHERE id = ANY(:ids)
                        """
                    )
                    with engine.begin() as conn:
                        conn.execute(q, {"ids": ids, "note": f"unpicked_risk_guard:{reason}"})
                    break

                side = OrderSide.BUY if side_s == "buy" else OrderSide.SELL

                if _recent_alpaca_dedupe(tc, sym, side, cfg.alpaca_dedupe_minutes):
                    mark_one(engine, sid, "skipped", f"dedupe_alpaca_{cfg.alpaca_dedupe_minutes}m")
                    LOG.info("skip | sid=%s %s %s | dedupe_alpaca_%sm", sid, sym, side_s, cfg.alpaca_dedupe_minutes)
                    continue

                limit_price = _latest_trade_price(data, sym)
                if limit_price is None:
                    mark_one(engine, sid, "skipped", "no_price")
                    LOG.warning("skip | sid=%s %s %s | no_price", sid, sym, side_s)
                    continue

                qty = _calc_qty(limit_price, cfg)
                if qty <= 0:
                    mark_one(engine, sid, "skipped", "qty_zero_by_limits")
                    LOG.info("skip | sid=%s %s %s | qty_zero_by_limits", sid, sym, side_s)
                    continue

                if cfg.cancel_opposite_open_orders:
                    _cancel_opposite_open_orders(tc, sym, side)

                try:
                    alpaca_id = submit_limit(tc, sym, side, qty, limit_price)
                    mark_one(engine, sid, "submitted", "order submitted")
                    LOG.info(
                        "submitted | sid=%s | %s %s qty=%s limit=%.2f | alpaca_id=%s",
                        sid, sym, side_s, qty, round(limit_price, 2), alpaca_id
                    )
                    last_ts[sym] = _now_utc()
                except Exception as e:
                    mark_one(engine, sid, "error", f"submit_error:{type(e).__name__}")
                    LOG.exception("submit failed | sid=%s %s %s | %s", sid, sym, side_s, e)

            time.sleep(cfg.poll_seconds)

        except Exception as e:
            LOG.exception("loop error: %s", e)
            time.sleep(max(5, cfg.poll_seconds))


if __name__ == "__main__":
    main()
