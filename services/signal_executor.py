import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import UUID

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

try:
    from services.market_gate import should_trade_now  # type: ignore
except Exception:  # pragma: no cover
    def should_trade_now(*_args, **_kwargs) -> bool:  # type: ignore
        return True

logger = logging.getLogger("signal_executor")


def _s(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _i(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _b(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def parse_side(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("buy", "long"):
        return "buy"
    if s in ("sell", "short"):
        return "sell"
    return s


@dataclass(frozen=True)
class Config:
    database_url: str
    symbols: List[str]
    min_strength: float
    portfolio_id: int
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

    data_base_url: str
    data_feed: str

    @staticmethod
    def from_env() -> "Config":
        return Config(
            database_url=_s("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@postgres:5432/trader"),
            symbols=parse_symbols(_s("SYMBOLS", "AAPL,MSFT,SPY,NVDA,AMD")),
            min_strength=_f("MIN_STRENGTH", 0.60),
            portfolio_id=_i("PORTFOLIO_ID", 1),
            poll_seconds=_i("SIGNAL_EXECUTOR_POLL_SECONDS", 20),

            allow_short=_b("ALLOW_SHORT", True),
            long_only=_b("LONG_ONLY", False),
            max_notional=_f("MAX_NOTIONAL", 1200.0),
            max_qty=_i("MAX_QTY", 5),

            alpaca_dedupe_minutes=_i("ALPACA_DEDUPE_MINUTES", 10),
            cancel_opposite_open_orders=_b("CANCEL_OPPOSITE_OPEN_ORDERS", True),

            max_open_positions=_i("MAX_OPEN_POSITIONS", 2),
            max_open_orders=_i("MAX_OPEN_ORDERS", 3),

            enable_daily_risk_guard=_b("ENABLE_DAILY_RISK_GUARD", False),
            daily_loss_stop_pct=_f("DAILY_LOSS_STOP_PCT", 1.0),
            max_daily_loss_usd=_f("MAX_DAILY_LOSS_USD", 200.0),

            symbol_cooldown_seconds=_i("SYMBOL_COOLDOWN_SECONDS", 0),
            pick_ttl_seconds=_i("PICK_TTL_SECONDS", 0),

            data_base_url=_s("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
            data_feed=_s("ALPACA_DATA_FEED", "iex"),
        )


def make_engine(cfg: Config) -> Engine:
    return create_engine(cfg.database_url, pool_pre_ping=True)


def fetch_unprocessed_signals(engine: Engine, cfg: Config, limit: int = 250) -> List[dict]:
    q = text(
        "select id, created_at, symbol, side, strength "
        "from signals "
        "where processed_at is null and portfolio_id=:pid and symbol=any(:symbols) and strength>=:min_strength "
        "order by created_at asc "
        "limit :limit"
    )
    with engine.begin() as con:
        rows = con.execute(
            q,
            {"pid": cfg.portfolio_id, "symbols": cfg.symbols, "min_strength": cfg.min_strength, "limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def select_signals(rows: List[dict]) -> List[dict]:
    best: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        k = (r["symbol"], r["side"])
        if k not in best or float(r["strength"]) > float(best[k]["strength"]):
            best[k] = r
    out = list(best.values())
    out.sort(key=lambda x: float(x["strength"]), reverse=True)
    return out


def mark_signal(engine: Engine, sid: int, status: str, note: str, alpaca_order_id: Optional[str] = None) -> None:
    q = text(
        "update signals set processed_at=now(), processed_status=:status, processed_note=:note, "
        "alpaca_order_id=coalesce(:alpaca_order_id, alpaca_order_id) where id=:id"
    )
    with engine.begin() as con:
        con.execute(q, {"id": sid, "status": status, "note": note, "alpaca_order_id": alpaca_order_id})


def mark_signals_picked(engine: Engine, ids: Sequence[int]) -> None:
    if not ids:
        return
    q = text(
        "update signals set processed_at=now(), processed_status='picked', processed_note='picked' "
        "where id=any(:ids) and processed_at is null"
    )
    with engine.begin() as con:
        con.execute(q, {"ids": list(ids)})


def unpick_signals(engine: Engine, ids: Sequence[int], note: str) -> int:
    if not ids:
        return 0
    q = text(
        "update signals set processed_at=null, processed_status=null, processed_note=:note "
        "where id=any(:ids) and processed_status='picked'"
    )
    with engine.begin() as con:
        res = con.execute(q, {"ids": list(ids), "note": note})
        return int(res.rowcount or 0)


def release_stale_picks(engine: Engine, cfg: Config) -> int:
    if cfg.pick_ttl_seconds <= 0:
        return 0
    q = text(
        "update signals set processed_at=null, processed_status=null, processed_note='pick_expired' "
        "where portfolio_id=:pid and processed_status='picked' and processed_at is not null "
        "and processed_at < (now() - make_interval(secs => :ttl))"
    )
    with engine.begin() as con:
        res = con.execute(q, {"pid": cfg.portfolio_id, "ttl": cfg.pick_ttl_seconds})
        return int(res.rowcount or 0)


def last_trade_ts(engine: Engine, cfg: Config, symbol: str) -> Optional[datetime]:
    # cooldown ONLY on submitted/filled (NOT on skipped), aby sa cooldown “nepredlžoval” skipmi
    q = text(
        "select max(processed_at) as last_ts from signals "
        "where portfolio_id=:pid and symbol=:symbol and processed_at is not null "
        "and processed_status in ('submitted','filled')"
    )
    with engine.begin() as con:
        row = con.execute(q, {"pid": cfg.portfolio_id, "symbol": symbol}).mappings().first()
    if not row or row["last_ts"] is None:
        return None
    return row["last_ts"]


def in_symbol_cooldown(engine: Engine, cfg: Config, symbol: str) -> bool:
    if cfg.symbol_cooldown_seconds <= 0:
        return False
    lt = last_trade_ts(engine, cfg, symbol)
    if lt is None:
        return False
    return (utcnow() - lt).total_seconds() < cfg.symbol_cooldown_seconds


def get_latest_price(cfg: Config, symbol: str) -> float:
    url = f"{cfg.data_base_url.rstrip('/')}/v2/stocks/{symbol}/trades/latest"
    headers = {"APCA-API-KEY-ID": _s("ALPACA_API_KEY"), "APCA-API-SECRET-KEY": _s("ALPACA_API_SECRET")}
    r = requests.get(url, headers=headers, params={"feed": cfg.data_feed}, timeout=10)
    r.raise_for_status()
    j = r.json()
    trade = j.get("trade") or {}
    if "p" in trade:
        return float(trade["p"])
    if "price" in trade:
        return float(trade["price"])
    raise RuntimeError(f"unexpected payload: {j}")


def open_orders(tc: TradingClient):
    return tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))


def count_open_orders(tc: TradingClient) -> Tuple[int, List[str]]:
    os_ = open_orders(tc)
    return len(os_), [f"{o.symbol}:{o.side.name}:{o.qty}:{o.type.name}:{o.status.name}" for o in os_]


def count_open_positions(tc: TradingClient) -> Tuple[int, List[str]]:
    ps = tc.get_all_positions()
    n = 0
    out: List[str] = []
    for p in ps:
        try:
            qty = float(p.qty)
        except Exception:
            qty = 0.0
        if abs(qty) > 0:
            n += 1
            out.append(f"{p.symbol}:{p.qty}")
    return n, out


def daily_loss_exceeded(tc: TradingClient, cfg: Config) -> Tuple[bool, str]:
    if not cfg.enable_daily_risk_guard:
        return (False, "")
    a = tc.get_account()
    if not hasattr(a, "equity") or not hasattr(a, "last_equity"):
        return (False, "daily_guard_unavailable_missing_last_equity")
    try:
        equity = float(a.equity)
        last_equity = float(a.last_equity)
    except Exception:
        return (False, "daily_guard_unavailable_parse_error")
    if last_equity <= 0:
        return (False, "daily_guard_unavailable_last_equity<=0")
    pl_usd = equity - last_equity
    pl_pct = (equity / last_equity - 1.0) * 100.0
    if cfg.max_daily_loss_usd > 0 and pl_usd <= -abs(cfg.max_daily_loss_usd):
        return (True, f"max_daily_loss_usd_exceeded pl_usd={pl_usd:.2f} <= -{abs(cfg.max_daily_loss_usd):.2f}")
    if cfg.daily_loss_stop_pct > 0 and pl_pct <= -abs(cfg.daily_loss_stop_pct):
        return (True, f"daily_loss_stop_pct_exceeded pl_pct={pl_pct:.2f}% <= -{abs(cfg.daily_loss_stop_pct):.2f}%")
    return (False, f"daily_pl_usd={pl_usd:.2f} daily_pl_pct={pl_pct:.2f}%")


def risk_guard_blocked(tc: TradingClient, cfg: Config) -> Tuple[bool, str]:
    blocked, reason = daily_loss_exceeded(tc, cfg)
    if blocked:
        return (True, reason)

    n_pos, pos_sum = count_open_positions(tc)
    if cfg.max_open_positions > 0 and n_pos >= cfg.max_open_positions:
        return (True, f"max_open_positions_reached:{n_pos}/{cfg.max_open_positions} pos={pos_sum}")

    n_ord, ord_sum = count_open_orders(tc)
    if cfg.max_open_orders > 0 and n_ord >= cfg.max_open_orders:
        return (True, f"max_open_orders_reached:{n_ord}/{cfg.max_open_orders} ord={ord_sum}")

    return (False, "")


def recent_orders(tc: TradingClient, minutes: int) -> List[object]:
    cutoff = utcnow() - timedelta(minutes=minutes)
    orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
    out = []
    for o in orders:
        sa = getattr(o, "submitted_at", None)
        if sa and sa >= cutoff:
            out.append(o)
    return out


def has_recent_order(tc: TradingClient, symbol: str, side: OrderSide, minutes: int) -> bool:
    if minutes <= 0:
        return False
    try:
        orders = recent_orders(tc, minutes)
    except Exception:
        logger.exception("recent orders fetch failed")
        return False
    for o in orders:
        if getattr(o, "symbol", None) == symbol and getattr(o, "side", None) == side:
            st = getattr(o, "status", None)
            st_name = getattr(st, "name", str(st)).lower()
            if st_name not in ("canceled", "expired", "rejected"):
                return True
    return False


def cancel_opposite_open_orders(tc: TradingClient, symbol: str, new_side: OrderSide) -> int:
    opposite = OrderSide.SELL if new_side == OrderSide.BUY else OrderSide.BUY
    canceled = 0
    for o in open_orders(tc):
        if o.symbol == symbol and o.side == opposite:
            try:
                tc.cancel_order_by_id(o.id)
                canceled += 1
            except Exception:
                logger.exception("cancel opposite order failed | %s %s %s", symbol, o.side, o.id)
    return canceled


def update_submitted_fills(engine: Engine, tc: TradingClient, cfg: Config, lookback_minutes: int = 180) -> int:
    q = text(
        "select id, alpaca_order_id from signals "
        "where portfolio_id=:pid and processed_status='submitted' and alpaca_order_id is not null "
        "and processed_at > (now() - make_interval(secs => :secs)) "
        "order by processed_at desc limit 200"
    )
    with engine.begin() as con:
        rows = con.execute(q, {"pid": cfg.portfolio_id, "secs": lookback_minutes * 60}).mappings().all()

    updated = 0
    for r in rows:
        sid = int(r["id"])
        oid_raw = str(r["alpaca_order_id"])
        try:
            oid: object = UUID(oid_raw)
        except Exception:
            oid = oid_raw
        try:
            o = tc.get_order_by_id(oid)
        except Exception:
            logger.exception("get_order_by_id failed | sid=%s oid=%s", sid, oid_raw)
            continue
        st = getattr(o, "status", None)
        st_name = getattr(st, "name", str(st)).lower()
        if st_name == "filled":
            mark_signal(engine, sid, "filled", "order filled", alpaca_order_id=oid_raw)
            updated += 1
        elif st_name in ("canceled", "expired"):
            mark_signal(engine, sid, "canceled", f"order {st_name}", alpaca_order_id=oid_raw)
            updated += 1
        elif st_name == "rejected":
            mark_signal(engine, sid, "error", "order rejected", alpaca_order_id=oid_raw)
            updated += 1
    return updated


def size_qty(cfg: Config, limit_price: float) -> int:
    # Allow at least 1 share even if price > MAX_NOTIONAL (bounded by MAX_QTY)
    if cfg.max_qty <= 0:
        return 0
    if cfg.max_notional <= 0:
        return cfg.max_qty
    qty_by_notional = max(1, int(cfg.max_notional // max(limit_price, 0.01)))
    return max(0, min(cfg.max_qty, qty_by_notional))


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = Config.from_env()
    engine = make_engine(cfg)

    tc = TradingClient(_s("ALPACA_API_KEY"), _s("ALPACA_API_SECRET"), paper=_b("ALPACA_PAPER", True))

    logger.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | ALPACA_DEDUPE_MINUTES=%s | "
        "CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | "
        "DAILY_LOSS_STOP_PCT=%s | MAX_DAILY_LOSS_USD=%s | ENABLE_DAILY_RISK_GUARD=%s | "
        "SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s",
        cfg.min_strength, cfg.symbols, cfg.portfolio_id, cfg.poll_seconds,
        cfg.allow_short, cfg.long_only, cfg.max_notional, cfg.max_qty,
        cfg.alpaca_dedupe_minutes, cfg.cancel_opposite_open_orders,
        cfg.max_open_positions, cfg.max_open_orders,
        cfg.daily_loss_stop_pct, cfg.max_daily_loss_usd, cfg.enable_daily_risk_guard,
        cfg.symbol_cooldown_seconds, cfg.pick_ttl_seconds,
    )

    last_block_reason: Optional[str] = None
    last_block_log_ts = 0.0

    while True:
        try:
            if not should_trade_now(tc):
                time.sleep(cfg.poll_seconds)
                continue

            released = release_stale_picks(engine, cfg)
            if released:
                logger.warning("released_stale_picks | released=%s (ttl=%ss)", released, cfg.pick_ttl_seconds)

            try:
                upd = update_submitted_fills(engine, tc, cfg, lookback_minutes=180)
                if upd:
                    logger.info("updated_submitted_fills | updated=%s", upd)
            except Exception:
                logger.exception("update_submitted_fills error")

            rows = fetch_unprocessed_signals(engine, cfg, limit=250)
            logger.info("fetch_new_signals | fetched %s rows", len(rows))
            if not rows:
                time.sleep(cfg.poll_seconds)
                continue

            selected = select_signals(rows)
            logger.info("select_signals | fetched=%s | selected=%s | unique_symbol_side=%s", len(rows), len(selected), len(selected))

            candidates: List[dict] = []
            for s in selected:
                sid = int(s["id"])
                symbol = str(s["symbol"]).upper()
                side = parse_side(str(s["side"]))

                if side == "sell" and (cfg.long_only or not cfg.allow_short):
                    mark_signal(engine, sid, "skipped", "short_disabled")
                    logger.info("skip | sid=%s %s %s | short_disabled", sid, symbol, side)
                    continue

                if in_symbol_cooldown(engine, cfg, symbol):
                    mark_signal(engine, sid, "skipped", f"symbol_cooldown_{cfg.symbol_cooldown_seconds}s")
                    logger.info("skip | sid=%s %s %s | symbol_cooldown_%ss", sid, symbol, side, cfg.symbol_cooldown_seconds)
                    continue

                candidates.append(s)

            if not candidates:
                time.sleep(cfg.poll_seconds)
                continue

            blocked, reason = risk_guard_blocked(tc, cfg)
            if blocked:
                now_ts = time.time()
                if reason != last_block_reason or (now_ts - last_block_log_ts) > 60:
                    logger.warning("risk_guard | blocked new entries | %s", reason)
                    last_block_reason = reason
                    last_block_log_ts = now_ts
                else:
                    logger.info("risk_guard | blocked new entries | %s", reason)

                for s in candidates:
                    mark_signal(engine, int(s["id"]), "skipped", f"risk_guard:{reason.split()[0]}")
                time.sleep(cfg.poll_seconds)
                continue

            last_block_reason = None

            ids = [int(s["id"]) for s in candidates]
            mark_signals_picked(engine, ids)

            for idx, s in enumerate(candidates):
                sid = int(s["id"])
                symbol = str(s["symbol"]).upper()
                side = parse_side(str(s["side"]))
                strength = float(s["strength"])

                blocked_mid, reason_mid = risk_guard_blocked(tc, cfg)
                if blocked_mid:
                    remaining = [int(x["id"]) for x in candidates[idx:]]
                    unpicked = unpick_signals(engine, remaining, f"unpicked_due_to_risk_guard:{reason_mid.split()[0]}")
                    logger.warning("risk_guard | blocked mid-batch | %s | unpicked_remaining=%s", reason_mid, unpicked)
                    break

                alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

                if has_recent_order(tc, symbol, alpaca_side, cfg.alpaca_dedupe_minutes):
                    mark_signal(engine, sid, "skipped", f"dedupe_alpaca_{cfg.alpaca_dedupe_minutes}m")
                    logger.info("skip | sid=%s %s %s | dedupe_alpaca_%sm", sid, symbol, side, cfg.alpaca_dedupe_minutes)
                    continue

                try:
                    px = get_latest_price(cfg, symbol)
                except Exception as e:
                    logger.exception("price fetch failed | sid=%s %s | %s", sid, symbol, e)
                    mark_signal(engine, sid, "error", f"price_fetch_failed:{type(e).__name__}")
                    continue

                limit_price = round(px, 2)
                qty = size_qty(cfg, limit_price)
                if qty <= 0:
                    mark_signal(engine, sid, "skipped", "qty_zero")
                    logger.info("skip | sid=%s %s %s | qty_zero", sid, symbol, side)
                    continue

                if cfg.cancel_opposite_open_orders:
                    canceled = cancel_opposite_open_orders(tc, symbol, alpaca_side)
                    if canceled:
                        logger.info("canceled_opposite_open_orders | %s | canceled=%s", symbol, canceled)

                try:
                    o = tc.submit_order(
                        LimitOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=alpaca_side,
                            time_in_force=TimeInForce.DAY,
                            limit_price=limit_price,
                        )
                    )
                    mark_signal(engine, sid, "submitted", "submitted", alpaca_order_id=str(o.id))
                    logger.info(
                        "submitted | sid=%s | %s %s qty=%s limit=%.2f strength=%.4f | alpaca_id=%s",
                        sid, symbol, side, qty, limit_price, strength, o.id
                    )
                except Exception as e:
                    logger.exception("submit_order failed | sid=%s %s %s | %s", sid, symbol, side, e)
                    mark_signal(engine, sid, "error", f"submit_failed:{type(e).__name__}")

            time.sleep(cfg.poll_seconds)

        except Exception:
            logger.exception("signal_executor loop error")
            time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
