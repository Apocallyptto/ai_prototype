# SPAM_GUARD_V1 + DRY_ORDERS_V1
import os
import time
import math
import uuid
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
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
    """
    Priority:
      1) TRADING_MODE=live|paper
      2) ALPACA_PAPER override (back-compat)
      3) default -> paper (fail-safe)
    """
    mode = (os.getenv("TRADING_MODE") or "paper").strip().lower()
    if mode not in ("live", "paper"):
        mode = "paper"

    paper = (mode != "live")

    if os.getenv("ALPACA_PAPER") is not None:
        paper = _env_bool("ALPACA_PAPER", paper)
        mode = "paper" if paper else "live"

    return mode, paper


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


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
    poll_seconds = int(os.getenv("POLL_SECONDS", "20"))
    min_strength = float(os.getenv("MIN_STRENGTH", "0.60"))
    symbols_raw = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))

    allow_short = _env_bool("ALLOW_SHORT", False)
    long_only = _env_bool("LONG_ONLY", True)

    max_notional = float(os.getenv("MAX_NOTIONAL", "100"))
    max_qty = float(os.getenv("MAX_QTY", "1"))
    max_position_qty = float(os.getenv("MAX_POSITION_QTY", "1"))
    allow_add_to_position = _env_bool("ALLOW_ADD_TO_POSITION", False)

    alpaca_dedupe_minutes = int(os.getenv("ALPACA_DEDUPE_MINUTES", "2"))
    cancel_opposite_open_orders = _env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True)

    max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
    max_open_orders = int(os.getenv("MAX_OPEN_ORDERS", "1"))

    daily_loss_stop_pct = float(os.getenv("DAILY_LOSS_STOP_PCT", "1.0"))
    max_daily_loss_usd = float(os.getenv("MAX_DAILY_LOSS_USD", "200.0"))
    enable_daily_risk_guard = _env_bool("ENABLE_DAILY_RISK_GUARD", True)

    symbol_cooldown_seconds = int(os.getenv("SYMBOL_COOLDOWN_SECONDS", "60"))
    pick_ttl_seconds = int(os.getenv("PICK_TTL_SECONDS", "120"))

    trade_only_when_market_open = _env_bool("TRADE_ONLY_WHEN_MARKET_OPEN", True)
    preopen_window_seconds = int(os.getenv("PREOPEN_WINDOW_SECONDS", "0"))
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
    nxt = getattr(clk, "next_open", None)
    if not nxt:
        return False

    now = _now_utc()
    if isinstance(nxt, str):
        try:
            nxt_dt = datetime.fromisoformat(nxt.replace("Z", "+00:00"))
        except Exception:
            return False
    else:
        nxt_dt = nxt

    delta = (nxt_dt - now).total_seconds()
    return 0 <= delta <= window_seconds


def _allowed_sides(cfg: Cfg) -> list[str]:
    # For entry orders: if long_only OR allow_short is False -> only BUY
    if cfg.long_only or not cfg.allow_short:
        return ["buy"]
    return ["buy", "sell"]


def _pick_signals(engine, cfg: Cfg) -> list[dict]:
    sides = _allowed_sides(cfg)

    with engine.begin() as con:
        rows = con.execute(
            text(
                """
                SELECT id, created_at, symbol, side, strength, price
                FROM signals
                WHERE portfolio_id=:pid
                  AND symbol = ANY(CAST(:symbols AS text[]))
                  AND side   = ANY(CAST(:sides   AS text[]))
                  AND strength >= :min_strength
                  AND created_at >= (NOW() - (:ttl * INTERVAL '1 second'))
                ORDER BY strength DESC, created_at DESC
                LIMIT 10
                """
            ),
            {
                "pid": cfg.portfolio_id,
                "symbols": list(cfg.symbols),
                "sides": list(sides),
                "min_strength": cfg.min_strength,
                "ttl": cfg.pick_ttl_seconds,
            },
        ).fetchall()

    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": int(r[0]),
                "created_at": r[1],
                "symbol": (r[2] or "").upper(),
                "side": (r[3] or "").lower(),
                "strength": float(r[4]),
                "price": float(r[5]) if r[5] is not None else None,
            }
        )
    return out


def _calc_qty(cfg: Cfg, price: float) -> float:
    if price <= 0:
        return 0.0
    qty = cfg.max_notional / price
    qty = min(qty, cfg.max_qty)
    qty = max(0.0, qty)
    qty = math.floor(qty * 1000) / 1000.0
    if 0 < qty < 1.0:
        qty = 1.0
    return float(qty)


def _dedupe_ok(engine, symbol: str, side: str, minutes: int) -> bool:
    """
    Returns False if same symbol+side exists in orders table within last N minutes.
    Works for both DRY_RUN (because we will write orders) and real mode.
    """
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


def _upsert_order_row(engine, cols: dict) -> None:
    """
    Upsert into orders table using the same column list as sync_orders.py.
    raw_json is passed as JSON string and cast to jsonb.
    """
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO orders (
                  id, created_at, updated_at, submitted_at, filled_at, expired_at, canceled_at, failed_at,
                  replaced_at, replaced_by, replaces, asset_id, symbol, asset_class, qty, filled_qty,
                  side, type, time_in_force, limit_price, stop_price, status, extended_hours,
                  client_order_id, order_class, raw_json
                )
                VALUES (
                  :id, :created_at, :updated_at, :submitted_at, :filled_at, :expired_at, :canceled_at, :failed_at,
                  :replaced_at, :replaced_by, :replaces, :asset_id, :symbol, :asset_class, :qty, :filled_qty,
                  :side, :type, :time_in_force, :limit_price, :stop_price, :status, :extended_hours,
                  :client_order_id, :order_class, CAST(:raw_json AS jsonb)
                )
                ON CONFLICT (id) DO UPDATE SET
                  created_at=EXCLUDED.created_at,
                  updated_at=EXCLUDED.updated_at,
                  submitted_at=EXCLUDED.submitted_at,
                  filled_at=EXCLUDED.filled_at,
                  expired_at=EXCLUDED.expired_at,
                  canceled_at=EXCLUDED.canceled_at,
                  failed_at=EXCLUDED.failed_at,
                  replaced_at=EXCLUDED.replaced_at,
                  replaced_by=EXCLUDED.replaced_by,
                  replaces=EXCLUDED.replaces,
                  asset_id=EXCLUDED.asset_id,
                  symbol=EXCLUDED.symbol,
                  asset_class=EXCLUDED.asset_class,
                  qty=EXCLUDED.qty,
                  filled_qty=EXCLUDED.filled_qty,
                  side=EXCLUDED.side,
                  type=EXCLUDED.type,
                  time_in_force=EXCLUDED.time_in_force,
                  limit_price=EXCLUDED.limit_price,
                  stop_price=EXCLUDED.stop_price,
                  status=EXCLUDED.status,
                  extended_hours=EXCLUDED.extended_hours,
                  client_order_id=EXCLUDED.client_order_id,
                  order_class=EXCLUDED.order_class,
                  raw_json=EXCLUDED.raw_json
                """
            ),
            cols,
        )


def _record_dry_run_order(
    engine,
    *,
    order_id: str,
    client_order_id: str,
    symbol: str,
    side: str,
    qty: float,
    limit_price: float,
    signal_id: int,
    strength: float,
    portfolio_id: int,
    mode: str,
    paper: bool,
) -> None:
    now = _now_utc()
    raw = {
        "dry_run": True,
        "signal_id": signal_id,
        "strength": strength,
        "portfolio_id": portfolio_id,
        "symbol": symbol,
        "side": side,
        "qty": float(qty),
        "limit_price": float(limit_price),
        "mode": mode,
        "paper": paper,
        "ts": now.isoformat(),
    }

    cols = {
        "id": order_id,
        "created_at": now,
        "updated_at": now,
        "submitted_at": now,
        "filled_at": None,
        "expired_at": None,
        "canceled_at": None,
        "failed_at": None,
        "replaced_at": None,
        "replaced_by": None,
        "replaces": None,
        "asset_id": None,
        "symbol": symbol,
        "asset_class": "us_equity",
        "qty": str(qty),
        "filled_qty": "0",
        "side": side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(round(float(limit_price), 2)),
        "stop_price": None,
        "status": "dry_run",
        "extended_hours": False,
        "client_order_id": client_order_id,
        "order_class": None,
        "raw_json": json.dumps(raw),
    }
    _upsert_order_row(engine, cols)


def _place_limit(tc: TradingClient, symbol: str, side: str, qty: float, limit_price: float) -> tuple[str, str]:
    client_order_id = f"ENTRY-{symbol}-{uuid.uuid4().hex[:10]}"
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=str(round(float(limit_price), 2)),
        client_order_id=client_order_id,
    )
    o = tc.submit_order(order_data=req)
    alpaca_id = str(getattr(o, "id", "")) or client_order_id
    return alpaca_id, client_order_id


def main() -> None:
    cfg = _load_cfg()
    engine = get_engine()
    tc, mode, paper, base = make_trading_client()

    # --- in-memory anti-spam (important even with DB dedupe, keeps logs clean) ---
    seen_signal_ids: dict[int, float] = {}
    last_action_by_symbol: dict[str, float] = {}
    seen_keep_seconds = max(600, cfg.pick_ttl_seconds * 20)

    LOG.info(
        "signal_executor starting | MODE=%s | paper=%s | base=%s | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | "
        "POLL=%ss | ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | MAX_POSITION_QTY=%s | "
        "ALLOW_ADD_TO_POSITION=%s | ALPACA_DEDUPE_MINUTES=%s | CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | "
        "MAX_OPEN_ORDERS=%s | DAILY_LOSS_STOP_PCT=%s | MAX_DAILY_LOSS_USD=%s | ENABLE_DAILY_RISK_GUARD=%s | "
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

    while True:
        try:
            now_ts = time.time()

            # prune seen cache
            if seen_signal_ids:
                cutoff = now_ts - seen_keep_seconds
                for k, v in list(seen_signal_ids.items()):
                    if v < cutoff:
                        seen_signal_ids.pop(k, None)

            if cfg.trading_paused:
                LOG.info("trading_paused | sleep=%ss", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            if cfg.trade_only_when_market_open:
                try:
                    if not _get_clock_is_open(tc):
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

            candidates = _pick_signals(engine, cfg)
            if not candidates:
                LOG.info("no_signal | sleep=%ss", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            picked: Optional[dict] = None
            for sig in candidates:
                sid = int(sig["id"])
                sym = sig["symbol"]

                if sid in seen_signal_ids:
                    continue

                last_ts = last_action_by_symbol.get(sym)
                if last_ts is not None and (now_ts - last_ts) < cfg.symbol_cooldown_seconds:
                    continue

                picked = sig
                break

            if not picked:
                LOG.info("no_eligible_signal (seen/cooldown) | sleep=%ss", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            sig_id = int(picked["id"])
            symbol = picked["symbol"]
            side = picked["side"]
            strength = float(picked["strength"])
            price = float(picked["price"]) if picked["price"] is not None else None

            if not price or price <= 0:
                LOG.info("skip no_price | %s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            # DB dedupe (now works for both DRY_RUN and real, because DRY_RUN writes to orders)
            if not _dedupe_ok(engine, symbol, side, cfg.alpaca_dedupe_minutes):
                LOG.info("dedupe_skip | %s %s", symbol, side)
                time.sleep(cfg.poll_seconds)
                continue

            qty = _calc_qty(cfg, price)
            if qty <= 0:
                LOG.info("skip qty<=0 | %s", symbol)
                time.sleep(cfg.poll_seconds)
                continue

            # mark processed BEFORE acting (prevents spam)
            seen_signal_ids[sig_id] = now_ts
            last_action_by_symbol[symbol] = now_ts

            limit_price = price

            if cfg.dry_run:
                oid = f"DRY-{symbol}-{uuid.uuid4().hex[:10]}"
                LOG.info(
                    "DRY_RUN would_submit | symbol=%s side=%s qty=%.4f limit=%.4f strength=%.4f oid=%s",
                    symbol, side, qty, float(limit_price), float(strength), oid
                )

                # record into orders table (audit + dedupe realism)
                _record_dry_run_order(
                    engine,
                    order_id=oid,
                    client_order_id=oid,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    limit_price=limit_price,
                    signal_id=sig_id,
                    strength=strength,
                    portfolio_id=cfg.portfolio_id,
                    mode=mode,
                    paper=paper,
                )
            else:
                alpaca_id, client_oid = _place_limit(tc, symbol, side, qty, limit_price)
                LOG.info(
                    "submitted | symbol=%s side=%s qty=%.4f limit=%.4f oid=%s client_order_id=%s strength=%.4f",
                    symbol, side, qty, float(limit_price), alpaca_id, client_oid, float(strength)
                )

                # optional: write "submitted" immediately (sync_orders will fill details later)
                now = _now_utc()
                raw = {
                    "dry_run": False,
                    "submitted_by": "signal_executor",
                    "signal_id": sig_id,
                    "strength": strength,
                    "portfolio_id": cfg.portfolio_id,
                    "symbol": symbol,
                    "side": side,
                    "qty": float(qty),
                    "limit_price": float(limit_price),
                    "mode": mode,
                    "paper": paper,
                    "ts": now.isoformat(),
                }
                cols = {
                    "id": alpaca_id,
                    "created_at": now,
                    "updated_at": now,
                    "submitted_at": now,
                    "filled_at": None,
                    "expired_at": None,
                    "canceled_at": None,
                    "failed_at": None,
                    "replaced_at": None,
                    "replaced_by": None,
                    "replaces": None,
                    "asset_id": None,
                    "symbol": symbol,
                    "asset_class": "us_equity",
                    "qty": str(qty),
                    "filled_qty": "0",
                    "side": side,
                    "type": "limit",
                    "time_in_force": "day",
                    "limit_price": str(round(float(limit_price), 2)),
                    "stop_price": None,
                    "status": "submitted",
                    "extended_hours": False,
                    "client_order_id": client_oid,
                    "order_class": None,
                    "raw_json": json.dumps(raw),
                }
                _upsert_order_row(engine, cols)

        except Exception as e:
            LOG.exception("loop_error: %r", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
