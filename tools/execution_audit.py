import os
from functools import lru_cache
from typing import Any, Dict, Optional, Set

from sqlalchemy import create_engine, text


_TABLE_READY = False


def _get_db_url() -> str:
    return (
        os.getenv("DB_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql+psycopg2://postgres:postgres@postgres:5432/trader"
    )


@lru_cache(maxsize=1)
def _engine():
    return create_engine(_get_db_url())


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_csv_set(key: str, default_csv: str) -> Set[str]:
    raw = os.getenv(key, default_csv)
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    return {x.lower() for x in parts}


def _normalize_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return str(value)


def _throttle_seconds() -> int:
    return max(0, _env_int("EXECUTION_AUDIT_THROTTLE_SECONDS", 300))


def _throttle_reasons() -> Set[str]:
    return _env_csv_set(
        "EXECUTION_AUDIT_THROTTLE_REASONS",
        "pdt_gate,market_closed,no_signal,max_open_positions",
    )


def ensure_table(force: bool = False) -> None:
    global _TABLE_READY
    if _TABLE_READY and not force:
        return

    sql = """
    CREATE TABLE IF NOT EXISTS execution_audit (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        event_type TEXT NOT NULL,
        symbol TEXT NULL,
        side TEXT NULL,
        strength DOUBLE PRECISION NULL,
        source TEXT NULL,
        portfolio_id TEXT NULL,
        reason TEXT NULL,
        detail TEXT NULL,
        signal_ts TIMESTAMPTZ NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS ix_execution_audit_ts
        ON execution_audit (ts DESC);

    CREATE INDEX IF NOT EXISTS ix_execution_audit_event_type_ts
        ON execution_audit (event_type, ts DESC);

    CREATE INDEX IF NOT EXISTS ix_execution_audit_symbol_ts
        ON execution_audit (symbol, ts DESC);

    CREATE INDEX IF NOT EXISTS ix_execution_audit_reason_ts
        ON execution_audit (reason, ts DESC);
    """
    with _engine().begin() as con:
        for stmt in [x.strip() for x in sql.split(";") if x.strip()]:
            con.execute(text(stmt))

    _TABLE_READY = True


def _should_throttle(
    event_type: str,
    symbol: Optional[str],
    side: Optional[str],
    reason: Optional[str],
    detail: Optional[str],
) -> bool:
    if event_type != "blocked_signal":
        return False

    seconds = _throttle_seconds()
    if seconds <= 0:
        return False

    reason_norm = _normalize_text(reason).strip().lower()
    if not reason_norm:
        return False

    allowed = _throttle_reasons()
    if "*" not in allowed and reason_norm not in allowed:
        return False

    q = text(
        """
        SELECT 1
        FROM execution_audit
        WHERE event_type = :event_type
          AND COALESCE(symbol, '') = :symbol
          AND COALESCE(side, '') = :side
          AND COALESCE(reason, '') = :reason
          AND COALESCE(detail, '') = :detail
          AND ts >= NOW() - (:window_seconds || ' seconds')::interval
        LIMIT 1
        """
    )

    params = {
        "event_type": _normalize_text(event_type),
        "symbol": _normalize_text(symbol),
        "side": _normalize_text(side),
        "reason": _normalize_text(reason),
        "detail": _normalize_text(detail),
        "window_seconds": seconds,
    }

    with _engine().begin() as con:
        row = con.execute(q, params).fetchone()
        return row is not None


def log_event(
    event_type: str,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    strength: Optional[float] = None,
    source: Optional[str] = None,
    portfolio_id: Optional[Any] = None,
    reason: Optional[str] = None,
    detail: Optional[str] = None,
    signal_ts: Optional[Any] = None,
) -> None:
    ensure_table()

    if _should_throttle(
        event_type=event_type,
        symbol=symbol,
        side=side,
        reason=reason,
        detail=detail,
    ):
        return

    q = text(
        """
        INSERT INTO execution_audit (
            event_type, symbol, side, strength, source, portfolio_id,
            reason, detail, signal_ts
        )
        VALUES (
            :event_type, :symbol, :side, :strength, :source, :portfolio_id,
            :reason, :detail, :signal_ts
        )
        """
    )
    with _engine().begin() as con:
        con.execute(
            q,
            {
                "event_type": event_type,
                "symbol": symbol,
                "side": side,
                "strength": strength,
                "source": source,
                "portfolio_id": None if portfolio_id is None else str(portfolio_id),
                "reason": reason,
                "detail": detail,
                "signal_ts": signal_ts,
            },
        )


def log_blocked_signal(
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    strength: Optional[float] = None,
    source: Optional[str] = None,
    portfolio_id: Optional[Any] = None,
    reason: Optional[str] = None,
    detail: Optional[str] = None,
    signal_ts: Optional[Any] = None,
) -> None:
    log_event(
        event_type="blocked_signal",
        symbol=symbol,
        side=side,
        strength=strength,
        source=source,
        portfolio_id=portfolio_id,
        reason=reason,
        detail=detail,
        signal_ts=signal_ts,
    )


def log_submitted_order(
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    strength: Optional[float] = None,
    source: Optional[str] = None,
    portfolio_id: Optional[Any] = None,
    detail: Optional[str] = None,
    signal_ts: Optional[Any] = None,
) -> None:
    log_event(
        event_type="submitted_order",
        symbol=symbol,
        side=side,
        strength=strength,
        source=source,
        portfolio_id=portfolio_id,
        reason=None,
        detail=detail,
        signal_ts=signal_ts,
    )


def row_to_signal_kwargs(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}

    def _get(name: str, default=None):
        try:
            return getattr(row, name, default)
        except Exception:
            return default

    return {
        "symbol": _get("symbol"),
        "side": _get("side"),
        "strength": _get("strength"),
        "source": _get("source"),
        "portfolio_id": _get("portfolio_id"),
        "signal_ts": _get("created_at") or _get("ts"),
    }