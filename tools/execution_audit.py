import os
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text


def _get_db_url() -> str:
    return (
        os.getenv("DB_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql+psycopg2://postgres:postgres@postgres:5432/trader"
    )


def _engine():
    return create_engine(_get_db_url())


def ensure_table() -> None:
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