# jobs/risk_limits.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskLimits:
    max_risk_per_trade_usd: float
    max_risk_per_trade_pct: float
    max_notional_per_trade_usd: float
    max_open_positions: int

    # nové polia pre denný limit
    max_daily_loss_usd: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None

    def __repr__(self) -> str:
        return (
            "RiskLimits("
            f"max_risk_per_trade_usd={self.max_risk_per_trade_usd}, "
            f"max_risk_per_trade_pct={self.max_risk_per_trade_pct}, "
            f"max_notional_per_trade_usd={self.max_notional_per_trade_usd}, "
            f"max_open_positions={self.max_open_positions}, "
            f"max_daily_loss_usd={self.max_daily_loss_usd}, "
            f"max_daily_loss_pct={self.max_daily_loss_pct}"
            ")"
        )


def _parse_optional_float(v: str) -> Optional[float]:
    v = (v or "").strip()
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    if f <= 0:
        return None
    return f


def load_limits_from_env() -> RiskLimits:
    """
    Načíta risk parametre z env premenných.
    Ak niečo nie je nastavené, použijú sa rozumné defaulty.
    """
    max_risk_per_trade_usd = float(os.getenv("MAX_RISK_PER_TRADE_USD", "50"))
    max_risk_per_trade_pct = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "0.003"))
    max_notional_per_trade_usd = float(os.getenv("MAX_NOTIONAL_PER_TRADE_USD", "1000"))
    max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

    max_daily_loss_usd = _parse_optional_float(os.getenv("MAX_DAILY_LOSS_USD", ""))
    max_daily_loss_pct = _parse_optional_float(os.getenv("MAX_DAILY_LOSS_PCT", ""))

    return RiskLimits(
        max_risk_per_trade_usd=max_risk_per_trade_usd,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        max_notional_per_trade_usd=max_notional_per_trade_usd,
        max_open_positions=max_open_positions,
        max_daily_loss_usd=max_daily_loss_usd,
        max_daily_loss_pct=max_daily_loss_pct,
    )


def can_open_new_position(current_open_positions: int, limits: RiskLimits) -> bool:
    """
    Jednoduchý guard: neotváraj viac pozícií, než povoľuje max_open_positions.
    """
    return current_open_positions < limits.max_open_positions


def compute_qty_for_long(
    entry_price: float,
    stop_loss_price: float,
    equity: float,
    limits: RiskLimits,
) -> int:
    """
    Spočíta veľkosť pozície tak, aby:
    - riziko na trade (distance entry->SL * qty) <= max_risk_per_trade_usd
      a zároveň aj percentuálny limit max_risk_per_trade_pct z equity,
    - notional (entry * qty) <= max_notional_per_trade_usd.

    Výsledok je celé číslo (ks akcií).
    """
    if entry_price <= 0:
        return 0

    # 1) limit v USD a v % z equity
    risk_usd = limits.max_risk_per_trade_usd
    if limits.max_risk_per_trade_pct > 0 and equity > 0:
        risk_usd = min(risk_usd, equity * limits.max_risk_per_trade_pct)

    if risk_usd <= 0:
        return 0

    # 2) vzdialenosť medzi entry a SL
    price_risk = max(entry_price - stop_loss_price, 0.0)

    # ak je SL >= entry (napr. fallback), radšej použijeme len notional cap
    if price_risk <= 0:
        max_notional = limits.max_notional_per_trade_usd
        if max_notional <= 0:
            return 0
        return int(max_notional // entry_price)

    # 3) raw qty podľa risk_usd
    raw_qty = int(risk_usd // price_risk)
    if raw_qty <= 0:
        return 0

    # 4) notional cap
    if limits.max_notional_per_trade_usd > 0:
        max_notional_qty = int(limits.max_notional_per_trade_usd // entry_price)
        if max_notional_qty <= 0:
            return 0
        qty = min(raw_qty, max_notional_qty)
    else:
        qty = raw_qty

    return max(qty, 0)


def is_daily_loss_ok(account, limits: RiskLimits) -> bool:
    """
    Denný loss guard.

    Použijeme jednoduchý model:
    - Alpaca Account má equity a last_equity (včera).
    - PnL = equity - last_equity.
    - Ak PnL < 0 a strata prekročí MAX_DAILY_LOSS_USD alebo MAX_DAILY_LOSS_PCT,
      vrátime False (nesmie sa otvoriť nový trade).

    Ak nie je nastavený žiadny denný limit, vždy vrátime True.
    """
    if limits.max_daily_loss_usd is None and limits.max_daily_loss_pct is None:
        return True

    try:
        equity = float(account.equity)
        last_equity = float(getattr(account, "last_equity", account.equity))
    except Exception:
        # keď nie je equity, radšej neblokujeme
        return True

    pnl = equity - last_equity
    if pnl >= 0:
        # sme v pluse, žiadne blokovanie
        return True

    loss = -pnl  # kladné číslo

    # limit v USD
    if limits.max_daily_loss_usd is not None and loss >= limits.max_daily_loss_usd:
        return False

    # percentuálny limit
    if limits.max_daily_loss_pct is not None:
        base = last_equity if last_equity > 0 else equity
        if base > 0 and (loss / base) >= limits.max_daily_loss_pct:
            return False

    return True
