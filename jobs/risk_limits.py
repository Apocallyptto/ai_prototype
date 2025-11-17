"""
jobs/risk_limits.py

Jednoduchá risk management vrstva:

- Čítanie risk limitov z env
- Výpočet veľkosti pozície podľa risku (entry vs SL, USD / % z equity)
- Kontrola max počtu otvorených pozícií

Integrovať do executor-a tak, aby KAŽDÝ nový obchod prešiel cez tieto pravidlá.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


# =============================
# Pomocné funkcie na env
# =============================

def _get_float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


# =============================
# Dataclass s limitmi
# =============================

@dataclass
class RiskLimits:
    """
    max_risk_per_trade_usd:
        Absolútny USD risk na jeden obchod.
        Príklad: 50 = ak SL zasiahne, strata max 50 USD.

    max_risk_per_trade_pct:
        Percento z equity na jeden obchod.
        Ak je > 0, risk_budget = min(max_risk_per_trade_usd, equity * pct)

    max_notional_per_trade_usd:
        Horný limit na notional (entry_price * qty).

    max_open_positions:
        Maximálny počet súčasne otvorených pozícií.
    """
    max_risk_per_trade_usd: float
    max_risk_per_trade_pct: float
    max_notional_per_trade_usd: float
    max_open_positions: int


def load_limits_from_env() -> RiskLimits:
    """
    Načíta risk limity z prostredia (env).
    Použi v executore: limits = load_limits_from_env()
    """
    return RiskLimits(
        max_risk_per_trade_usd=_get_float_env("MAX_RISK_PER_TRADE_USD", 50.0),
        max_risk_per_trade_pct=_get_float_env("MAX_RISK_PER_TRADE_PCT", 0.003),  # 0.3 % z equity
        max_notional_per_trade_usd=_get_float_env("MAX_NOTIONAL_PER_TRADE_USD", 1000.0),
        max_open_positions=_get_int_env("MAX_OPEN_POSITIONS", 3),
    )


# =============================
# Risk funkcie
# =============================

def compute_qty_for_long(
    entry_price: float,
    stop_loss_price: float,
    equity: float,
    limits: RiskLimits,
) -> int:
    """
    Vypočíta veľkosť LONG pozície na základe:

    - vzdialenosť SL od entry
    - USD risk na obchod
    - % z equity risk na obchod
    - max_notional_per_trade_usd

    Výsledok je celý počet akcií (int). 0 = žiadny obchod (risk moc veľký, alebo zlé ceny).
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0
    if stop_loss_price >= entry_price:
        # SL nesmie byť nad entry pri long pozícii
        return 0

    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        return 0

    # 1) základný risk budget v USD
    risk_budget_usd = max(limits.max_risk_per_trade_usd, 0.0)

    # 2) ak máme aj percento z equity, sprísnime podľa neho
    if limits.max_risk_per_trade_pct > 0 and equity > 0:
        pct_budget = equity * limits.max_risk_per_trade_pct
        risk_budget_usd = min(risk_budget_usd, pct_budget)

    if risk_budget_usd <= 0:
        return 0

    # 3) qty podľa risku
    qty_by_risk = risk_budget_usd / risk_per_share

    # 4) qty podľa notional limitu
    if limits.max_notional_per_trade_usd > 0:
        qty_by_notional = limits.max_notional_per_trade_usd / entry_price
        raw_qty = min(qty_by_risk, qty_by_notional)
    else:
        raw_qty = qty_by_risk

    qty = int(math.floor(max(0.0, raw_qty)))
    return qty


def can_open_new_position(
    current_open_positions: int,
    limits: RiskLimits,
) -> bool:
    """
    Vráti True, ak ešte môžeme otvoriť novú pozíciu pri danom počte existujúcich pozícií.
    """
    if limits.max_open_positions <= 0:
        # 0 alebo menej = prakticky žiadny limit (neodporúčam, ale nechávam otvorené)
        return True
    return current_open_positions < limits.max_open_positions
