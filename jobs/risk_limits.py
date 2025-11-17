"""
jobs/risk_limits.py

Risk management layer for live trading.

Provides:
- Loading risk limits from environment variables
- Per-trade risk-based sizing (USD + % equity + notional caps)
- Max open positions guard
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional


# ------------------------------
# ENV HELPERS
# ------------------------------

def _get_float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ------------------------------
# LIMITS DATACLASS
# ------------------------------

@dataclass
class RiskLimits:
    max_risk_per_trade_usd: float
    max_risk_per_trade_pct: float
    max_notional_per_trade_usd: float
    max_open_positions: int


def load_limits_from_env() -> RiskLimits:
    """
    Reads all risk parameters from env.
    """
    return RiskLimits(
        max_risk_per_trade_usd=_get_float_env("MAX_RISK_PER_TRADE_USD", 50.0),
        max_risk_per_trade_pct=_get_float_env("MAX_RISK_PER_TRADE_PCT", 0.003),  # 0.3%
        max_notional_per_trade_usd=_get_float_env("MAX_NOTIONAL_PER_TRADE_USD", 1000.0),
        max_open_positions=_get_int_env("MAX_OPEN_POSITIONS", 3),
    )


# ------------------------------
# CHECK: ALLOW NEW POSITION?
# ------------------------------

def can_open_new_position(current_open_positions: int, limits: RiskLimits) -> bool:
    if limits.max_open_positions <= 0:
        return True
    return current_open_positions < limits.max_open_positions


# ------------------------------
# POSITION SIZING
# ------------------------------

def compute_qty_for_long(
    entry_price: float,
    stop_loss_price: float,
    equity: float,
    limits: RiskLimits,
) -> int:
    """
    Calculates position size based on:
    - distance to SL
    - max risk USD
    - max risk %
    - max notional per trade
    """

    if entry_price <= 0 or stop_loss_price <= 0:
        return 0
    if stop_loss_price >= entry_price:
        return 0

    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        return 0

    # Budget per-trade in USD
    risk_budget_usd = limits.max_risk_per_trade_usd

    # If percentage rule also defined, take min
    if limits.max_risk_per_trade_pct > 0 and equity > 0:
        pct_budget = equity * limits.max_risk_per_trade_pct
        risk_budget_usd = min(risk_budget_usd, pct_budget)

    if risk_budget_usd <= 0:
        return 0

    qty_by_risk = risk_budget_usd / risk_per_share

    # Notional limit
    if limits.max_notional_per_trade_usd > 0:
        qty_by_notional = limits.max_notional_per_trade_usd / entry_price
        raw_qty = min(qty_by_risk, qty_by_notional)
    else:
        raw_qty = qty_by_risk

    qty = int(math.floor(max(0.0, raw_qty)))
    return qty
