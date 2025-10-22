# lib/risk.py
from __future__ import annotations
import os
from dataclasses import dataclass

# Config knobs (defaults are safe)
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "10"))   # $ risk target per trade
MAX_POSITION_USD   = float(os.getenv("MAX_POSITION_USD", "500"))    # cap position notional
MIN_POSITION_USD   = float(os.getenv("MIN_POSITION_USD", "50"))     # floor to avoid dust
MIN_ATR_PCT        = float(os.getenv("MIN_ATR_PCT", "0.10"))        # skip if ATR% < 0.10%
MAX_ATR_PCT        = float(os.getenv("MAX_ATR_PCT", "3.00"))        # skip if ATR% > 3.0%

@dataclass
class AtrInputs:
    last_price: float   # e.g., from quotes or last trade
    atr: float          # ATR (same units as price), e.g., 14-period
    side: str           # 'buy' or 'sell'

def should_trade_by_volatility(inp: AtrInputs) -> bool:
    if inp.last_price <= 0 or inp.atr <= 0:
        return False
    atr_pct = 100.0 * (inp.atr / inp.last_price)
    return (MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT)

def compute_qty_from_atr(inp: AtrInputs) -> int:
    """
    Risk-based sizing: qty ≈ RISK_PER_TRADE_USD / ATR.
    Then clamp by notional caps.
    """
    if inp.last_price <= 0 or inp.atr <= 0:
        return 0

    raw_qty = max(int(RISK_PER_TRADE_USD / inp.atr), 0)
    # Notional clamps
    max_qty = int(MAX_POSITION_USD // inp.last_price)
    min_qty = 1 if (inp.last_price * 1) >= MIN_POSITION_USD else max(int(MIN_POSITION_USD // inp.last_price), 0)
    qty = max(min(raw_qty, max_qty), min_qty)
    return max(qty, 0)
