"""
services/bracket_public.py

Public, stable wrapper over services.bracket_helper.

It avoids importing underscored members from bracket_helper in your tools.
If your bracket_helper already defines the public names, these shims will
bind to them; otherwise we alias to the internal (underscored) versions.
"""

from services import bracket_helper as _bh  # your existing implementation


def _alias(name_public: str, name_private: str):
    """
    Return the attribute with public name if present in bracket_helper,
    otherwise alias to the private (underscored) one.
    Raises AttributeError only if neither exists (which would be a code issue).
    """
    if hasattr(_bh, name_public):
        return getattr(_bh, name_public)
    if hasattr(_bh, name_private):
        return getattr(_bh, name_private)
    raise AttributeError(
        f"services.bracket_helper is missing both '{name_public}' and '{name_private}'"
    )


# ---- Public API (stable names) ----
# Pricing / market data
get_last_price       = _alias("get_last_price",       "_get_last_price")

# ATR + exits
atr_value            = _alias("atr_value",            "_atr")
risk_per_share       = _alias("risk_per_share",       "_risk_per_share")

# Position sizing
compute_dynamic_qty  = _alias("compute_dynamic_qty",  "_compute_dynamic_qty")

# Order submitter (used by executor & CLI)
submit_bracket       = _alias("submit_bracket",       "submit_bracket")

__all__ = [
    "get_last_price",
    "atr_value",
    "risk_per_share",
    "compute_dynamic_qty",
    "submit_bracket",
]
