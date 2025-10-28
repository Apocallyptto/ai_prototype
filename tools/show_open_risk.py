# tools/show_open_risk.py
import os
from datetime import datetime, timezone

from services.risk_budget import (
    equity, max_portfolio_risk_pct, active_portfolio_risk,
)

def main():
    eq = equity()
    cap_pct = max_portfolio_risk_pct()
    cap_abs = eq * cap_pct
    total, lines = active_portfolio_risk()

    print("Open Risk Snapshot\n")
    print(f"time={datetime.now(timezone.utc).isoformat()}")
    print(f"equity={eq:.2f}")
    print(f"max_portfolio_risk={cap_pct*100:.2f}%  → cap_abs≈{cap_abs:.2f}\n")
    print(f"active_risk≈{total:.2f}")
    for ln in lines:
        print(" - " + ln)

if __name__ == "__main__":
    main()
