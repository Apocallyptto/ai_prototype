# tools/show_atr.py
from __future__ import annotations
import os, sys, requests
from typing import Optional
from lib.atr_utils import last_atr

ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
DATA_FEED       = os.getenv("ALPACA_DATA_FEED", "iex")

API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

ATR_PERIOD        = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP       = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL       = float(os.getenv("ATR_MULT_SL", "1.0"))

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS","AAPL,MSFT,SPY").split(",") if s.strip()]

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})

def latest_trade(symbol: str) -> Optional[float]:
    try:
        r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest",
                  params={"feed": DATA_FEED}, timeout=15)
        r.raise_for_status()
        p = r.json().get("trade",{}).get("p")
        return float(p) if p is not None else None
    except Exception:
        return None

def main():
    print(f"ATR params: period={ATR_PERIOD}, lookback_days={ATR_LOOKBACK_DAYS}, tp×{ATR_MULT_TP}, sl×{ATR_MULT_SL}")
    for sym in SYMBOLS:
        try:
            atr = last_atr(sym, period=ATR_PERIOD, lookback_days=ATR_LOOKBACK_DAYS)
        except Exception as e:
            print(f"{sym}: ATR error -> {e}")
            continue

        px = latest_trade(sym)
        if px is None:
            print(f"{sym}: last price unavailable; ATR={atr:.4f}")
            continue

        tp_buy  = px + ATR_MULT_TP * atr
        sl_buy  = px - ATR_MULT_SL * atr
        tp_sell = px - ATR_MULT_TP * atr
        sl_sell = px + ATR_MULT_SL * atr

        print(f"{sym}: px={px:.2f}  ATR={atr:.4f}  |  BUY: TP={tp_buy:.2f} SL={sl_buy:.2f}  |  SELL: TP={tp_sell:.2f} SL={sl_sell:.2f}")

if __name__ == "__main__":
    main()
