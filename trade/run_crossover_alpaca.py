import os, time, pandas as pd, yfinance as yf
from datetime import datetime, timedelta, timezone
from alpaca_trade_api import REST

API = REST(
    key_id=os.environ["ALPACA_KEY_ID"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    base_url=os.environ.get("ALPACA_BASE_URL","https://paper-api.alpaca.markets"),
)

TICKER = os.environ.get("TICKER","AAPL")
QTY    = int(os.environ.get("QTY","1"))
FAST   = int(os.environ.get("SMA_FAST","50"))
SLOW   = int(os.environ.get("SMA_SLOW","200"))

def latest_signal(ticker=TICKER, fast=FAST, slow=SLOW):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=slow*3)
    data = yf.download(ticker, start=start.date(), end=end.date(), interval="1d", progress=False)
    if len(data) < slow+1:
        return None
    data["fast"] = data["Close"].rolling(fast).mean()
    data["slow"] = data["Close"].rolling(slow).mean()
    cur = data.iloc[-1]; prev = data.iloc[-2]
    # Cross up -> buy; cross down -> sell; else hold
    if prev["fast"] <= prev["slow"] and cur["fast"] > cur["slow"]:
        return "BUY"
    if prev["fast"] >= prev["slow"] and cur["fast"] < cur["slow"]:
        return "SELL"
    return "HOLD"

def run_once():
    sig = latest_signal()
    if sig is None or sig == "HOLD":
        print("No trade.")
        return
    position = next((p for p in API.list_positions() if p.symbol==TICKER), None)
    if sig == "BUY":
        if position:
            print("Already long/position exists.")
        else:
            print(f"Placing BUY {QTY} {TICKER}")
            API.submit_order(symbol=TICKER, qty=QTY, side="buy", type="market", time_in_force="day")
    elif sig == "SELL":
        if position and float(position.qty) > 0:
            print(f"Closing {position.qty} {TICKER}")
            API.submit_order(symbol=TICKER, qty=position.qty, side="sell", type="market", time_in_force="day")
        else:
            print("No long to close.")

if __name__ == "__main__":
    run_once()
