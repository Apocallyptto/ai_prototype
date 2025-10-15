# jobs/download_bars.py
import os, requests, datetime as dt, time
import pandas as pd

ALPACA_DATA = os.environ.get("ALPACA_DATA_URL","https://data.alpaca.markets").rstrip("/")
KEY  = os.environ["ALPACA_API_KEY"]
SEC  = os.environ["ALPACA_API_SECRET"]
HDR  = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC, "accept": "application/json"}

# quick helper to get 5m bars for a symbol for the last N days
def fetch_5m(symbol: str, days: int = 10) -> pd.DataFrame:
    end   = dt.datetime.utcnow().replace(microsecond=0)
    start = end - dt.timedelta(days=days)
    url = (f"{ALPACA_DATA}/v2/stocks/{symbol}/bars"
           f"?timeframe=5Min&start={start.isoformat()}Z&end={end.isoformat()}Z&limit=10000&adjustment=raw&feed=iex")
    r = requests.get(url, headers=HDR, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = js.get("bars", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df = df[["time","open","high","low","close","volume"]].sort_values("time")
    return df

def main():
    symbols = os.environ.get("ML_SYMBOLS","AAPL,MSFT,SPY").split(",")
    days = int(os.environ.get("ML_DAYS","15"))
    outdir = os.path.join("data","bars_5m")
    os.makedirs(outdir, exist_ok=True)
    for s in symbols:
        print(f"downloading {s}…")
        df = fetch_5m(s.strip(), days=days)
        if not df.empty:
            df.to_csv(os.path.join(outdir, f"{s.strip()}_5m.csv"), index=False)
            print(f"saved {s} {len(df)} rows")
        time.sleep(0.3)

if __name__ == "__main__":
    main()
