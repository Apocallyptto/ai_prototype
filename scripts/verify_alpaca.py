import os, requests

base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
key  = os.environ["ALPACA_API_KEY"]
sec  = os.environ["ALPACA_API_SECRET"]

r = requests.get(
    f"{base}/v2/account",
    headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
    timeout=15,
)
r.raise_for_status()
acct = r.json()
print("✅ Alpaca account status:", acct.get("status"), "| buying_power:", acct.get("buying_power"))
