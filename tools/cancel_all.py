# tools/cancel_all.py
import os, requests
b = os.getenv("ALPACA_BASE_URL","https://paper-api.alpaca.markets").rstrip("/")
k = os.environ["ALPACA_API_KEY"]; s = os.environ["ALPACA_API_SECRET"]
h = {"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":s}
r = requests.delete(f"{b}/v2/orders", headers=h, timeout=20)
print(r.status_code, r.text[:300])
