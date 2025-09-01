# lib/alerts.py
import os, json, urllib.request, urllib.error

def slack(msg: str) -> None:
    """
    Send a plain-text message to a Slack Incoming Webhook.
    Requires env var SLACK_WEBHOOK to be set. Silently no-ops if missing.
    """
    url = os.environ.get("SLACK_WEBHOOK")
    if not url:
        return  # no webhook configured -> skip

    payload = {"text": msg}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()  # drain
    except urllib.error.URLError:
        # Fail closed: don't crash ETL just because Slack is down
        pass

def slack_ok(title: str, details: str = "") -> None:
    slack(f"✅ *{title}*\n{details}")

def slack_err(title: str, details: str = "") -> None:
    slack(f"❌ *{title}*\n{details}")
