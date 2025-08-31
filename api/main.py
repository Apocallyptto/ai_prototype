from fastapi import FastAPI
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.engine import URL
import os

app = FastAPI()

url = URL.create(
    "postgresql+psycopg2",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", 5432)),
    database=os.getenv("DB_NAME"),
    query={"sslmode":"require","channel_binding":"require"},
)
engine = sa.create_engine(url, pool_pre_ping=True)

class SignalIn(BaseModel):
    ticker: str
    timeframe: str = "1d"
    model: str = "api"
    side: str
    strength: float

@app.post("/signals")
def post_signal(s: SignalIn):
    with engine.begin() as conn:
        sym_id = conn.execute(sa.text(
            "INSERT INTO symbols(ticker) VALUES(:t) "
            "ON CONFLICT (ticker) DO UPDATE SET ticker=EXCLUDED.ticker "
            "RETURNING id"
        ), {"t": s.ticker}).scalar_one()
        conn.execute(sa.text("""
            INSERT INTO signals(symbol_id,timeframe,model,signal)
            VALUES(:sid,:tf,:m, jsonb_build_object('side',:side,'strength',:str))
        """), {"sid": sym_id, "tf": s.timeframe, "m": s.model, "side": s.side, "str": s.strength})
    return {"ok": True}
