# backtests/crossover.py
# Simple daily SMA crossover backtest with optional RSI filter, SL/TP, commissions,
# and either fixed-allocation or volatility-target sizing. Writes daily equity to
# daily_pnl so it appears in your Streamlit PnL page.

import os
import math
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ---------- DB engine (CLI-friendly; reads ENV like your migrations/scripts) ----------
def engine_from_env():
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    pwd  = os.environ["DB_PASSWORD"]
    url = (
        f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
        "?sslmode=require&channel_binding=require"
    )
    return create_engine(url, pool_pre_ping=True)

# ---------- Helpers ----------
def cagr(equity, freq=252):
    if len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    years = len(equity) / freq
    if start <= 0 or end <= 0 or years <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1

def sharpe(returns, freq=252):
    r = returns.dropna()
    if r.std() == 0 or len(r) == 0:
        return 0.0
    return r.mean() / r.std() * math.sqrt(freq)

def max_drawdown(equity):
    e = equity.astype(float)
    peak = e.cummax()
    dd = e / peak - 1
    return float(dd.min())

def profit_factor(trade_pnl):
    g = trade_pnl[trade_pnl > 0].sum()
    l = -trade_pnl[trade_pnl < 0].sum()
    if l == 0:
        return np.inf if g > 0 else 0.0
    return float(g / l)

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - 100 / (1 + rs)

# ---------- Data ----------
def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to ts,o,h,l,c,v (+ optional symbol/timeframe) regardless of source casing."""
    mapping = {}

    def first_present(names):
        for n in names:
            if n in df.columns:
                return n
        return None

    # timestamp
    t_src = first_present(["ts", "time", "timestamp", "datetime", "date", "Date"])
    if t_src:
        mapping[t_src] = "ts"

    # OHLCV
    o_src = first_present(["o", "open", "Open", "OPEN"])
    h_src = first_present(["h", "high", "High", "HIGH"])
    l_src = first_present(["l", "low", "Low", "LOW"])
    c_src = first_present(["c", "close", "Close", "CLOSE", "adj_close", "Adj Close", "Adj_Close"])
    v_src = first_present(["v", "volume", "Volume", "VOLUME"])
    if o_src: mapping[o_src] = "o"
    if h_src: mapping[h_src] = "h"
    if l_src: mapping[l_src] = "l"
    if c_src: mapping[c_src] = "c"
    if v_src: mapping[v_src] = "v"

    # symbol/ticker + timeframe
    s_src = first_present(["symbol", "ticker", "s"])
    tf_src = first_present(["timeframe", "tf", "interval"])
    if s_src: mapping[s_src] = "symbol"
    if tf_src: mapping[tf_src] = "timeframe"

    return df.rename(columns=mapping)

def _try_read(eng, sql, params):
    try:
        with eng.connect() as c:
            return pd.read_sql(text(sql), c, params=params)
    except Exception:
        return pd.DataFrame()

def load_candles(ticker: str, timeframe: str, start: str | None, end: str | None, eng):
    """
    Try DB first with several schema variants, then fallback to yfinance.
    Normalizes columns to: ts,o,h,l,c,v
    """
    # try a few WHERE variants depending on column names
    attempts = [
        "SELECT * FROM candles WHERE symbol=:t AND timeframe=:tf AND (:start IS NULL OR ts>=:start) AND (:end IS NULL OR ts<=:end) ORDER BY ts",
        "SELECT * FROM candles WHERE ticker=:t AND timeframe=:tf AND (:start IS NULL OR ts>=:start) AND (:end IS NULL OR ts<=:end) ORDER BY ts",
        "SELECT * FROM candles WHERE symbol=:t AND tf=:tf        AND (:start IS NULL OR ts>=:start) AND (:end IS NULL OR ts<=:end) ORDER BY ts",
        "SELECT * FROM candles WHERE ticker=:t AND tf=:tf        AND (:start IS NULL OR ts>=:start) AND (:end IS NULL OR ts<=:end) ORDER BY ts",
        # absolute fallback: read all and filter in pandas (avoid if table is huge)
        "SELECT * FROM candles ORDER BY ts",
    ]
    params = {"t": ticker, "tf": timeframe, "start": start, "end": end}

    df = pd.DataFrame()
    for sql in attempts:
        df = _try_read(eng, sql, params)
        if not df.empty:
            break

    if not df.empty:
        df = _rename_columns(df)

        # if the WHERE didn't filter (e.g., last attempt), filter here if those cols exist
        if "symbol" in df.columns:
            df = df[df["symbol"] == ticker]
        if "timeframe" in df.columns:
            df = df[df["timeframe"] == timeframe]

        # keep only what we need
        required = {"ts", "o", "h", "l", "c", "v"}
        missing = required.difference(df.columns)
        if missing:
            # Sometimes the table stores Open/High/... with original names; map again after reading
            df = _rename_columns(df)
            missing = required.difference(df.columns)
        if missing:
            # give a clear message before falling back
            print(f"[warn] DB candles missing {missing}. Columns available: {list(df.columns)}")
            df = pd.DataFrame()

    if df.empty:
        # fallback to yfinance (daily only)
        import yfinance as yf
        yf_tf = "1d" if timeframe == "1d" else timeframe
        yf_df = yf.download(ticker, start=start, end=end, interval=yf_tf, auto_adjust=False, progress=False)
        if yf_df.empty:
            raise RuntimeError(f"No candles for {ticker} in DB and yfinance fallback is empty.")
        yf_df = yf_df.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})
        yf_df["ts"] = yf_df.index.tz_localize(None)
        df = yf_df.reset_index(drop=True)

    # final normalization & sorting
    df = _rename_columns(df)
    required = {"ts", "o", "h", "l", "c", "v"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Candles missing required columns {missing}. Got: {list(df.columns)}")

    df = df.dropna(subset=["c"]).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df

# ---------- Backtest ----------
def run_backtest(df: pd.DataFrame,
                 sma_fast=50, sma_slow=200,
                 rsi_filter=False, rsi_lo=55.0,
                 sl_pct=0.0, tp_pct=0.0,
                 sizing="fixed", alloc=1.0, vol_target=0.15, vol_window=20,
                 commission_bps=1.0,
                 initial_equity=100_000.0):
    """
    Two sizing modes:
      - 'fixed': position = alloc (0..1) when fast > slow else 0
      - 'vol':   position = min(alloc, vol_target / rolling_vol) when fast > slow else 0

    SL/TP are optional simple close-to-close checks (approx). Commission in bps of notional turnover.
    """
    df = df.copy()
    px = df["c"].astype(float)
    df["ret"] = px.pct_change()

    # signals
    df["sma_f"] = px.rolling(sma_fast, min_periods=sma_fast).mean()
    df["sma_s"] = px.rolling(sma_slow, min_periods=sma_slow).mean()
    long_sig = (df["sma_f"] > df["sma_s"]).astype(int)

    if rsi_filter:
        df["rsi"] = rsi(px, period=14)
        long_sig = (long_sig & (df["rsi"] >= rsi_lo)).astype(int)

    # sizing
    if sizing == "vol":
        vol = df["ret"].rolling(vol_window).std()
        raw_w = (vol_target / (vol + 1e-12)).clip(upper=alloc)
        position = (raw_w * long_sig).fillna(0.0)
    else:
        position = (alloc * long_sig).fillna(0.0)

    # baseline strategy returns (no SL/TP) with commission on turnover
    turn = position.diff().abs().fillna(position.abs())  # first bar open = position
    trade_cost = turn * (commission_bps / 10_000.0)
    strat_ret = position.shift(1).fillna(0.0) * df["ret"] - trade_cost

    # Optional crude SL/TP using close-to-close checks
    if sl_pct > 0 or tp_pct > 0:
        # Track entry price when entering a long
        entry = px.where(position.diff() > 0)  # entry on days where pos increases
        entry = entry.ffill()
        active = (position > 0).astype(int)
        hit_sl = (active == 1) & (sl_pct > 0) & ((px / (entry + 1e-12) - 1) <= -sl_pct)
        hit_tp = (active == 1) & (tp_pct > 0) & ((px / (entry + 1e-12) - 1) >=  tp_pct)
        # Force flat return day on SL/TP hit (approx) and close position
        hit_any = hit_sl | hit_tp
        strat_ret = np.where(hit_any, -trade_cost, strat_ret)  # flatten that day's ret
        # Next day position -> 0
        flat_next = hit_any.shift(1, fill_value=False)
        position = np.where(flat_next, 0.0, position)
        position = pd.Series(position, index=df.index)
        # recompute turnover/cost after sl/tp flattening
        turn = position.diff().abs().fillna(position.abs())
        trade_cost = turn * (commission_bps / 10_000.0)
        strat_ret = position.shift(1).fillna(0.0) * df["ret"] - trade_cost

    equity = (1.0 + strat_ret.fillna(0.0)).cumprod() * float(initial_equity)

    # Trades (approx): whenever position crosses from 0 to >0 or >0 to 0
    trades = []
    pos_prev = 0.0
    ep = None
    for i in range(len(df)):
        pos = float(position.iloc[i])
        if pos_prev == 0.0 and pos > 0.0:
            ep = float(px.iloc[i])  # entry at close (approx)
        if pos_prev > 0.0 and pos == 0.0 and ep is not None:
            exitp = float(px.iloc[i])
            trades.append(exitp / ep - 1.0)
            ep = None
        pos_prev = pos
    trade_pnl = pd.Series(trades, dtype=float)

    metrics = {
        "CAGR": cagr(equity),
        "Sharpe": sharpe(strat_ret),
        "MaxDD": max_drawdown(equity),
        "WinRate": float((trade_pnl > 0).mean()) if len(trade_pnl) else 0.0,
        "ProfitFactor": profit_factor(trade_pnl) if len(trade_pnl) else 0.0,
        "Trades": int(len(trade_pnl)),
        "FinalEquity": float(equity.iloc[-1]),
    }

    out = df[["ts","o","h","l","c","v"]].copy()
    out["position"] = position
    out["strategy_ret"] = strat_ret
    out["equity"] = equity
    return out, pd.Series(metrics)

# ---------- Persist daily equity to daily_pnl ----------
def upsert_daily_pnl(equity_df: pd.DataFrame, portfolio_id: int, eng):
    """
    equity_df: index or column 'ts' (datetime), column 'equity' (float)
    Writes unrealized = equity; realized/fees = 0.
    """
    if "ts" in equity_df.columns:
        tmp = equity_df.set_index("ts")
    else:
        tmp = equity_df.copy()
    by_day = tmp["equity"].groupby(tmp.index.date).last().reset_index()
    by_day.columns = ["date", "equity"]

    up = text("""
        INSERT INTO daily_pnl (portfolio_id, date, realized, unrealized, fees)
        VALUES (:pid, :d, 0, :u, 0)
        ON CONFLICT (portfolio_id, date)
        DO UPDATE SET unrealized = EXCLUDED.unrealized
    """)
    with eng.begin() as c:
        for _, r in by_day.iterrows():
            c.execute(up, {"pid": portfolio_id, "d": r["date"], "u": float(r["equity"])})

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="SMA crossover backtest → daily_pnl")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--sma-fast", type=int, default=50)
    ap.add_argument("--sma-slow", type=int, default=200)
    ap.add_argument("--rsi-filter", action="store_true")
    ap.add_argument("--rsi-lo", type=float, default=55.0)
    ap.add_argument("--sl-pct", type=float, default=0.0)
    ap.add_argument("--tp-pct", type=float, default=0.0)
    ap.add_argument("--sizing", choices=["fixed","vol"], default="fixed")
    ap.add_argument("--alloc", type=float, default=1.0)
    ap.add_argument("--vol-target", type=float, default=0.15)
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--initial-equity", type=float, default=100_000.0)
    ap.add_argument("--portfolio-id", type=int, default=1)
    ap.add_argument("--no-write", action="store_true", help="Don't write to daily_pnl")
    args = ap.parse_args()

    eng = engine_from_env()
    df = load_candles(args.ticker, args.tf, args.start, args.end, eng)
    out, metrics = run_backtest(
        df,
        sma_fast=args.sma_fast, sma_slow=args.sma_slow,
        rsi_filter=args.rsi_filter, rsi_lo=args.rsi_lo,
        sl_pct=args.sl_pct, tp_pct=args.tp_pct,
        sizing=args.sizing, alloc=args.alloc,
        vol_target=args.vol_target, vol_window=args.vol_window,
        commission_bps=args.commission_bps,
        initial_equity=args.initial_equity,
    )

    # print metrics
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>12}: {v:.4f}")
        else:
            print(f"{k:>12}: {v}")

    # persist to daily_pnl
    if not args.no_write:
        upsert_daily_pnl(out, args.portfolio_id, eng)
        print(f"\nWrote {args.ticker} equity to daily_pnl for portfolio_id={args.portfolio_id}")

if __name__ == "__main__":
    main()
