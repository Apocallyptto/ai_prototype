#!/usr/bin/env python
"""
jobs/backtest_ensemble.py

Jednoduchý backtester pre tvoj 5m "ensemble / ML" prístup.

- Sťahuje historické 5m sviečky (yfinance)
- Vypočíta technické featury (ATR, EMA, RSI, returns)
- Pokúsi sa načítať model models/gbc_5m.pkl (ak to zlyhá, ide bez modelu)
- Simuluje len LONG obchody s ATR-based TP/SL
- Ukladá výsledky do CSV a vypisuje štatistiky

Toto je prvá verzia – neskôr ju prispôsobíme presne tvojim live pravidlám.
"""

import os
import math
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import load as joblib_load

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Konfigurácia / env premenné
# -----------------------------

def get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default


BACKTEST_SYMBOLS = get_env("BACKTEST_SYMBOLS", "AAPL,MSFT,SPY").split(",")
BACKTEST_DAYS = int(get_env("BACKTEST_DAYS", "60"))  # koľko dní dozadu
BACKTEST_INTERVAL = get_env("BACKTEST_INTERVAL", "5m")

# ATR-based exits
ATR_PERIOD = int(get_env("BACKTEST_ATR_PERIOD", "14"))
TP_ATR_MULT = float(get_env("BACKTEST_TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(get_env("BACKTEST_SL_ATR_MULT", "1.0"))

# Model
MODEL_DIR = get_env("MODEL_DIR", "models")
MODEL_FILENAME = get_env("MODEL_FILENAME", "gbc_5m.pkl")
MIN_STRENGTH = float(get_env("BACKTEST_MIN_STRENGTH", "0.55"))  # prah pre "silný" signál

# Obchodovanie
MAX_HOLD_BARS = int(get_env("BACKTEST_MAX_HOLD_BARS", "30"))  # max počet barov v obchode
RISK_PER_TRADE_USD = float(get_env("BACKTEST_RISK_PER_TRADE_USD", "1000"))  # len pre sizing v backteste
START_EQUITY = float(get_env("BACKTEST_START_EQUITY", "100000"))

OUT_TRADES_CSV = get_env("BACKTEST_TRADES_CSV", "backtest_trades.csv")
OUT_EQUITY_CSV = get_env("BACKTEST_EQUITY_CSV", "backtest_equity_curve.csv")


# -----------------------------
# Technické indikátory
# -----------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ATR.
    Očakáva stĺpce: High, Low, Close.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jednoduché featury – možno nebude 100% matchovať presné featury
    pôvodného tréningu, ale dá nám základ.

    Ak tvoj model očakáva iné featury, neskôr túto funkciu zladíme
    s jobs/make_signals_ml.py.
    """
    out = df.copy()

    # ATR
    out["atr"] = compute_atr(out, ATR_PERIOD)
    out["atr_pct"] = out["atr"] / out["Close"]

    # EMAs
    out["ema_fast"] = out["Close"].ewm(span=10, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=30, adjust=False).mean()
    out["ema_diff_pct"] = (out["ema_fast"] - out["ema_slow"]) / out["Close"]

    # Returns
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_5"] = out["Close"].pct_change(5)

    # Volatility
    out["vol_10"] = out["Close"].pct_change().rolling(10).std()

    # RSI
    out["rsi_14"] = compute_rsi(out["Close"], 14)

    # Volume relative
    vol_ma = out["Volume"].rolling(50).mean()
    out["vol_rel"] = out["Volume"] / vol_ma

    out = out.dropna()
    return out


# -----------------------------
# Model scoring
# -----------------------------

class EnsembleModel:
    """
    Jednoduchý wrapper – pokúsi sa použiť GBC model, ak zlyhá,
    spadne na rule-based skórovanie.
    """

    def __init__(self):
        self.model = None
        self.used_model = False

        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if os.path.exists(model_path):
            try:
                self.model = joblib_load(model_path)
                self.used_model = True
                print(f"[backtest] Loaded model from {model_path}")
            except Exception as e:
                print(f"[backtest] Warning: failed to load model ({e}), using rule-based only")
                self.model = None
        else:
            print(f"[backtest] Model file not found at {model_path}, using rule-based only")

    def score(self, row: pd.Series) -> float:
        """
        Vráti "strength" v intervale približne [-1, 1] (long / short edge).
        Teraz implementujeme len LONG edge (0..1), short ignorujeme.

        - Ak je model k dispozícii a predikcia prebehne OK, použijeme jeho pravdepodobnosť
        - Inak použijeme rule-based kombináciu EMA + RSI
        """
        # Rule-based fallback (EMA + RSI)
        rule_strength = 0.0
        if row["ema_fast"] > row["ema_slow"] and 50 < row["rsi_14"] < 70:
            # základný edge
            rule_strength = 0.6
        elif row["ema_fast"] > row["ema_slow"] and 40 < row["rsi_14"] <= 50:
            rule_strength = 0.5
        elif row["ema_fast"] < row["ema_slow"] and row["rsi_14"] < 40:
            rule_strength = 0.2
        else:
            rule_strength = 0.3

        # Ak nemáme model – len pravidlo
        if not self.used_model or self.model is None:
            return rule_strength

        # Skús model
        try:
            # vyberieme numeric featury (môžeš zmeniť podľa reálneho tréningu)
            feature_cols = [
                "atr_pct",
                "ema_diff_pct",
                "ret_1",
                "ret_3",
                "ret_5",
                "vol_10",
                "rsi_14",
                "vol_rel",
            ]
            X = row[feature_cols].values.reshape(1, -1)
            proba = self.model.predict_proba(X)[0][1]  # pravdepodobnosť "up"
            # Ensemble: 50% model, 50% rule
            strength = 0.5 * rule_strength + 0.5 * float(proba)
            return float(strength)
        except Exception as e:
            # Ak nastane chyba (napr. iný počet featur) – log a fallback
            # (v backteste nechceme crashnúť)
            print(f"[backtest] Model predict error: {e}, using rule_strength only")
            self.used_model = False
            return rule_strength


# -----------------------------
# Backtest jadro
# -----------------------------

def position_size(entry_px: float, sl_px: float, risk_usd: float) -> float:
    """Jednoduchý sizing: risk_usd / (entry - SL)."""
    if sl_px >= entry_px:
        return 0.0
    risk_per_share = entry_px - sl_px
    if risk_per_share <= 0:
        return 0.0
    qty = risk_usd / risk_per_share
    return max(0.0, math.floor(qty))


def backtest_symbol(symbol: str, model: EnsembleModel) -> (pd.DataFrame, pd.DataFrame):
    """
    Backtest pre jeden symbol.
    Vráti:
      - trades_df
      - equity_df
    """
    print(f"[backtest] Downloading data for {symbol}...")
    end = datetime.utcnow()
    start = end - timedelta(days=BACKTEST_DAYS)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=BACKTEST_INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"[backtest] No data for {symbol}, skipping.")
        return pd.DataFrame(), pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = build_features(df)

    if df.empty:
        print(f"[backtest] After feature building, no data for {symbol}, skipping.")
        return pd.DataFrame(), pd.DataFrame()

    df["strength"] = df.apply(model.score, axis=1)

    trades: List[Dict] = []
    equity_curve: List[Dict] = []

    equity = START_EQUITY
    position_open = False
    entry_idx: Optional[pd.Timestamp] = None
    entry_px = 0.0
    sl_px = 0.0
    tp_px = 0.0
    qty = 0.0
    bars_in_trade = 0

    for ts, row in df.iterrows():
        high = row["High"]
        low = row["Low"]
        close = row["Close"]
        atr = row["atr"]

        # aktualizuj equity curve
        equity_curve.append({"time": ts, "symbol": symbol, "equity": equity})

        # Ak máme otvorenú pozíciu – kontrola TP/SL/time exit
        if position_open:
            bars_in_trade += 1
            exit_reason = None
            exit_px = None

            # SL first
            if low <= sl_px:
                exit_px = sl_px
                exit_reason = "SL"
            # potom TP
            elif high >= tp_px:
                exit_px = tp_px
                exit_reason = "TP"
            # time exit
            elif bars_in_trade >= MAX_HOLD_BARS:
                exit_px = close
                exit_reason = "TIME"

            if exit_px is not None:
                pnl = (exit_px - entry_px) * qty
                r_multiple = pnl / (entry_px - sl_px) / qty if qty > 0 and entry_px > sl_px else np.nan
                equity += pnl

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_time": entry_idx,
                        "exit_time": ts,
                        "entry_px": entry_px,
                        "exit_px": exit_px,
                        "sl_px": sl_px,
                        "tp_px": tp_px,
                        "qty": qty,
                        "pnl": pnl,
                        "r_mult": r_multiple,
                        "reason": exit_reason,
                    }
                )

                # reset pozície
                position_open = False
                entry_idx = None
                entry_px = sl_px = tp_px = 0.0
                qty = 0.0
                bars_in_trade = 0

            # ak sme ešte neexituje, pokračujeme na ďalší bar
            continue

        # Ak sme bez pozície – hľadáme nový long setup
        strength = row["strength"]

        if strength < MIN_STRENGTH:
            continue  # žiadny vstup

        if pd.isna(atr) or atr <= 0:
            continue

        # ATR-based levels
        entry_px = close  # vstup na close bar-u
        sl_px = entry_px - SL_ATR_MULT * atr
        tp_px = entry_px + TP_ATR_MULT * atr

        qty = position_size(entry_px, sl_px, RISK_PER_TRADE_USD)
        if qty <= 0:
            continue

        # otvoríme pozíciu
        position_open = True
        entry_idx = ts
        bars_in_trade = 0

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    return trades_df, equity_df


def summarize_trades(trades: pd.DataFrame):
    if trades.empty:
        print("[backtest] No trades generated.")
        return

    total_trades = len(trades)
    wins = (trades["pnl"] > 0).sum()
    losses = (trades["pnl"] < 0).sum()
    winrate = wins / total_trades * 100 if total_trades > 0 else 0

    gross_pnl = trades["pnl"].sum()
    avg_pnl = trades["pnl"].mean()
    avg_r = trades["r_mult"].replace([np.inf, -np.inf], np.nan).dropna().mean()

    # jednoduchý max drawdown z equity z trade-ov
    eq = trades["pnl"].cumsum()
    running_max = eq.cummax()
    drawdown = eq - running_max
    max_dd = drawdown.min()

    print("\n========== BACKTEST SUMMARY ==========")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses} | Winrate: {winrate:.1f}%")
    print(f"Gross PnL: {gross_pnl:.2f} USD")
    print(f"Avg PnL per trade: {avg_pnl:.2f} USD")
    print(f"Avg R multiple: {avg_r:.3f}")
    print(f"Max drawdown (from trade PnL curve): {max_dd:.2f} USD")
    print("======================================\n")


def main():
    print("[backtest] Starting backtest...")
    model = EnsembleModel()

    all_trades = []
    all_equity = []

    for sym in BACKTEST_SYMBOLS:
        sym = sym.strip().upper()
        if not sym:
            continue
        trades_df, equity_df = backtest_symbol(sym, model)
        if not trades_df.empty:
            all_trades.append(trades_df)
        if not equity_df.empty:
            all_equity.append(equity_df)

    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True).sort_values("entry_time")
    else:
        trades = pd.DataFrame()

    if all_equity:
        equity = pd.concat(all_equity, ignore_index=True)
    else:
        equity = pd.DataFrame()

    # uložiť CSV
    if not trades.empty:
        trades.to_csv(OUT_TRADES_CSV, index=False)
        print(f"[backtest] Saved trades to {OUT_TRADES_CSV}")
    if not equity.empty:
        equity.to_csv(OUT_EQUITY_CSV, index=False)
        print(f"[backtest] Saved equity curve to {OUT_EQUITY_CSV}")

    summarize_trades(trades)


if __name__ == "__main__":
    main()
