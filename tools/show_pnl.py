# tools/show_pnl.py

import pandas as pd
from sqlalchemy import text

from utils import get_engine


def load_pnl():
    engine = get_engine()

    sql = text(
        """
        SELECT
            as_of_date,
            portfolio_id,
            equity,
            cash,
            buying_power,
            portfolio_value,
            long_market_value,
            short_market_value,
            created_at
        FROM daily_pnl
        ORDER BY as_of_date ASC, portfolio_id ASC
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()

    if not rows:
        print("⚠️  Table daily_pnl is empty.")
        return None

    df = pd.DataFrame(rows)
    return df


def add_changes(df: pd.DataFrame) -> pd.DataFrame:
    # Sort pre istotu
    df = df.sort_values(["portfolio_id", "as_of_date"]).reset_index(drop=True)

    # Pre každý portfolio_id spočítať day-over-day zmenu equity
    df["equity_change"] = df.groupby("portfolio_id")["equity"].diff()
    df["equity_change_pct"] = (
        df["equity_change"] / df.groupby("portfolio_id")["equity"].shift(1) * 100.0
    )

    return df


def print_report(df: pd.DataFrame):
    print("\n==== DAILY PnL (equity by day) ====\n")

    # Kratší pohľad
    cols = [
        "as_of_date",
        "portfolio_id",
        "equity",
        "equity_change",
        "equity_change_pct",
        "cash",
        "buying_power",
        "portfolio_value",
    ]

    df_print = df[cols].copy()

    # Zaokrúhlenie
    df_print["equity"] = df_print["equity"].round(2)
    df_print["equity_change"] = df_print["equity_change"].round(2)
    df_print["equity_change_pct"] = df_print["equity_change_pct"].round(2)
    df_print["cash"] = df_print["cash"].round(2)
    df_print["buying_power"] = df_print["buying_power"].round(2)
    df_print["portfolio_value"] = df_print["portfolio_value"].round(2)

    # Vytlačiť
    print(df_print.to_string(index=False))

    # Zhrnutie
    print("\n==== SUMMARY ====\n")
    latest = df.sort_values(["as_of_date", "portfolio_id"]).groupby("portfolio_id").tail(1)

    for _, row in latest.iterrows():
        pid = int(row["portfolio_id"])
        eq = float(row["equity"])
        cash = float(row["cash"])
        bp = float(row["buying_power"])

        print(
            f"Portfolio {pid}: equity={eq:.2f}, cash={cash:.2f}, buying_power={bp:.2f}"
        )


def main():
    df = load_pnl()
    if df is None:
        return

    df = add_changes(df)
    print_report(df)


if __name__ == "__main__":
    main()
