# ai_prototype – Mini Trading System (Alpaca Paper)

Tento projekt je malý, ale plnohodnotný **trading systém** postavený na:

- **Alpaca paper účte**
- **Python + Docker Compose**
- **PostgreSQL** databáze
- ML modele (Gradient Boosting na 5-minútových baroch) + jednoduché rules signály

Aktuálne obchodujeme **3 tickery**: `AAPL`, `MSFT`, `SPY` na jednom portfóliu (`portfolio_id = 1`).

---

## Architektúra

Systém pozostáva z niekoľkých služieb (kontajnerov):

1. **postgres**  
   - databáza `trader`  
   - tabuľky: `signals`, `orders`, `executions`, `daily_pnl`, …

2. **signal_maker**  
   - periodicky spúšťa `jobs.make_signals_ml`  
   - generuje signály (BUY/SELL) pre AAPL, MSFT, SPY  
   - ukladá ich do tabuľky `signals`

3. **signal_executor**  
   - číta nové signály zo `signals`  
   - filtruje podľa:
     - `MIN_STRENGTH`
     - `SYMBOLS`
     - `portfolio_id`
   - pre každý (symbol, side) vyberie **najsilnejší signál**
   - rieši:
     - wash-trade ochranu (ruší opačné open ordery, skipuje rovnaké)
     - ATR-based vstup (limit price okolo poslednej ceny)
     - position sizing podľa risk parametrov
     - cooldown po chybách typu “insufficient buying power/qty”
     - voliteľný denný risk guard (max denná strata / max drawdown v %)

4. **sync_orders**  
   - synchronizuje Alpaca orders/positions do DB  
   - slúži na historickú analýzu, audit a monitoring

5. **oco_exit_monitor**  
   - sleduje fillnuté vstupné ordery v DB  
   - vytvára OCO exit objednávky (TP/SL) podľa ATR logiky

6. **pnl_recorder**  
   - raz denne uloží snapshot účtu z Alpaca do tabuľky `daily_pnl`  
   - hodnoty: `equity`, `cash`, `buying_power`, `portfolio_value`, `long_market_value`, `short_market_value`  
   - na základe toho beží script `tools.show_pnl` (denný PnL report)

---

## Hlavné služby (Docker Compose)

Spúšťanie:

```bash
docker compose up -d
