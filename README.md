# US Signal Desk

US Signal Desk is a local market-research system that ranks a
version-controlled universe of liquid U.S. large-cap stocks after each market
session. It combines transparent technical analysis with a chronological
21-session probability model, stores every run in DuckDB, and explains the
positive and negative evidence behind each candidate.

The output is a research shortlist, not personalized investment advice or a
promise that a stock will rise.

## What it answers

- Which stock has the strongest risk-adjusted setup today?
- Is the pick supported by trend, momentum, relative strength, and volume?
- Is the stock liquid, above its long-term trend, and not excessively extended?
- What is the model-estimated probability of a positive 21-session return?
- What evidence supports the pick, and what could invalidate it?
- How did the model behave on a chronological holdout period?

## Ranking methodology

The selection process is:

```text
1. Apply transparent technical, liquidity, trend, and extension gates
2. Rank eligible stocks using:
   90% logistic-model probability of a positive 21-session return
   10% technical strength
```

The technical layer evaluates:

- Price relative to 20-, 50-, and 200-session simple moving averages
- Moving-average trend alignment
- MACD direction
- 21- and 63-session momentum
- 63-session relative performance versus SPY
- RSI timing
- Volume confirmation
- ATR and realized volatility
- Proximity to the 52-week high
- Average dollar volume and overextension penalties

Eligibility requires a price of at least $5, 20-session average dollar volume
of at least $20 million, sufficient price history, price above the 200-session
average, RSI below 80, and limited distance above the 20-session average.

The statistical model uses a chronological holdout with a 21-session gap
between the training and validation windows. This reduces, but cannot eliminate,
look-ahead and selection risk. The model is intentionally logistic regression
so its role stays understandable and auditable.

## Signal labels

| Label | Meaning |
|---|---|
| Top pick | Highest-ranked eligible setup in the current universe |
| Buy candidate | Eligible setup in the top decile of the daily selection score |
| Research | Eligible setup in the top quartile of the daily selection score |
| Watch | Eligible setup outside the top quartile |
| Avoid | Failed one or more technical or portfolio gates |

“Buy candidate” means the model found a qualifying research setup. It does not
incorporate the user's objectives, financial situation, taxes, current
portfolio, earnings calendar, breaking news, or execution costs.

## Database

The local database is stored at:

```text
data/us_signal_lab.duckdb
```

It contains:

- `prices` — adjusted daily OHLCV observations
- `features` — technical features and future labels used for research
- `rankings` — current cross-sectional ranking
- `ranking_history` — one stored ranking snapshot per market session
- `validation_predictions` — chronological holdout predictions
- `model_metrics` — current model diagnostics
- `metadata` — run timestamp, market date, universe coverage, and data source

## Run on another computer

The project works with Python 3.11–3.13.

### macOS or Linux

```bash
git clone https://github.com/amirahhha/us-stock-signal-lab.git
cd us-stock-signal-lab
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 pipeline.py --period 5y
python3 -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

### Windows PowerShell

```powershell
git clone https://github.com/amirahhha/us-stock-signal-lab.git
cd us-stock-signal-lab
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py pipeline.py --period 5y
py -m streamlit run app.py
```

The first pipeline run downloads roughly five years of adjusted daily market
data and creates `data/us_signal_lab.duckdb`. That generated database remains
local to the computer and is intentionally excluded from Git.

## Refresh the database

The refresh is fail-safe: a download or model error raises before the database
tables are replaced, preserving the previous usable database.

```bash
python3 pipeline.py --period 5y
```

On macOS, the convenience launcher can reopen the dashboard:

```bash
./scripts/start_dashboard.sh
```

## Daily automation on macOS

The included LaunchAgent refreshes the database at 6:00 PM local time every
Monday through Friday:

```bash
chmod +x scripts/*.sh
./scripts/install_daily_job.sh
```

Check the refresh log at:

```text
logs/daily_refresh.log
```

Remove the schedule without deleting the database:

```bash
./scripts/uninstall_daily_job.sh
```

## Data and validation limitations

- Market data is downloaded through the open-source `yfinance` package for
  personal research and educational use. It is not an exchange-grade feed.
- Adjusted prices can change after corporate-action corrections.
- The curated universe is not a commercial index and introduces selection and
  survivorship bias.
- Twenty-one-session outcome labels overlap, so headline validation metrics are not
  a standalone trading backtest.
- Results exclude spreads, slippage, commissions, taxes, liquidity at the
  intended order size, and portfolio-level risk.
- Technical relationships can fail abruptly around earnings, guidance,
  macroeconomic releases, or unexpected news.
- A single-stock ranking creates concentration risk. Position sizing and
  diversification require separate analysis.

## Primary technical references

- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [scikit-learn time-series validation guidance](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [DuckDB Python API](https://duckdb.org/docs/stable/clients/python/overview)
- [Streamlit documentation](https://docs.streamlit.io/)
