# Stock Predictor

A stock-ranking project that downloads market data for about 500 stocks from several markets, builds features, trains a LightGBM model, and produces a **daily ranking** of the stocks inside each market. A small web interface shows the results.

> **What it predicts:** the probability that a stock **beats the median stock of its own market** over the next 5 (or 20) trading days. It is a **ranking, not a price forecast**.
> This is a student project for learning. It is **not financial advice**.

## What it does

1. **Downloads data**: daily prices for the stocks, world indexes, currencies and commodities (Yahoo Finance via `yfinance`), macro series (FRED), earnings reports (Yahoo), and company fundamentals for US stocks (SEC EDGAR).
2. **Builds features**: price and volume indicators (returns, volatility, RSI, moving averages, drawdown), how each stock compares with its market peers and sector on the same day, earnings-surprise features, and fundamentals (growth, margins, debt, ROE). The same functions are used for training and for live predictions, so both see identical features.
3. **Avoids look-ahead**: a report or filing is only used from the first trading day **after** it became public, and training/test splits are chronological with a gap (purge) around the cutoff.
4. **Trains and tests the model**: LightGBM (binary and ranking objectives), tuned on a validation set, then tested with a **walk-forward** test (refit every month, predict only the following month). Metrics include within-day AUC and the return of the top versus the bottom stocks.
5. **Predicts every day**: ranks the stocks inside each market and logs the predictions, so they can later be compared with what really happened.
6. **Retrains monthly**: a candidate model is trained and checked; it only replaces the main model if it passes the checks (the old model is archived).
7. **Web interface**: a Flask app that reads the daily results (top picks, search, tracked stocks, notifications).

## Main files

| File | Role |
|---|---|
| `stock_predictor.py` | Core data loading, context (indexes, macro) and feature functions |
| `stock_features.py` | Shared feature engineering and evaluation for the models |
| `fundamental_features.py` | Earnings and SEC fundamentals features (point-in-time) |
| `download_earnings.py`, `download_fundamentals.py`, `download_names.py`, `download_sectors.py` | Data downloads |
| `expand_universe.py`, `fix_universe.py` | Build and fix the list of stocks (S&P 500, ASX 200, FTSE 100, ...) |
| `final_matrix_builder.py`, `make_horizon.py`, `make_split.py` | Build the final feature matrix, the 5/20-day targets and the chronological split |
| `run_lightgbm.py` | Main model: tuning, walk-forward test, final model |
| `run_tabpfn3.py`, `colab_tabpfn3.ipynb`, `kaggle_tabpfn3.ipynb` | Experiments with the TabPFN-3 model (GPU) for comparison |
| `predict_today.py` | Daily live prediction, ranking and evaluation of past predictions |
| `monthly_retrain.py` | Automatic monthly retraining with safety checks |
| `run_daily.bat`, `run_monthly.bat` | Windows Task Scheduler launchers |
| `app.py`, `data.py`, `notify.py`, `templates/`, `static/` | Web interface and notifications (in my local copy these files are in an `app/` folder) |
| `check_matrix.py`, `test_*.py`, `news_*.py`, `repair_context_dates.py` | Data checks, tests and news experiments |

## Setup

```bash
pip install pandas numpy scikit-learn lightgbm yfinance requests flask
# optional: pip install plyer      (Windows pop-up notifications)
# optional: pip install torch      (only for the TabPFN-3 experiments)
```

Optional API keys, read from **environment variables** (nothing is stored in the code):

| Variable | Used for |
|---|---|
| `FRED_API_KEY` | Macro data (FRED) |
| `SEC_EMAIL` | Contact e-mail required by SEC EDGAR for downloading fundamentals |
| `ALPHAVANTAGE_API_KEY`, `GEMINI_API_KEY` | News experiments |
| `TABPFN_TOKEN` | TabPFN-3 experiments |

## Usage (overview)

```bash
python expand_universe.py                 # build the list of stocks
python download_earnings.py               # earnings history
python download_fundamentals.py --email you@example.com   # US fundamentals (SEC)
python final_matrix_builder.py            # feature matrix
python make_split.py                      # chronological split
python run_lightgbm.py --data-dir "data\split" --objective rank   # train + walk-forward test
python predict_today.py --top 10          # today's ranking
python predict_today.py --evaluate        # score past predictions
python app.py                             # web interface on http://127.0.0.1:5000
```

Each script has a docstring at the top that explains its options. The scripts expect the `data/` folder layout created by the download scripts.

## Not included in this repository

The large downloaded data files (prices, earnings, fundamentals, news), the trained model files and the generated predictions are not stored here because of their size. They are recreated by running the scripts above.

## Notes and limits

- The model gives a ranking inside each market; it does not predict prices and it has not been proven profitable.
- Data comes from free sources (Yahoo Finance, FRED, SEC EDGAR), which can have gaps, delays or format changes.
