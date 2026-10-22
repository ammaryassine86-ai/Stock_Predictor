"""Daily live prediction with the trained LightGBM model.

What it does
  1. Downloads the last ~400 days of prices for the universe and the world
     indexes / FX / commodities (same Yahoo calls as the training data), and
     refreshes the FRED macro cache if FRED_API_KEY is set.
  2. Builds the features with the SAME functions used for training
     (stock_predictor.py for stock/context/macro, stock_features.py for
     peers and sectors), so live and training features are identical.
  3. Loads lightgbm_model.txt, predicts for each stock's latest trading day,
     ranks stocks inside each market and saves the ranking.
  4. Appends the ranking to predictions/prediction_log.csv, so --evaluate can
     later check the predictions against what really happened.

The model predicts: probability that the stock BEATS THE MEDIAN STOCK OF ITS
OWN MARKET over the next 5 trading days. It is a ranking, not a price forecast.

Usage
  python predict_today.py                     # download fresh data and predict
  python predict_today.py --top 10            # show 10 best/worst per market
  python predict_today.py --ticker ASML.AS    # one stock, with the reasons
  python predict_today.py --evaluate          # score past logged predictions
  python predict_today.py --include-today     # late evening, after all markets closed
  python predict_today.py --offline --as-of 2026-09-11
        # rebuild a past date from the saved training files (consistency test)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import fundamental_features as ff
import stock_features as sf
import stock_predictor as sp

LIVE_DIR = Path("data") / "live"
PRED_DIR = Path("predictions")
LOG_FILE = PRED_DIR / "prediction_log.csv"

# Markets where the backtest showed no edge: still computed and logged (so live
# tracking can tell if that changes), but hidden from the printed ranking.
UNVALIDATED_MARKETS = {"Hong Kong", "Japan"}   # Japan: until a model trained on it is tested


def model_version(model_path) -> str:
    """'_v3' for lightgbm_model_h20_rank_v3.txt, '_main' for the monthly-retrained main
    model (one continuous log across retrains), '' when the file name has no version."""
    import re
    m = re.search(r"_(v\d+|main)$", Path(model_path).stem)
    return f"_{m.group(1)}" if m else ""


def files_for(horizon: int, stamp: str = "", objective: str = "binary", version: str = "") -> tuple[Path, Path]:
    """Each kind of model gets its own log and ranking files, so models never mix.
    The 5-day yes/no model keeps the original names (prediction_log.csv, ranking_<date>.csv);
    e.g. the 20-day ranking model uses prediction_log_h20_rank.csv."""
    tag = ("" if horizon == 5 else f"_h{horizon}") + ("_rank" if objective == "rank" else "") + version
    return PRED_DIR / f"prediction_log{tag}.csv", PRED_DIR / f"ranking{tag}_{stamp}.csv"
HISTORY_DAYS = 400          # enough for 60-day windows + 20-day returns + lags
ID_COLS = ["date", "ticker", "country", "region", "benchmark"]


# ------------------------------------------------------------------ data ----
def download_live_data(universe: pd.DataFrame, days: int) -> tuple[Path, Path]:
    """Save fresh Yahoo data in the same format as data/global/{stocks,context}."""
    start = (date.today() - timedelta(days=days)).isoformat()
    end = (date.today() + timedelta(days=1)).isoformat()
    stock_dir, context_dir = LIVE_DIR / "stocks", LIVE_DIR / "context"
    stock_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    symbols = universe["symbol"].tolist()
    print(f"Downloading {len(symbols)} stocks from {start}...")
    ok = 0
    for i in range(0, len(symbols), 25):
        chunk = symbols[i:i + 25]
        try:
            data = sp._download_yahoo_chunk(chunk, start, end)
        except Exception as error:
            print(f"  batch {i // 25 + 1} failed: {error}")
            continue
        for symbol in chunk:
            try:
                frame = data[symbol].dropna(how="all") if isinstance(data.columns, pd.MultiIndex) \
                    else data.dropna(how="all")
                if frame.empty or "Close" not in frame:
                    continue
                frame.to_csv(stock_dir / f"{sp._safe_global_filename(symbol)}.csv")
                ok += 1
            except Exception:
                continue
        print(f"  {min(i + 25, len(symbols))}/{len(symbols)} requested, {ok} saved")

    print("Downloading world indexes / FX / commodities...")
    for symbol in sp.GLOBAL_CONTEXT_SYMBOLS:
        try:
            history = sp.cached_ticker(symbol).history(start=start, end=end, auto_adjust=False)
            if not history.empty:
                history.to_csv(context_dir / f"{sp._safe_global_filename(symbol)}.csv")
        except Exception as error:
            print(f"  [FAILED] {symbol}: {error}")
    return stock_dir, context_dir


def refresh_macro():
    if os.getenv("FRED_API_KEY"):
        print("Refreshing FRED macro data...")
        sp.ensure_macro_data(macro_start="2015-01-01", refresh=True)
    else:
        print("FRED_API_KEY not set: using the cached macro data in data/macro "
              "(fine for a few weeks; set the key to keep it fresh).")


def build_panel(universe, stock_dir: Path, context_dir: Path, as_of, horizon) -> pd.DataFrame:
    """Same steps as build_global_training_dataset, for recent history only."""
    sp.GLOBAL_V2_CONTEXT_DIR = str(context_dir)          # read context from here
    context = sp._load_global_context_frame()
    if as_of is not None:
        context = context[context.index <= as_of]
    rows = []
    for _, meta in universe.iterrows():
        path = stock_dir / f"{sp._safe_global_filename(meta['symbol'])}.csv"
        if not path.exists():
            continue
        history = sp._read_global_stock_file(str(path))
        if as_of is not None:
            history = history[history["date"] <= as_of]
        if len(history) < 80:
            continue
        feats = sp._global_stock_features(history, horizon)
        feats = sp._attach_global_features(feats, meta.to_dict(), context).reset_index()
        feats["ticker"] = str(meta["symbol"])
        feats["country"] = str(meta.get("country", ""))
        feats["region"] = str(meta.get("region", ""))
        feats["benchmark"] = str(meta.get("benchmark", "^GSPC"))
        rows.append(feats)
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.replace([np.inf, -np.inf], np.nan)


def illiquid_stocks() -> set:
    """Stocks removed by the matrix cleaning step (never seen by the model)."""
    meta = Path("data") / "final" / "final_feature_matrix_metadata.json"
    try:
        report = json.loads(meta.read_text(encoding="utf-8"))
        return set(report["cleaning"]["illiquid_stocks_removed"])
    except Exception:
        return set()


def explain_feature(name: str) -> str:
    """Short plain-language label for a feature name."""
    n = name.replace("_lag1", "")
    for key, label in (("world_sector_", "its sector worldwide (prev. day) "),
                       ("rel_sector_", "stock minus its sector "),
                       ("sector_", "its sector in its market "),
                       ("rel_", "stock minus rest of its market "),
                       ("peer_", "rest of its market ")):
        if n.startswith(key):
            return label + n[len(key):].replace("_", " ")
    if n.endswith("_sector_xrank"):
        return "rank vs its sector: " + n[:-len("_sector_xrank")].replace("_", " ")
    if n.endswith("_xrank"):
        return "rank vs its market: " + n[:-len("_xrank")].replace("_", " ")
    return n.replace("_", " ")


def show_tickers(tickers, today, ranking, booster, features, horizon, is_rank=False):
    """Detailed view for the requested stocks, with the main reasons."""
    wanted = [t.strip().upper() for t in tickers]
    found = today[today["ticker"].str.upper().isin(wanted)]
    for t in wanted:
        if t not in set(found["ticker"].str.upper()):
            print(f"\n{t}: not found. It must be in data/global/universe.csv and have price data "
                  "(check the spelling, e.g. SAP.DE, ASML.AS, 7203.T).")
    if found.empty:
        return
    contrib = booster.predict(found[features].astype(np.float32), pred_contrib=True)
    for (idx, row), c in zip(found.iterrows(), contrib):
        r = ranking.loc[idx]
        n = int(r["stocks_in_market"])
        pct = 100 * (1 - (int(r["rank_in_market"]) - 1) / max(n - 1, 1))
        verdict = ("likely to BEAT its market" if r["quintile"] == 5 else
                   "leaning to beat its market" if r["quintile"] == 4 else
                   "likely to LAG its market" if r["quintile"] == 1 else
                   "leaning to lag its market" if r["quintile"] == 2 else "no clear edge (middle of the pack)")
        print("\n" + "=" * 70)
        print(f"{row['ticker']}  |  {row['country']}  |  {r['sector']}  |  data up to {row['date'].date()}")
        if is_rank:
            print(f"Ranking score in its market over the next {horizon} trading days: {r['prob']:.3f} "
                  "(0 = worst, 1 = best)")
        else:
            print(f"Probability of beating the median {row['country']} stock over the next {horizon} "
                  f"trading days: {r['prob']:.3f}")
        if row["country"] in UNVALIDATED_MARKETS:
            print(f"WARNING: the model showed NO edge on {row['country']} stocks in the backtest. "
                  "Do not rely on this ranking.")
        print(f"Rank in its market: {int(r['rank_in_market'])} of {n} (better than {pct:.0f}% of them), "
              f"quintile {int(r['quintile'])}/5 -> {verdict}")
        effects = pd.Series(c[:-1], index=features)
        up = effects.sort_values(ascending=False).head(4)
        down = effects.sort_values().head(4)
        print("Main reasons pushing it UP:")
        for f, v in up.items():
            if v > 0:
                print(f"   +{v:.3f}  {explain_feature(f):55s} = {row[f]:.4g}")
        print("Main reasons pushing it DOWN:")
        for f, v in down.items():
            if v < 0:
                print(f"   {v:.3f}  {explain_feature(f):55s} = {row[f]:.4g}")
        if r["missing_features_pct"] > 20:
            print(f"Warning: {r['missing_features_pct']:.0f}% of this stock's features are missing.")
    unit = "model-score units" if is_rank else "log-odds units"
    print(f"\n(Reasons are in {unit}: bigger = stronger push. A weak research signal, "
          "not investment advice.)")


def save_reasons(today, booster, features, out_file, n=4):
    """For every stock: the features that pushed its score up/down the most today.
    Saved next to the ranking (reasons_*.csv) so the web interface can show them
    without rebuilding anything."""
    try:
        contrib = booster.predict(today[features].astype(np.float32), pred_contrib=True)[:, :-1]
        rows = []
        for (_, row), c in zip(today.iterrows(), contrib):
            order = np.argsort(c)
            for kind, idx in (("up", order[::-1][:n]), ("down", order[:n])):
                for i in idx:
                    if (c[i] > 0) if kind == "up" else (c[i] < 0):
                        v = row[features[i]]
                        rows.append({"ticker": row["ticker"], "kind": kind, "feature": features[i],
                                     "label": explain_feature(features[i]),
                                     "value": float(v) if pd.notna(v) else np.nan, "effect": float(c[i])})
        path = out_file.with_name(out_file.name.replace("ranking", "reasons", 1))
        pd.DataFrame(rows).to_csv(path, index=False)
    except Exception as error:      # reasons are a bonus: never stop the prediction for them
        print(f"(Could not save reasons: {error})")


# ------------------------------------------------------------ predict ----
def predict(args):
    model_path = Path(args.model)
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    features, horizon = meta["features"], meta["horizon"]
    booster = lgb.Booster(model_file=str(model_path))
    print(f"Model: {model_path} (trained up to {meta['trained_up_to']}, "
          f"target: {meta['target']}, {len(features)} features)")

    universe = pd.read_csv(sp.GLOBAL_V2_UNIVERSE)
    universe = universe[~universe["symbol"].isin(illiquid_stocks())]
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    if as_of is None and not args.include_today:
        # Yahoo returns today's UNFINISHED price while markets are open; use
        # only completed trading days (yesterday and before) by default.
        as_of = pd.Timestamp(date.today()) - pd.Timedelta(days=1)
        print(f"Using completed trading days up to {as_of.date()} "
              "(--include-today to use today's closes after all markets have closed)")

    if args.offline:
        stock_dir, context_dir = Path(sp.GLOBAL_V2_STOCK_DIR), Path(sp.GLOBAL_V2_CONTEXT_DIR)
        print(f"Offline: using saved files in {stock_dir} and {context_dir}")
    elif args.skip_download:
        stock_dir, context_dir = LIVE_DIR / "stocks", LIVE_DIR / "context"
        print(f"Using the data already downloaded in {LIVE_DIR}")
    else:
        stock_dir, context_dir = download_live_data(universe, args.history_days)
        refresh_macro()

    panel = build_panel(universe, stock_dir, context_dir, as_of, horizon)
    if any(f.startswith(("earn_", "fund_")) for f in features):
        panel = ff.add_company_features(panel)   # same earnings/fundamentals features as in training
    panel = sf.add_peer_features(panel)
    sectors_path = Path(args.sectors)
    sectors = pd.read_csv(sectors_path) if sectors_path.exists() else pd.DataFrame(columns=["symbol", "sector"])
    panel = sf.add_sector_features(panel, sectors)

    # Latest trading day of each market (markets close at different times).
    last_day = panel.groupby("country")["date"].transform("max")
    today = panel[panel["date"] == last_day].copy()
    missing = [f for f in features if f not in today.columns]
    if missing:
        print(f"WARNING: {len(missing)} model features could not be built and are left empty: {missing[:8]}")
        for f in missing:
            today[f] = np.nan
    empty_share = today[features].isna().mean(axis=1)
    today["prob"] = booster.predict(today[features].astype(np.float32))
    is_rank = meta.get("objective") == "rank"
    if is_rank:
        # a ranking model gives scores, not probabilities: show the position in the market (0-1)
        today["prob"] = today.groupby("country")["prob"].rank(pct=True)
    today["missing_features_pct"] = (empty_share * 100).round(1)

    sector_map = sectors.set_index("symbol")["sector"].to_dict() if len(sectors) else {}
    today["sector"] = today["ticker"].map(sector_map).fillna("Unknown")
    today["rank_in_market"] = today.groupby("country")["prob"].rank(ascending=False, method="first").astype(int)
    today["stocks_in_market"] = today.groupby("country")["prob"].transform("size")
    today["quintile"] = today.groupby("country")["prob"].transform(
        lambda p: pd.qcut(p.rank(method="first"), 5, labels=False) + 1 if len(p) >= 5 else np.nan)

    stamp = (pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(date.today())).date().isoformat()
    PRED_DIR.mkdir(exist_ok=True)
    out_cols = ["date", "country", "ticker", "sector", "prob", "rank_in_market",
                "stocks_in_market", "quintile", "missing_features_pct"]
    ranking = today[out_cols].sort_values(["country", "rank_in_market"])
    log_file, out_file = files_for(horizon, stamp, meta.get("objective", "binary"), model_version(args.model))
    ranking.to_csv(out_file, index=False)
    save_reasons(today, booster, features, out_file)

    if not args.offline:
        logged = ranking.assign(run_date=stamp, model_trained_up_to=meta["trained_up_to"])
        logged.to_csv(log_file, mode="a", header=not log_file.exists(), index=False)

    if args.ticker:
        show_tickers(args.ticker, today, ranking, booster, features, horizon, is_rank)
        print(f"\nFull ranking saved: {out_file.resolve()}")
        return

    # ---- Print --------------------------------------------------------------
    print("\n" + "=" * 78)
    if is_rank:
        print(f"RANKING - score of each stock within its market for the next {horizon} trading days "
              "(0 = worst, 1 = best)")
        print("(A ranking, not a forecast. Not investment advice.)")
    else:
        print(f"RANKING - probability of beating the median stock of its market over the next "
              f"{horizon} trading days")
        print("(A ranking, not a forecast: values near 0.50 are normal. Not investment advice.)")
    hidden = []
    for country, grp in ranking.groupby("country"):
        if country in UNVALIDATED_MARKETS and not args.show_all_markets:
            hidden.append(country)
            continue
        d = grp["date"].iloc[0]
        age = (pd.Timestamp(stamp) - pd.Timestamp(d)).days
        stale = f"  <-- data {age} days old" if age > 4 else ""
        print(f"\n{country} - last trading day {pd.Timestamp(d).date()} - {len(grp)} stocks{stale}")
        top, bottom = grp.head(args.top), grp.tail(args.top).iloc[::-1]
        print(f"  {'TOP (most likely to beat)':38s}  {'BOTTOM (most likely to lag)':38s}")
        for (_, a), (_, b) in zip(top.iterrows(), bottom.iterrows()):
            print(f"  {a['ticker']:12s} {a['prob']:.3f} {a['sector'][:18]:18s}   "
                  f"{b['ticker']:12s} {b['prob']:.3f} {b['sector'][:18]:18s}")
    if hidden:
        print(f"\nHidden (no edge in the backtest): {', '.join(hidden)}. "
              "Still saved in the CSV; --show-all-markets to print them.")
    bad = ranking[ranking["missing_features_pct"] > 20]
    if len(bad):
        print(f"\nNote: {len(bad)} stocks have >20% of features missing (short history or data gaps).")
    print(f"\nSaved: {out_file.resolve()}")
    if not args.offline:
        print(f"Logged for later evaluation: {log_file.resolve()}  "
              f"(can be scored after {horizon} trading days)")


# ----------------------------------------------------------- evaluate ----
def evaluate(args):
    """Check logged predictions against what happened over the model's horizon."""
    meta = json.loads(Path(args.model).with_suffix(".json").read_text(encoding="utf-8"))
    horizon, objective = meta["horizon"], meta.get("objective", "binary")
    log_file, _ = files_for(horizon, objective=objective, version=model_version(args.model))
    print(f"--- Evaluating the {horizon}-day {objective} model ({log_file.name}) ---")
    if not log_file.exists():
        print(f"No log yet ({log_file}). Run predictions first.")
        return
    log = pd.read_csv(log_file, parse_dates=["date"]).drop_duplicates(["date", "ticker"], keep="last")
    stock_dir = LIVE_DIR / "stocks"
    rows = []
    for ticker, grp in log.groupby("ticker"):
        path = stock_dir / f"{sp._safe_global_filename(ticker)}.csv"
        if not path.exists():
            continue
        px = sp._read_global_stock_file(str(path)).set_index("date")["Close"]
        fut = (px.shift(-horizon) / px - 1).rename("future_return")
        rows.append(grp.merge(fut, left_on="date", right_index=True, how="left"))
    res = pd.concat(rows).dropna(subset=["future_return"]) if rows else pd.DataFrame()
    if res.empty:
        print(f"No logged prediction is old enough yet (needs {horizon} trading days of new prices).")
        return
    res["target"] = (res["future_return"] >
                     res.groupby(["date", "country"])["future_return"].transform("median")).astype(int)
    print(f"Evaluating {len(res):,} predictions from {res['date'].nunique()} day(s)...")
    sf.stock_picking(res, res["prob"].to_numpy(), 1)
    by_q = res.groupby("quintile")["future_return"].mean() * 100
    print(f"Average {horizon}-day return by predicted quintile (1 = bottom, 5 = top):",
          {int(k): round(v, 3) for k, v in by_q.items()})
    print("With only a few days logged this is noisy; judge it after several weeks.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lightgbm_model.txt")
    p.add_argument("--sectors", default=str(Path("data") / "global" / "sectors.csv"))
    p.add_argument("--top", type=int, default=5, help="Best/worst stocks to print per market.")
    p.add_argument("--ticker", nargs="+",
                   help="Only show these stocks, with the reasons (e.g. --ticker ASML.AS AAPL). "
                        "The whole market is still computed: the prediction compares the stock to its peers.")
    p.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    p.add_argument("--offline", action="store_true",
                   help="Use the saved training files instead of downloading (for testing).")
    p.add_argument("--skip-download", action="store_true",
                   help="Reuse the data already downloaded in data/live (no new download).")
    p.add_argument("--as-of", help="Predict as of this past date (YYYY-MM-DD), e.g. with --offline.")
    p.add_argument("--evaluate", action="store_true", help="Score past logged predictions.")
    p.add_argument("--show-all-markets", action="store_true",
                   help=f"Also print markets where the model showed no edge ({', '.join(sorted(UNVALIDATED_MARKETS))}).")
    p.add_argument("--include-today", action="store_true",
                   help="Also use today's prices (only after every market has closed, e.g. late evening).")
    args = p.parse_args()
    t0 = time.time()
    if args.evaluate:
        evaluate(args)
    else:
        predict(args)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
