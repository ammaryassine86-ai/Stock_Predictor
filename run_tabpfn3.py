"""TabPFN-3 global stock model: train on the chronological split, evaluate on
validation and test, and save metrics + per-row predictions.

Tuned for a small GPU (e.g. RTX 3050, 6 GB):
  * uses the most recent --context-rows training rows (default 80k) and
    automatically halves that number if the GPU runs out of memory
  * predicts in small batches, halving the batch on out-of-memory
  * drops redundant/empty columns by default (--all-features keeps them)
  * frees the first model before refitting on train+validation

Also:
  * purge gap: drops the last `--horizon` trading dates before each cutoff
  * target: 'relative' (beat the median stock of the same market) by default
  * peer features: each stock ranked / compared with its market that day
  * stock-picking evaluation: within-day AUC and top-minus-bottom returns
  * ROC AUC, logistic + LightGBM baselines on ALL rows (CPU), a leak check
  * per-row predictions saved to CSV for later return / trading backtests
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

ID_COLUMNS = ["date", "ticker", "country", "region", "benchmark"]
LEAKAGE_COLUMNS = ["future_return", "target", "target_up"]

# Stock features ranked against the other stocks of the same market that day.
RANK_FEATURES = [
    "return_1", "return_5", "return_20", "return_60", "volatility_20",
    "volatility_60", "rsi_14", "volume_z_20", "drawdown_60", "sma_20_ratio",
    "sma_60_ratio", "range_1", "volume_change_1",
]
PEER_GROUP = ["date", "country"]  # peers = same market, same day (same closing time)
NEWS_COLUMNS = [
    "news_score", "news_count", "positive_count", "negative_count", "neutral_count",
    "news_available", "gemini_sentiment_score", "gemini_relevance",
    "gemini_confidence", "gemini_news_count", "gemini_available",
]


# ---------------------------------------------------------------- data ----
def load_split(path: Path) -> pd.DataFrame:
    print(f"Loading: {path}")
    df = pd.read_csv(path, compression="gzip", low_memory=False, parse_dates=["date"])
    num = df.select_dtypes(include=[np.number]).columns.drop("target", errors="ignore")
    df[num] = df[num].astype(np.float32)
    print(f"  rows={len(df):,}  columns={len(df.columns):,}  "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def purge_tail(df: pd.DataFrame, horizon: int, name: str) -> pd.DataFrame:
    """Drop the last `horizon` trading dates: their targets use prices that
    fall inside the next split."""
    dates = np.sort(df["date"].unique())
    if horizon <= 0 or len(dates) <= horizon:
        return df
    cutoff = dates[-horizon]
    out = df[df["date"] < cutoff]
    print(f"  Purged {len(df) - len(out):,} rows from end of {name} "
          f"(dates >= {pd.Timestamp(cutoff).date()})")
    return out


CONTEXT_YEARS = 0.0  # set from --context-years in main()
EVAL_EVERY = 1       # set from --eval-every in main()


def recent_context(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Choose the TabPFN context rows.
    CONTEXT_YEARS == 0: the most recent `max_rows` rows (whole dates).
    CONTEXT_YEARS > 0 : `max_rows` rows sampled at random from the last
                        CONTEXT_YEARS years, so the model sees several market
                        periods instead of only the last few months."""
    if CONTEXT_YEARS > 0:
        end = df["date"].max()
        window = df[df["date"] > end - pd.DateOffset(days=int(CONTEXT_YEARS * 365))]
        out = window.sample(n=min(max_rows, len(window)), random_state=0) if len(window) > max_rows else window
        print(f"  Context: {len(out):,} rows sampled from "
              f"{window['date'].min().date()} -> {end.date()}")
        return out
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    per_date = df.groupby("date").size().sort_index(ascending=False).cumsum()
    first_date = per_date[per_date <= max_rows].index.min()
    out = df[df["date"] >= first_date]
    print(f"  Context: {len(out):,} most recent rows "
          f"({first_date.date()} -> {df['date'].max().date()})")
    return out


def split_xy(df: pd.DataFrame, feature_columns=None):
    if feature_columns is None:
        excluded = set(ID_COLUMNS + LEAKAGE_COLUMNS)
        feature_columns = [c for c in df.columns if c not in excluded]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = df["target"].astype(int).to_numpy()
    return X, y, feature_columns


def add_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    """How each stock compares with the OTHER stocks of its market on that day.
    Uses only same-day, same-market data, so nothing comes from the future."""
    g = df.groupby(PEER_GROUP)
    new = {}
    for f in RANK_FEATURES:
        if f in df:
            new[f"{f}_xrank"] = g[f].rank(pct=True)
    n = g["return_1"].transform("size")
    for f in ("return_1", "return_5", "return_20"):
        peer_mean = (g[f].transform("sum") - df[f]) / (n - 1).clip(lower=1)
        new[f"peer_{f}"] = peer_mean                 # what the rest of the market did
        new[f"rel_{f}"] = df[f] - peer_mean          # stock vs the rest of the market
    new["peer_count"] = n
    return pd.concat([df, pd.DataFrame(new, index=df.index).astype(np.float32)], axis=1)


def add_sector_features(df: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """Sector peers, using data/global/sectors.csv (from download_sectors.py).

    - Same market, same sector, same day (same closing time): what the other
      stocks of the sector did, and how this stock compares (needs >= 3 stocks).
    - Same sector across ALL markets, PREVIOUS day: world-wide sector moves.
      Lagged one day because other markets may close after this stock."""
    sector_map = sectors.set_index("symbol")["sector"].fillna("").to_dict()
    sec = df["ticker"].map(sector_map).fillna("")
    sec = sec.where(sec != "", "Unknown")
    known = sec != "Unknown"
    key = [df["date"], df["country"], sec]
    new = {}

    n = df.groupby(key)["return_1"].transform("size")
    enough = known & (n >= 3)
    for f in ("return_1", "return_5", "return_20"):
        total = df.groupby(key)[f].transform("sum")
        peer = ((total - df[f]) / (n - 1).clip(lower=1)).where(enough)
        new[f"sector_{f}"] = peer
        new[f"rel_sector_{f}"] = (df[f] - peer).where(enough)
    for f in ("return_5", "return_20"):
        new[f"{f}_sector_xrank"] = df.groupby(key)[f].rank(pct=True).where(enough)
    new["sector_count"] = n.where(known)

    # World-wide sector average, previous trading day.
    tmp = pd.DataFrame({"date": df["date"], "sector": sec})
    for f in ("return_1", "return_5", "return_20"):
        tmp[f] = df[f]
    daily = tmp[known].groupby(["date", "sector"])[["return_1", "return_5", "return_20"]].mean()
    lagged = pd.DataFrame(
        daily.unstack("sector").sort_index().shift(1).stack("sector", future_stack=True))
    lagged = lagged.rename(columns=lambda c: f"world_sector_{c}_lag1")
    joined = pd.DataFrame({"date": df["date"].values, "sector": sec.values}).merge(
        lagged.reset_index(), on=["date", "sector"], how="left")
    for c in lagged.columns:
        new[c] = joined[c].to_numpy()

    out = pd.concat([df, pd.DataFrame(new, index=df.index).astype(np.float32)], axis=1)
    coverage = known.mean() * 100
    print(f"  Sector features added: {len(new)} columns; {coverage:.1f}% of rows have a known sector")
    return out


def set_target(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """absolute: will the stock go up?  relative: will it beat the median stock
    of its own market over the same 5 days (market direction removed)?"""
    df = df.copy()
    df["target_up"] = df["target"].astype(int)
    if kind == "relative":
        median = df.groupby(PEER_GROUP)["future_return"].transform("median")
        df["target"] = (df["future_return"] > median).astype(int)
    return df


def compact_features(features):
    """Fewer columns = more rows fit in GPU memory.
    Drops the 5-day return of every global index/FX/commodity (the 1-day and
    20-day versions stay) and the news columns (they cover <1% of rows)."""
    keep = [
        c for c in features
        if c not in NEWS_COLUMNS
        and not (c.endswith("_return_5") and c != "return_5"
                 and not c.startswith(("peer_", "rel_", "sector_")))
    ]
    print(f"  Compact features: {len(features)} -> {len(keep)} columns")
    return keep


# ----------------------------------------------------------------- GPU ----
def is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batched_proba(model, X: pd.DataFrame, batch_size: int) -> tuple[np.ndarray, int]:
    """Predict in batches; if the GPU runs out of memory, halve the batch."""
    out, start, step = [], 0, 0
    while start < len(X):
        stop = min(start + batch_size, len(X))
        try:
            out.append(model.predict_proba(X.iloc[start:stop])[:, 1])
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if not is_oom(e) or batch_size <= 100:
                raise
            free_gpu()
            batch_size //= 2
            print(f"  GPU out of memory while predicting -> batch size {batch_size:,}")
            continue
        step += 1
        if step % 10 == 0 or stop == len(X):
            print(f"  Predicted {stop:,} / {len(X):,}")
        start = stop
    return np.concatenate(out), batch_size


def fit_with_backoff(make_model, pool, features, rows, probe_X, batch_size,
                     min_rows=1000, min_batch=250) -> tuple[Any, int, float, int]:
    """Fit on the most recent `rows` rows and run one probe prediction.
    On GPU out-of-memory: first shrink the prediction batch (down to
    `min_batch`), then shrink the context rows. Returns the model, rows used,
    fit seconds and the batch size that worked."""
    while True:
        ctx = recent_context(pool, rows)
        X_tr, y_tr, _ = split_xy(ctx, features)
        print(f"  Trying TabPFN-3 with {len(X_tr):,} rows x {len(features)} features, "
              f"batch {batch_size:,}...")
        model = make_model()
        try:
            t = time.time()
            model.fit(X_tr, y_tr)
            fit_seconds = time.time() - t
            while True:  # memory usually peaks at prediction time
                try:
                    model.predict_proba(probe_X.iloc[:batch_size])
                    break
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if not is_oom(e) or batch_size <= min_batch:
                        raise
                    free_gpu()
                    batch_size //= 2
                    print(f"  GPU out of memory while predicting -> batch size {batch_size:,}")
            print(f"  OK - fit took {fit_seconds:.0f}s")
            return model, len(X_tr), fit_seconds, batch_size
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if not is_oom(e) or rows <= min_rows:
                raise
            del model
            free_gpu()
            rows //= 2
            print(f"  GPU out of memory -> retrying with {rows:,} rows")


def walk_forward_proba(make_model, history, eval_df, features, rows, batch, horizon,
                       label) -> tuple[np.ndarray, int, int]:
    """Rolling refit: for each calendar month of `eval_df`, refit TabPFN on the
    most recent `rows` rows known BEFORE that month and predict only that month.

    Rows whose 5-day target is not yet known at the start of the month (the
    last `horizon` trading days) are left out of the context, so nothing from
    the future is used. This is also how the model would run live."""
    known = pd.concat([history, eval_df], ignore_index=True).sort_values("date")
    all_dates = np.sort(known["date"].unique())
    prob = np.full(len(eval_df), np.nan)
    months = eval_df["date"].dt.to_period("M")
    n_months = months.nunique()
    t_start = time.time()
    for i, (month, idx) in enumerate(eval_df.groupby(months).groups.items(), 1):
        first_day = eval_df.loc[idx, "date"].min()
        before = all_dates[all_dates < np.datetime64(first_day)]
        if len(before) <= horizon:
            raise ValueError(f"Not enough history before {month}")
        last_known = before[-horizon - 1]  # target of this date ends before the month starts
        pool = known[known["date"] <= last_known]
        pos = eval_df.index.get_indexer(idx)
        X_month, _, _ = split_xy(eval_df.loc[idx], features)
        print(f"\n[{label} {i}/{n_months}] {month}: context up to {pd.Timestamp(last_known).date()}, "
              f"predicting {len(X_month):,} rows")
        model, rows, _, batch = fit_with_backoff(make_model, pool, features, rows, X_month, batch)
        prob[pos], batch = batched_proba(model, X_month, batch)
        del model
        free_gpu()
        done = time.time() - t_start
        print(f"  elapsed {done / 60:.1f} min, about {done / i * (n_months - i) / 60:.0f} min left for {label}")
    return prob, rows, batch


# ------------------------------------------------------------- metrics ----
def stock_picking(df: pd.DataFrame, prob: np.ndarray, horizon: int) -> dict:
    """Does the model rank stocks correctly WITHIN a market on a given day?
    - within_day_auc: AUC computed inside each (date, market), then averaged
    - top_minus_bottom: each day, average 5-day return of the model's top 20%
      minus its bottom 20% (same market). Measured every `horizon` days so the
      periods don't overlap; t_stat > 2 means unlikely to be luck."""
    x = df[PEER_GROUP + ["future_return", "target"]].copy()
    x["p"] = prob
    groups = [z for _, z in x.groupby(PEER_GROUP) if len(z) >= 10]
    aucs = [roc_auc_score(z["target"], z["p"]) for z in groups if z["target"].nunique() > 1]
    rows = []
    for z in groups:
        q = pd.qcut(z["p"].rank(method="first"), 5, labels=False)
        rows.append((z["date"].iloc[0],
                     z.loc[q == 4, "future_return"].mean() - z.loc[q == 0, "future_return"].mean()))
    spread = pd.DataFrame(rows, columns=["date", "s"]).groupby("date")["s"].mean().sort_index()
    # non-overlapping 5-day periods (dates are already every EVAL_EVERY-th day)
    spread = spread.iloc[::max(1, round(horizon / EVAL_EVERY))]
    t = float(spread.mean() / spread.std() * np.sqrt(len(spread))) if len(spread) > 2 else float("nan")
    out = {
        "within_day_auc": float(np.asarray(aucs, dtype=float).mean() * 100),
        "top_minus_bottom_pct": float(spread.mean() * 100),
        "t_stat": t,
        "periods": int(len(spread)),
        "hit_rate": float((spread > 0).mean() * 100),
    }
    print(f"{'within_day_auc':20s} {out['within_day_auc']:.2f}")
    print(f"{'top-bottom (5d)':20s} {out['top_minus_bottom_pct']:+.3f}%   t={t:+.2f}   "
          f"positive in {out['hit_rate']:.0f}% of {out['periods']} periods")
    return out


def evaluate(name: str, y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5):
    y_pred = (prob >= threshold).astype(int)
    m = {
        "rows": int(len(y_true)),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "roc_auc": roc_auc_score(y_true, prob) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "always_up_accuracy": float(np.mean(y_true == 1) * 100),
        "predicted_up_pct": float(np.mean(y_pred == 1) * 100),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    print(f"\n=== {name} ===")
    for k in ("rows", "accuracy", "balanced_accuracy", "roc_auc", "precision",
              "recall", "f1", "always_up_accuracy", "predicted_up_pct"):
        v = m[k]
        print(f"{k:20s} {v:,}" if k == "rows" else f"{k:20s} {v:.2f}")
    print(f"{'confusion_matrix':20s} {m['confusion_matrix']}")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def leak_check(df: pd.DataFrame, feature_columns) -> dict:
    """Any single feature with |corr| > 0.05 to the 5-day future return is
    suspicious for daily stock data and should be investigated."""
    corr = df[feature_columns].corrwith(df["future_return"]).abs().sort_values(ascending=False)
    top = corr.head(10)
    print("\nLeak check - top |corr(feature, future_return)| on train:")
    for k, v in top.items():
        one_day = k.endswith("_return_1") or k == "return_1"
        flag = ("  <-- SUSPICIOUS (1-day feature)" if one_day
                else "  (slow feature, likely market regime)") if v > 0.05 else ""
        print(f"  {k:32s} {v:.3f}{flag}")
    return {k: float(v) for k, v in top.items()}


# ---------------------------------------------------------------- main ----
def make_baselines():
    """Logistic regression always; LightGBM too if it is installed
    (pip install lightgbm). Both train on ALL rows on the CPU."""
    models: dict[str, Any] = {"logistic": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))}
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
    except ImportError:
        print("  (LightGBM not installed - only the logistic baseline will run. "
              "`pip install lightgbm` to add it.)")
    return models


def fit_baselines(models, X, y):
    for name, m in models.items():
        print(f"  Fitting {name} on {len(X):,} rows...")
        m.fit(X.fillna(0) if name == "logistic" else X, y)


def baseline_proba(name, model, X):
    return model.predict_proba(X.fillna(0) if name == "logistic" else X)[:, 1]


def report(name, df, y, prob, horizon):
    m = evaluate(name, y, prob)
    m.update(stock_picking(df, prob, horizon))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", default="tabpfn3_results.json")
    p.add_argument("--predictions", default="tabpfn3_predictions.csv.gz")
    p.add_argument("--horizon", type=int, default=5, help="Target horizon in trading days (purge gap).")
    p.add_argument("--target", choices=["relative", "absolute"], default="relative",
                   help="relative = beat the median stock of the same market (default); "
                        "absolute = go up.")
    p.add_argument("--no-peers", action="store_true",
                   help="Do not add the peer / cross-sectional features.")
    p.add_argument("--context-rows", type=int, default=80_000,
                   help="Most recent training rows given to TabPFN. Halved automatically on GPU out-of-memory.")
    p.add_argument("--n-estimators", type=int, default=2)
    p.add_argument("--prediction-batch-size", type=int, default=2_000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--all-features", action="store_true",
                   help="Keep every column (default drops redundant/empty ones to save GPU memory).")
    p.add_argument("--refit-on-train-val", action="store_true",
                   help="For the test evaluation, refit on purged train+validation.")
    p.add_argument("--walk-forward", action="store_true",
                   help="Refit TabPFN every month on the latest --context-rows rows known at that "
                        "time and predict only that month (recommended; same total prediction time).")
    p.add_argument("--sectors", default=str(Path("data") / "global" / "sectors.csv"),
                   help="Sector file from download_sectors.py (used if it exists).")
    p.add_argument("--no-sectors", action="store_true", help="Do not add sector features.")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Predict only every Nth trading day of validation/test (e.g. 5 = 5x faster). "
                        "The stock-picking test keeps its power: it already uses every 5th day.")
    p.add_argument("--context-years", type=float, default=0.0,
                   help="0 = context is the most recent rows only (default). "
                        "e.g. 3 = sample the context rows from the last 3 years.")
    args = p.parse_args()
    global CONTEXT_YEARS, EVAL_EVERY
    CONTEXT_YEARS = args.context_years
    EVAL_EVERY = max(1, args.eval_every)
    if args.walk_forward:
        args.refit_on_train_val = True  # baselines also get the train+validation refit for test

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: install the CUDA build of PyTorch "
                         "(see pytorch.org) or pass --device cpu (very slow).")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB")

    d = Path(args.data_dir)
    t0 = time.time()
    names = ("train", "validation", "test")
    parts = []
    for name in names:
        part = load_split(d / f"{name}_matrix.csv.gz")
        part["_split"] = name
        parts.append(part)
    everything = pd.concat(parts, ignore_index=True)
    del parts
    if not args.no_peers:
        everything = add_peer_features(everything)
    sectors_path = Path(args.sectors)
    if args.no_sectors:
        print("Sector features: off")
    elif sectors_path.exists():
        everything = add_sector_features(everything, pd.read_csv(sectors_path))
    else:
        print(f"Sector features: skipped ({sectors_path} not found - run download_sectors.py)")
    splits = {name: set_target(everything[everything["_split"] == name].drop(columns="_split"),
                               args.target) for name in names}
    del everything
    train, val, test = splits["train"], splits["validation"], splits["test"]
    if args.eval_every > 1:
        keep_val = np.sort(val["date"].unique())[::args.eval_every]
        keep_test = np.sort(test["date"].unique())[::args.eval_every]
        val = val[val["date"].isin(keep_val)]
        test = test[test["date"].isin(keep_test)]
        print(f"Evaluating on every {args.eval_every}th trading day: "
              f"{len(val):,} validation rows, {len(test):,} test rows")
    print(f"\nTarget: {args.target}"
          + (" (beat the median stock of the same market over the next "
             f"{args.horizon} days)" if args.target == "relative" else " (price goes up)"))
    print(f"Peer features: {'off' if args.no_peers else 'on'}")

    print("\nPurging overlap between splits...")
    train_p = purge_tail(train, args.horizon, "train")

    _, _, features = split_xy(train_p)
    leaks = leak_check(train_p, features)
    if not args.all_features:
        features = compact_features(features)

    def make_model():
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3,
            device=args.device,
            n_estimators=args.n_estimators,
            ignore_pretraining_limits=True,
        )

    X_val, y_val, _ = split_xy(val, features)
    X_te, y_te, _ = split_xy(test, features)

    # ---- Baselines on CPU with ALL training rows ----------------------------
    print(f"\nBaselines on all {len(train_p):,} training rows (CPU)...")
    baselines = make_baselines()
    X_full, y_full, _ = split_xy(train_p, features)
    fit_baselines(baselines, X_full, y_full)
    del X_full
    results_val = {f"{n}_baseline": report(f"VALIDATION - {n} baseline", val, y_val,
                                           baseline_proba(n, m, X_val), args.horizon)
                   for n, m in baselines.items()}

    model = None
    fit_seconds = float("nan")
    test_prob: np.ndarray = np.array([])
    if args.walk_forward:
        # ---- Monthly rolling refit (validation, then test) -------------------
        print("\nTabPFN-3 walk-forward: refit every month on the latest known rows...")
        val_prob, used_rows, batch = walk_forward_proba(
            make_model, train, val, features, args.context_rows,
            args.prediction_batch_size, args.horizon, "validation")
    else:
        # ---- Validation: TabPFN on the most recent training rows ------------
        print("\nFitting TabPFN-3 (validation run)...")
        model, used_rows, fit_seconds, batch = fit_with_backoff(
            make_model, train_p, features, args.context_rows, X_val, args.prediction_batch_size)
        val_prob, batch = batched_proba(model, X_val, batch)
    results_val["tabpfn"] = report("VALIDATION - TabPFN-3", val, y_val, val_prob, args.horizon)

    # ---- Test ---------------------------------------------------------------
    if args.walk_forward:
        test_prob, used_rows, batch = walk_forward_proba(
            make_model, pd.concat([train, val], ignore_index=True), test, features,
            used_rows, batch, args.horizon, "test")
        print("\nRefitting baselines on purged train + validation for the test evaluation...")
        tv = pd.concat([train, purge_tail(val, args.horizon, "validation")], ignore_index=True)
        X_tv, y_tv, _ = split_xy(tv, features)
        fit_baselines(baselines, X_tv, y_tv)
        del X_tv, tv
    elif args.refit_on_train_val:
        print("\nRefitting on purged train + validation for the test evaluation...")
        del model
        free_gpu()  # release the first model before loading the second
        tv = pd.concat([train, purge_tail(val, args.horizon, "validation")], ignore_index=True)
        model, used_rows, _, batch = fit_with_backoff(
            make_model, tv, features, used_rows, X_te, batch)
        X_tv, y_tv, _ = split_xy(tv, features)
        fit_baselines(baselines, X_tv, y_tv)
        del X_tv, tv
    else:
        print("\nNote: test uses the train-only fit, so the context ends ~2.3 years before test starts.")

    if not args.walk_forward:
        test_prob, _ = batched_proba(model, X_te, batch)
    results_test = {"tabpfn": report("TEST - TabPFN-3", test, y_te, test_prob, args.horizon)}
    for n, m in baselines.items():
        results_test[f"{n}_baseline"] = report(f"TEST - {n} baseline", test, y_te,
                                               baseline_proba(n, m, X_te), args.horizon)

    keep = ["date", "ticker", "country", "future_return", "target", "target_up"]
    preds = pd.concat([
        val[keep].assign(split="validation", prob=val_prob),
        test[keep].assign(split="test", prob=test_prob),
    ])
    preds.to_csv(args.predictions, index=False, compression="gzip")

    # ---- Summary table --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"SUMMARY  (target: {args.target}{', TabPFN walk-forward' if args.walk_forward else ''})")
    print(f"{'':12s}{'model':22s}{'AUC':>7s}{'within-day AUC':>16s}{'top-bottom 5d':>15s}{'t':>7s}")
    for split_name, res in (("validation", results_val), ("test", results_test)):
        for model_name, m in res.items():
            print(f"{split_name:12s}{model_name:22s}{m['roc_auc']:7.2f}{m['within_day_auc']:16.2f}"
                  f"{m['top_minus_bottom_pct']:+14.3f}%{m['t_stat']:+7.2f}")
    print("=" * 78)
    print("Stock-picking works if top-bottom is positive on BOTH validation and test "
          "(t > 2 = strong evidence).")

    results = {
        "model": "Prior-Labs/tabpfn_3",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "target": args.target,
        "peer_features": not args.no_peers,
        "horizon": args.horizon,
        "context_rows_used": int(used_rows),
        "n_estimators": args.n_estimators,
        "refit_on_train_val": args.refit_on_train_val,
        "walk_forward": args.walk_forward,
        "context_years": args.context_years,
        "eval_every": args.eval_every,
        "sector_features": (not args.no_sectors) and Path(args.sectors).exists(),
        "n_features": len(features),
        "features": features,
        "leak_check_top_corr": leaks,
        "fit_seconds": fit_seconds,
        "validation": results_val,
        "test": results_test,
        "total_seconds": time.time() - t0,
    }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nTabPFN context rows actually used: {used_rows:,}")
    print(f"Results: {Path(args.output).resolve()}")
    print(f"Predictions: {Path(args.predictions).resolve()}")
    print(f"Total runtime: {results['total_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
