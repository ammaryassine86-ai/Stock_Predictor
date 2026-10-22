"""TabPFN-3 global stock model: train on the chronological split, evaluate on
validation and test, and save metrics + per-row predictions.

Tuned for a small GPU (e.g. RTX 3050, 6 GB):
  * uses the most recent --context-rows training rows (default 40k) and
    automatically halves that number if the GPU runs out of memory
  * predicts in small batches, halving the batch on out-of-memory
  * drops redundant/empty columns by default (--all-features keeps them)
  * frees the first model before refitting on train+validation

Also:
  * purge gap: drops the last `--horizon` trading dates before each cutoff
  * ROC AUC, a logistic baseline trained on ALL rows (CPU), a leak check
  * per-row predictions saved to CSV for later return / trading backtests
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

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
LEAKAGE_COLUMNS = ["future_return", "target"]
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


def recent_context(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep the most recent `max_rows` rows (whole dates) as TabPFN context."""
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


def compact_features(features):
    """Fewer columns = more rows fit in GPU memory.
    Drops the 5-day return of every global index/FX/commodity (the 1-day and
    20-day versions stay) and the news columns (they cover <1% of rows)."""
    keep = [
        c for c in features
        if c not in NEWS_COLUMNS
        and not (c.endswith("_return_5") and c != "return_5")
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


def batched_proba(model, X: pd.DataFrame, batch_size: int):
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
                     min_rows=1000, min_batch=250):
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


# ------------------------------------------------------------- metrics ----
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
        flag = "  <-- SUSPICIOUS" if v > 0.05 else ""
        print(f"  {k:32s} {v:.3f}{flag}")
    return {k: float(v) for k, v in top.items()}


# ---------------------------------------------------------------- main ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", default="tabpfn3_results.json")
    p.add_argument("--predictions", default="tabpfn3_predictions.csv.gz")
    p.add_argument("--horizon", type=int, default=5, help="Target horizon in trading days (purge gap).")
    p.add_argument("--context-rows", type=int, default=40_000,
                   help="Most recent training rows given to TabPFN. Halved automatically on GPU out-of-memory.")
    p.add_argument("--n-estimators", type=int, default=2)
    p.add_argument("--prediction-batch-size", type=int, default=2_000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--all-features", action="store_true",
                   help="Keep every column (default drops redundant/empty ones to save GPU memory).")
    p.add_argument("--refit-on-train-val", action="store_true",
                   help="For the test evaluation, refit on purged train+validation.")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: install the CUDA build of PyTorch "
                         "(see pytorch.org) or pass --device cpu (very slow).")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB")

    d = Path(args.data_dir)
    t0 = time.time()
    train = load_split(d / "train_matrix.csv.gz")
    val = load_split(d / "validation_matrix.csv.gz")
    test = load_split(d / "test_matrix.csv.gz")

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

    # Baseline on CPU uses ALL training rows (cheap), so it shows what the
    # full dataset gives without TabPFN.
    print(f"\nBaseline: logistic regression on all {len(train_p):,} training rows (CPU)...")
    X_full, y_full, _ = split_xy(train_p, features)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    base.fit(X_full.fillna(0), y_full)
    del X_full
    base_val = evaluate("VALIDATION - logistic baseline", y_val, base.predict_proba(X_val.fillna(0))[:, 1])

    # ---- Validation: fit TabPFN on the most recent training rows -----------
    print("\nFitting TabPFN-3 (validation run)...")
    model, used_rows, fit_seconds, batch = fit_with_backoff(
        make_model, train_p, features, args.context_rows, X_val, args.prediction_batch_size)
    val_prob, batch = batched_proba(model, X_val, batch)
    val_metrics = evaluate("VALIDATION - TabPFN-3", y_val, val_prob)

    # ---- Test ---------------------------------------------------------------
    if args.refit_on_train_val:
        print("\nRefitting on purged train + validation for the test evaluation...")
        del model
        free_gpu()  # release the first model before loading the second
        tv = pd.concat([train, purge_tail(val, args.horizon, "validation")], ignore_index=True)
        model, used_rows, _, batch = fit_with_backoff(
            make_model, tv, features, used_rows, X_te, batch)
        X_tv, y_tv, _ = split_xy(tv, features)
        base.fit(X_tv.fillna(0), y_tv)
        del X_tv, tv
    else:
        print("\nNote: test uses the train-only fit, so the context ends ~2.3 years before test starts.")

    test_prob, _ = batched_proba(model, X_te, batch)
    test_metrics = evaluate("TEST - TabPFN-3", y_te, test_prob)
    base_test = evaluate("TEST - logistic baseline", y_te, base.predict_proba(X_te.fillna(0))[:, 1])

    preds = pd.concat([
        val[["date", "ticker", "future_return", "target"]].assign(split="validation", prob_up=val_prob),
        test[["date", "ticker", "future_return", "target"]].assign(split="test", prob_up=test_prob),
    ])
    preds.to_csv(args.predictions, index=False, compression="gzip")

    results = {
        "model": "Prior-Labs/tabpfn_3",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "horizon": args.horizon,
        "context_rows_used": int(used_rows),
        "n_estimators": args.n_estimators,
        "refit_on_train_val": args.refit_on_train_val,
        "n_features": len(features),
        "features": features,
        "leak_check_top_corr": leaks,
        "fit_seconds": fit_seconds,
        "validation": val_metrics,
        "validation_baseline": base_val,
        "test": test_metrics,
        "test_baseline": base_test,
        "total_seconds": time.time() - t0,
    }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nTabPFN context rows actually used: {used_rows:,}")
    print(f"Results: {Path(args.output).resolve()}")
    print(f"Predictions: {Path(args.predictions).resolve()}")
    print(f"Total runtime: {results['total_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()