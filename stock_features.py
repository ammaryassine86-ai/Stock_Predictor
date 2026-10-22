"""Shared data loading, feature engineering and evaluation for the stock models.

Used by run_lightgbm.py (extracted from run_tabpfn3.py so both use the
exact same features, targets and stock-picking test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

EVAL_EVERY = 1  # set by the calling script when only every Nth day is evaluated

ID_COLUMNS = ["date", "ticker", "country", "region", "benchmark"]


LEAKAGE_COLUMNS = ["future_return", "target", "target_up"]


RANK_FEATURES = [
    "return_1", "return_5", "return_20", "return_60", "volatility_20",
    "volatility_60", "rsi_14", "volume_z_20", "drawdown_60", "sma_20_ratio",
    "sma_60_ratio", "range_1", "volume_change_1",
]


PEER_GROUP = ["date", "country"]  # peers = same market, same day (same closing time)


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
    # (the number of stocks in the group is NOT a feature: it only identifies
    #  the market / period, it says nothing about the stock itself)
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


def stock_picking(df: pd.DataFrame, prob: np.ndarray, horizon: int) -> dict:
    """Does the model rank stocks correctly WITHIN a market on a given day?
    - within_day_auc: AUC computed inside each (date, market), then averaged
    - top_minus_bottom: each day, average `horizon`-day return of the model's
      top 20% minus its bottom 20% (same market).

    The t-stat needs NON-overlapping periods (one every `horizon` days). Which
    day you start on changes the result a lot (e.g. t 2.34 vs 1.30 for the same
    model), so the test is done for EVERY possible start day and averaged:
      t_stat            = average t over all start days (the honest number)
      t_stat_one_start  = t when starting on the first day (the old method)
      top_minus_bottom  = average spread over all days."""
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
    step = max(1, round(horizon / EVAL_EVERY))      # dates may already be every EVAL_EVERY-th day

    def t_of(v: pd.Series) -> float:
        return float(v.mean() / v.std() * np.sqrt(len(v))) if len(v) > 2 and v.std() > 0 else float("nan")

    t_each = [t_of(spread.iloc[k::step]) for k in range(min(step, len(spread)))]
    out = {
        "within_day_auc": float(np.asarray(aucs, dtype=float).mean() * 100),
        "top_minus_bottom_pct": float(spread.mean() * 100),
        "t_stat": float(np.nanmean(t_each)) if t_each else float("nan"),
        "t_stat_one_start": t_each[0] if t_each else float("nan"),
        "t_stat_min": float(np.nanmin(t_each)) if t_each else float("nan"),
        "t_stat_max": float(np.nanmax(t_each)) if t_each else float("nan"),
        "periods": int(len(spread.iloc[::step])),
        "hit_rate": float((spread > 0).mean() * 100),
    }
    print(f"{'within_day_auc':20s} {out['within_day_auc']:.2f}")
    print(f"{f'top-bottom ({horizon}d)':20s} {out['top_minus_bottom_pct']:+.3f}%   "
          f"t={out['t_stat']:+.2f} (average over {len(t_each)} start days; "
          f"range {out['t_stat_min']:+.2f} to {out['t_stat_max']:+.2f})   "
          f"positive on {out['hit_rate']:.0f}% of days, {out['periods']} periods")
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
