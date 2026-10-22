"""LightGBM stock model: the main model of the project.

Same data, features and tests as run_tabpfn3.py (via stock_features.py),
but LightGBM learns from ALL rows on the CPU, so no GPU and no row limit.

Steps
  1. Load the split, add peer + sector features, relative target.
  2. Tune: try a few settings, fit on train, score on VALIDATION
     (within-day AUC). The test set is never used for choosing.
  3. Test: walk-forward. Every month (--refit-months), refit on ALL rows
     known before that month (5-day purge) and predict only that month.
  4. Final model: fit on everything, save it for live predictions, and
     print which features matter most.

Two kinds of model (--objective):
  binary  (default) answers "will this stock beat its market's median?" for each stock
  rank    learns to ORDER the stocks of a market on a given day (LightGBM lambdarank):
          each (day, market) is one list; the label is the quintile (0-4) of the
          stock's future return in that list. Its scores are turned into a 0-1
          position within the market, so every test below works the same way.

Usage:
  python run_lightgbm.py --data-dir "data\\split"
  python run_lightgbm.py --data-dir "data\\split" --no-tune        (skip step 2)
  python run_lightgbm.py --data-dir "data\\split" --objective rank
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import stock_features as sf

BASE_PARAMS = dict(
    learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=1.0, verbose=-1,
)
# (num_leaves, min_child_samples, n_estimators): small, regularised trees work
# best on noisy stock data; the grid goes from simple to more complex.
GRID = [
    (15, 1000, 300),
    (31, 500, 300),
    (31, 2000, 500),
    (63, 1000, 300),
    (63, 3000, 600),
    (127, 3000, 400),
]
DEFAULT = (31, 2000, 500)


RANK_GRADES = 5          # rank labels: quintile 0 (worst) .. 4 (best) of the future return
RANK_PARAMS = dict(
    # linear gains: the bottom of the list matters too (stocks to avoid), not only the top
    label_gain=list(range(RANK_GRADES)),
    lambdarank_truncation_level=100,     # whole list (a market has <= ~65 stocks)
)


def make_model(leaves, min_child, n_trees, n_jobs, objective="binary"):
    if objective == "rank":
        return lgb.LGBMRanker(objective="lambdarank", num_leaves=leaves, min_child_samples=min_child,
                              n_estimators=n_trees, n_jobs=n_jobs, **RANK_PARAMS, **BASE_PARAMS)
    return lgb.LGBMClassifier(num_leaves=leaves, min_child_samples=min_child,
                              n_estimators=n_trees, n_jobs=n_jobs, **BASE_PARAMS)


def _in_list_order(df: pd.DataFrame) -> pd.DataFrame:
    """A ranker needs the rows of each (date, market) list next to each other."""
    ids = df.groupby(sf.PEER_GROUP, sort=True).ngroup().to_numpy()
    if len(ids) and np.all(np.diff(ids) >= 0):
        return df
    return df.sort_values(sf.PEER_GROUP, kind="stable")


def rank_labels(df: pd.DataFrame) -> np.ndarray:
    pct = df.groupby(sf.PEER_GROUP)["future_return"].rank(pct=True, method="first")
    return np.minimum((pct * RANK_GRADES).astype(int), RANK_GRADES - 1).to_numpy()


def fit(df: pd.DataFrame, features, params, args):
    """Fit one model on the rows of df (binary or ranking)."""
    if args.objective == "rank":
        df = _in_list_order(df)
        X, _, _ = sf.split_xy(df, features)
        group = df.groupby(sf.PEER_GROUP, sort=False).size().to_numpy()
        return make_model(*params, args.n_jobs, "rank").fit(X, rank_labels(df), group=group)
    X, y, _ = sf.split_xy(df, features)
    return make_model(*params, args.n_jobs).fit(X, y)


def score(model, df: pd.DataFrame, features, args) -> np.ndarray:
    """Probability (binary) or 0-1 position within its market that day (rank)."""
    X, _, _ = sf.split_xy(df, features)
    if args.objective == "rank":
        raw = pd.Series(model.predict(X), index=df.index)
        return raw.groupby([df["date"], df["country"]]).rank(pct=True).to_numpy()
    return model.predict_proba(X)[:, 1]


def quiet_stock_picking(df, prob, horizon):
    """stock_picking without its prints (used inside the tuning loop)."""
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return sf.stock_picking(df, prob, horizon)


def load_everything(args):
    names = ("train", "validation", "test")
    parts = []
    for name in names:
        part = sf.load_split(Path(args.data_dir) / f"{name}_matrix.csv.gz")
        part["_split"] = name
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df = sf.add_peer_features(df)
    sectors = Path(args.sectors)
    if sectors.exists():
        df = sf.add_sector_features(df, pd.read_csv(sectors))
    else:
        print(f"Sector features skipped ({sectors} not found)")
    return {n: sf.set_target(df[df["_split"] == n].drop(columns="_split"), args.target)
            for n in names}


def known_before(history: pd.DataFrame, first_day, horizon: int) -> pd.DataFrame:
    """Rows whose 5-day target was already known before `first_day`."""
    dates = np.sort(history["date"].unique())
    before = dates[dates < np.datetime64(first_day)]
    return history[history["date"] <= before[-horizon - 1]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sectors", default=str(Path("data") / "global" / "sectors.csv"))
    p.add_argument("--target", choices=["relative", "absolute"], default="relative")
    p.add_argument("--objective", choices=["binary", "rank"], default="binary",
                   help="binary = beat the median yes/no; rank = learn to order the stocks of each market/day.")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--refit-months", type=int, default=1,
                   help="Walk-forward on test: refit every N months (1 = monthly).")
    p.add_argument("--no-tune", action="store_true", help=f"Skip tuning, use {DEFAULT}.")
    p.add_argument("--stock-level-only", action="store_true",
                   help="Drop features that are identical for every stock of a market on a given day "
                        "(world indexes, FX, commodities, macro). With the relative target these can "
                        "only help through interactions and may let the model memorise time periods.")
    p.add_argument("--drop-features", nargs="+", default=[], metavar="PREFIX",
                   help="Leave out features starting with these prefixes, e.g. --drop-features earn_ fund_ "
                        "(to measure what a feature group adds, on exactly the same rows).")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU cores (-1 = all).")
    p.add_argument("--output", default="lightgbm_results.json")
    p.add_argument("--predictions", default="lightgbm_predictions.csv.gz")
    p.add_argument("--importance", default="lightgbm_feature_importance.csv")
    p.add_argument("--model-out", default="lightgbm_model.txt")
    args = p.parse_args()
    t0 = time.time()

    splits = load_everything(args)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    train_p = sf.purge_tail(train, args.horizon, "train")
    _, _, features = sf.split_xy(train_p)
    if args.stock_level_only:
        sample = train_p.sample(min(len(train_p), 200_000), random_state=0)
        within = sample.groupby(["date", "country"])[features].std().mean()
        overall = sample[features].std().replace(0, np.nan)
        # market-wide = (almost) no variation between stocks of the same market that day
        market_level = [f for f in features if not (within[f] / overall[f] > 0.05)]
        features = [f for f in features if f not in market_level]
        print(f"Stock-level only: dropped {len(market_level)} market-wide features, e.g. {market_level[:6]}")
    if args.drop_features:
        left_out = [f for f in features if f.startswith(tuple(args.drop_features))]
        features = [f for f in features if f not in left_out]
        print(f"Left out {len(left_out)} features starting with {args.drop_features}")
    print(f"\nObjective: {args.objective} | target: {args.target} | features: {len(features)} | "
          f"train {len(train_p):,} / validation {len(val):,} / test {len(test):,} rows")

    y_val = val["target"].astype(int).to_numpy()

    # ---- 1. Tuning on validation ------------------------------------------
    tuning = []
    if args.no_tune:
        best = DEFAULT
    else:
        print("\nTuning on validation (fit on train, score on validation)...")
        print(f"{'leaves':>7}{'min_child':>10}{'trees':>7}{'AUC':>8}{'within-day':>12}"
              f"{'top-bottom':>12}{'t':>7}{'time':>7}")
        for leaves, min_child, n_trees in GRID:
            t = time.time()
            m = fit(train_p, features, (leaves, min_child, n_trees), args)
            prob = score(m, val, features, args)
            sp = quiet_stock_picking(val, prob, args.horizon)
            auc = sf.roc_auc_score(y_val, prob) * 100
            tuning.append({"num_leaves": leaves, "min_child_samples": min_child,
                           "n_estimators": n_trees, "auc": auc, **sp})
            print(f"{leaves:7d}{min_child:10d}{n_trees:7d}{auc:8.2f}{sp['within_day_auc']:12.2f}"
                  f"{sp['top_minus_bottom_pct']:+11.3f}%{sp['t_stat']:+7.2f}{time.time() - t:6.0f}s")
        top = max(tuning, key=lambda r: (r["within_day_auc"], r["t_stat"]))
        best = (top["num_leaves"], top["min_child_samples"], top["n_estimators"])
    print(f"\nChosen settings: num_leaves={best[0]}, min_child_samples={best[1]}, n_estimators={best[2]}")

    model = fit(train_p, features, best, args)
    val_prob = score(model, val, features, args)
    val_metrics = sf.evaluate("VALIDATION - LightGBM (fit on train)", y_val, val_prob)
    val_metrics.update(sf.stock_picking(val, val_prob, args.horizon))

    # ---- 2. Walk-forward on test ------------------------------------------
    print(f"\nWalk-forward on test: refit every {args.refit_months} month(s) on all known rows...")
    history = pd.concat([train, val], ignore_index=True)
    known = pd.concat([history, test], ignore_index=True)
    months = test["date"].dt.to_period("M")
    blocks = sorted(months.unique())
    blocks = [blocks[i:i + args.refit_months] for i in range(0, len(blocks), args.refit_months)]
    test_prob = np.full(len(test), np.nan)
    for i, block in enumerate(blocks, 1):
        mask = months.isin(block).to_numpy()
        first_day = test.loc[mask, "date"].min()
        pool = known_before(known, first_day, args.horizon)
        t = time.time()
        m = fit(pool, features, best, args)
        test_prob[mask] = score(m, test.loc[mask], features, args)
        print(f"  [{i}/{len(blocks)}] {block[0]}..{block[-1]}: trained on {len(pool):,} rows "
              f"up to {pool['date'].max().date()}, predicted {mask.sum():,} rows ({time.time() - t:.0f}s)")
    y_te = test["target"].astype(int).to_numpy()
    test_metrics = sf.evaluate("TEST - LightGBM walk-forward", y_te, test_prob)
    test_metrics.update(sf.stock_picking(test, test_prob, args.horizon))

    # ---- 3. Final model on everything + feature importance ------------------
    print("\nFitting the final model on all data (for live predictions)...")
    final_pool = sf.purge_tail(known.sort_values("date", kind="stable"), args.horizon, "all data")
    final = fit(final_pool, features, best, args)
    final.booster_.save_model(args.model_out)
    meta = {"features": features, "target": args.target, "horizon": args.horizon,
            "objective": args.objective,
            "params": {"num_leaves": best[0], "min_child_samples": best[1],
                       "n_estimators": best[2], **BASE_PARAMS},
            "trained_up_to": str(final_pool["date"].max().date()), "rows": int(len(final_pool))}
    Path(Path(args.model_out).with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    gain = pd.Series(final.booster_.feature_importance("gain"), index=features)
    imp = (gain / gain.sum() * 100).sort_values(ascending=False).rename("importance_pct")
    imp.to_csv(args.importance, header=True)
    print("\nTop 20 features (share of total gain):")
    for name, v in imp.head(20).items():
        print(f"  {name:32s} {v:5.1f}%")
    peer = imp[[f for f in features if f.startswith(("peer_", "rel_")) or ("_xrank" in f and "sector" not in f)]].sum()
    sector = imp[[f for f in features if "sector" in f]].sum()
    world = imp[[f for f in features if f.startswith(("us_", "uk_", "germany", "france", "japan", "hong", "australia",
                                                      "india", "canada", "spain", "netherlands", "gold", "wti", "brent",
                                                      "eur_", "gbp_", "usd_", "aud_", "cad_", "chf_", "benchmark"))]].sum()
    company = imp[[f for f in features if f.startswith(("earn_", "fund_"))]].sum()
    macro = imp[[f for f in features if any(k in f for k in ("fed_funds", "cpi", "unemployment", "yield", "vix", "dollar"))
                 and not f.startswith(("us_vix", "earn_", "fund_"))]].sum()
    own = 100 - peer - sector - world - macro - company
    print(f"\nBy group: stock's own price/volume/technicals {own:.0f}% | peers {peer:.0f}% | sector {sector:.0f}% | "
          f"company results {company:.0f}% | world markets/FX/commodities {world:.0f}% | macro {macro:.0f}%")

    # ---- Save ----------------------------------------------------------------
    keep = ["date", "ticker", "country", "future_return", "target", "target_up"]
    pd.concat([val[keep].assign(split="validation", prob=val_prob),
               test[keep].assign(split="test", prob=test_prob)]).to_csv(
        args.predictions, index=False, compression="gzip")
    results = {"model": "LightGBM", "objective": args.objective, "target": args.target, "horizon": args.horizon,
               "stock_level_only": args.stock_level_only, "dropped_feature_prefixes": args.drop_features,
               "chosen_params": meta["params"], "tuning": tuning,
               "refit_months": args.refit_months, "n_features": len(features),
               "validation": val_metrics, "test_walk_forward": test_metrics,
               "feature_groups_pct": {"own": own, "peers": peer, "sector": sector,
                                      "company": company, "world": world, "macro": macro},
               "total_seconds": time.time() - t0}
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY - LightGBM")
    label = f"top-bottom {args.horizon}d"
    print(f"{'':14s}{'AUC':>7s}{'within-day AUC':>16s}{label:>15s}{'t (avg)':>9s}{'t range':>17s}")
    for name, m in (("validation", val_metrics), ("test (w-f)", test_metrics)):
        print(f"{name:14s}{m['roc_auc']:7.2f}{m['within_day_auc']:16.2f}"
              f"{m['top_minus_bottom_pct']:+14.3f}%{m['t_stat']:+9.2f}"
              f"   {m['t_stat_min']:+.2f} to {m['t_stat_max']:+.2f}")
    print("=" * 70)
    print(f"Model saved: {Path(args.model_out).resolve()} (+ .json with features/settings)")
    print(f"Results: {Path(args.output).resolve()}")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
