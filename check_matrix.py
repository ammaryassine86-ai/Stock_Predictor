"""Health check for the stock feature matrix.

Usage:
    python check_matrix.py --data-dir "C:\\stock predictor\\data\\split"
    python check_matrix.py --matrix "data\\final\\final_feature_matrix.csv.gz"

Prints PASS / WARN / FAIL for each check with a suggested fix, and saves
everything to matrix_report.json. Nothing is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ["date", "ticker", "country", "region", "benchmark"]
TARGET_COLUMNS = ["future_return", "target"]
NEWS_COLUMNS = ["news_available", "gemini_available"]
MONTHLY_MACRO = ["cpi", "unemployment", "fed_funds_rate"]
LEVEL_MACRO = ["fed_funds_rate", "cpi", "unemployment", "yield_10y", "yield_2y",
               "vix", "dollar_index", "yield_curve_spread"]

RESULTS: list[dict] = []
ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[INFO]"}


def report(status: str, check: str, detail: str, fix: str = ""):
    RESULTS.append({"status": status, "check": check, "detail": detail, "fix": fix})
    print(f"\n{ICON[status]} {check}")
    for line in detail.splitlines():
        print(f"       {line}")
    if fix and status in ("WARN", "FAIL"):
        print(f"       FIX: {fix}")


def load(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False, parse_dates=["date"])
    num = df.select_dtypes(include=[np.number]).columns.drop("target", errors="ignore")
    df[num] = df[num].astype(np.float32)
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLUMNS + TARGET_COLUMNS]


# ------------------------------------------------------------------ checks --
def check_structure(df):
    missing = [c for c in ID_COLUMNS + TARGET_COLUMNS if c not in df.columns]
    dups = int(df.duplicated(["date", "ticker"]).sum())
    num = df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(num.to_numpy()).sum())
    nan_frac = df[feature_columns(df)].isna().mean()
    high_nan = nan_frac[nan_frac > 0.2]
    const = [c for c in feature_columns(df)
             if c in num and df[c].nunique(dropna=True) <= 1]
    problems = []
    if missing: problems.append(f"missing columns: {missing}")
    if dups: problems.append(f"{dups:,} duplicate (date, ticker) rows")
    if n_inf: problems.append(f"{n_inf:,} infinite values")
    if len(high_nan): problems.append(f"{len(high_nan)} columns >20% missing: {list(high_nan.index)[:8]}")
    if const: problems.append(f"constant columns (useless): {const}")
    detail = (f"{len(df):,} rows, {df['ticker'].nunique():,} tickers, "
              f"{df['date'].nunique():,} dates, {len(feature_columns(df))} features, "
              f"{df['date'].min().date()} -> {df['date'].max().date()}")
    if problems:
        report("FAIL" if (missing or dups) else "WARN", "Structure",
               detail + "\n" + "\n".join(problems),
               "Drop duplicate/constant columns; fix the source of inf/NaN values.")
    else:
        report("PASS", "Structure", detail)


def check_target(df):
    agree = float((df["target"] == (df["future_return"] > 0).astype(int)).mean())
    up = float(df["target"].mean())
    status = "PASS" if agree > 0.999 else "FAIL"
    report(status, "Target consistency",
           f"target == (future_return > 0) on {agree:.2%} of rows; UP share {up:.1%}",
           "Target must be built only from future_return.")


def infer_horizon(df, max_h=20, n_tickers=60):
    """Find h such that future_return(t) == price change from t to t+h."""
    rng = np.random.default_rng(0)
    tickers = rng.choice(df["ticker"].unique(), size=min(n_tickers, df["ticker"].nunique()), replace=False)
    sub = df[df["ticker"].isin(tickers)].sort_values(["ticker", "date"])
    logret = np.log1p(sub["return_1"].astype(float))
    target = sub["future_return"].astype(float)
    best, scores = None, {}
    g = logret.groupby(sub["ticker"])
    cum = pd.Series(0.0, index=sub.index)
    for h in range(1, max_h + 1):
        cum = cum + g.shift(-h)
        fwd = np.expm1(cum)
        m = fwd.notna() & target.notna()
        scores[h] = float(np.corrcoef(fwd[m], target[m])[0, 1])
    best = max(scores, key=lambda h: scores[h])
    report("PASS" if scores[best] > 0.95 else "WARN", "Target horizon",
           f"future_return matches the next {best} trading days (corr {scores[best]:.3f})",
           "If no horizon matches, future_return is computed on a different price "
           "series or dates are misaligned.")
    return best


def check_split_gaps(splits, horizon):
    if len(splits) < 2:
        report("INFO", "Split gaps", "Single matrix given - split check skipped.")
        return
    names = list(splits)
    lines, bad = [], False
    for a, b in zip(names, names[1:]):
        end_a, start_b = splits[a]["date"].max(), splits[b]["date"].min()
        overlap = end_a >= start_b
        # Weekdays strictly between the two splits (the purged dates are not
        # in any file, so they must be counted on the calendar).
        gap = 0 if overlap else int(np.busday_count(
            (end_a + pd.Timedelta(days=1)).date(), start_b.date()))
        lines.append(f"{a} ends {end_a.date()}, {b} starts {start_b.date()}: "
                     f"{gap} trading days between (need >= {horizon})"
                     + ("  OVERLAP!" if overlap else ""))
        bad |= gap < horizon or overlap
    report("FAIL" if bad else "PASS", "Gap between splits (purge)", "\n".join(lines),
           f"Leave {horizon} trading days out between splits, or let run_tabpfn3.py purge them.")


def check_leakage(df, feats):
    """No single feature should correlate strongly with the future return."""
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num].corrwith(df["future_return"]).abs().sort_values(ascending=False)
    bad = corr[corr > 0.05]
    # 1-day returns carry no memory: a strong link to the NEXT 5 days means
    # future data. Slow features (20/60-day volatility, macro levels, VIX) can
    # correlate through market regimes (e.g. 2020 crash then rebound).
    fast = [k for k in bad.index if k.endswith("_return_1") or k == "return_1"]
    top = "\n".join(f"{k:32s} |corr|={v:.3f}" for k, v in corr.head(8).items())
    if fast:
        report("FAIL", "Leak scan (feature vs future return)",
               f"{len(bad)} features with |corr| > 0.05; 1-day features among them: {fast}\n{top}",
               "1-day features contain future information. See the alignment check below.")
    elif len(bad):
        report("WARN", "Leak scan (feature vs future return)",
               f"{len(bad)} slow features with |corr| > 0.05 (no 1-day features)\n{top}",
               "Likely market-regime effects (e.g. 2020), not leaks. Check they don't "
               "carry the same sign in every split before trusting them.")
    else:
        report("PASS", "Leak scan (feature vs future return)", f"No feature above 0.05\n{top}")
    return {k: float(v) for k, v in corr.head(20).items()}


def check_market_alignment(df):
    """For each country, compare every index's 1-day return on date t with that
    country's average stock return on t and on t+1. If an index lines up better
    with t+1, it closes AFTER that market (or its dates are shifted) and must be
    lagged one day."""
    ctx_cols = [c for c in df.columns if c.endswith("_return_1") and c != "return_1"]
    if "country" not in df or not ctx_cols:
        report("INFO", "Market alignment", "No country or context columns - skipped.")
        return {}
    flagged, table = [], {}
    for country, x in df.groupby("country"):
        if x["ticker"].nunique() < 5:
            continue
        local = x.groupby("date")["return_1"].mean()
        ctx = x.groupby("date")[ctx_cols].first()
        for c in ctx_cols:
            same = local.corr(ctx[c])
            nxt = local.shift(-1).corr(ctx[c])
            table[f"{country}|{c}"] = (round(float(same), 2), round(float(nxt), 2))
            if nxt > 0.25 and nxt > same:
                flagged.append((country, c, same, nxt))
    if flagged:
        lines = [f"{'country':15s} {'feature':28s} corr(t) corr(t+1)"]
        lines += [f"{a:15s} {b:28s} {s:6.2f} {n:8.2f}" for a, b, s, n in flagged[:25]]
        if len(flagged) > 25:
            lines.append(f"... and {len(flagged) - 25} more")
        own = [f for f in flagged if f[1] == "benchmark_return_1"]
        msg = ("The stock's OWN benchmark lines up with t+1: dates are shifted "
               "(timezone bug when parsing Yahoo dates). Use the fixed pipeline."
               if own else
               "These markets close after the stock's market on the same date. "
               "Lag them one day (the fixed pipeline does this).")
        report("FAIL", "Market alignment (time zones / closing times)",
               f"{len(flagged)} country/feature pairs see the NEXT day's market\n" + "\n".join(lines), msg)
    else:
        report("PASS", "Market alignment (time zones / closing times)",
               "No context column lines up with the next day's local returns.")
    return table


def check_ffilled_returns(df):
    """Returns forward-filled over holidays repeat the previous day's value;
    the correct value on a closed day is 0."""
    ctx_cols = [c for c in df.columns if c.endswith("_return_1") and c != "return_1"]
    per_date = df.groupby("date")[ctx_cols].first().sort_index()
    rep = ((per_date == per_date.shift(1)) & (per_date != 0)).mean().sort_values(ascending=False)
    bad = rep[rep > 0.005]
    if len(bad):
        report("WARN", "Repeated returns on market holidays",
               "\n".join(f"{k:32s} {v:.1%} of dates repeat yesterday's return" for k, v in bad.head(10).items()),
               "Forward-fill PRICES, then compute returns (or fill returns with 0), "
               "not ffill() after pct_change().")
    else:
        report("PASS", "Repeated returns on market holidays", "No repeated non-zero returns.")


def check_macro(df):
    lines, bad = [], False
    per_date = df.groupby("date").first().sort_index()
    for c in MONTHLY_MACRO:
        if c not in per_date:
            continue
        changes = per_date.index[per_date[c].diff().fillna(0) != 0]
        if len(changes) < 5:
            continue
        early = float(np.mean(changes.day <= 5))
        lines.append(f"{c:16s} changes on days 1-5 of the month in {early:.0%} of updates")
        bad |= early > 0.5
    if not lines:
        report("INFO", "FRED release dates", "No monthly macro columns found.")
    elif bad:
        report("FAIL", "FRED release dates", "\n".join(lines) +
               "\nValues switch at the start of the month = FRED reference date, "
               "but the data is only published ~2-6 weeks later.",
               "Shift CPI ~+45 days, UNRATE ~+35 days, FEDFUNDS ~+30 days before merging "
               "(or use ALFRED vintages).")
    else:
        report("PASS", "FRED release dates", "\n".join(lines))


def check_levels_drift(splits):
    if len(splits) < 2:
        return
    names = list(splits)
    tr, te = splits[names[0]], splits[names[-1]]
    feats = [c for c in feature_columns(tr) if pd.api.types.is_numeric_dtype(tr[c])]
    sd = tr[feats].std().replace(0, np.nan)
    shift = ((te[feats].mean() - tr[feats].mean()).abs() / sd).sort_values(ascending=False)
    top = shift[shift > 1.0]
    levels = [c for c in top.index if c in LEVEL_MACRO]
    if len(top):
        report("WARN", f"Distribution shift ({names[0]} -> {names[-1]})",
               "\n".join(f"{k:28s} moved {v:.1f} std" for k, v in top.head(10).items()),
               "Raw levels (rates, CPI, VIX level) never repeat across periods, so the model "
               "uses them as a date stamp. Prefer changes / z-scores"
               + (f"; e.g. drop {levels}" if levels else "") + ".")
    else:
        report("PASS", "Distribution shift", "No feature moved more than 1 std between first and last split.")


def check_news(splits):
    lines, low = [], False
    for n, s in splits.items():
        parts = []
        for c in NEWS_COLUMNS:
            if c in s:
                share = float((s[c] > 0).mean())
                parts.append(f"{c}={share:.2%}")
                low |= share < 0.05
        if parts:
            lines.append(f"{n:12s} " + "  ".join(parts))
    if not lines:
        report("INFO", "News coverage", "No news columns.")
        return
    report("WARN" if low else "PASS", "News coverage (rows with any news)", "\n".join(lines),
           "Below ~5% the model cannot learn from news. Collect more history, or drop "
           "the columns until coverage improves.")


def check_universe(splits):
    names = list(splits)
    first, last = splits[names[0]], splits[names[-1]]
    only_last = set(last["ticker"]) - set(first["ticker"])
    full = pd.concat(splits.values())
    last_date = full["date"].max()
    ends = full.groupby("ticker")["date"].max()
    delisted = int((ends < last_date - pd.Timedelta(days=30)).sum())
    per_date = full.groupby("date").size()
    detail = (f"{len(only_last)} tickers appear only after {names[0]}; "
              f"{delisted} tickers stop trading before the end\n"
              f"stocks per date: min {per_date.min()}, median {int(per_date.median())}, max {per_date.max()}")
    if delisted == 0:
        report("WARN", "Survivorship bias", detail +
               "\nNo stock ever disappears: the universe is today's survivors only.",
               "Results are optimistic; mention it, or add delisted stocks if you can get them.")
    else:
        report("PASS", "Survivorship bias", detail)
    thin = per_date[per_date < 0.5 * per_date.median()]
    if len(thin):
        report("WARN", "Thin dates",
               f"{len(thin)} dates have < half the usual number of stocks, e.g. "
               f"{[d.date().isoformat() for d in thin.index[:5]]}",
               "Usually holidays in one market; fine unless there are many.")


def check_single_rules(df, feats):
    """Best balanced accuracy from 'predict UP if feature > 0'. A single raw
    feature should never beat ~51.5%."""
    from sklearn.metrics import balanced_accuracy_score
    y = df["target"].to_numpy()
    scores = {}
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() < 3:
            continue
        pred = (df[c].fillna(0).to_numpy() > 0).astype(int)
        s = balanced_accuracy_score(y, pred)
        scores[c] = max(s, 1 - s)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
    strong = [k for k, v in scores.items() if v > 0.515]
    # A 1-day return that predicts the next 5 days on its own = leak.
    # A 20/60-day return can be genuine trend/momentum in one period.
    leaky = [k for k in strong if k.endswith("_return_1") or k == "return_1"]
    status = "FAIL" if leaky else ("WARN" if strong else "PASS")
    report(status, "One-feature rules (last split)",
           "\n".join(f"{k:32s} balanced acc {v:.2%}" for k, v in top),
           ("1-day features beating ~51.5% alone leak future data: " + ", ".join(leaky[:6]))
           if leaky else "Long-horizon features beating 51.5% may just be a trend in this "
           "period; check they don't hold in every split.")


# -------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", help="Folder with train/validation/test_matrix.csv.gz")
    p.add_argument("--matrix", help="A single (unsplit) matrix file")
    p.add_argument("--output", default="matrix_report.json")
    args = p.parse_args()

    if args.matrix:
        splits = {"matrix": load(Path(args.matrix))}
    elif args.data_dir:
        d = Path(args.data_dir)
        splits = {n: load(d / f"{n}_matrix.csv.gz") for n in ("train", "validation", "test")}
    else:
        p.error("give --data-dir or --matrix")

    first = next(iter(splits.values()))
    last = list(splits.values())[-1]
    feats = feature_columns(first)
    print("=" * 70 + "\nMATRIX HEALTH CHECK\n" + "=" * 70)

    for n, s in splits.items():
        print(f"\n--- {n} ---", end="")
        check_structure(s)
    check_target(first)
    horizon = infer_horizon(first)
    check_split_gaps(splits, horizon)
    leak_top = check_leakage(first, feats)
    align = check_market_alignment(first)
    check_ffilled_returns(first)
    check_macro(first)
    check_levels_drift(splits)
    check_news(splits)
    check_universe(splits)
    check_single_rules(last, feats)

    counts = {k: sum(r["status"] == k for r in RESULTS) for k in ("FAIL", "WARN", "PASS")}
    print("\n" + "=" * 70)
    print(f"SUMMARY: {counts['FAIL']} FAIL, {counts['WARN']} WARN, {counts['PASS']} PASS")
    for r in RESULTS:
        if r["status"] == "FAIL":
            print(f"  FAIL - {r['check']}")
    print("=" * 70)
    Path(args.output).write_text(json.dumps(
        {"summary": counts, "horizon": horizon, "checks": RESULTS,
         "leak_top_corr": leak_top, "alignment_corr_t_t1": align}, indent=2), encoding="utf-8")
    print(f"Report saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()