"""Company-results features: earnings reports (all markets) and SEC fundamentals (US).

Used by BOTH final_matrix_builder.py (training) and predict_today.py (live),
so the model always sees exactly the same features in both places.

Point-in-time rule (no look-ahead):
  * an earnings report dated D (New York date) is used from the first trading
    row of the stock whose date is strictly AFTER D;
  * the 2-day price reaction to a report is only used once those 2 days are over;
  * an SEC number is used from the first trading row strictly after its FIRST
    filing date (later restatements are ignored).

Stocks without data simply get empty values (LightGBM handles them).

Columns added (prefix earn_ / fund_):
  earn_days_since       calendar days since the last report (empty if > 200)
  earn_surprise_last    last EPS surprise %, clipped to [-100, 100]
  earn_surprise_avg4    average of the last 4 surprises
  earn_reaction_2d      stock return minus its index over the 2 trading days after the last report
  fund_revenue_growth   latest revenue vs the same period a year earlier
  fund_profit_growth    latest net income vs a year earlier, divided by |year-earlier value|
  fund_gross_margin     gross profit / revenue (latest period)
  fund_operating_margin operating income / revenue
  fund_roe              net income (annualised) / shareholders' equity
  fund_debt_to_assets   liabilities / assets
  fund_current_ratio    current assets / current liabilities
  fund_days_since_filing calendar days since the last SEC filing
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARNINGS_DIR = Path("data") / "global" / "earnings"
FUNDAMENTALS_DIR = Path("data") / "global" / "fundamentals"

EARN_COLUMNS = ["earn_days_since", "earn_surprise_last", "earn_surprise_avg4", "earn_reaction_2d"]
FUND_COLUMNS = ["fund_revenue_growth", "fund_profit_growth", "fund_gross_margin", "fund_operating_margin",
                "fund_roe", "fund_debt_to_assets", "fund_current_ratio", "fund_days_since_filing"]
ALL_COLUMNS = EARN_COLUMNS + FUND_COLUMNS

MAX_DAYS_SINCE_REPORT = 200     # older than this = data gap, not a real "last report"
MAX_DAYS_SINCE_FILING = 200
REVENUE_CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet", "RevenuesNetOfInterestExpense"]


def safe_name(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.=-" else "_" for ch in str(symbol))


# ------------------------------------------------------------ earnings ----
def _earnings_for_stock(rows: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """rows: one stock's panel rows sorted by date (date, return_1, benchmark_return_1)."""
    out = pd.DataFrame(index=rows.index, columns=EARN_COLUMNS, dtype=float)
    reports = reports[reports["eps_reported"].notna()].copy()
    if reports.empty or rows.empty:
        return out
    reports["date_ny"] = pd.to_datetime(reports["date_ny"])
    reports = reports.sort_values("date_ny").drop_duplicates("date_ny")
    reports["surprise"] = pd.to_numeric(reports["surprise_pct"], errors="coerce").clip(-100, 100)
    reports["surprise_avg4"] = reports["surprise"].rolling(4, min_periods=1).mean()

    dates = rows["date"].to_numpy()
    # first trading row strictly after the report date
    first_row = np.searchsorted(dates, reports["date_ny"].to_numpy(), side="right")

    # 2-day reaction: rows first_row and first_row+1, known from row first_row+1 on
    excess = (rows["return_1"] - rows.get("benchmark_return_1", 0.0)).to_numpy(dtype=float)
    reaction = np.full(len(reports), np.nan)
    for k, i in enumerate(first_row):
        if i + 1 < len(rows):
            reaction[k] = np.nansum(excess[i:i + 2])

    # Report k is "the last report" for rows [first_row[k], first_row[k+1]).
    report_idx = np.searchsorted(first_row, np.arange(len(rows)), side="right") - 1
    has = report_idx >= 0
    idx = np.where(has, report_idx, 0)
    days = (dates - reports["date_ny"].to_numpy()[idx]).astype("timedelta64[D]").astype(float)
    valid = has & (days <= MAX_DAYS_SINCE_REPORT)

    out["earn_days_since"] = np.where(valid, days, np.nan)
    out["earn_surprise_last"] = np.where(valid, reports["surprise"].to_numpy()[idx], np.nan)
    out["earn_surprise_avg4"] = np.where(valid, reports["surprise_avg4"].to_numpy()[idx], np.nan)
    row_pos = np.arange(len(rows))
    reaction_known = valid & (row_pos >= first_row[idx] + 1)
    out["earn_reaction_2d"] = np.where(reaction_known, reaction[idx], np.nan)
    return out


# -------------------------------------------------------- fundamentals ----
def _prepare_facts(facts: pd.DataFrame) -> pd.DataFrame:
    facts = facts.copy()
    facts["end"] = pd.to_datetime(facts["end"], errors="coerce")
    facts["start"] = pd.to_datetime(facts["start"], errors="coerce")
    facts["filed"] = pd.to_datetime(facts["filed"], errors="coerce")
    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")
    facts = facts.dropna(subset=["end", "filed", "value"])
    facts["days"] = (facts["end"] - facts["start"]).dt.days
    # all revenue tags are merged into one; for each period keep the FIRST published value
    facts["concept"] = facts["concept"].where(~facts["concept"].isin(REVENUE_CONCEPTS), "Revenue")
    facts = facts.sort_values("filed").drop_duplicates(["concept", "start", "end"], keep="first")
    return facts


def _duration_kind(days) -> pd.Series:
    days = pd.Series(days)
    return pd.Series(np.select([days.between(80, 100), days.between(350, 380)], ["Q", "Y"], ""),
                     index=days.index)


def _latest_flow(known: pd.DataFrame, concept: str):
    """Latest quarterly (or else annual) value of an income-statement item, and
    the same period one year earlier. Returns (value, prior_value, kind, start, end)."""
    f = known[known["concept"] == concept]
    if f.empty:
        return None
    f = f.assign(kind=_duration_kind(f["days"]).to_numpy())
    f = f[f["kind"] != ""]
    if f.empty:
        return None
    last_end = f["end"].max()
    cur = f[f["end"] == last_end].sort_values("kind")       # "Q" before "Y"
    cur = cur.iloc[0]
    same = f[(f["kind"] == cur["kind"]) &
             ((cur["end"] - f["end"]).dt.days.between(345, 385))]
    prior = same["value"].iloc[-1] if len(same) else np.nan
    return cur["value"], prior, cur["kind"], cur["start"], cur["end"]


def _period_value(known: pd.DataFrame, concept: str, start, end):
    f = known[(known["concept"] == concept) & (known["start"] == start) & (known["end"] == end)]
    return f["value"].iloc[0] if len(f) else np.nan


def _latest_instant(known: pd.DataFrame, concept: str):
    f = known[(known["concept"] == concept) & known["start"].isna()]
    return f.sort_values("end")["value"].iloc[-1] if len(f) else np.nan


def _ratio(a, b):
    return a / b if b and np.isfinite(b) and b != 0 and np.isfinite(a) else np.nan


def _fundamental_snapshots(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per filing date with the metrics known right after that filing."""
    facts = _prepare_facts(facts)
    snaps = []
    for filed in np.sort(facts["filed"].unique()):
        known = facts[facts["filed"] <= filed]
        snap = {"filed": pd.Timestamp(filed)}
        rev = _latest_flow(known, "Revenue")
        if rev:
            value, prior, kind, start, end = rev
            snap["fund_revenue_growth"] = _ratio(value - prior, abs(prior)) if np.isfinite(prior) else np.nan
            snap["fund_gross_margin"] = _ratio(_period_value(known, "GrossProfit", start, end), value)
            snap["fund_operating_margin"] = _ratio(_period_value(known, "OperatingIncomeLoss", start, end), value)
        ni = _latest_flow(known, "NetIncomeLoss")
        equity = _latest_instant(known, "StockholdersEquity")
        if ni:
            value, prior, kind, _, _ = ni
            snap["fund_profit_growth"] = _ratio(value - prior, abs(prior)) if np.isfinite(prior) else np.nan
            snap["fund_roe"] = _ratio(value * (4 if kind == "Q" else 1), equity) if equity > 0 else np.nan
        assets = _latest_instant(known, "Assets")
        liabilities = _latest_instant(known, "Liabilities")
        if not np.isfinite(liabilities) and np.isfinite(equity):
            liabilities = assets - equity
        snap["fund_debt_to_assets"] = _ratio(liabilities, assets)
        snap["fund_current_ratio"] = _ratio(_latest_instant(known, "AssetsCurrent"),
                                            _latest_instant(known, "LiabilitiesCurrent"))
        snaps.append(snap)
    out = pd.DataFrame(snaps)
    for c in FUND_COLUMNS:
        if c not in out:
            out[c] = np.nan
    # growth rates of tiny bases explode; clip to a sane range
    for c in ("fund_revenue_growth", "fund_profit_growth"):
        out[c] = out[c].clip(-5, 5)
    out["fund_roe"] = out["fund_roe"].clip(-2, 2)
    out["fund_current_ratio"] = out["fund_current_ratio"].clip(0, 20)
    return out


def _fundamentals_for_stock(rows: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index, columns=FUND_COLUMNS, dtype=float)
    if snaps.empty or rows.empty:
        return out
    left = rows[["date"]].reset_index().sort_values("date")
    merged = pd.merge_asof(left, snaps.sort_values("filed"), left_on="date", right_on="filed",
                           allow_exact_matches=False)   # strictly after the filing date
    merged = merged.set_index("index")
    days = (merged["date"] - merged["filed"]).dt.days
    fresh = days <= MAX_DAYS_SINCE_FILING
    for c in FUND_COLUMNS:
        if c == "fund_days_since_filing":
            out[c] = days.where(fresh)
        else:
            out[c] = merged[c].where(fresh)
    return out


# ---------------------------------------------------------------- main ----
def add_company_features(panel: pd.DataFrame, earnings_dir: Path = EARNINGS_DIR,
                         fundamentals_dir: Path = FUNDAMENTALS_DIR, verbose: bool = True) -> pd.DataFrame:
    """Add the earn_* and fund_* columns to a panel with date, ticker, return_1
    (and benchmark_return_1). Rows keep their order and index."""
    earnings_dir, fundamentals_dir = Path(earnings_dir), Path(fundamentals_dir)
    if not pd.api.types.is_datetime64_any_dtype(panel["date"]):
        panel = panel.assign(date=pd.to_datetime(panel["date"]))   # (no full copy otherwise: the panel is big)
    result = pd.DataFrame(index=panel.index, columns=ALL_COLUMNS, dtype=float)
    n_earn = n_fund = 0
    snapshot_cache: dict[str, pd.DataFrame] = {}

    for ticker, rows in panel.groupby("ticker", sort=False):
        rows = rows.sort_values("date")
        name = safe_name(ticker)
        e_path = earnings_dir / f"{name}.csv"
        if e_path.exists():
            reports = pd.read_csv(e_path)
            if len(reports):
                result.loc[rows.index, EARN_COLUMNS] = _earnings_for_stock(rows, reports).to_numpy()
                n_earn += 1
        f_path = fundamentals_dir / f"{name}.csv"
        if f_path.exists():
            if name not in snapshot_cache:
                facts = pd.read_csv(f_path)
                snapshot_cache[name] = _fundamental_snapshots(facts) if len(facts) else pd.DataFrame()
            result.loc[rows.index, FUND_COLUMNS] = _fundamentals_for_stock(rows, snapshot_cache[name]).to_numpy()
            n_fund += 1

    for c in ALL_COLUMNS:
        panel[c] = result[c].astype(float)
    if verbose:
        cover = {c: f"{panel[c].notna().mean():.0%}" for c in ("earn_days_since", "earn_surprise_last",
                                                               "fund_revenue_growth", "fund_roe")}
        print(f"[COMPANY] Earnings files: {n_earn} stocks, fundamentals: {n_fund} stocks. "
              f"Rows filled: {cover}")
    return panel
