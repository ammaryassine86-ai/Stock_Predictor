"""Download each stock's earnings history from Yahoo Finance (dates, EPS estimate,
reported EPS, surprise %).

Reads the tickers from data/global/universe.csv and writes one file per stock:
  data/global/earnings/<ticker>.csv
  columns: earnings_time_ny, date_ny, eps_estimate, eps_reported, surprise_pct

Yahoo gives every date in New York time. When features are built later, an
earnings report is only used from the first trading day AFTER date_ny, so a
report released after the close (or overnight for Asia) is never seen early.

Only downloads; the matrix and the models are not touched.
Safe to stop and re-run: tickers already downloaded are skipped (--refresh to redo).

Usage:  python download_earnings.py
        python download_earnings.py --due   # daily refresh: only stocks with a new report
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE = Path("data") / "global" / "universe.csv"
OUT_DIR = Path("data") / "global" / "earnings"
PAUSE_SECONDS = 1.0    # Yahoo blocks fast scraping of this page
RETRIES = 3
COLUMNS = ["earnings_time_ny", "date_ny", "eps_estimate", "eps_reported", "surprise_pct"]


def safe_name(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.=-" else "_" for ch in symbol)


def find_col(df: pd.DataFrame, *words: str):
    for col in df.columns:
        name = str(col).lower()
        if all(w in name for w in words):
            return col
    return None


def fetch(symbol: str) -> pd.DataFrame | None:
    for attempt in range(1, RETRIES + 1):
        try:
            raw = yf.Ticker(symbol).get_earnings_dates(limit=100)
            if raw is None or raw.empty:
                return pd.DataFrame(columns=COLUMNS)
            raw = raw.reset_index()
            when = pd.to_datetime(raw.iloc[:, 0], utc=True).dt.tz_convert("America/New_York")
            est, rep, sur = (find_col(raw, "estimate"), find_col(raw, "reported"),
                             find_col(raw, "surprise"))
            out = pd.DataFrame({
                "earnings_time_ny": when.dt.strftime("%Y-%m-%d %H:%M"),
                "date_ny": when.dt.strftime("%Y-%m-%d"),
                "eps_estimate": pd.to_numeric(raw[est], errors="coerce") if est is not None else float("nan"),
                "eps_reported": pd.to_numeric(raw[rep], errors="coerce") if rep is not None else float("nan"),
                "surprise_pct": pd.to_numeric(raw[sur], errors="coerce") if sur is not None else float("nan"),
            })
            return out.drop_duplicates("date_ny", keep="first").sort_values("date_ny").reset_index(drop=True)
        except ImportError as error:
            raise SystemExit(f"Missing package ({error}). Run:  pip install lxml beautifulsoup4")
        except Exception as error:
            if attempt == RETRIES:
                print(f"  [FAILED] {symbol}: {error}")
                return None
            time.sleep(5 * attempt)
    return None


def is_due(path: Path, today: str, recheck_days: int = 30) -> bool:
    """A stock needs a refresh when a scheduled report date has passed (its EPS
    is not in the file yet), or when its file is older than recheck_days."""
    if not path.exists():
        return True
    if time.time() - path.stat().st_mtime > recheck_days * 86400:
        return True
    df = pd.read_csv(path)
    month_ago = (pd.Timestamp(today) - pd.Timedelta(days=recheck_days)).strftime("%Y-%m-%d")
    dates = df["date_ny"].astype(str)
    waiting = df[df["eps_reported"].isna() & (dates < today) & (dates >= month_ago)]
    return len(waiting) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="Download again even if the file exists.")
    p.add_argument("--tickers", nargs="+", help="Only these tickers (for a quick test).")
    p.add_argument("--due", action="store_true",
                   help="Only stocks whose scheduled report date has passed (fast daily refresh).")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE)
    if args.tickers:
        universe = universe[universe["symbol"].isin(args.tickers)]
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.due:
        todo = [s for s in universe["symbol"] if is_due(OUT_DIR / f"{safe_name(s)}.csv", today)]
    else:
        todo = [s for s in universe["symbol"]
                if args.refresh or not (OUT_DIR / f"{safe_name(s)}.csv").exists()]
    print(f"{len(universe)} tickers, {len(todo)} to download into {OUT_DIR}")

    failed = []
    for i, symbol in enumerate(todo, 1):
        df = fetch(symbol)
        if df is None:
            failed.append(symbol)
        else:
            df.to_csv(OUT_DIR / f"{safe_name(symbol)}.csv", index=False)
            past = df[df["eps_reported"].notna()]
            first = past["date_ny"].iloc[0] if len(past) else "-"
            print(f"  [{i}/{len(todo)}] {symbol:12s} {len(past):3d} past reports since {first}")
        time.sleep(PAUSE_SECONDS)

    # ---- Coverage summary per market --------------------------------------
    rows = []
    for symbol, country in zip(universe["symbol"], universe["country"]):
        path = OUT_DIR / f"{safe_name(symbol)}.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        past = df[df["eps_reported"].notna()]
        since_2015 = past[past["date_ny"].astype(str) >= "2015-01-01"]
        rows.append({"country": country, "has_data": len(past) > 0,
                     "reports_since_2015": len(since_2015),
                     "with_surprise": int(since_2015["surprise_pct"].notna().sum())})
    cov = pd.DataFrame(rows).groupby("country").agg(
        stocks=("has_data", "size"), with_data=("has_data", "sum"),
        median_reports_since_2015=("reports_since_2015", "median"),
        median_with_surprise=("with_surprise", "median"))
    print("\nCoverage per market (a full history since 2015 is about 40 quarterly reports):")
    print(cov.to_string())
    if failed:
        print(f"\nFailed ({len(failed)}): {failed}\nRe-run the script to retry them.")


if __name__ == "__main__":
    main()
