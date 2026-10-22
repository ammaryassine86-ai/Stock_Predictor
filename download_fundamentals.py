"""Download company fundamentals for the US stocks from SEC EDGAR (free, official).

Every number in EDGAR comes with its FILING date ("filed"): the day the report
became public. Features will only use a number from the trading day after it
was filed, so the model never sees results before the market could.

Reads the US tickers from data/global/universe.csv and writes one file per stock:
  data/global/fundamentals/<ticker>.csv
  columns: concept, unit, start, end, value, form, fiscal_year, fiscal_period, filed

The SEC asks every script to identify itself with a contact e-mail (it is only
sent to sec.gov in the request header). Give yours once with --email, or set
the SEC_EMAIL environment variable.

Only downloads; the matrix and the models are not touched.
Safe to re-run: tickers already downloaded are skipped (--refresh to redo).

Usage:  python download_fundamentals.py --email you@example.com
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

UNIVERSE = Path("data") / "global" / "universe.csv"
OUT_DIR = Path("data") / "global" / "fundamentals"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
PAUSE_SECONDS = 0.2           # SEC limit is 10 requests per second
FIRST_PERIOD = "2013-01-01"   # a bit before 2015, so growth rates exist from the start
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

# Concepts kept. Several names for the same thing: companies use different tags.
CONCEPTS = {
    "us-gaap": [
        # income statement
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
        "RevenuesNetOfInterestExpense", "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss",
        "EarningsPerShareDiluted", "EarningsPerShareBasic",
        # balance sheet
        "Assets", "Liabilities", "StockholdersEquity", "AssetsCurrent", "LiabilitiesCurrent",
        "CashAndCashEquivalentsAtCarryingValue", "LongTermDebt", "LongTermDebtNoncurrent",
        # cash flow
        "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
    ],
    "dei": ["EntityCommonStockSharesOutstanding"],
}


def safe_name(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.=-" else "_" for ch in symbol)


def get_json(session: requests.Session, url: str) -> dict | None:
    for attempt in range(1, 4):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as error:
            if attempt == 3:
                print(f"  [FAILED] {url}: {error}")
                return None
            time.sleep(3 * attempt)
    return None


def facts_to_table(facts: dict) -> pd.DataFrame:
    rows = []
    for taxonomy, names in CONCEPTS.items():
        group = facts.get("facts", {}).get(taxonomy, {})
        for concept in names:
            for unit, entries in group.get(concept, {}).get("units", {}).items():
                for e in entries:
                    if e.get("form") not in FORMS or str(e.get("end", "")) < FIRST_PERIOD:
                        continue
                    rows.append({
                        "concept": concept, "unit": unit,
                        "start": e.get("start", ""),   # empty for balance-sheet values
                        "end": e.get("end"), "value": e.get("val"),
                        "form": e.get("form"), "fiscal_year": e.get("fy"),
                        "fiscal_period": e.get("fp"), "filed": e.get("filed"),
                    })
    df = pd.DataFrame(rows, columns=["concept", "unit", "start", "end", "value", "form",
                                     "fiscal_year", "fiscal_period", "filed"])
    # The same number is repeated in later reports (as last year's comparison).
    # Keep only its FIRST publication: that is when the market learned it.
    df = df.sort_values("filed").drop_duplicates(["concept", "unit", "start", "end"], keep="first")
    return df.sort_values(["concept", "end", "filed"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--email", default=os.environ.get("SEC_EMAIL"),
                   help="Contact e-mail the SEC requires in the request header.")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--tickers", nargs="+", help="Only these tickers (for a quick test).")
    args = p.parse_args()
    if not args.email or "@" not in args.email:
        raise SystemExit("The SEC requires a contact e-mail: python download_fundamentals.py --email you@example.com")

    session = requests.Session()
    session.headers.update({"User-Agent": f"StockAgent research {args.email}",
                            "Accept-Encoding": "gzip, deflate"})

    universe = pd.read_csv(UNIVERSE)
    us = universe[universe["country"] == "USA"]["symbol"].tolist()
    if args.tickers:
        us = [t for t in us if t in args.tickers] or args.tickers
    ticker_map = get_json(session, TICKER_MAP_URL)
    if not ticker_map:
        raise SystemExit("Could not download the SEC ticker list. Check the internet connection.")
    cik_of = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in ticker_map.values()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, failed = [], []
    for i, symbol in enumerate(us, 1):
        out = OUT_DIR / f"{safe_name(symbol)}.csv"
        if out.exists() and not args.refresh:
            df = pd.read_csv(out)
        else:
            cik = cik_of.get(symbol.upper().replace(".", "-"))
            if cik is None:
                print(f"  [{i}/{len(us)}] {symbol:8s} not found in the SEC list")
                failed.append(symbol)
                continue
            facts = get_json(session, FACTS_URL.format(cik=cik))
            time.sleep(PAUSE_SECONDS)
            if not facts:
                failed.append(symbol)
                continue
            df = facts_to_table(facts)
            df.to_csv(out, index=False)
        revenue = df[df["concept"].isin(CONCEPTS["us-gaap"][:4])]
        summary.append({"ticker": symbol, "values": len(df), "concepts": df["concept"].nunique(),
                        "first_filed": df["filed"].min() if len(df) else "-",
                        "revenue_reports": revenue["end"].nunique()})
        print(f"  [{i}/{len(us)}] {symbol:8s} {len(df):5d} values, {df['concept'].nunique():2d} concepts, "
              f"first filed {summary[-1]['first_filed']}")

    s = pd.DataFrame(summary)
    if len(s):
        print(f"\n{len(s)}/{len(us)} US stocks have fundamentals.")
        print(f"Median: {s['concepts'].median():.0f} concepts, "
              f"{s['revenue_reports'].median():.0f} revenue periods since {FIRST_PERIOD[:4]}.")
        thin = s[s["revenue_reports"] < 20]["ticker"].tolist()
        if thin:
            print(f"Little revenue history (<20 periods, e.g. banks or recent listings): {thin}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
    print(f"Saved to {OUT_DIR}")
    print("Next: send me the coverage summary of this and of download_earnings.py")


if __name__ == "__main__":
    main()
