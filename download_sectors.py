"""Download each stock's sector and industry from Yahoo Finance (one time).

Reads the tickers from data/global/universe.csv and writes
data/global/sectors.csv with columns: symbol, sector, industry.

Safe to stop and re-run: tickers already in sectors.csv are skipped.
Prices and the matrix are not touched.

Usage:  python download_sectors.py
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE = Path("data") / "global" / "universe.csv"
OUT = Path("data") / "global" / "sectors.csv"
PAUSE_SECONDS = 0.4   # be gentle with Yahoo
RETRIES = 3


def fetch(symbol: str) -> dict:
    for attempt in range(1, RETRIES + 1):
        try:
            info = yf.Ticker(symbol).info or {}
            return {
                "symbol": symbol,
                "sector": info.get("sector") or info.get("sectorDisp") or "",
                "industry": info.get("industry") or info.get("industryDisp") or "",
            }
        except Exception as error:
            if attempt == RETRIES:
                print(f"  [FAILED] {symbol}: {error}")
                return {"symbol": symbol, "sector": "", "industry": ""}
            time.sleep(2 * attempt)
    return {"symbol": symbol, "sector": "", "industry": ""}


def main():
    symbols = pd.read_csv(UNIVERSE)["symbol"].astype(str).str.strip().tolist()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["symbol", "sector", "industry"])
    have = set(done.loc[done["sector"].fillna("") != "", "symbol"])
    todo = [s for s in symbols if s not in have]
    print(f"{len(symbols)} tickers in universe, {len(have)} already have a sector, {len(todo)} to download")

    rows = [r for r in done.to_dict("records") if r["symbol"] in have]
    for i, symbol in enumerate(todo, 1):
        row = fetch(symbol)
        rows.append(row)
        print(f"  [{i}/{len(todo)}] {symbol:16s} {row['sector'] or '-':25s} {row['industry']}")
        if i % 25 == 0:  # save progress regularly
            pd.DataFrame(rows).to_csv(OUT, index=False)
        time.sleep(PAUSE_SECONDS)

    out = pd.DataFrame(rows).drop_duplicates("symbol", keep="last").sort_values("symbol")
    out.to_csv(OUT, index=False)
    missing = (out["sector"].fillna("") == "").sum()
    print(f"\nSaved {len(out)} tickers to {OUT}")
    print("Sectors:", out.loc[out["sector"] != "", "sector"].value_counts().to_dict())
    if missing:
        print(f"{missing} tickers have no sector (they get 'Unknown'). Re-run to retry them.")


if __name__ == "__main__":
    main()
