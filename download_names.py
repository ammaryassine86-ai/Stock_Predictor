"""Download each stock's company name from Yahoo Finance (for the web interface search).

Reads data/global/universe.csv and writes data/global/names.csv
(columns: symbol, name). Safe to stop and re-run: names already found are skipped.

Usage:  python download_names.py
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE = Path("data") / "global" / "universe.csv"
OUT = Path("data") / "global" / "names.csv"
PAUSE_SECONDS = 0.3


def fetch(symbol: str) -> str:
    for attempt in range(3):
        try:
            info = yf.Ticker(symbol).info or {}
            return info.get("longName") or info.get("shortName") or ""
        except Exception:
            time.sleep(2 * (attempt + 1))
    return ""


def main():
    symbols = pd.read_csv(UNIVERSE)["symbol"].astype(str).tolist()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["symbol", "name"])
    have = dict(zip(done["symbol"], done["name"].fillna("")))
    todo = [s for s in symbols if not have.get(s)]
    print(f"{len(symbols)} stocks, {len(symbols) - len(todo)} names already known, {len(todo)} to download")
    for i, s in enumerate(todo, 1):
        have[s] = fetch(s)
        print(f"  [{i}/{len(todo)}] {s:12s} {have[s] or '-'}")
        if i % 50 == 0:
            pd.DataFrame(sorted(have.items()), columns=["symbol", "name"]).to_csv(OUT, index=False)
        time.sleep(PAUSE_SECONDS)
    pd.DataFrame(sorted(have.items()), columns=["symbol", "name"]).to_csv(OUT, index=False)
    missing = [s for s in symbols if not have.get(s)]
    print(f"Saved {OUT}. Without a name: {len(missing)} {missing[:10]}")


if __name__ == "__main__":
    main()
