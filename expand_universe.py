"""Add the missing stocks of the S&P 500, ASX 200 and FTSE 100 to the universe.

The original universe took stocks round-robin from each index until it had 500,
so these three big indexes were cut off alphabetically after ~63 stocks
(the S&P 500 stopped at "B"). The other indexes are already complete.

What it does:
  1. Reads the current member lists from Wikipedia (same scraper as stock_predictor.py).
  2. Fixes ticker formats for Yahoo (BRK.B -> BRK-B, BT.A -> BT-A.L, ...).
  3. Appends ONLY the stocks that are not in data/global/universe.csv yet
     (backup: data/global/universe_before_expand.csv, made once).
  4. Downloads the 2015 -> today price history of the new stocks.
Existing stocks, price files, the matrix and the models are not touched.

Usage:
  python expand_universe.py --dry-run     # only show what would be added
  python expand_universe.py
  python expand_universe.py --indexes "S&P 500"     # only one index
  python expand_universe.py --indexes "Nikkei 225"  # add Japan (official Nikkei list)
"""
import argparse
import re
import shutil
import time
from datetime import date
from pathlib import Path

import pandas as pd

import stock_predictor as sp

UNIVERSE = Path(sp.GLOBAL_V2_UNIVERSE)
BACKUP = UNIVERSE.with_name("universe_before_expand.csv")
STOCK_DIR = Path(sp.GLOBAL_V2_STOCK_DIR)
DEFAULT_INDEXES = ["S&P 500", "ASX 200", "FTSE 100"]


def yahoo_symbol(symbol: str, country: str) -> str:
    s = str(symbol).strip().upper()
    hk = re.fullmatch(r"(?:SEHK:)?(\d+)\.HK", s)
    if hk:
        return f"{int(hk.group(1)):04d}.HK"
    if country == "United Kingdom":
        base = s[:-2] if s.endswith(".L") else s
        base = base.rstrip(".").replace(".", "-")          # BT.A -> BT-A, RR. -> RR
        return base + ".L"
    if country == "Canada" and not s.endswith(".TO"):
        return s.replace(".", "-") + ".TO"
    return s


NIKKEI_URL = "https://indexes.nikkei.co.jp/en/nkave/index/component"


def scrape_nikkei225() -> list[dict]:
    """The official Nikkei 225 list (Wikipedia has no usable table for it).
    Codes like 7203 (Toyota) or 285A become 7203.T / 285A.T on Yahoo."""
    from io import StringIO
    from urllib.request import Request, urlopen
    req = Request(NIKKEI_URL, headers={"User-Agent": "Mozilla/5.0 stock-predictor/1.0"})
    with urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="ignore")
    rows = []
    for table in pd.read_html(StringIO(html)):
        cols = {str(c).strip().lower(): c for c in table.columns}
        if "code" not in cols:
            continue
        for code in table[cols["code"]].astype(str).str.strip():
            if re.fullmatch(r"[0-9A-Z]{4}", code):
                rows.append({"symbol": f"{code}.T", "country": "Japan", "region": "Asia-Pacific",
                             "benchmark": "^N225", "index": "Nikkei 225"})
    return rows


def download(symbols, start):
    end = date.today().isoformat()
    ok, failed = [], []
    for i in range(0, len(symbols), 25):
        chunk = symbols[i:i + 25]
        print(f"\nDownloading {i + 1}-{i + len(chunk)} / {len(symbols)} ...")
        try:
            data = sp._download_yahoo_chunk(chunk, start, end)
        except Exception as error:
            print(f"  batch failed: {error}")
            failed.extend(chunk)
            continue
        for symbol in chunk:
            frame = pd.DataFrame()
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    if symbol in set(data.columns.get_level_values(0)):
                        frame = pd.DataFrame(data[symbol]).dropna(how="all")
                else:
                    frame = pd.DataFrame(data).dropna(how="all")
            if frame.empty or "Close" not in frame.columns:
                failed.append(symbol)
                continue
            frame.to_csv(STOCK_DIR / f"{sp._safe_global_filename(symbol)}.csv")
            ok.append(symbol)
        print(f"  ok so far: {len(ok)}, failed so far: {len(failed)}")
        time.sleep(1)
    return ok, failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indexes", nargs="+", default=DEFAULT_INDEXES)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--start", default="2015-01-01")
    args = p.parse_args()

    uni = pd.read_csv(UNIVERSE)
    have = set(uni["symbol"])
    sources = [s for s in sp.GLOBAL_INDEX_SOURCES if s["name"] in args.indexes]
    missing = set(args.indexes) - {s["name"] for s in sources} - {"Nikkei 225"}
    if missing:
        raise SystemExit(f"Unknown index name(s): {missing}. "
                         f"Choose from: {[s['name'] for s in sp.GLOBAL_INDEX_SOURCES]}")

    new_rows = []
    for source in sources:
        rows = (scrape_nikkei225() if source["name"] == "Nikkei 225"
                else sp._scrape_global_index_source(source))
        for r in rows:
            r["symbol"] = yahoo_symbol(r["symbol"], r["country"])
        fresh = [r for r in rows if r["symbol"] not in have]
        print(f"{source['name']:10s} {len(rows):4d} members on Wikipedia, "
              f"{len(rows) - len(fresh):3d} already in the universe, {len(fresh):3d} to add")
        for r in fresh:
            have.add(r["symbol"])
            new_rows.append(r)
    if not new_rows:
        print("Nothing to add.")
        return
    print("Examples:", [r["symbol"] for r in new_rows[:12]])
    if args.dry_run:
        print("\nDry run: nothing changed.")
        return

    new = pd.DataFrame(new_rows)[uni.columns.tolist()]
    ok, failed = download(new["symbol"].tolist(), args.start)
    new = new[new["symbol"].isin(ok)]            # only keep stocks that really have prices

    if not BACKUP.exists():
        shutil.copy(UNIVERSE, BACKUP)
        print(f"\nBackup of the old universe: {BACKUP}")
    out = pd.concat([uni, new], ignore_index=True)
    out.to_csv(UNIVERSE, index=False)
    print(f"Universe: {len(uni)} -> {len(out)} stocks ({len(new)} added)")
    print(out["country"].value_counts().to_string())
    if failed:
        print(f"\nNo price data on Yahoo, not added ({len(failed)}): {failed}")
    print("\nNext: python download_sectors.py")


if __name__ == "__main__":
    main()
