"""Fix the ticker formats in data/global/universe.csv that Yahoo does not recognise,
then download the price history of ONLY those fixed tickers.

Wrong format            Yahoo format
  SEHK:5.HK        ->   0005.HK      (Hong Kong: 4 digits, no "SEHK:")
  TECK.B           ->   TECK-B.TO    (Canada: class shares use "-" and need ".TO")
  BIP.UN           ->   BIP-UN.TO
  BT.A             ->   BT-A.L       (UK)

What it changes:
  - data/global/universe.csv          corrected tickers (same rows, same order)
  - data/global/universe_original.csv backup of the old file (only made once)
  - data/global/stocks/<new>.csv      price files for the fixed tickers (2015 -> today)
Nothing else is touched: the other ~370 price files, the matrix and the models stay as they are.

Usage:
  python fix_universe.py --dry-run   # only show what would change
  python fix_universe.py
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
BACKUP = UNIVERSE.with_name("universe_original.csv")
STOCK_DIR = Path(sp.GLOBAL_V2_STOCK_DIR)

# Exceptions that do not follow the simple rules below.
MANUAL = {
    "BT.A": "BT-A.L",
}


def yahoo_symbol(symbol: str, country: str) -> str:
    symbol = symbol.strip()
    if symbol in MANUAL:
        return MANUAL[symbol]
    hk = re.fullmatch(r"SEHK:(\d+)\.HK", symbol)
    if hk:
        return f"{int(hk.group(1)):04d}.HK"
    if country == "Canada" and not symbol.endswith(".TO"):
        return symbol.replace(".", "-") + ".TO"
    return symbol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--start", default="2015-01-01")
    args = p.parse_args()

    uni = pd.read_csv(UNIVERSE)
    fixed = [yahoo_symbol(s, c) for s, c in zip(uni["symbol"], uni["country"])]
    changes = [(old, new) for old, new in zip(uni["symbol"], fixed) if old != new]
    print(f"{len(changes)} of {len(uni)} tickers need a new format:")
    for old, new in changes:
        print(f"  {old:14s} -> {new}")
    if not changes:
        print("Nothing to fix.")
        return
    if pd.Series(fixed).duplicated().any():
        raise SystemExit("Two tickers map to the same new name; nothing was changed.")
    if args.dry_run:
        print("\nDry run: nothing changed.")
        return

    if not BACKUP.exists():
        shutil.copy(UNIVERSE, BACKUP)
        print(f"\nBackup of the old universe: {BACKUP}")
    uni["symbol"] = fixed
    uni.to_csv(UNIVERSE, index=False)
    print(f"Saved corrected universe: {UNIVERSE}")

    # Download only the fixed tickers, exactly like stock_predictor.py --download-global-data
    new_symbols = [new for _, new in changes]
    end = date.today().isoformat()
    ok, failed = [], []
    for i in range(0, len(new_symbols), 20):
        chunk = new_symbols[i:i + 20]
        print(f"\nDownloading {i + 1}-{i + len(chunk)} / {len(new_symbols)} ...")
        data = sp._download_yahoo_chunk(chunk, args.start, end)
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
                print(f"  [FAILED] {symbol}")
                continue
            frame.to_csv(STOCK_DIR / f"{sp._safe_global_filename(symbol)}.csv")
            ok.append(symbol)
            print(f"  [OK] {symbol}: {len(frame)} rows, from {pd.Timestamp(frame.index[0]).date()}")
        time.sleep(1)

    print(f"\nDownloaded {len(ok)}/{len(new_symbols)} fixed tickers.")
    if failed:
        print(f"Still failing ({len(failed)}): {failed}")
        print("These may really be delisted or renamed; send me the list.")
    print("\nNext: python download_sectors.py   (adds the sectors of the new tickers)")


if __name__ == "__main__":
    main()
