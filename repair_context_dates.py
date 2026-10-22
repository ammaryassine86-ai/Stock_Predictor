"""Rebuild data/global/context/*.csv from data/global/raw_backup/context/.

The files in context/ had their dates converted to UTC, which moved every
European/Asian date back one day (DAX on 2015-01-01 and on Sundays). The
backup still has Yahoo's original stamps like '2015-01-02 00:00:00+01:00';
the first 10 characters are the exchange's real trading date.

The broken files are moved to data/global/context_broken/ (nothing deleted).

Usage:  python repair_context_dates.py
"""
import shutil
from pathlib import Path

import pandas as pd

GLOBAL = Path("data") / "global"
BACKUP = GLOBAL / "raw_backup" / "context"
CONTEXT = GLOBAL / "context"
BROKEN = GLOBAL / "context_broken"


def main():
    files = sorted(BACKUP.glob("*.csv"))
    if not files:
        raise SystemExit(f"No backup files found in {BACKUP}")
    BROKEN.mkdir(parents=True, exist_ok=True)

    for src in files:
        raw = pd.read_csv(src)
        date_col = next((c for c in raw.columns if c.lower() in ("date", "datetime")), raw.columns[0])
        out = pd.DataFrame({
            "Date": pd.to_datetime(raw[date_col].astype(str).str.strip().str[:10], errors="coerce"),
            "Close": pd.to_numeric(raw["Close"], errors="coerce"),
        })
        out = out.dropna(subset=["Date", "Close"])
        out = out[out["Close"] > 0]                      # prices must be positive for returns
        out = pd.DataFrame(out).drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

        dst = CONTEXT / src.name
        if dst.exists():
            shutil.move(str(dst), str(BROKEN / src.name))
        out.to_csv(dst, index=False)

        weekend = int((pd.to_datetime(out["Date"]).dt.dayofweek >= 5).sum())
        print(f"{src.name:16s} {len(out):5d} rows  {out['Date'].iloc[0]} -> {out['Date'].iloc[-1]}"
              f"  weekend dates: {weekend}")

    print(f"\nRepaired {len(files)} files in {CONTEXT}")
    print(f"Old (shifted) files moved to {BROKEN}")
    print("Weekend dates should be 0 for stock indexes (FX can have a few).")


if __name__ == "__main__":
    main()
