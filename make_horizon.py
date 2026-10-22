"""Create a copy of the final matrix with a different target horizon.

The features do not depend on the horizon; only future_return and target do.
So instead of rebuilding the whole panel, this recomputes the target from the
saved price files exactly like stock_predictor.py does
(future_return = close.shift(-horizon) / close - 1) and writes a NEW file.
The original 5-day matrix is not touched.

Usage:
  python make_horizon.py --horizon 20
      -> data/final/final_feature_matrix_h20.csv.gz
  python make_horizon.py --horizon 5 --verify
      -> recomputes the 5-day target and checks it equals the existing one
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import stock_predictor as sp

MATRIX = Path("data") / "final" / "final_feature_matrix.csv.gz"


def future_returns(ticker: str, horizon: int) -> pd.Series:
    path = Path(sp.GLOBAL_V2_STOCK_DIR) / f"{sp._safe_global_filename(ticker)}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    close = sp._read_global_stock_file(str(path)).set_index("date")["Close"]
    return close.shift(-horizon) / close - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--matrix", default=str(MATRIX))
    p.add_argument("--out", help="Default: final_feature_matrix_h<N>.csv.gz next to the input")
    p.add_argument("--verify", action="store_true",
                   help="Only compare the recomputed target with the existing one (no file written).")
    args = p.parse_args()

    print(f"Loading {args.matrix} ...")
    df = pd.read_csv(args.matrix, low_memory=False, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} stocks")

    parts, missing = [], []
    for ticker, grp in df.groupby("ticker", sort=False):
        fut = future_returns(ticker, args.horizon)
        if fut.empty:
            missing.append(ticker)
        parts.append(grp["date"].map(fut))
    new_future = pd.concat(parts).reindex(df.index)
    if missing:
        print(f"  No price file for {len(missing)} stocks: {missing[:10]}")

    if args.verify:
        both = df["future_return"].notna() & new_future.notna()
        diff = (df.loc[both, "future_return"] - new_future[both]).abs()
        print(f"Compared {both.sum():,} rows: max difference {diff.max():.2e}, "
              f"rows differing by > 1e-6: {(diff > 1e-6).sum():,}")
        return

    out = df.copy()
    out["future_return"] = new_future
    out["target"] = (out["future_return"] > 0).astype(int)
    before = len(out)
    out = out.dropna(subset=["future_return"]).reset_index(drop=True)
    target = args.out or str(Path(args.matrix).with_name(
        Path(args.matrix).name.replace(".csv.gz", f"_h{args.horizon}.csv.gz")))
    out.to_csv(target, index=False, compression="gzip")
    print(f"Horizon {args.horizon}: {len(out):,} rows kept "
          f"({before - len(out):,} dropped at the end: their {args.horizon}-day future is not known yet)")
    print(f"UP share: {out['target'].mean():.1%}")
    print(f"Saved: {target}")
    print(f"\nNext:\n  python make_split.py --matrix {target} --out data/split_h{args.horizon} "
          f"--horizon {args.horizon}")


if __name__ == "__main__":
    main()
