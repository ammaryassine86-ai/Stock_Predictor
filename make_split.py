"""Split the final matrix chronologically into train / validation / test
(60 / 20 / 20 by trading date) with a purge gap between splits.

The gap removes `--horizon` trading dates before each cutoff, because those
rows' 5-day targets use prices from the next split.

Usage:
    python make_split.py
    python make_split.py --matrix data\\final\\final_feature_matrix.csv.gz --out data\\split
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=str(Path("data") / "final" / "final_feature_matrix.csv.gz"))
    p.add_argument("--out", default=str(Path("data") / "split"))
    p.add_argument("--train", type=float, default=0.60)
    p.add_argument("--validation", type=float, default=0.20)
    p.add_argument("--horizon", type=int, default=5, help="Purge gap in trading days")
    args = p.parse_args()

    print(f"Loading {args.matrix} ...")
    df = pd.read_csv(args.matrix, low_memory=False, parse_dates=["date"])
    dates = np.sort(df["date"].unique())
    n = len(dates)
    i_train = int(n * args.train)
    i_val = int(n * (args.train + args.validation))
    h = args.horizon

    # Last `h` dates before each cutoff are dropped (the purge gap).
    train_dates = dates[: i_train - h]
    val_dates = dates[i_train: i_val - h]
    test_dates = dates[i_val:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, ds in (("train", train_dates), ("validation", val_dates), ("test", test_dates)):
        part = df[df["date"].isin(ds)].sort_values(["date", "ticker"])
        path = out / f"{name}_matrix.csv.gz"
        part.to_csv(path, index=False, compression="gzip")
        print(f"{name:10s} {len(part):>9,} rows  {pd.Timestamp(ds[0]).date()} -> "
              f"{pd.Timestamp(ds[-1]).date()}  -> {path}")
    print(f"Purge gap: {h} trading dates dropped before validation and before test.")


if __name__ == "__main__":
    main()