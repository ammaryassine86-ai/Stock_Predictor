import os
import pandas as pd

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL"]
NEWS_DIR = "data/news"

for ticker in TICKERS:
    path = os.path.join(NEWS_DIR, f"{ticker}_processed.csv")

    if not os.path.exists(path):
        print(f"\n{ticker}: file not found -> {path}")
        continue

    df = pd.read_csv(path)

    if "date" not in df.columns:
        print(f"\n{ticker}: no date column")
        continue

    counts = df.groupby("date").size()

    print("\n" + "=" * 60)
    print(ticker)
    print("=" * 60)

    print(f"Total headlines: {len(df):,}")
    print(f"Days with news: {len(counts):,}")

    print("\nHeadlines per day:")
    print(f"Mean:   {counts.mean():.2f}")
    print(f"Median: {counts.median():.2f}")
    print(f"P75:    {counts.quantile(.75):.2f}")
    print(f"P90:    {counts.quantile(.90):.2f}")
    print(f"P95:    {counts.quantile(.95):.2f}")
    print(f"P99:    {counts.quantile(.99):.2f}")
    print(f"Max:    {counts.max()}")

    print("\nDistribution:")

    ranges = [
        ("0", counts == 0),
        ("1", counts == 1),
        ("2", counts == 2),
        ("3", counts == 3),
        ("4-5", counts.between(4, 5)),
        ("6-10", counts.between(6, 10)),
        (">10", counts > 10),
    ]

    for label, condition in ranges:
        n = condition.sum()
        pct = n / len(counts) * 100
        print(f"{label:>5}: {n:5d} days ({pct:6.2f}%)")

    print("\nHeadlines retained with a daily cap:")

    for n in [1, 2, 3, 5, 10]:
        retained = counts.clip(upper=n).sum()
        pct = retained / len(df) * 100
        print(f"Top {n:2d}/day: {retained:6d} headlines ({pct:6.2f}%)")

    if "relevance_score" in df.columns:
        relevance = pd.to_numeric(
            df["relevance_score"], errors="coerce"
        ).dropna()

        print("\nAlpha Vantage relevance_score:")
        print(f"Available: {len(relevance):,}/{len(df):,}")
        print(f"Mean:      {relevance.mean():.4f}")
        print(f"Median:    {relevance.median():.4f}")
        print(f"P25:       {relevance.quantile(.25):.4f}")
        print(f"P75:       {relevance.quantile(.75):.4f}")
        print(f"P90:       {relevance.quantile(.90):.4f}")
        print(f"Min:       {relevance.min():.4f}")
        print(f"Max:       {relevance.max():.4f}")
    else:
        print("\nNo relevance_score column found.")
        print("Columns:", list(df.columns))