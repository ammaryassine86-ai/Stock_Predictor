import argparse
import json
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd

import fundamental_features as ff

PANEL = os.path.join('data', 'global', 'training_panel.csv.gz')
NEWS_DIR = os.path.join('data', 'news')
FINAL_DIR = os.path.join('data', 'final')
FINAL_MATRIX = os.path.join(FINAL_DIR, 'final_feature_matrix.csv.gz')
FINAL_METADATA = os.path.join(FINAL_DIR, 'final_feature_matrix_metadata.json')

RULE_NEWS_COLS = [
    'news_score', 'news_count', 'positive_count', 'negative_count', 'neutral_count'
]
GEMINI_NEWS_COLS = [
    'gemini_sentiment_score', 'gemini_relevance', 'gemini_confidence', 'gemini_news_count'
]


def _ticker_from_news_file(name):
    if not name.lower().endswith('.csv'):
        return None
    stem = name[:-4]
    suffixes = ['_processed', '_daily', '_gemini', '_gemini_daily', '_gemini_input', '_gemini_input_daily']
    for suffix in suffixes:
        if stem.endswith(suffix):
            return None
    if not stem or stem.startswith('_'):
        return None
    return stem.upper()


def _read_daily_news():
    frames = []
    if not os.path.isdir(NEWS_DIR):
        return pd.DataFrame()

    raw_files = sorted(os.listdir(NEWS_DIR))
    tickers = [t for t in (_ticker_from_news_file(n) for n in raw_files) if t]
    for ticker in tickers:
        daily_path = os.path.join(NEWS_DIR, f'{ticker}_daily.csv')
        gemini_path = os.path.join(NEWS_DIR, f'{ticker}_gemini_daily.csv')
        rule = None
        gemini = None

        if os.path.exists(daily_path):
            try:
                rule = pd.read_csv(daily_path)
                if 'date' in rule.columns:
                    rule['date'] = pd.to_datetime(rule['date'], errors='coerce').dt.normalize()
                    rule = rule.dropna(subset=['date'])
                    for c in RULE_NEWS_COLS:
                        if c not in rule.columns:
                            rule[c] = 0.0
                        rule[c] = pd.to_numeric(rule[c], errors='coerce').fillna(0.0)
                    rule['news_available'] = (rule['news_count'] > 0).astype('int8')
                    rule = rule[['date'] + RULE_NEWS_COLS + ['news_available']]
            except Exception as e:
                print(f'[FINAL NEWS] Could not read {daily_path}: {e}')
                rule = None

        if os.path.exists(gemini_path):
            try:
                gemini = pd.read_csv(gemini_path)
                if 'date' in gemini.columns:
                    gemini['date'] = pd.to_datetime(gemini['date'], errors='coerce').dt.normalize()
                    gemini = gemini.dropna(subset=['date'])
                    for c in GEMINI_NEWS_COLS:
                        if c not in gemini.columns:
                            gemini[c] = 0.0
                        gemini[c] = pd.to_numeric(gemini[c], errors='coerce').fillna(0.0)
                    gemini['gemini_available'] = (gemini['gemini_news_count'] > 0).astype('int8')
                    gemini = gemini[['date'] + GEMINI_NEWS_COLS + ['gemini_available']]
            except Exception as e:
                print(f'[FINAL NEWS] Could not read {gemini_path}: {e}')
                gemini = None

        if rule is None and gemini is None:
            continue

        if rule is not None and gemini is not None:
            daily = rule.merge(gemini, on='date', how='outer')
        elif rule is not None:
            daily = rule.copy()
        elif gemini is not None:
            daily = gemini.copy()
        else:
            continue

        daily['ticker'] = ticker
        frames.append(daily)

    if not frames:
        return pd.DataFrame()

    news = pd.concat(frames, ignore_index=True, sort=False)
    for c in RULE_NEWS_COLS + GEMINI_NEWS_COLS + ['news_available', 'gemini_available']:
        if c not in news.columns:
            news[c] = 0
    news = news.sort_values(['ticker', 'date'])
    news = news.drop_duplicates(['ticker', 'date'], keep='last')
    return news


def _shift_news_to_next_stock_day(panel, news):
    """Use only news known before the prediction date.

    The existing daily news files use a 16:00 New York cutoff. Because the
    final matrix is global (different exchanges close at different local
    times), we conservatively attach each day's news to the next available
    trading row for that ticker. This avoids same-day cross-market leakage.
    """
    news_cols = [c for c in RULE_NEWS_COLS + GEMINI_NEWS_COLS + ['news_available', 'gemini_available'] if c in news.columns]
    if not news_cols or news.empty:
        for c in news_cols:
            panel[c] = 0.0
        panel['news_available'] = 0
        panel['gemini_available'] = 0
        return panel

    panel = panel.sort_values(['ticker', 'date']).reset_index(drop=True)
    news = news.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Merge same-day news first, then shift one trading row within each stock.
    merged = panel.merge(news, on=['ticker', 'date'], how='left', suffixes=('', '_news'))
    for c in news_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0.0)

    shifted_parts = []
    for ticker, group in merged.groupby('ticker', sort=False):
        group = group.copy()
        group[news_cols] = group[news_cols].shift(1).fillna(0.0)
        shifted_parts.append(group)
    merged = pd.concat(shifted_parts, ignore_index=True)
    return merged


# ----------------------------------------------------------- cleaning ----
ID_AND_TARGET = ['date', 'ticker', 'country', 'region', 'benchmark', 'future_return', 'target']

# Raw levels never repeat across periods (CPI in 2016 vs 2025), so a model can
# use them as a date stamp. Their *_change versions are kept.
RAW_LEVEL_COLUMNS = ['cpi', 'fed_funds_rate', 'yield_10y', 'yield_2y', 'dollar_index']

# Known near-duplicates (|corr| > 0.95). The first column is kept.
KNOWN_REDUNDANT = {
    'neutral_count': 'news_count',
    'gemini_confidence': 'gemini_available',
    'ema_12_ratio': 'sma_20_ratio',
    'ema_26_ratio': 'sma_20_ratio',
}

# Stocks whose price doesn't move on >15% of days. Many normal low-priced ASX
# stocks sit at 9-14% (price often moves less than one tick), so 10% would
# remove real companies such as AMP.AX; 15% only removes the truly illiquid.
MAX_ZERO_RETURN_SHARE = 0.15
MIN_NEWS_COVERAGE = 0.02       # news block dropped if <2% of rows have news
REDUNDANT_CORR = 0.95


def clean_matrix(panel):
    """Remove illiquid stocks and useless/redundant columns. Returns the
    cleaned panel and a report of everything removed."""
    report = {}
    print('\n=== CLEANING ===')

    # 1. Illiquid stocks: price unchanged on many days = stale quotes.
    if 'return_1' in panel.columns:
        zero_share = panel.groupby('ticker')['return_1'].apply(lambda s: float((s == 0).mean()))
        illiquid = sorted(zero_share[zero_share > MAX_ZERO_RETURN_SHARE].index)
        if illiquid:
            before = len(panel)
            panel = panel[~panel['ticker'].isin(illiquid)]
            print(f'[CLEAN] Removed {len(illiquid)} illiquid stocks '
                  f'(>{MAX_ZERO_RETURN_SHARE:.0%} days without a price change), '
                  f'{before - len(panel):,} rows: {illiquid}')
        report['illiquid_stocks_removed'] = {t: round(zero_share[t], 3) for t in illiquid}

    features = [c for c in panel.columns if c not in ID_AND_TARGET]
    dropped = {}

    # 2. News block: useless when almost no row has news.
    news_cols = [c for c in RULE_NEWS_COLS + GEMINI_NEWS_COLS + ['news_available', 'gemini_available']
                 if c in features]
    if news_cols and 'news_available' in panel.columns:
        coverage = float((panel['news_available'] > 0).mean())
        report['news_coverage'] = coverage
        if coverage < MIN_NEWS_COVERAGE:
            for c in news_cols:
                dropped[c] = f'news coverage only {coverage:.2%} of rows'
        else:
            print(f'[CLEAN] News kept: {coverage:.2%} of rows have news')

    # 3. Raw macro levels.
    for c in RAW_LEVEL_COLUMNS:
        if c in features:
            dropped[c] = 'raw level (acts as a date stamp); *_change kept'

    # 4. Known redundant pairs.
    for c, keep in KNOWN_REDUNDANT.items():
        if c in features and c not in dropped:
            dropped[c] = f'near-duplicate of {keep}'

    # 5. Constant columns.
    for c in features:
        if c not in dropped and panel[c].nunique(dropna=True) <= 1:
            dropped[c] = 'constant'

    # 6. Any other near-duplicates found in the data (keeps the first column).
    remaining = [c for c in features if c not in dropped and pd.api.types.is_numeric_dtype(panel[c])]
    sample = panel[remaining].sample(min(len(panel), 200_000), random_state=0)
    corr = sample.corr().abs()
    for i, a in enumerate(remaining):
        if a in dropped:
            continue
        for b in remaining[i + 1:]:
            if b not in dropped and corr.loc[a, b] > REDUNDANT_CORR:
                dropped[b] = f'near-duplicate of {a} (|corr|={corr.loc[a, b]:.3f})'

    for c, why in dropped.items():
        print(f'[CLEAN] Drop {c:28s} {why}')
    panel = panel.drop(columns=list(dropped))
    kept = [c for c in panel.columns if c not in ID_AND_TARGET]
    print(f'[CLEAN] Features: {len(features)} -> {len(kept)}')
    report['columns_dropped'] = dropped
    report['features_kept'] = kept
    return panel, report


def build_final_matrix(rebuild=False, clean=True, company=True):
    if not os.path.exists(PANEL):
        raise FileNotFoundError(
            f'Intermediate global panel not found: {PANEL}. '
            'Run the global dataset build first.'
        )

    os.makedirs(FINAL_DIR, exist_ok=True)
    if os.path.exists(FINAL_MATRIX) and not rebuild:
        print(f'[FINAL] Existing matrix found: {FINAL_MATRIX}')
        print('[FINAL] Use --rebuild-final-matrix to rebuild it.')
        return

    print('=== BUILDING FINAL FEATURE MATRIX ===')
    print(f'Base panel: {PANEL}')
    print('Reading intermediate global panel...')
    panel = pd.read_csv(PANEL, compression='gzip')
    if 'date' not in panel.columns or 'ticker' not in panel.columns:
        raise ValueError('Panel must contain date and ticker columns.')
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel = panel.dropna(subset=['date', 'ticker'])
    panel['ticker'] = panel['ticker'].astype(str).str.strip().str.upper()

    print(f'[FINAL] Base rows: {len(panel):,}')
    print(f'[FINAL] Base columns: {len(panel.columns)}')

    news = _read_daily_news()
    if news.empty:
        print('[FINAL NEWS] No daily news files found. News features will be zero with availability=0.')
    else:
        print(f'[FINAL NEWS] Daily news rows: {len(news):,}')
        print(f'[FINAL NEWS] Tickers with news: {news["ticker"].nunique()}')

    panel = _shift_news_to_next_stock_day(panel, news)

    # Earnings reports (all markets) and SEC fundamentals (US), point-in-time.
    if company:
        panel = ff.add_company_features(panel)
    else:
        print('[COMPANY] Skipped (--no-company).')

    # Final numerical hygiene. Do not silently alter target values.
    panel = panel.replace([np.inf, -np.inf], np.nan)

    # Clean BEFORE dropping rows with missing values, so a removed column can't
    # cause rows to be thrown away.
    cleaning_report = {'skipped': True}
    if clean:
        panel, cleaning_report = clean_matrix(panel)
    else:
        print('[CLEAN] Skipped (--no-clean).')

    protected =[c for c in ['date', 'ticker', 'country', 'region', 'benchmark', 'future_return', 'target'] if c in panel.columns]
    numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {'target'}]

    # Drop rows only when an actual model feature or target is missing.
    # Company-results columns are empty for many stocks by nature (no data for that
    # market); they must not cause rows to be dropped.
    required_numeric = [c for c in feature_cols if c != 'future_return' and c not in ff.ALL_COLUMNS] + (['future_return'] if 'future_return' in panel.columns else [])
    before = len(panel)
    if required_numeric:
        panel = panel.dropna(subset=required_numeric)
    dropped = before - len(panel)

    panel = panel.sort_values(['date', 'ticker']).reset_index(drop=True)

    # Exact key validation.
    duplicate_keys = int(panel.duplicated(['date', 'ticker']).sum())
    if duplicate_keys:
        raise ValueError(f'Final matrix contains {duplicate_keys:,} duplicate date/ticker rows.')

    # Metadata and quality summary.
    news_cols_present = [c for c in RULE_NEWS_COLS + GEMINI_NEWS_COLS + ['news_available', 'gemini_available'] if c in panel.columns]
    quality = {
        'created_at_utc': datetime.utcnow().isoformat() + 'Z',
        'rows': int(len(panel)),
        'columns': int(len(panel.columns)),
        'stocks': int(panel['ticker'].nunique()),
        'date_start': panel['date'].min().date().isoformat() if len(panel) else None,
        'date_end': panel['date'].max().date().isoformat() if len(panel) else None,
        'duplicate_date_ticker_rows': duplicate_keys,
        'dropped_rows_for_missing_features': int(dropped),
        'infinite_values': int(np.isinf(panel.select_dtypes(include=[np.number]).to_numpy()).sum()) if len(panel) else 0,
        'missing_cells': int(panel.isna().sum().sum()),
        'news_columns': news_cols_present,
        'news_rows_with_information': int((panel.get('news_available', pd.Series(0, index=panel.index)) > 0).sum()),
        'gemini_rows_with_information': int((panel.get('gemini_available', pd.Series(0, index=panel.index)) > 0).sum()),
        'target_up': int((panel['target'] == 1).sum()) if 'target' in panel.columns else None,
        'target_down': int((panel['target'] == 0).sum()) if 'target' in panel.columns else None,
    }

    # Write compressed final matrix.
    panel.to_csv(FINAL_MATRIX, index=False, compression='gzip')
    with open(FINAL_METADATA, 'w', encoding='utf-8') as f:
        json.dump({
            'matrix': FINAL_MATRIX,
            'quality': quality,
            'cleaning': cleaning_report,
            'note': 'News features are shifted to the next available stock trading row to avoid same-day cross-market leakage. No train/test split is performed here.'
        }, f, indent=2)

    print('\n=== FINAL FEATURE MATRIX COMPLETE ===')
    print(f'Rows: {quality["rows"]:,}')
    print(f'Columns: {quality["columns"]:,}')
    print(f'Stocks: {quality["stocks"]:,}')
    print(f'Date range: {quality["date_start"]} -> {quality["date_end"]}')
    print(f'Dropped for missing features: {quality["dropped_rows_for_missing_features"]:,}')
    print(f'Duplicate date/ticker rows: {quality["duplicate_date_ticker_rows"]:,}')
    print(f'Missing cells: {quality["missing_cells"]:,}')
    print(f'News-informed rows: {quality["news_rows_with_information"]:,}')
    print(f'Gemini-informed rows: {quality["gemini_rows_with_information"]:,}')
    print(f'Target UP: {quality["target_up"]:,}')
    print(f'Target DOWN: {quality["target_down"]:,}')
    print(f'Matrix: {FINAL_MATRIX}')
    print(f'Metadata: {FINAL_METADATA}')
    print('\nNo train/test split was performed.')


def validate_final_matrix():
    if not os.path.exists(FINAL_MATRIX):
        raise FileNotFoundError(f'Final matrix not found: {FINAL_MATRIX}')
    df = pd.read_csv(FINAL_MATRIX, compression='gzip')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    nums = df.select_dtypes(include=[np.number])
    print('\n=== FINAL MATRIX VALIDATION ===')
    print(f'Rows: {len(df):,}')
    print(f'Columns: {len(df.columns):,}')
    print(f'Stocks: {df["ticker"].nunique():,}')
    print(f'Dates: {df["date"].min().date()} -> {df["date"].max().date()}')
    print(f'Duplicate date/ticker rows: {df.duplicated(["date", "ticker"]).sum():,}')
    print(f'Missing cells: {df.isna().sum().sum():,}')
    print(f'Infinite values: {np.isinf(nums.to_numpy()).sum():,}')
    if 'target' in df:
        print(f'Target missing: {df["target"].isna().sum():,}')
        print(f'Target UP: {(df["target"] == 1).sum():,}')
        print(f'Target DOWN: {(df["target"] == 0).sum():,}')
    if 'news_available' in df:
        print(f'Rows with news information: {(df["news_available"] > 0).sum():,}')
    if 'gemini_available' in df:
        print(f'Rows with Gemini information: {(df["gemini_available"] > 0).sum():,}')
    print('No train/test split was performed.')


def main():
    parser = argparse.ArgumentParser(description='Build and validate the final stock ML feature matrix.')
    parser.add_argument('--build-final-matrix', action='store_true')
    parser.add_argument('--rebuild-final-matrix', action='store_true')
    parser.add_argument('--validate-final-matrix', action='store_true')
    parser.add_argument('--no-clean', action='store_true', help='Skip the cleaning step (illiquid stocks, redundant/useless columns).')
    parser.add_argument('--no-company', action='store_true', help='Do not add the earnings/fundamentals features.')
    args = parser.parse_args()
    if args.build_final_matrix or args.rebuild_final_matrix:
        build_final_matrix(rebuild=args.rebuild_final_matrix, clean=not args.no_clean, company=not args.no_company)
    elif args.validate_final_matrix:
        validate_final_matrix()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
