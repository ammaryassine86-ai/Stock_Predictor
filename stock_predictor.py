import argparse
from io import StringIO
import pycountry
from babel.numbers import get_territory_currencies
import csv
import json
import os
import re
from difflib import SequenceMatcher
import sys
import time as sleep_time
from collections import Counter
from datetime import date, datetime, time, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
from typing import Any
import pandas as pd
import yfinance as yf
import requests_cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statistics
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
YF_CACHE_DIR = "data/yfinance_cache"
os.makedirs(YF_CACHE_DIR, exist_ok=True)

YF_CACHE_DB = os.path.join(YF_CACHE_DIR, "yfinance_requests")
YF_CACHE_EXPIRE_SECONDS = 24 * 60 * 60

yf_session = requests_cache.CachedSession(
    YF_CACHE_DB,
    expire_after=YF_CACHE_EXPIRE_SECONDS,
)
GLOBAL_UNIVERSE_EXTRA = ["2330.TW", "9984.T", "CL=F", "USDTWD=X"]
GLOBAL_UNIVERSE_CACHE = os.path.join(YF_CACHE_DIR, "global_universe.csv")
GLOBAL_DATA_DIR = os.path.join(YF_CACHE_DIR, "historical")

def cached_ticker(symbol: str):
    return yf.Ticker(symbol)
def load_sp500_universe():
    """Load the S&P 500 symbols from Wikipedia and add the requested global instruments."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/153.0 Safari/537.36"
            },
        )
        with urlopen(request, timeout=20) as response:
            html = response.read().decode("utf-8")
        tables = pd.read_html(StringIO(html))
        symbols = tables[0]["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    except Exception as error:
        if os.path.exists(GLOBAL_UNIVERSE_CACHE):
            return pd.read_csv(GLOBAL_UNIVERSE_CACHE)["symbol"].dropna().astype(str).tolist()
        raise ValueError(f"Could not load S&P 500 universe: {error}")
    universe = list(dict.fromkeys(symbols + GLOBAL_UNIVERSE_EXTRA))
    os.makedirs(YF_CACHE_DIR, exist_ok=True)
    pd.DataFrame({"symbol": universe}).to_csv(GLOBAL_UNIVERSE_CACHE, index=False)
    return universe

class GeminiNewsClassification(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    relevance: float = Field(description="How relevant the headline is to the specified ticker/company, from 0 to 1")
    event_type: str = Field(description="One of earnings, lawsuit, product_launch, regulation, management, acquisition, macroeconomic, or other")
    confidence: float = Field(description="Confidence in the classification, from 0 to 1")

class GeminiNewsBatch(BaseModel):
    classifications: list[GeminiNewsClassification]

def fetch_wikidata_context(ticker: str):
    """Fetch optional Wikidata company context for descriptive live output."""
    try:
        company_name = cached_ticker(ticker).info.get('longName')
        if not company_name:
            return {}
        search_url = 'https://www.wikidata.org/w/api.php?' + urlencode({'action': 'wbsearchentities', 'search': company_name, 'language': 'en', 'format': 'json', 'limit': 5})
        search_request = Request(search_url, headers={'User-Agent': 'stock-predictor/1.0'})
        with urlopen(search_request, timeout=8) as response:
            search_results = json.load(response).get('search', [])
        result = next((item for item in search_results if any((word in item.get('description', '').lower() for word in ('company', 'corporation', 'business')))), search_results[0] if search_results else None)
        if not result:
            return {}
        entity_id = result['id']
        entity_url = f'https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json'
        entity_request = Request(entity_url, headers={'User-Agent': 'stock-predictor/1.0'})
        with urlopen(entity_request, timeout=8) as response:
            entity = json.load(response)['entities'][entity_id]
        claim_ids = []
        for property_id in ('P31', 'P452', 'P1056'):
            for claim in entity.get('claims', {}).get(property_id, []):
                value = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {})
                if isinstance(value, dict) and 'id' in value:
                    claim_ids.append(value['id'])
        labels = {}
        if claim_ids:
            labels_url = 'https://www.wikidata.org/w/api.php?' + urlencode({'action': 'wbgetentities', 'ids': '|'.join(claim_ids[:20]), 'props': 'labels', 'languages': 'en', 'format': 'json'})
            labels_request = Request(labels_url, headers={'User-Agent': 'stock-predictor/1.0'})
            with urlopen(labels_request, timeout=8) as response:
                entities = json.load(response).get('entities', {})
            labels = {item['labels']['en']['value'] for item in entities.values() if 'en' in item.get('labels', {})}
        return {'company': company_name, 'wikidata_id': entity_id, 'description': result.get('description', ''), 'topics': sorted(labels)}
    except Exception:
        return {}
FRED_SERIES_URL = 'https://api.stlouisfed.org/fred/series/observations'

# Multi-indicator macro block.
FRED_SERIES = {
    'fed_funds_rate': 'FEDFUNDS',
    'cpi': 'CPIAUCSL',
    'unemployment': 'UNRATE',
    'yield_10y': 'DGS10',
    'yield_2y': 'DGS2',
    'vix': 'VIXCLS',
    'dollar_index': 'DTWEXBGS',
}

MACRO_FEATURE_NAMES = list(FRED_SERIES.keys()) + [
    'fed_funds_rate_change',
    'cpi_change',
    'unemployment_change',
    'yield_10y_change',
    'yield_2y_change',
    'vix_change',
    'dollar_index_change',
    'yield_curve_spread',
    'yield_curve_change',
]

FRED_SERIES_ID = FRED_SERIES['fed_funds_rate']

def fetch_fred_series(series_id: str, api_key: str, start_date: date, end_date: date | None = None, cache_dir: str='data/macro'):
    """Fetch a FRED series and cache the returned observations locally."""
    if not api_key:
        return {}
    end_date = end_date or date.today()
    query = urlencode({'series_id': series_id, 'api_key': api_key, 'file_type': 'json', 'observation_start': start_date.isoformat(), 'observation_end': end_date.isoformat()})
    request = Request(f'{FRED_SERIES_URL}?{query}', headers={'User-Agent': 'stock-predictor/1.0'})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as error:
        print(f'Could not reach FRED API ({error}); macro feature skipped.')
        return {}
    if 'error_message' in payload:
        print(f"FRED API error: {payload['error_message']}; macro feature skipped.")
        return {}
    values = {}
    for obs in payload.get('observations', []):
        raw_value = obs.get('value')
        if raw_value in (None, '.'):
            continue
        values[obs['date']] = float(raw_value)
    if values:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{series_id}.csv')
        with open(cache_path, 'w', newline='', encoding='utf-8') as cache_file:
            writer = csv.writer(cache_file)
            writer.writerow(['date', 'value'])
            for obs_date in sorted(values):
                writer.writerow([obs_date, values[obs_date]])
    return values

def load_cached_fred_series(series_id: str, cache_dir: str='data/macro'):
    """Load a previously cached FRED series without making a network request."""
    cache_path = os.path.join(cache_dir, f'{series_id}.csv')
    if not os.path.exists(cache_path):
        return {}
    values = {}
    with open(cache_path, newline='', encoding='utf-8') as cache_file:
        for row in csv.DictReader(cache_file):
            try:
                values[row['date']] = float(row['value'])
            except (KeyError, ValueError):
                continue
    return values

def get_fred_series(series_id: str, api_key: str, start_date: date, cache_dir: str='data/macro'):
    """Use live FRED data when available and otherwise fall back to the local cache."""
    if api_key:
        values = fetch_fred_series(series_id, api_key, start_date, cache_dir=cache_dir)
        if values:
            return values
    return load_cached_fred_series(series_id, cache_dir)

def fetch_macro_signal(fred_api_key: str):
    """Summarize the current multi-indicator macro environment."""
    latest_values = {}
    series_values = {}

    for feature_name, series_id in FRED_SERIES.items():
        values = get_fred_series(
            series_id,
            fred_api_key or '',
            date.today() - timedelta(days=120)
        )
        if values:
            series_values[feature_name] = values
            latest_values[feature_name] = sorted(values.items())[-1][1]

    if not latest_values:
        return (0, 'Macro data unavailable.')

    score = 0
    explanations = []

    fed_values = series_values.get('fed_funds_rate', {})
    if len(fed_values) >= 2:
        ordered = sorted(fed_values.items())
        previous, latest = ordered[-2][1], ordered[-1][1]
        if latest > previous:
            score -= 1
            explanations.append(f'Fed funds rose ({previous:.2f}% -> {latest:.2f}%)')
        elif latest < previous:
            score += 1
            explanations.append(f'Fed funds fell ({previous:.2f}% -> {latest:.2f}%)')

    vix = latest_values.get('vix')
    if vix is not None:
        if vix >= 30:
            score -= 1
            explanations.append(f'VIX elevated ({vix:.1f})')
        elif vix <= 15:
            score += 1
            explanations.append(f'VIX subdued ({vix:.1f})')

    y10 = latest_values.get('yield_10y')
    y2 = latest_values.get('yield_2y')
    if y10 is not None and y2 is not None:
        spread = y10 - y2
        if spread > 0.25:
            score += 1
            explanations.append(f'10Y-2Y spread positive ({spread:+.2f} pp)')
        elif spread < -0.25:
            score -= 1
            explanations.append(f'10Y-2Y spread inverted ({spread:+.2f} pp)')

    if score >= 2:
        label = 'Supportive'
    elif score <= -2:
        label = 'Restrictive'
    else:
        label = 'Mixed'

    details = '; '.join(explanations) if explanations else 'No strong macro changes detected.'
    return (score, f'{label}: {details}.')


def attach_macro_feature(data, fred_values: dict, column_name: str):
    """Attach one FRED series using a backward as-of merge."""
    if not fred_values:
        data[column_name] = float('nan')
        return data

    fred_dates = pd.to_datetime(list(fred_values.keys())).as_unit('ns')
    fred_frame = pd.DataFrame(
        {'fred_date': fred_dates, 'value': list(fred_values.values())}
    ).sort_values('fred_date')

    trading_dates = pd.to_datetime(data.index)
    if trading_dates.tz is not None:
        trading_dates = trading_dates.tz_localize(None)
    trading_dates = trading_dates.as_unit('ns')

    trading_frame = pd.DataFrame({'trade_date': trading_dates}).sort_values('trade_date')
    merged = pd.merge_asof(
        trading_frame,
        fred_frame,
        left_on='trade_date',
        right_on='fred_date',
        direction='backward'
    )
    data[column_name] = merged['value'].values
    return data


# FRED dates are REFERENCE periods (August CPI is dated 2026-08-01) but the
# numbers are only published later. Shift each series to when it was actually
# known, so a row never uses a figure that had not been released yet.
FRED_PUBLICATION_LAG_DAYS = {
    'fed_funds_rate': 32,   # monthly average, out early the next month
    'cpi': 45,              # released mid next month
    'unemployment': 38,     # first Friday of the next month
    'yield_10y': 1,         # daily, available the next day
    'yield_2y': 1,
    'vix': 1,
    'dollar_index': 7,      # H.10 release is weekly
}


def _shift_to_release_date(values: dict, lag_days: int) -> dict:
    shifted = {}
    for obs_date, value in values.items():
        release = pd.Timestamp(obs_date) + pd.Timedelta(days=lag_days)
        shifted[release.date().isoformat()] = value
    return shifted


def _add_macro_columns(data, fred_by_feature: dict):
    """Attach all macro series and derive changes and yield-curve features."""
    for feature_name, values in fred_by_feature.items():
        lag = FRED_PUBLICATION_LAG_DAYS.get(feature_name, 1)
        values = _shift_to_release_date(values, lag)
        data = attach_macro_feature(data, values, feature_name)
        data[feature_name] = data[feature_name].ffill()

    for feature_name in FRED_SERIES:
        data[f'{feature_name}_change'] = data[feature_name].diff()

    data['yield_curve_spread'] = data['yield_10y'] - data['yield_2y']
    data['yield_curve_change'] = data['yield_curve_spread'].diff()

    return data


def add_macro_features(data, start_date: date, feature_names: list):
    """Add live/cached multi-indicator macro features."""
    fred_by_feature = {}
    api_key = os.getenv('FRED_API_KEY') or ''

    for feature_name, series_id in FRED_SERIES.items():
        values = get_fred_series(series_id, api_key, start_date)
        if values:
            fred_by_feature[feature_name] = values

    if len(fred_by_feature) != len(FRED_SERIES):
        return (data, feature_names)

    data = _add_macro_columns(data, fred_by_feature)
    return (
        data,
        feature_names + [
            name for name in MACRO_FEATURE_NAMES
            if name not in feature_names
        ]
    )


def ensure_macro_data(series_id: str = FRED_SERIES_ID, years: int = 10, cache_dir: str = 'data/macro', macro_start: str | None = None, refresh: bool = False):
    """Ensure every configured FRED macro series is cached from the requested start date.

    Existing caches are preserved unless they do not reach the requested start
    date or ``refresh`` is explicitly requested. This prevents a short recent
    cache (for example 2021-2026) from silently truncating the historical
    stock panel when the project needs 2015-2026 data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    api_key = os.getenv('FRED_API_KEY')
    requested_start = datetime.strptime(macro_start, '%Y-%m-%d').date() if macro_start else date.today() - timedelta(days=years * 365)

    print(f"Macro cache directory: {cache_dir}")
    print(f"Requested macro start: {requested_start.isoformat()}")
    print(f"Checking {len(FRED_SERIES)} macro series...")

    statuses = {}

    for feature_name, configured_series_id in FRED_SERIES.items():
        cache_path = os.path.join(cache_dir, f'{configured_series_id}.csv')
        needs_download = refresh or not os.path.exists(cache_path)

        if os.path.exists(cache_path) and not refresh:
            try:
                cached = pd.read_csv(cache_path, usecols=['date'])
                cached_dates = pd.to_datetime(cached['date'], errors='coerce').dropna()
                cached_start = cached_dates.min().date() if not cached_dates.empty else None
                if cached_start is not None and cached_start <= requested_start:
                    statuses[configured_series_id] = 'cached'
                    print(f"  [CACHED]     {configured_series_id:<10} ({feature_name}) | start={cached_start}")
                    continue
                print(f"  [EXTENDING]  {configured_series_id:<10} ({feature_name}) | cached_start={cached_start}")
                needs_download = True
            except Exception as error:
                print(f"  [REPAIR]     {configured_series_id:<10} ({feature_name}) | {error}")
                needs_download = True

        if not needs_download:
            continue

        if not api_key:
            statuses[configured_series_id] = 'missing_api_key'
            print(f"  [NO KEY]     {configured_series_id:<10} ({feature_name})")
            continue

        values = fetch_fred_series(
            configured_series_id,
            api_key,
            requested_start,
            cache_dir=cache_dir
        )

        if values:
            statuses[configured_series_id] = 'downloaded'
            print(f"  [DOWNLOADED]  {configured_series_id:<10} ({len(values)} observations)")
        else:
            statuses[configured_series_id] = 'failed'
            print(f"  [FAILED]      {configured_series_id:<10} ({feature_name})")

    missing = [
        series_id for series_id in FRED_SERIES.values()
        if not os.path.exists(os.path.join(cache_dir, f'{series_id}.csv'))
    ]

    if missing:
        print()
        print("Macro cache is INCOMPLETE.")
        print("Missing series:", ", ".join(missing))
        return 'incomplete'

    print()
    print("Macro cache is COMPLETE.")
    print(f"All configured macro series are available from the requested start where FRED provides observations.")
    return 'complete'

def add_macro_features_frozen(data, feature_names: list, cache_dir: str = 'data/macro'):
    """Add only frozen local macro caches for reproducible historical experiments."""
    fred_by_feature = {}

    for feature_name, series_id in FRED_SERIES.items():
        values = load_cached_fred_series(series_id, cache_dir)
        if not values:
            raise ValueError(
                f'No cached macro data found for {series_id} ({feature_name}). '
                'Run with --download-macro first (requires FRED_API_KEY the first time; cached afterward).'
            )
        fred_by_feature[feature_name] = values

    data = _add_macro_columns(data, fred_by_feature)
    return (
        data,
        feature_names + [
            name for name in MACRO_FEATURE_NAMES
            if name not in feature_names
        ]
    )


def fetch_price_data(ticker: str, days: int=60):
    """Fetch recent Yahoo Finance price history for live analysis."""
    stock = cached_ticker(ticker)
    hist = stock.history(period=f'{days + 10}d')
    if hist.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. Check the symbol.")
    hist = hist.dropna(subset=['Close'])
    if hist.empty:
        raise ValueError(f"No valid price data found for ticker '{ticker}'. Check the symbol.")
    return hist.tail(days)

def download_historical_news(ticker: str, start_date: date, end_date: date, output_dir: str='data/news'):
    """Download historical Alpha Vantage news for a ticker."""
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        raise ValueError('Set the ALPHAVANTAGE_API_KEY environment variable before downloading news.')
    if start_date > end_date:
        raise ValueError('News start date must be on or before the end date.')
    query = urlencode({'function': 'NEWS_SENTIMENT', 'tickers': ticker.upper(), 'time_from': start_date.strftime('%Y%m%dT0000'), 'time_to': end_date.strftime('%Y%m%dT2359'), 'sort': 'EARLIEST', 'limit': 1000, 'apikey': api_key})
    request = Request(f'https://www.alphavantage.co/query?{query}', headers={'User-Agent': 'stock-predictor/1.0'})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if 'Error Message' in payload:
        raise ValueError(payload['Error Message'])
    if 'Note' in payload:
        raise ValueError(f"Alpha Vantage rate limit: {payload['Note']}")
    rows = {}
    for item in payload.get('feed', []):
        published_at = item.get('time_published')
        headline = (item.get('title') or '').strip()
        source = (item.get('source') or '').strip()
        if not published_at or not headline:
            continue
        key = (published_at, headline, source)
        rows[key] = {'ticker': ticker.upper(), 'published_at': published_at, 'headline': headline, 'source': source}
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{ticker.upper()}.csv')
    existing_rows = {}
    if os.path.exists(output_path):
        with open(output_path, newline='', encoding='utf-8') as input_file:
            for row in csv.DictReader(input_file):
                key = (row.get('published_at', ''), row.get('headline', '').casefold(), row.get('source', '').casefold())
                existing_rows[key] = row
    existing_rows.update(rows)
    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=['ticker', 'published_at', 'headline', 'source'])
        writer.writeheader()
        writer.writerows(sorted(existing_rows.values(), key=lambda row: row['published_at']))
    print(f'Added {len(rows)} headlines; file now contains {len(existing_rows)} rows: {output_path}')
    print('Alpha Vantage sentiment fields were intentionally ignored.')

def download_historical_news_yearly(ticker: str, start_date: date, end_date: date, output_dir: str='data/news'):
    """Download historical Alpha Vantage news one calendar year at a time and merge it."""
    if start_date > end_date:
        raise ValueError('News start date must be on or before the end date.')
    current_year = start_date.year
    request_count = 0
    while current_year <= end_date.year:
        chunk_start = max(start_date, date(current_year, 1, 1))
        chunk_end = min(end_date, date(current_year, 12, 31))
        print(f'Request {request_count + 1}: {chunk_start} to {chunk_end}')
        download_historical_news(ticker, chunk_start, chunk_end, output_dir)
        request_count += 1
        current_year += 1
    print(f'Completed {request_count} yearly requests for {ticker.upper()}.')

def classify_headline_sentiment(headline: str) -> int:
    """Classify a headline with the deterministic keyword rule used by the historical-news control."""
    words = set(re.findall('[a-z]+', headline.lower()))
    positive_hits = len(words & POSITIVE_WORDS)
    negative_hits = len(words & NEGATIVE_WORDS)
    if positive_hits > negative_hits:
        return 1
    if negative_hits > positive_hits:
        return -1
    return 0

def process_historical_news(ticker: str, input_dir: str='data/news', cutoff_hour: int=16):
    """Normalize cached historical headlines and create daily rule-based news features."""
    input_path = os.path.join(input_dir, f'{ticker.upper()}.csv')
    processed_path = os.path.join(input_dir, f'{ticker.upper()}_processed.csv')
    daily_path = os.path.join(input_dir, f'{ticker.upper()}_daily.csv')
    if not os.path.exists(input_path):
        raise ValueError(f'Raw news file not found: {input_path}')
    if not 0 <= cutoff_hour <= 23:
        raise ValueError('Cutoff hour must be between 0 and 23.')
    timezone = ZoneInfo('America/New_York')
    rows = {}
    with open(input_path, newline='', encoding='utf-8') as input_file:
        for row in csv.DictReader(input_file):
            published_text = (row.get('published_at') or '').strip()
            headline = (row.get('headline') or '').strip()
            source = (row.get('source') or '').strip()
            if not published_text or not headline:
                continue
            try:
                published_at = datetime.strptime(published_text, '%Y%m%dT%H%M%S').replace(tzinfo=ZoneInfo('UTC'))
            except ValueError:
                continue
            local_time = published_at.astimezone(timezone)
            local_date = local_time.date()
            cutoff = datetime.combine(local_date, time(cutoff_hour), tzinfo=timezone)
            if local_time > cutoff:
                continue
            key = (published_at.isoformat(), headline.casefold(), source.casefold())
            rows[key] = {'ticker': ticker.upper(), 'published_at': published_at.isoformat(), 'headline': headline, 'source': source, 'sentiment': classify_headline_sentiment(headline), 'date': local_date.isoformat()}
    with open(processed_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=['ticker', 'published_at', 'headline', 'source', 'sentiment', 'date'])
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: row['published_at']))
    daily = {}
    for row in rows.values():
        day = daily.setdefault(row['date'], Counter())
        sentiment = int(row['sentiment'])
        day['news_score'] += sentiment
        day['news_count'] += 1
        day['positive_count'] += sentiment == 1
        day['negative_count'] += sentiment == -1
        day['neutral_count'] += sentiment == 0
    with open(daily_path, 'w', newline='', encoding='utf-8') as output_file:
        fieldnames = ['date', 'news_score', 'news_count', 'positive_count', 'negative_count', 'neutral_count']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for day, values in sorted(daily.items()):
            writer.writerow({'date': day, **values})
    print(f'Processed {len(rows)} headlines to {processed_path}')
    print(f'Aggregated {len(daily)} days to {daily_path}')
    print(f'Cutoff: {cutoff_hour:02d}:00 America/New_York')

def ensure_news_data(ticker: str, news_dir: str='data/news', years: int=5, cutoff_hour: int=16):
    """Ensure the required historical news cache exists without silently changing research settings."""
    ticker = ticker.upper()
    raw_path = os.path.join(news_dir, f'{ticker}.csv')
    daily_path = os.path.join(news_dir, f'{ticker}_daily.csv')
    if os.path.exists(daily_path):
        return 'cached'
    if os.path.exists(raw_path):
        process_historical_news(ticker, news_dir, cutoff_hour)
        return 'processed'
    if not os.getenv('ALPHAVANTAGE_API_KEY'):
        return 'unavailable'
    start_date = date.today() - timedelta(days=years * 365)
    download_historical_news_yearly(ticker, start_date, date.today(), news_dir)
    process_historical_news(ticker, news_dir, cutoff_hour)
    return 'downloaded'

def classify_news_with_gemini(ticker: str, news_dir: str='data/news', model_name: str='gemini-3.5-flash-lite', limit: int=100, batch_size: int=10, request_delay: float=4.2, request_timeout_ms: int=60000):
    """Classify uncached historical headlines with Gemini in batches, persist progress, and aggregate daily features."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError('Set the GEMINI_API_KEY environment variable before classification.')
    processed_path = os.path.join(news_dir, f'{ticker.upper()}_processed.csv')
    output_path = os.path.join(news_dir, f'{ticker.upper()}_gemini.csv')
    if not os.path.exists(processed_path):
        raise ValueError(f'Processed news file not found: {processed_path}')
    if batch_size < 1:
        raise ValueError('gemini batch size must be at least 1.')
    cached = {}
    if os.path.exists(output_path):
        with open(output_path, newline='', encoding='utf-8') as input_file:
            for row in csv.DictReader(input_file):
                key = (row.get('published_at', ''), row.get('headline', '').casefold(), row.get('source', '').casefold())
                cached[key] = row
    headlines = []
    with open(processed_path, newline='', encoding='utf-8') as input_file:
        for row in csv.DictReader(input_file):
            key = (row.get('published_at', ''), row.get('headline', '').casefold(), row.get('source', '').casefold())
            if key not in cached:
                headlines.append((key, row))
    pending = headlines if limit == 0 else headlines[:limit]
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=request_timeout_ms))
    stopped_early = False
    last_error = None
    classified_this_run = 0
    total_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_number, batch_start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[batch_start:batch_start + batch_size]
        prompt = (
            f"Classify each financial headline for ticker {ticker.upper()}. "
            "Return exactly one classification for each numbered headline, in the same order. "
            "Do not predict the stock price.\n\n" +
            "\n".join(f"{i}. Headline: {row['headline']}" for i, (_, row) in enumerate(batch, start=1))
        )
        response = None
        result = None
        batch_error = None
        for attempt in range(1, 5):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json',
                        response_schema=GeminiNewsBatch,
                    ),
                )
                result = GeminiNewsBatch.model_validate_json(response.text or '')
                if len(result.classifications) != len(batch):
                    raise ValueError(f'Gemini returned {len(result.classifications)} classifications for a batch of {len(batch)} headlines.')
                batch_error = None
                break
            except Exception as error:
                batch_error = error
                error_text = str(error)
                retry_match = re.search(r'retry in ([0-9]+(?:\.[0-9]+)?)s', error_text, re.IGNORECASE)
                is_rate_limit = '429' in error_text or 'RESOURCE_EXHAUSTED' in error_text
                is_connection_reset = '10054' in error_text or 'forcibly closed by the remote host' in error_text.lower()
                is_timeout = isinstance(error, TimeoutError) or 'timeout' in error_text.lower() or 'timed out' in error_text.lower()
                is_transient = is_rate_limit or is_connection_reset or is_timeout or 'temporarily unavailable' in error_text.lower()
                if not is_transient or attempt == 4:
                    break
                if is_rate_limit:
                    wait_seconds = max(request_delay, float(retry_match.group(1)) + 1 if retry_match else 40.0)
                elif is_connection_reset:
                    wait_seconds = 10.0 * attempt
                else:
                    wait_seconds = 8.0 * attempt
                print(f'Gemini transient error on batch {batch_number}/{total_batches}. Retry {attempt}/3 in {wait_seconds:.0f}s...')
                sleep_time.sleep(wait_seconds)
                if is_connection_reset or is_timeout:
                    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=request_timeout_ms))

        if batch_error is not None or result is None:
            error = batch_error or RuntimeError('Gemini returned no result after retries.')
            print(f'Gemini stopped after {classified_this_run} new headlines: {error}')
            stopped_early = True
            last_error = error
            break

        for (key, row), classification in zip(batch, result.classifications):
            sentiment = classification.sentiment.lower()
            if sentiment not in {'positive', 'negative', 'neutral'}:
                sentiment = 'neutral'
            cached[key] = {
                'ticker': ticker.upper(),
                'published_at': row.get('published_at', ''),
                'headline': row.get('headline', ''),
                'source': row.get('source', ''),
                'date': row.get('date', ''),
                'sentiment': sentiment,
                'relevance': max(0.0, min(1.0, classification.relevance)),
                'event_type': classification.event_type,
                'confidence': max(0.0, min(1.0, classification.confidence)),
            }

        classified_this_run += len(batch)
        print(f'Gemini classified {classified_this_run}/{len(pending)} headlines ({batch_number}/{total_batches} requests)')

        os.makedirs(news_dir, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
            fieldnames = ['ticker', 'published_at', 'headline', 'source', 'date', 'sentiment', 'relevance', 'event_type', 'confidence']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(cached.values(), key=lambda row: row['published_at']))

        if batch_number < total_batches and request_delay > 0:
            sleep_time.sleep(request_delay)

        os.makedirs(news_dir, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
            fieldnames = ['ticker', 'published_at', 'headline', 'source', 'date', 'sentiment', 'relevance', 'event_type', 'confidence']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(cached.values(), key=lambda row: row['published_at']))

    daily_path = os.path.join(news_dir, f'{ticker.upper()}_gemini_daily.csv')
    sentiment_values = {'positive': 1, 'negative': -1, 'neutral': 0}
    daily = {}
    for row in cached.values():
        day = daily.setdefault(row.get('date', ''), {'gemini_sentiment_score': 0.0, 'gemini_relevance_sum': 0.0, 'gemini_confidence_sum': 0.0, 'gemini_news_count': 0})
        day['gemini_sentiment_score'] += sentiment_values.get(row.get('sentiment', 'neutral'), 0) * float(row.get('relevance', 0))
        day['gemini_relevance_sum'] += float(row.get('relevance', 0))
        day['gemini_confidence_sum'] += float(row.get('confidence', 0))
        day['gemini_news_count'] += 1
    with open(daily_path, 'w', newline='', encoding='utf-8') as output_file:
        fieldnames = ['date', 'gemini_sentiment_score', 'gemini_relevance', 'gemini_confidence', 'gemini_news_count']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for day, values in sorted(daily.items()):
            count = values['gemini_news_count']
            writer.writerow({'date': day, 'gemini_sentiment_score': values['gemini_sentiment_score'], 'gemini_relevance': values['gemini_relevance_sum'] / count if count else 0, 'gemini_confidence': values['gemini_confidence_sum'] / count if count else 0, 'gemini_news_count': count})
    print(f'Cached {len(cached)} Gemini classifications to {output_path}')
    print(f'Aggregated Gemini features to {daily_path}')
    if stopped_early:
        remaining = len(pending) - classified_this_run
        print(f'INCOMPLETE: {remaining} headline(s) were not classified this run.')
        raise ValueError(f'Gemini classification run did not finish: {last_error}')

def backtest_technical_rule(ticker: str, years: int=5, horizon: int=5):
    """Evaluate the simple moving-average technical control on chronological historical data."""
    start = date.today() - timedelta(days=years * 365)
    hist = cached_ticker(ticker).history(start=start.isoformat())
    if hist.empty:
        raise ValueError(f"No historical data found for ticker '{ticker}'.")
    closes = hist['Close']
    short_average = closes.rolling(10).mean()
    long_average = closes.rolling(30).mean()
    gap_pct = (short_average - long_average) / long_average * 100
    signal = gap_pct.where(gap_pct.abs() > 1, 0).apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    future_return = closes.shift(-horizon) / closes - 1
    actual = future_return.apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    previous_day = closes.pct_change().apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    results = signal.to_frame('signal').join([actual.rename('actual'), previous_day.rename('previous_day')]).dropna(subset=['signal', 'actual', 'previous_day'])
    results = results[results['signal'] != 0]
    if results.empty:
        raise ValueError('Not enough historical data to evaluate the rule.')
    accuracy = (results['signal'] == results['actual']).mean() * 100
    always_up = (results['actual'] == 1).mean() * 100
    previous_day_accuracy = (results['previous_day'] == results['actual']).mean() * 100
    print(f'\n=== Backtest for {ticker.upper()} ===')
    print(f'Period: {start.isoformat()} to today')
    print(f'Forecast horizon: {horizon} trading days')
    print(f'Evaluated signals: {len(results)}')
    print(f'Moving-average rule accuracy: {accuracy:.2f}%')
    print(f'Always-UP baseline: {always_up:.2f}%')
    print(f'Previous-day direction baseline: {previous_day_accuracy:.2f}%')
    if accuracy > max(always_up, previous_day_accuracy):
        print('Result: the rule beat both baselines in this period.')
    else:
        print('Result: the rule did not beat both baselines in this period.')

def backtest_logistic_model(ticker: str, years: int=5, horizon: int=5):
    """Evaluate the technical Logistic Regression using a chronological development/test split."""
    start = date.today() - timedelta(days=years * 365)
    stock_history = cached_ticker(ticker).history(start=start.isoformat())
    market_history = cached_ticker('^GSPC').history(start=start.isoformat())
    if stock_history.empty or market_history.empty:
        raise ValueError('Could not download enough stock and market history.')
    data = stock_history[['Close', 'Volume']].copy()
    data['daily_return'] = data['Close'].pct_change()
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_10'] = data['Close'].rolling(10).mean()
    data['sma_30'] = data['Close'].rolling(30).mean()
    delta = data['Close'].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float('nan'))
    data['rsi_14'] = 100 - 100 / (1 + relative_strength)
    data['volatility_20'] = data['daily_return'].rolling(20).std()
    data['volume_change'] = data['Volume'].pct_change()
    data['market_return'] = market_history['Close'].pct_change()
    data['future_return'] = data['Close'].shift(-horizon) / data['Close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    feature_names = ['daily_return', 'sma_5', 'sma_10', 'sma_30', 'rsi_14', 'volatility_20', 'volume_change', 'market_return']
    data = data.dropna(subset=feature_names + ['future_return'])
    train_end = int(len(data) * 0.6)
    validation_end = int(len(data) * 0.8)
    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]
    if train.empty or validation.empty or test.empty or (train['target'].nunique() < 2):
        raise ValueError('Not enough varied historical data for a chronological model split.')

    def create_model():
        """Create the standardized Logistic Regression pipeline used by the research backtests."""
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def print_metrics(label, frame, predictions):
        """Print the standard classification metrics and baselines used in model evaluation."""
        always_up = [1] * len(frame)
        previous_day = (frame['daily_return'] > 0).astype(int)
        print(f"{label} accuracy: {accuracy_score(frame['target'], predictions) * 100:.2f}%")
        print(f"{label} balanced accuracy: {balanced_accuracy_score(frame['target'], predictions) * 100:.2f}%")
        print(f"{label} precision: {precision_score(frame['target'], predictions, zero_division=0) * 100:.2f}%")
        print(f"{label} recall: {recall_score(frame['target'], predictions, zero_division=0) * 100:.2f}%")
        print(f"{label} always-UP accuracy: {accuracy_score(frame['target'], always_up) * 100:.2f}%")
        print(f"{label} previous-day accuracy: {accuracy_score(frame['target'], previous_day) * 100:.2f}%")
        print(f"{label} confusion matrix [[TN, FP], [FN, TP]]: {[list(row) for row in confusion_matrix(frame['target'], predictions)]}")
    development_model = create_model()
    development_model.fit(train[feature_names], train['target'])
    validation_predictions = development_model.predict(validation[feature_names])
    final_model = create_model()
    development_data = data.iloc[:validation_end]
    final_model.fit(development_data[feature_names], development_data['target'])
    test_predictions = final_model.predict(test[feature_names])
    print(f'\n=== Logistic Regression backtest for {ticker.upper()} ===')
    print(f'Training rows: {len(train)}')
    print(f'Validation rows: {len(validation)}')
    print(f'Test rows: {len(test)}')
    print(f'Forecast horizon: {horizon} trading days')
    print(f"Features used: {', '.join(feature_names)}")
    print(f"Overall target distribution: {data['target'].value_counts(normalize=True).sort_index().to_dict()}")
    print(f"Validation target distribution: {validation['target'].value_counts(normalize=True).sort_index().to_dict()}")
    print(f"Test target distribution: {test['target'].value_counts(normalize=True).sort_index().to_dict()}")
    print_metrics('Validation model', validation, validation_predictions)
    print_metrics('Final test model', test, test_predictions)

def walk_forward_logistic_model(ticker: str, years: int=5, horizon: int=5, test_window: int=63, verbose: bool=True):
    """Evaluate the frozen technical Logistic Regression with expanding, time-ordered test windows."""
    start = date.today() - timedelta(days=years * 365)
    stock_history = cached_ticker(ticker).history(start=start.isoformat())
    market_history = cached_ticker('^GSPC').history(start=start.isoformat())
    if stock_history.empty or market_history.empty:
        raise ValueError('Could not download enough stock and market history.')
    data = stock_history[['Close', 'Volume']].copy()
    data['daily_return'] = data['Close'].pct_change()
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_10'] = data['Close'].rolling(10).mean()
    data['sma_30'] = data['Close'].rolling(30).mean()
    delta = data['Close'].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float('nan'))
    data['rsi_14'] = 100 - 100 / (1 + relative_strength)
    data['volatility_20'] = data['daily_return'].rolling(20).std()
    data['volume_change'] = data['Volume'].pct_change()
    data['market_return'] = market_history['Close'].pct_change()
    data['future_return'] = data['Close'].shift(-horizon) / data['Close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    feature_names = ['daily_return', 'sma_5', 'sma_10', 'sma_30', 'rsi_14', 'volatility_20', 'volume_change', 'market_return']
    data = data.dropna(subset=feature_names + ['future_return'])
    initial_train = int(len(data) * 0.5)
    if initial_train < 2 or len(data) - initial_train < 1:
        raise ValueError('Not enough historical data for walk-forward testing.')
    actual_values = []
    prediction_values = []
    window_results = []
    test_start = initial_train
    while test_start < len(data):
        test_end = min(test_start + test_window, len(data))
        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]
        if train['target'].nunique() < 2:
            test_start = test_end
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[feature_names], train['target'])
        predictions = model.predict(test[feature_names])
        actual_values.extend(list(test['target']))
        prediction_values.extend(list(predictions))
        window_results.append((test.index[0], test.index[-1], test, predictions))
        test_start = test_end
    if not actual_values:
        raise ValueError('No valid walk-forward windows were produced.')
    if verbose:
        print(f'\n=== Walk-forward Logistic Regression for {ticker.upper()} ===')
        print(f'Initial training rows: {initial_train}')
        print(f'Test window: {test_window} trading days')
        print(f'Forecast horizon: {horizon} trading days')
        print(f"Features used: {', '.join(feature_names)}")
        for window_start, window_end, frame, predictions in window_results:
            score = balanced_accuracy_score(frame['target'], predictions) * 100
            print(f'{window_start.date()} to {window_end.date()}: {score:.2f}% balanced accuracy')
    actual = data.iloc[initial_train:initial_train + len(actual_values)]['target']
    always_up = [1] * len(actual_values)
    previous_day = (data.iloc[initial_train:initial_train + len(actual_values)]['daily_return'] > 0).astype(int)
    summary = {'ticker': ticker.upper(), 'predictions': len(actual_values), 'accuracy': accuracy_score(actual, prediction_values) * 100, 'balanced_accuracy': balanced_accuracy_score(actual, prediction_values) * 100, 'always_up': accuracy_score(actual, always_up) * 100, 'previous_day': accuracy_score(actual, previous_day) * 100}
    if verbose:
        print(f"Out-of-sample rows: {summary['predictions']}")
        print(f"Overall accuracy: {summary['accuracy']:.2f}%")
        print(f"Overall balanced accuracy: {summary['balanced_accuracy']:.2f}%")
        print(f'Overall precision: {precision_score(actual, prediction_values, zero_division=0) * 100:.2f}%')
        print(f'Overall recall: {recall_score(actual, prediction_values, zero_division=0) * 100:.2f}%')
        print(f"Always-UP accuracy: {summary['always_up']:.2f}%")
        print(f"Previous-day accuracy: {summary['previous_day']:.2f}%")
        print(f'Confusion matrix [[TN, FP], [FN, TP]]: {[list(row) for row in confusion_matrix(actual, prediction_values)]}')
    return summary

def walk_forward_macro_model(ticker: str, years: int=5, horizon: int=5, test_window: int=63, verbose: bool=True):
    """Evaluate technical + multi-indicator frozen-macro features with walk-forward validation."""
    start = date.today() - timedelta(days=years * 365)
    stock_history = cached_ticker(ticker).history(start=start.isoformat())
    market_history = cached_ticker('^GSPC').history(start=start.isoformat())
    if stock_history.empty or market_history.empty:
        raise ValueError('Could not download enough stock and market history.')
    data = stock_history[['Close', 'Volume']].copy()
    data['daily_return'] = data['Close'].pct_change()
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_10'] = data['Close'].rolling(10).mean()
    data['sma_30'] = data['Close'].rolling(30).mean()
    delta = data['Close'].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float('nan'))
    data['rsi_14'] = 100 - 100 / (1 + relative_strength)
    data['volatility_20'] = data['daily_return'].rolling(20).std()
    data['volume_change'] = data['Volume'].pct_change()
    data['market_return'] = market_history['Close'].pct_change()
    data['future_return'] = data['Close'].shift(-horizon) / data['Close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    feature_names = ['daily_return', 'sma_5', 'sma_10', 'sma_30', 'rsi_14', 'volatility_20', 'volume_change', 'market_return']
    data, feature_names = add_macro_features_frozen(data, feature_names)
    data = data.dropna(subset=feature_names + ['future_return'])
    initial_train = int(len(data) * 0.5)
    if initial_train < 2 or len(data) - initial_train < 1:
        raise ValueError('Not enough historical data for walk-forward testing.')
    actual_values = []
    prediction_values = []
    window_results = []
    test_start = initial_train
    while test_start < len(data):
        test_end = min(test_start + test_window, len(data))
        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]
        if train['target'].nunique() < 2:
            test_start = test_end
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[feature_names], train['target'])
        predictions = model.predict(test[feature_names])
        actual_values.extend(list(test['target']))
        prediction_values.extend(list(predictions))
        window_results.append((test.index[0], test.index[-1], test, predictions))
        test_start = test_end
    if not actual_values:
        raise ValueError('No valid walk-forward windows were produced.')
    if verbose:
        print(f'\n=== Walk-forward technical + macro for {ticker.upper()} ===')
        print(f'Initial training rows: {initial_train}')
        print(f'Test window: {test_window} trading days')
        print(f'Forecast horizon: {horizon} trading days')
        print(f"Features used: {', '.join(feature_names)}")
        for window_start, window_end, frame, predictions in window_results:
            score = balanced_accuracy_score(frame['target'], predictions) * 100
            print(f'{window_start.date()} to {window_end.date()}: {score:.2f}% balanced accuracy')
    actual = data.iloc[initial_train:initial_train + len(actual_values)]['target']
    always_up = [1] * len(actual_values)
    previous_day = (data.iloc[initial_train:initial_train + len(actual_values)]['daily_return'] > 0).astype(int)
    summary = {'ticker': ticker.upper(), 'predictions': len(actual_values), 'accuracy': accuracy_score(actual, prediction_values) * 100, 'balanced_accuracy': balanced_accuracy_score(actual, prediction_values) * 100, 'always_up': accuracy_score(actual, always_up) * 100, 'previous_day': accuracy_score(actual, previous_day) * 100}
    if verbose:
        print(f"Out-of-sample rows: {summary['predictions']}")
        print(f"Overall accuracy: {summary['accuracy']:.2f}%")
        print(f"Overall balanced accuracy: {summary['balanced_accuracy']:.2f}%")
        print(f"Always-UP accuracy: {summary['always_up']:.2f}%")
        print(f"Previous-day accuracy: {summary['previous_day']:.2f}%")
        print(f'Confusion matrix [[TN, FP], [FN, TP]]: {[list(row) for row in confusion_matrix(actual, prediction_values)]}')
        print("Compare this balanced accuracy directly against --walk-forward's technical-only result to see whether the macro feature actually helps for this ticker.")
    return summary


def walk_forward_news_macro_model(ticker: str, years: int=5, horizon: int=5, test_window: int=63, news_dir: str='data/news', verbose: bool=True):
    """Evaluate technical + rule-based news + frozen macro features with the same walk-forward protocol."""
    news_path = os.path.join(news_dir, f'{ticker.upper()}_daily.csv')
    if not os.path.exists(news_path):
        raise ValueError(f'Daily news file not found: {news_path}')

    news_by_date = {}
    with open(news_path, newline='', encoding='utf-8') as news_file:
        for row in csv.DictReader(news_file):
            news_date = (row.get('date') or '').strip()
            if news_date:
                news_by_date[news_date] = {
                    'news_score': float(row.get('news_score') or 0),
                    'news_count': float(row.get('news_count') or 0),
                }

    start = date.today() - timedelta(days=years * 365)
    stock_history = cached_ticker(ticker).history(start=start.isoformat())
    market_history = cached_ticker('^GSPC').history(start=start.isoformat())
    if stock_history.empty or market_history.empty:
        raise ValueError('Could not download enough stock and market history.')

    data = stock_history[['Close', 'Volume']].copy()
    data['date'] = [index.date().isoformat() for index in data.index]
    data['news_score'] = data['date'].map(
        lambda value: news_by_date.get(value, {}).get('news_score', 0)
    )
    data['news_count'] = data['date'].map(
        lambda value: news_by_date.get(value, {}).get('news_count', 0)
    )
    data['daily_return'] = data['Close'].pct_change()
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_10'] = data['Close'].rolling(10).mean()
    data['sma_30'] = data['Close'].rolling(30).mean()

    delta = data['Close'].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float('nan'))
    data['rsi_14'] = 100 - 100 / (1 + relative_strength)

    data['volatility_20'] = data['daily_return'].rolling(20).std()
    data['volume_change'] = data['Volume'].pct_change()
    data['market_return'] = market_history['Close'].pct_change()
    data['future_return'] = data['Close'].shift(-horizon) / data['Close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)

    feature_names = [
        'daily_return', 'sma_5', 'sma_10', 'sma_30', 'rsi_14',
        'volatility_20', 'volume_change', 'market_return',
        'news_score', 'news_count'
    ]
    data, feature_names = add_macro_features_frozen(data, feature_names)
    data = data.dropna(subset=feature_names + ['future_return'])

    initial_train = int(len(data) * 0.5)
    if initial_train < 2 or len(data) - initial_train < 1:
        raise ValueError('Not enough historical data for walk-forward testing.')

    actual_values = []
    prediction_values = []
    window_results = []
    test_start = initial_train

    while test_start < len(data):
        test_end = min(test_start + test_window, len(data))
        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]

        if train['target'].nunique() < 2:
            test_start = test_end
            continue

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )
        model.fit(train[feature_names], train['target'])
        predictions = model.predict(test[feature_names])

        actual_values.extend(list(test['target']))
        prediction_values.extend(list(predictions))
        window_results.append(
            (test.index[0], test.index[-1], test, predictions)
        )
        test_start = test_end

    if not actual_values:
        raise ValueError('No valid walk-forward windows were produced.')

    actual = data.iloc[
        initial_train:initial_train + len(actual_values)
    ]['target']
    out_of_sample = data.iloc[
        initial_train:initial_train + len(actual_values)
    ]

    always_up = [1] * len(actual_values)
    previous_day = (out_of_sample['daily_return'] > 0).astype(int)

    summary = {
        'ticker': ticker.upper(),
        'predictions': len(actual_values),
        'news_days': int((data['news_count'] > 0).sum()),
        'out_of_sample_news_days': int(
            (out_of_sample['news_count'] > 0).sum()
        ),
        'accuracy': accuracy_score(actual, prediction_values) * 100,
        'balanced_accuracy': balanced_accuracy_score(
            actual, prediction_values
        ) * 100,
        'always_up': accuracy_score(actual, always_up) * 100,
        'previous_day': accuracy_score(actual, previous_day) * 100,
    }

    if verbose:
        print(f'\n=== Walk-forward technical + news + macro for {ticker.upper()} ===')
        print(f'Daily news records: {len(news_by_date)}')
        print(f"Trading days with news: {(data['news_count'] > 0).sum()}")
        print(
            f"Out-of-sample days with news: "
            f"{(out_of_sample['news_count'] > 0).sum()}"
        )
        print(f'Initial training rows: {initial_train}')
        print(f'Test window: {test_window} trading days')
        print(f'Forecast horizon: {horizon} trading days')
        print(f"Features used: {', '.join(feature_names)}")

        for window_start, window_end, frame, predictions in window_results:
            score = balanced_accuracy_score(
                frame['target'], predictions
            ) * 100
            print(
                f'{window_start.date()} to {window_end.date()}: '
                f'{score:.2f}% balanced accuracy'
            )

        print(f"Out-of-sample rows: {summary['predictions']}")
        print(f"Overall accuracy: {summary['accuracy']:.2f}%")
        print(
            f"Overall balanced accuracy: "
            f"{summary['balanced_accuracy']:.2f}%"
        )
        print(
            f"Overall precision: "
            f"{precision_score(actual, prediction_values, zero_division=0) * 100:.2f}%"
        )
        print(
            f"Overall recall: "
            f"{recall_score(actual, prediction_values, zero_division=0) * 100:.2f}%"
        )
        print(f"Always-UP accuracy: {summary['always_up']:.2f}%")
        print(
            f"Previous-day accuracy: "
            f"{summary['previous_day']:.2f}%"
        )
        print(
            f"Confusion matrix [[TN, FP], [FN, TP]]: "
            f"{confusion_matrix(actual, prediction_values).tolist()}"
        )


    return summary


def backtest_multiple_news_macro_tickers(tickers, years: int=5, horizon: int=5, test_window: int=63, news_dir: str='data/news'):
    """Compare technical-only, rule-news, and technical+news+macro walk-forward results across multiple tickers."""
    print('\n=== Multi-stock technical + news + macro comparison ===')
    print('Ticker | Predictions | Technical BA | News BA | News+Macro BA | News Delta | Macro Delta')
    print('-------|-------------|--------------|---------|---------------|------------|-----------')
    for ticker in tickers:
        try:
            news_status = ensure_news_data(ticker, news_dir, years)
            if news_status == 'unavailable':
                print(f'{ticker.upper()}: skipped because historical news is unavailable.')
                continue
        except ValueError as error:
            print(f'{ticker.upper()}: historical news unavailable: {error}')
            continue

        try:
            technical = walk_forward_logistic_model(
                ticker, years, horizon, test_window, verbose=False
            )
            news = walk_forward_news_model(
                ticker, years, horizon, test_window, news_dir, verbose=False
            )
            news_macro = walk_forward_news_macro_model(
                ticker, years, horizon, test_window, news_dir, verbose=False
            )
        except ValueError as error:
            print(f'{ticker.upper()}: skipped: {error}')
            continue

        technical_ba = technical['balanced_accuracy']
        news_ba = news['balanced_accuracy']
        news_macro_ba = news_macro['balanced_accuracy']

        print(
            f"{ticker.upper():6} | "
            f"{news_macro['predictions']:11} | "
            f"{technical_ba:12.2f}% | "
            f"{news_ba:7.2f}% | "
            f"{news_macro_ba:13.2f}% | "
            f"{news_ba - technical_ba:+10.2f} pp | "
            f"{news_macro_ba - news_ba:+10.2f} pp"
        )


def backtest_multiple_news_tickers(tickers, years: int=5, horizon: int=5, test_window: int=63, news_dir: str='data/news'):
    """Compare technical-only and rule-news walk-forward results across multiple tickers."""
    print('\n=== Multi-stock Phase 3A comparison ===')
    print('Ticker | Predictions | Technical BA | News BA | Delta BA | News days | OOS news days | News accuracy | Always-UP | Previous-day')
    print('-------|-------------|--------------|---------|----------|-----------|---------------|---------------|-----------|--------------')
    for ticker in tickers:
        try:
            news_status = ensure_news_data(ticker, news_dir, years)
            if news_status == 'unavailable':
                print(f'{ticker.upper()}: skipped because historical news is unavailable.')
                continue
        except ValueError as error:
            print(f'{ticker.upper()}: historical news unavailable: {error}')
            continue
        technical = walk_forward_logistic_model(ticker, years, horizon, test_window, verbose=False)
        news = walk_forward_news_model(ticker, years, horizon, test_window, news_dir, verbose=False)
        print(f"{ticker.upper():6} | {news['predictions']:11} | {technical['balanced_accuracy']:12.2f}% | {news['balanced_accuracy']:7.2f}% | {news['balanced_accuracy'] - technical['balanced_accuracy']:8.2f} pp | {news['news_days']:9} | {news['out_of_sample_news_days']:13} | {news['accuracy']:13.2f}% | {news['always_up']:9.2f}% | {news['previous_day']:12.2f}%")

def walk_forward_news_model(ticker: str, years: int=5, horizon: int=5, test_window: int=63, news_dir: str='data/news', verbose: bool=True):
    """Evaluate technical features plus deterministic daily news features with the same walk-forward protocol."""
    news_path = os.path.join(news_dir, f'{ticker.upper()}_daily.csv')
    if not os.path.exists(news_path):
        raise ValueError(f'Daily news file not found: {news_path}')
    news_by_date = {}
    with open(news_path, newline='', encoding='utf-8') as news_file:
        for row in csv.DictReader(news_file):
            news_date = (row.get('date') or '').strip()
            if news_date:
                news_by_date[news_date] = {'news_score': float(row.get('news_score') or 0), 'news_count': float(row.get('news_count') or 0)}
    start = date.today() - timedelta(days=years * 365)
    stock_history = cached_ticker(ticker).history(start=start.isoformat())
    market_history = cached_ticker('^GSPC').history(start=start.isoformat())
    if stock_history.empty or market_history.empty:
        raise ValueError('Could not download enough stock and market history.')
    data = stock_history[['Close', 'Volume']].copy()
    data['date'] = [index.date().isoformat() for index in data.index]
    data['news_score'] = data['date'].map(lambda value: news_by_date.get(value, {}).get('news_score', 0))
    data['news_count'] = data['date'].map(lambda value: news_by_date.get(value, {}).get('news_count', 0))
    data['daily_return'] = data['Close'].pct_change()
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_10'] = data['Close'].rolling(10).mean()
    data['sma_30'] = data['Close'].rolling(30).mean()
    delta = data['Close'].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float('nan'))
    data['rsi_14'] = 100 - 100 / (1 + relative_strength)
    data['volatility_20'] = data['daily_return'].rolling(20).std()
    data['volume_change'] = data['Volume'].pct_change()
    data['market_return'] = market_history['Close'].pct_change()
    data['future_return'] = data['Close'].shift(-horizon) / data['Close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    feature_names = ['daily_return', 'sma_5', 'sma_10', 'sma_30', 'rsi_14', 'volatility_20', 'volume_change', 'market_return', 'news_score', 'news_count']
    data = data.dropna(subset=feature_names + ['future_return'])
    initial_train = int(len(data) * 0.5)
    if initial_train < 2 or len(data) - initial_train < 1:
        raise ValueError('Not enough historical data for walk-forward testing.')
    actual_values = []
    prediction_values = []
    window_results = []
    test_start = initial_train
    while test_start < len(data):
        test_end = min(test_start + test_window, len(data))
        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]
        if train['target'].nunique() < 2:
            test_start = test_end
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[feature_names], train['target'])
        predictions = model.predict(test[feature_names])
        actual_values.extend(list(test['target']))
        prediction_values.extend(list(predictions))
        window_results.append((test.index[0], test.index[-1], test, predictions))
        test_start = test_end
    if not actual_values:
        raise ValueError('No valid walk-forward windows were produced.')
    actual = data.iloc[initial_train:initial_train + len(actual_values)]['target']
    out_of_sample = data.iloc[initial_train:initial_train + len(actual_values)]
    always_up = [1] * len(actual_values)
    previous_day = (out_of_sample['daily_return'] > 0).astype(int)
    summary = {'ticker': ticker.upper(), 'predictions': len(actual_values), 'news_days': int((data['news_count'] > 0).sum()), 'out_of_sample_news_days': int((out_of_sample['news_count'] > 0).sum()), 'accuracy': accuracy_score(actual, prediction_values) * 100, 'balanced_accuracy': balanced_accuracy_score(actual, prediction_values) * 100, 'always_up': accuracy_score(actual, always_up) * 100, 'previous_day': accuracy_score(actual, previous_day) * 100}
    if verbose:
        print(f'\n=== Walk-forward technical + news for {ticker.upper()} ===')
        print(f'Daily news records: {len(news_by_date)}')
        print(f"Trading days with news: {(data['news_count'] > 0).sum()}")
        print(f"Out-of-sample days with news: {(out_of_sample['news_count'] > 0).sum()}")
        print(f'Initial training rows: {initial_train}')
        print(f'Test window: {test_window} trading days')
        print(f'Forecast horizon: {horizon} trading days')
        print(f"Features used: {', '.join(feature_names)}")
        for window_start, window_end, frame, predictions in window_results:
            score = balanced_accuracy_score(frame['target'], predictions) * 100
            print(f'{window_start.date()} to {window_end.date()}: {score:.2f}% balanced accuracy')
        print(f"Out-of-sample rows: {summary['predictions']}")
        print(f"Overall accuracy: {summary['accuracy']:.2f}%")
        print(f"Overall balanced accuracy: {summary['balanced_accuracy']:.2f}%")
        print(f'Overall precision: {precision_score(actual, prediction_values, zero_division=0) * 100:.2f}%')
        print(f'Overall recall: {recall_score(actual, prediction_values, zero_division=0) * 100:.2f}%')
        print(f"Always-UP accuracy: {summary['always_up']:.2f}%")
        print(f"Previous-day accuracy: {summary['previous_day']:.2f}%")
        print(f'Confusion matrix [[TN, FP], [FN, TP]]: {[list(row) for row in confusion_matrix(actual, prediction_values)]}')
    return summary

def walk_forward_gemini_model(ticker: str, years: int=5, horizon: int=5, test_window: int=63, news_dir: str='data/news', verbose: bool=True):
    """Evaluate technical features plus cached Gemini news features without making API calls during backtesting."""
    gemini_daily_path = os.path.join(news_dir, f'{ticker.upper()}_gemini_daily.csv')
    if not os.path.exists(gemini_daily_path):
        raise ValueError(f'Gemini daily file not found: {gemini_daily_path}. Run --download-news, --process-news, then --process-news-gemini first.')
    gemini_by_date = {}
    with open(gemini_daily_path, newline='', encoding='utf-8') as gemini_file:
        for row in csv.DictReader(gemini_file):
            g_date = (row.get('date') or '').strip()
            if g_date:
                gemini_by_date[g_date] = {'gemini_sentiment_score': float(row.get('gemini_sentiment_score') or 0), 'gemini_relevance': float(row.get('gemini_relevance') or 0), 'gemini_confidence': float(row.get('gemini_confidence') or 0), 'gemini_news_count': float(row.get('gemini_news_count') or 0)}
    start = date.today() - timedelta(days=years * 365)
    stock_history = cached_ticker(ticker).history(start=start.isoformat())
    market_history = cached_ticker('^GSPC').history(start=start.isoformat())
    if stock_history.empty or market_history.empty:
        raise ValueError('Could not download enough stock and market history.')
    data = stock_history[['Close', 'Volume']].copy()
    data['date'] = [index.date().isoformat() for index in data.index]
    data['gemini_sentiment_score'] = data['date'].map(lambda value: gemini_by_date.get(value, {}).get('gemini_sentiment_score', 0))
    data['gemini_relevance'] = data['date'].map(lambda value: gemini_by_date.get(value, {}).get('gemini_relevance', 0))
    data['gemini_confidence'] = data['date'].map(lambda value: gemini_by_date.get(value, {}).get('gemini_confidence', 0))
    data['gemini_news_count'] = data['date'].map(lambda value: gemini_by_date.get(value, {}).get('gemini_news_count', 0))
    data['daily_return'] = data['Close'].pct_change()
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_10'] = data['Close'].rolling(10).mean()
    data['sma_30'] = data['Close'].rolling(30).mean()
    delta = data['Close'].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float('nan'))
    data['rsi_14'] = 100 - 100 / (1 + relative_strength)
    data['volatility_20'] = data['daily_return'].rolling(20).std()
    data['volume_change'] = data['Volume'].pct_change()
    data['market_return'] = market_history['Close'].pct_change()
    data['future_return'] = data['Close'].shift(-horizon) / data['Close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    feature_names = ['daily_return', 'sma_5', 'sma_10', 'sma_30', 'rsi_14', 'volatility_20', 'volume_change', 'market_return', 'gemini_sentiment_score', 'gemini_relevance', 'gemini_confidence', 'gemini_news_count']
    data = data.dropna(subset=feature_names + ['future_return'])
    initial_train = int(len(data) * 0.5)
    if initial_train < 2 or len(data) - initial_train < 1:
        raise ValueError('Not enough historical data for walk-forward testing.')
    actual_values = []
    prediction_values = []
    window_results = []
    test_start = initial_train
    while test_start < len(data):
        test_end = min(test_start + test_window, len(data))
        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]
        if train['target'].nunique() < 2:
            test_start = test_end
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[feature_names], train['target'])
        predictions = model.predict(test[feature_names])
        actual_values.extend(list(test['target']))
        prediction_values.extend(list(predictions))
        window_results.append((test.index[0], test.index[-1], test, predictions))
        test_start = test_end
    if not actual_values:
        raise ValueError('No valid walk-forward windows were produced.')
    actual = data.iloc[initial_train:initial_train + len(actual_values)]['target']
    out_of_sample = data.iloc[initial_train:initial_train + len(actual_values)]
    always_up = [1] * len(actual_values)
    previous_day = (out_of_sample['daily_return'] > 0).astype(int)
    summary = {'ticker': ticker.upper(), 'predictions': len(actual_values), 'gemini_days': int((data['gemini_news_count'] > 0).sum()), 'out_of_sample_gemini_days': int((out_of_sample['gemini_news_count'] > 0).sum()), 'accuracy': accuracy_score(actual, prediction_values) * 100, 'balanced_accuracy': balanced_accuracy_score(actual, prediction_values) * 100, 'always_up': accuracy_score(actual, always_up) * 100, 'previous_day': accuracy_score(actual, previous_day) * 100}
    if verbose:
        print(f'\n=== Walk-forward technical + Gemini news for {ticker.upper()} ===')
        print(f"Days with Gemini-classified news: {(data['gemini_news_count'] > 0).sum()}")
        print(f"Out-of-sample days with Gemini news: {(out_of_sample['gemini_news_count'] > 0).sum()}")
        print(f'Initial training rows: {initial_train}')
        print(f'Test window: {test_window} trading days')
        print(f'Forecast horizon: {horizon} trading days')
        print(f"Features used: {', '.join(feature_names)}")
        for window_start, window_end, frame, predictions in window_results:
            score = balanced_accuracy_score(frame['target'], predictions) * 100
            print(f'{window_start.date()} to {window_end.date()}: {score:.2f}% balanced accuracy')
        print(f"Out-of-sample rows: {summary['predictions']}")
        print(f"Overall accuracy: {summary['accuracy']:.2f}%")
        print(f"Overall balanced accuracy: {summary['balanced_accuracy']:.2f}%")
        print(f'Overall precision: {precision_score(actual, prediction_values, zero_division=0) * 100:.2f}%')
        print(f'Overall recall: {recall_score(actual, prediction_values, zero_division=0) * 100:.2f}%')
        print(f"Always-UP accuracy: {summary['always_up']:.2f}%")
        print(f"Previous-day accuracy: {summary['previous_day']:.2f}%")
        print(f'Confusion matrix [[TN, FP], [FN, TP]]: {[list(row) for row in confusion_matrix(actual, prediction_values)]}')
    return summary

def backtest_three_way_comparison(tickers, years: int=5, horizon: int=5, test_window: int=63, news_dir: str='data/news'):
    """Compare technical-only, rule-news, and Gemini-news models using the same walk-forward protocol."""
    print('\n=== Three-way comparison: technical vs rule-news vs Gemini-news ===')
    print('Ticker | Predictions | Technical BA | Rule-news BA | Gemini BA | Delta rule | Delta Gemini | Gemini days (OOS)')
    print('-------|-------------|--------------|--------------|-----------|------------|--------------|-------------------')
    for ticker in tickers:
        try:
            news_status = ensure_news_data(ticker, news_dir, years)
            if news_status == 'unavailable':
                print(f'{ticker.upper()}: skipped -- rule-based news unavailable (set ALPHAVANTAGE_API_KEY or run --download-news first).')
                continue
        except ValueError as error:
            print(f'{ticker.upper()}: skipped -- {error}')
            continue
        gemini_daily_path = os.path.join(news_dir, f'{ticker.upper()}_gemini_daily.csv')
        if not os.path.exists(gemini_daily_path):
            print(f'{ticker.upper()}: skipped -- no Gemini classification cached. Run --process-news-gemini first.')
            continue
        try:
            technical = walk_forward_logistic_model(ticker, years, horizon, test_window, verbose=False)
            rule_news = walk_forward_news_model(ticker, years, horizon, test_window, news_dir, verbose=False)
            gemini_news = walk_forward_gemini_model(ticker, years, horizon, test_window, news_dir, verbose=False)
        except ValueError as error:
            print(f'{ticker.upper()}: skipped -- {error}')
            continue
        print(f"{ticker.upper():6} | {gemini_news['predictions']:11} | {technical['balanced_accuracy']:12.2f}% | {rule_news['balanced_accuracy']:12.2f}% | {gemini_news['balanced_accuracy']:9.2f}% | {rule_news['balanced_accuracy'] - technical['balanced_accuracy']:10.2f} pp | {gemini_news['balanced_accuracy'] - technical['balanced_accuracy']:12.2f} pp | {gemini_news['out_of_sample_gemini_days']:19}")

def backtest_multiple_tickers(tickers, years: int=5, horizon: int=5, test_window: int=63):
    """Compare the technical walk-forward control across multiple tickers."""
    summaries = [walk_forward_logistic_model(ticker, years, horizon, test_window, verbose=False) for ticker in tickers]
    print('\n=== Multi-stock walk-forward comparison ===')
    print('Ticker | Predictions | Balanced accuracy | Always-UP | Previous-day')
    print('-------|-------------|--------------------|-----------|--------------')
    for summary in summaries:
        print(f"{summary['ticker']:6} | {summary['predictions']:11} | {summary['balanced_accuracy']:18.2f}% | {summary['always_up']:9.2f}% | {summary['previous_day']:12.2f}%")

def compute_technical_signal(hist):
    """Compute the compact live technical trend signal and momentum."""
    closes = list(hist['Close'])
    latest_price = closes[-1]
    short_window = closes[-10:] if len(closes) >= 10 else closes
    long_window = closes[-30:] if len(closes) >= 30 else closes
    short_ma = statistics.mean(short_window)
    long_ma = statistics.mean(long_window)
    momentum_pct = (latest_price - closes[0]) / closes[0] * 100
    gap_pct = (short_ma - long_ma) / long_ma * 100
    if gap_pct > 1:
        technical_score = 1
        trend_desc = 'short-term average is above the longer-term average (bullish trend)'
    elif gap_pct < -1:
        technical_score = -1
        trend_desc = 'short-term average is below the longer-term average (bearish trend)'
    else:
        technical_score = 0
        trend_desc = 'short-term and longer-term averages are close (no clear trend)'
    explanation = f'Latest close: {latest_price:.2f}. 10-day avg: {short_ma:.2f}, 30-day avg: {long_ma:.2f}. Momentum over window: {momentum_pct:+.2f}%. Signal: {trend_desc}.'
    return (technical_score, explanation)

def _news_cache_path(ticker: str, news_dir: str='data/news'):
    """Return the local path used for cached recent Yahoo headlines."""
    cache_dir = os.path.join(news_dir, 'live')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{ticker.upper()}_recent.json')

def fetch_news(ticker: str, limit: int=8, news_dir: str='data/news', cache_hours: float=6, refresh: bool=False):
    """Fetch recent Yahoo headlines with a local time-based cache to reduce repeated requests."""
    cache_path = _news_cache_path(ticker, news_dir)
    now = datetime.now().timestamp()
    if not refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding='utf-8') as cache_file:
                cached = json.load(cache_file)
            cached_at = float(cached.get('cached_at', 0))
            if now - cached_at <= cache_hours * 3600:
                return [item for item in cached.get('headlines', []) if item][:limit]
        except (OSError, ValueError, TypeError):
            pass
    stock = cached_ticker(ticker)
    try:
        raw_news = stock.news
    except Exception:
        raw_news = []
    headlines = []
    for item in raw_news[:max(limit, 8)]:
        title = item.get('title') or item.get('content', {}).get('title')
        if title:
            headlines.append(title.strip())
    payload = {'ticker': ticker.upper(), 'cached_at': now, 'headlines': headlines}
    try:
        with open(cache_path, 'w', encoding='utf-8') as cache_file:
            json.dump(payload, cache_file, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return headlines[:limit]
POSITIVE_WORDS = {'beat', 'beats', 'surge', 'surges', 'rally', 'rallies', 'growth', 'upgrade', 'upgraded', 'record', 'strong', 'outperform', 'bullish', 'profit', 'gain', 'gains', 'soar', 'soars', 'expansion', 'buyback'}
NEGATIVE_WORDS = {'miss', 'misses', 'plunge', 'plunges', 'drop', 'drops', 'downgrade', 'downgraded', 'weak', 'underperform', 'bearish', 'loss', 'losses', 'fall', 'falls', 'lawsuit', 'investigation', 'recall', 'layoffs', 'cut'}

def find_context_matches(headlines, context):
    """Find recent headlines containing topics retrieved from optional company context."""
    topics = [topic.lower() for topic in context.get('topics', [])]
    return [headline for headline in headlines if any((topic in headline.lower() for topic in topics))]

def score_news_sentiment(headlines):
    """Score recent headlines using the deterministic keyword control."""
    if not headlines:
        return (0, [])
    pos_hits = 0
    neg_hits = 0
    matched = []
    for headline in headlines:
        words = set(headline.lower().split())
        pos_found = words & POSITIVE_WORDS
        neg_found = words & NEGATIVE_WORDS
        if pos_found or neg_found:
            matched.append(headline)
        pos_hits += len(pos_found)
        neg_hits += len(neg_found)
    if pos_hits == neg_hits:
        news_score = 0
    elif pos_hits > neg_hits:
        news_score = 1
    else:
        news_score = -1
    return (news_score, matched)

def make_prediction(technical_score, news_score):
    """Return the legacy heuristic lean and confidence; this is not a validated ML prediction."""
    total = technical_score + news_score
    if total >= 2:
        return "UP", "High"
    if total == 1:
        return "UP", "Moderate"
    if total == 0:
        return "UNCLEAR", "Low"
    if total == -1:
        return "DOWN", "Moderate"
    return "DOWN", "High"

def discover_assets(query: str, limit: int=20, asset_type: str | None=None):
    """Discover supported Yahoo Finance assets from natural-language search terms."""
    clean_query = query.strip()
    if not clean_query:
        return []
    search_url = 'https://query2.finance.yahoo.com/v1/finance/search?' + urlencode({'q': clean_query, 'quotesCount': max(limit * 2, 20), 'newsCount': 0})
    request = Request(search_url, headers={'User-Agent': 'Mozilla/5.0 stock-predictor/1.0'})
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.load(response)
    except Exception:
        return []
    allowed = {'EQUITY': 'equity', 'ETF': 'fund', 'MUTUALFUND': 'fund', 'INDEX': 'index', 'CURRENCY': 'currency'}
    results = []
    seen = set()
    for item in payload.get('quotes', []):
        symbol = (item.get('symbol') or '').strip()
        quote_type = (item.get('quoteType') or '').upper()
        if not symbol or symbol in seen:
            continue
        if asset_type and allowed.get(quote_type) != asset_type:
            continue
        seen.add(symbol)
        results.append({'symbol': symbol, 'name': item.get('longname') or item.get('shortname') or symbol, 'type': allowed.get(quote_type, quote_type.lower()), 'exchange': item.get('exchange') or item.get('exchDisp') or ''})
        if len(results) >= limit:
            break
    return results

def infer_currency_code(query: str):
    
    text = str(query).strip()
    upper = text.upper()

    direct = re.findall(r"\b[A-Z]{3}\b", upper)

    for code in direct:
        try:
            if pycountry.currencies.get(alpha_3=code):
                return code
        except Exception:
            pass

    cleaned = re.sub(
        r"\b(currency|currencies|money|exchange|exchange rate|rate|forex|fx)\b",
        " ",
        text,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return None

    try:
        search_fuzzy = getattr(pycountry.countries, "search_fuzzy", None)

        if callable(search_fuzzy):
            country_matches: Any = search_fuzzy(cleaned)

            if country_matches:
                for country in country_matches:
                    try:
                        currencies = get_territory_currencies(
                            country.alpha_2,
                            tender=True,
                        )

                        if currencies:
                            return currencies[0]
                    except Exception:
                        pass

    except Exception:
        pass

    currency_aliases = {
        "eurozone": "EUR",
        "euro": "EUR",
        "sterling": "GBP",
        "pound sterling": "GBP",
        "pound": "GBP",
        "yen": "JPY",
        "yuan": "CNY",
        "renminbi": "CNY",
        "franc": "CHF",
        "swiss franc": "CHF",
        "dollar": "USD",
        "us dollar": "USD",
        "american dollar": "USD",
        "canadian dollar": "CAD",
        "australian dollar": "AUD",
        "new zealand dollar": "NZD",
        "singapore dollar": "SGD",
        "hong kong dollar": "HKD",
        "swedish krona": "SEK",
        "swedish krona": "SEK",
        "norwegian krone": "NOK",
        "danish krone": "DKK",
        "polish zloty": "PLN",
        "polish złoty": "PLN",
        "czech koruna": "CZK",
        "hungarian forint": "HUF",
        "romanian leu": "RON",
        "bulgarian lev": "BGN",
        "turkish lira": "TRY",
        "tunisian dinar": "TND",
        "algerian dinar": "DZD",
        "moroccan dirham": "MAD",
        "egyptian pound": "EGP",
        "south african rand": "ZAR",
        "indian rupee": "INR",
        "russian ruble": "RUB",
        "russian rouble": "RUB",
        "brazilian real": "BRL",
        "mexican peso": "MXN",
        "south korean won": "KRW",
        "korean won": "KRW",
        "thai baht": "THB",
        "indonesian rupiah": "IDR",
        "malaysian ringgit": "MYR",
        "philippine peso": "PHP",
        "vietnamese dong": "VND",
    }

    lowered = cleaned.lower()

    for alias, code in sorted(
        currency_aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if alias in lowered:
            return code

    try:
        matches = pycountry.currencies.get(
            name=cleaned
        )

        if matches:
            return matches.alpha_3
    except Exception:
        pass

    return None

def resolve_currency_query(query: str):
    """Resolve a natural-language currency query to a Yahoo Finance FX symbol."""
    code = infer_currency_code(query)
    if not code:
        return []

    code = code.upper()

    try:
        currency = pycountry.currencies.get(alpha_3=code)
        name = currency.name if currency is not None else code
    except Exception:
        name = code

    if code == "EUR":
        candidates = ["EURUSD=X", "EUR=X"]
    elif code == "JPY":
        candidates = ["JPY=X", "JPYUSD=X"]
    else:
        candidates = [f"{code}=X", f"{code}USD=X", f"USD{code}=X"]

    for symbol in candidates:
        try:
            hist = cached_ticker(symbol).history(period="5d")
            if not hist.empty:
                return [{
                    "symbol": symbol,
                    "name": name,
                    "type": "currency",
                    "exchange": "",
                }]
        except Exception:
            pass

    return [{
        "symbol": candidates[0],
        "name": name,
        "type": "currency",
        "exchange": "",
    }]


def resolve_asset_query(query: str):
    """Resolve direct symbols and natural-language asset queries without a hard-coded asset universe."""
    text = query.strip()
    if not text:
        return (None, [])
    if re.fullmatch('[A-Za-z0-9.$=^_-]{1,20}', text):
        symbol = text.upper()
        try:
            hist = cached_ticker(symbol).history(period='5d')
            if not hist.empty:
                asset_type = 'currency' if symbol.endswith('=X') else 'asset'
                return (
                    symbol,
                    [{
                        'symbol': symbol,
                        'name': symbol,
                        'type': asset_type,
                        'exchange': '',
                    }],
                )
        except Exception:
            pass
        if symbol.endswith('=X'):
            return (
                symbol,
                [{
                    'symbol': symbol,
                    'name': symbol,
                    'type': 'currency',
                    'exchange': '',
                }],
            )
    lower = text.lower()
    currency_hint = any(word in lower for word in ('currency', 'currencies', 'dinar', 'dollar', 'euro', 'pound', 'yen', 'franc', 'rupee', 'ruble', 'yuan', 'won', 'peso', 'real', 'rand', 'krone', 'krona', 'lira', 'dirham', 'riyal', 'ringgit', 'baht', 'dong', 'naira', 'cedi', 'shilling', 'zloty', 'forint', 'koruna'))
    if currency_hint:
        currency_results = resolve_currency_query(text)
        if currency_results:
            return (text, currency_results)
    asset_type = None
    if any((word in lower for word in ('stock', 'stocks', 'equity', 'equities', 'share', 'shares', 'company', 'companies', 'miner', 'miners'))):
        asset_type = 'equity'
    elif any((word in lower for word in ('etf', 'fund', 'funds'))):
        asset_type = 'fund'
    results = discover_assets(text, limit=20, asset_type=asset_type)
    if asset_type in ('equity', 'fund') and results:
        results = results[:1]
    if not results:
        cleaned = re.sub('\\b(show|find|give|me|the|best|top|stocks|stock|assets|asset|related)\\b', ' ', lower)
        cleaned = re.sub('\\s+', ' ', cleaned).strip()
        if cleaned:
            results = discover_assets(cleaned, limit=20, asset_type=asset_type)
            if asset_type in ('equity', 'fund') and results:
                results = results[:1]
    return (text if results else None, results)

def analyze_asset_query(query: str, days: int, news_dir: str, years: int, cutoff_hour: int, news_cache_hours: float=6, refresh_news: bool=False, show_headlines: bool=False):
    """Resolve a query dynamically and analyze each discovered asset without historical-news downloads."""
    label, assets = resolve_asset_query(query)
    if not assets or not label:
        raise ValueError(f'No supported asset found for: {query}')
    print(f'\n=== {(label or "").upper()} ===')
    print('Discovered: ' + ', '.join((item['symbol'] for item in assets)))
    for item in assets:
        try:
            analyze_ticker(item['symbol'], days, news_dir, years, cutoff_hour, auto_historical_news=False, news_cache_hours=news_cache_hours, refresh_news=refresh_news, show_headlines=show_headlines, asset_type=item.get('type'))
        except Exception as error:
            print(f"{item['symbol']}: skipped ({error})")

def predict_live_direction(
    ticker: str,
    years: int = 5,
    horizon: int = 5,
    news_dir: str = "data/news",
    min_rows: int = 60,
):
    """Train the equity Logistic Regression with technical + multi-macro features."""
    start = date.today() - timedelta(days=years * 365)

    try:
        stock_history = cached_ticker(ticker).history(start=start.isoformat())
        market_history = cached_ticker("^GSPC").history(start=start.isoformat())
    except Exception:
        return None

    if stock_history.empty or market_history.empty:
        return None

    data = stock_history[["Close", "Volume"]].copy()
    data["daily_return"] = data["Close"].pct_change()
    data["sma_5"] = data["Close"].rolling(5).mean()
    data["sma_10"] = data["Close"].rolling(10).mean()
    data["sma_30"] = data["Close"].rolling(30).mean()

    delta = data["Close"].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float("nan"))
    data["rsi_14"] = 100 - (100 / (1 + relative_strength))
    data["volatility_20"] = data["daily_return"].rolling(20).std()
    data["volume_change"] = data["Volume"].pct_change()

    market_return = market_history["Close"].pct_change()
    stock_dates = pd.to_datetime(data.index).tz_localize(None).normalize()
    market_dates = pd.to_datetime(market_return.index).tz_localize(None).normalize()
    market_series = pd.Series(market_return.to_numpy(), index=market_dates)
    market_series = market_series[~market_series.index.duplicated(keep="last")]
    data["market_return"] = market_series.reindex(stock_dates).to_numpy()

    data["future_return"] = data["Close"].shift(-horizon) / data["Close"] - 1
    data["target"] = (data["future_return"] > 0).astype(int)
    data = data.replace([float("inf"), float("-inf")], float("nan"))

    technical_features = [
        "daily_return", "sma_5", "sma_10", "sma_30", "rsi_14",
        "volatility_20", "volume_change", "market_return",
    ]

    data, feature_names = add_macro_features(
        data,
        start,
        technical_features.copy()
    )

    # Do not silently run a different model if one macro series is missing.
    if not all(name in data.columns for name in MACRO_FEATURE_NAMES):
        return None

    data = data.dropna(subset=feature_names + ["Close", "Volume"]).copy()
    if data.empty:
        return None

    latest_row = data.iloc[[-1]]
    if latest_row[feature_names].isna().any(axis=None):
        return None

    trainable = data.iloc[:-1].dropna(
        subset=feature_names + ["future_return"]
    )

    if len(trainable) < min_rows or trainable["target"].nunique() < 2:
        return None

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
    model.fit(trainable[feature_names], trainable["target"])

    prediction = model.predict(latest_row[feature_names])[0]
    probability_up = model.predict_proba(latest_row[feature_names])[0][1]

    macro_snapshot = {
        name: float(latest_row[name].iloc[0])
        for name in MACRO_FEATURE_NAMES
        if name in latest_row.columns and pd.notna(latest_row[name].iloc[0])
    }

    return {
        "direction": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "features_used": feature_names,
        "training_rows": len(trainable),
        "macro_snapshot": macro_snapshot,
        "macro_features_used": [
            name for name in feature_names
            if name in MACRO_FEATURE_NAMES
        ],
    }


def build_fx_dataset(ticker: str, years: int = 5, horizon: int = 5):
    """Build a currency-specific dataset; optional DXY features are used only when available."""
    start = date.today() - timedelta(days=years * 365)
    try:
        fx_history = cached_ticker(ticker).history(start=start.isoformat(), auto_adjust=False)
    except Exception:
        fx_history = None

    if fx_history is None or fx_history.empty or len(fx_history) < 315:
        try:
            fallback = cached_ticker(ticker).history(period="max", auto_adjust=False)
            if not fallback.empty:
                cutoff = pd.Timestamp(start)
                fallback_tz = getattr(fallback.index, "tz", None)
                if fallback_tz is not None and cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize(fallback_tz)
                fallback = fallback[fallback.index >= cutoff]
                current_len = 0 if fx_history is None else len(fx_history)
                if len(fallback) > current_len:
                    fx_history = fallback
        except Exception:
            pass

    if fx_history is None or fx_history.empty:
        return None, []

    data = fx_history[["Close"]].copy()
    feature_names = [
        "daily_return", "return_5", "sma_5_ratio", "sma_20_ratio",
        "rsi_14", "volatility_20",
    ]

    try:
        dxy_history = cached_ticker("DX-Y.NYB").history(start=start.isoformat(), auto_adjust=False)
        if dxy_history.empty:
            dxy_history = cached_ticker("DX-Y.NYB").history(period="max", auto_adjust=False)
            if not dxy_history.empty:
                cutoff = pd.Timestamp(start)
                fallback_tz = getattr(dxy_history.index, "tz", None)
                if fallback_tz is not None and cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize(fallback_tz)
                dxy_history = dxy_history[dxy_history.index >= cutoff]
        if not dxy_history.empty:
            fx_dates = pd.to_datetime(data.index).tz_localize(None).normalize()
            dxy_dates = pd.to_datetime(dxy_history.index).tz_localize(None).normalize()
            dxy = pd.Series(
                dxy_history["Close"].to_numpy(),
                index=dxy_dates,
                name="dxy_close",
            )
            dxy = dxy[~dxy.index.duplicated(keep="last")]
            aligned_dxy = dxy.reindex(fx_dates).ffill()
            data["dxy_close"] = aligned_dxy.to_numpy()
            coverage = data["dxy_close"].notna().mean()
            if coverage >= 0.8:
                data["dxy_return"] = data["dxy_close"].pct_change()
                data["dxy_return_5"] = data["dxy_close"].pct_change(5)
                feature_names += ["dxy_return", "dxy_return_5"]
            else:
                data = data.drop(columns=["dxy_close"])
    except Exception:
        pass

    data["daily_return"] = data["Close"].pct_change()
    data["return_5"] = data["Close"].pct_change(5)
    data["sma_5_ratio"] = data["Close"] / data["Close"].rolling(5).mean() - 1
    data["sma_20_ratio"] = data["Close"] / data["Close"].rolling(20).mean() - 1
    delta = data["Close"].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float("nan"))
    data["rsi_14"] = 100 - (100 / (1 + relative_strength))
    data["volatility_20"] = data["daily_return"].rolling(20).std()
    # The target is defined strictly from future prices; feature rows never use future information.
    data["future_return"] = data["Close"].shift(-horizon) / data["Close"] - 1
    data["target"] = (data["future_return"] > 0).astype(int)
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    return data, feature_names

FX_VALIDATION_CACHE = "data/fx_validation.json"
FX_VALIDATION_TEST_WINDOW = 63


def load_validated_fx_pairs(years: int, horizon: int, test_window: int = FX_VALIDATION_TEST_WINDOW):
    try:
        with open(FX_VALIDATION_CACHE, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if (
            payload.get("years") != years
            or payload.get("horizon") != horizon
            or payload.get("test_window") != test_window
        ):
            return set()
        return {str(pair).upper() for pair in payload.get("validated_pairs", [])}
    except (OSError, ValueError, TypeError):
        return set()


def save_fx_validation(summaries, years: int, horizon: int, test_window: int):
    validated = [summary["ticker"] for summary in summaries if summary["beats_both"]]
    payload = {
        "years": years,
        "horizon": horizon,
        "test_window": test_window,
        "validated_pairs": validated,
    }
    os.makedirs(os.path.dirname(FX_VALIDATION_CACHE), exist_ok=True)
    with open(FX_VALIDATION_CACHE, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return validated


def predict_live_fx_direction(
    ticker: str,
    years: int = 5,
    horizon: int = 5,
):
    ticker = ticker.upper()
    validated_pairs = load_validated_fx_pairs(years, horizon)
    if ticker not in validated_pairs:
        return None

    data, feature_names = build_fx_dataset(ticker, years, horizon)
    if data is None or not all(name in feature_names for name in ("dxy_return", "dxy_return_5")):
        return None

    data = data.replace([float("inf"), float("-inf")], float("nan"))
    latest_row = data.iloc[[-1]]
    if latest_row[feature_names].isna().any(axis=None):
        return None

    trainable = data.iloc[:-1].dropna(subset=feature_names + ["future_return"])
    if len(trainable) < 252 or trainable["target"].nunique() < 2:
        return None

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(trainable[feature_names], trainable["target"])
    prediction = model.predict(latest_row[feature_names])[0]
    probability_up = model.predict_proba(latest_row[feature_names])[0][1]

    return {
        "direction": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "features_used": feature_names,
        "training_rows": len(trainable),
    }


def walk_forward_fx_model(ticker: str, years: int = 5, horizon: int = 5, test_window: int = 63, verbose: bool = True):
    """Evaluate the FX model with expanding chronological walk-forward windows."""
    data, feature_names = build_fx_dataset(ticker, years, horizon)
    if data is None:
        raise ValueError(f"No usable FX data for {ticker.upper()}.")

    data = data.dropna(subset=feature_names + ["future_return"]).copy()
    initial_train = max(252, test_window)
    if len(data) <= initial_train + test_window:
        raise ValueError(
            f"Not enough FX history for {ticker.upper()}: {len(data)} usable rows; "
            f"at least {initial_train + test_window + 1} are required."
        )
    if data["target"].nunique() < 2:
        raise ValueError(f"FX target has only one class for {ticker.upper()} after preprocessing.")

    actual_values = []
    prediction_values = []
    test_start = initial_train
    while test_start < len(data):
        test_end = min(test_start + test_window, len(data))
        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]
        if train["target"].nunique() < 2:
            test_start = test_end
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[feature_names], train["target"])
        predictions = model.predict(test[feature_names])
        actual_values.extend(list(test["target"]))
        prediction_values.extend(list(predictions))
        test_start = test_end

    if not actual_values:
        raise ValueError(f"No valid FX walk-forward windows for {ticker.upper()}.")

    actual = pd.Series(actual_values)
    predictions = pd.Series(prediction_values)
    always_up = pd.Series([1] * len(actual))
    previous_day = data.iloc[initial_train:initial_train + len(actual)]["daily_return"].gt(0).astype(int).reset_index(drop=True)
    summary = {
        "ticker": ticker.upper(),
        "predictions": len(actual),
        "accuracy": accuracy_score(actual, predictions) * 100,
        "balanced_accuracy": balanced_accuracy_score(actual, predictions) * 100,
        "always_up": accuracy_score(actual, always_up) * 100,
        "previous_day": accuracy_score(actual, previous_day) * 100,
        "features": feature_names,
    }
    if verbose:
        print(f"\n=== FX walk-forward Logistic Regression for {ticker.upper()} ===")
        print(f"Out-of-sample rows: {summary['predictions']}")
        print(f"Overall accuracy: {summary['accuracy']:.2f}%")
        print(f"Overall balanced accuracy: {summary['balanced_accuracy']:.2f}%")
        print(f"Always-UP accuracy: {summary['always_up']:.2f}%")
        print(f"Previous-day accuracy: {summary['previous_day']:.2f}%")
        print(f"Features: {', '.join(feature_names)}")
        print(f"Confusion matrix [[TN, FP], [FN, TP]]: {[list(row) for row in confusion_matrix(actual, predictions)]}")
    return summary


def backtest_multiple_fx_pairs(pairs, years=5, horizon=5, test_window=63):
    summaries = []
    for pair in pairs:
        try:
            summary = walk_forward_fx_model(pair, years, horizon, test_window, verbose=False)
            summary["beats_both"] = (
                summary["balanced_accuracy"] > summary["always_up"]
                and summary["balanced_accuracy"] > summary["previous_day"]
            )
            summaries.append(summary)
        except ValueError as error:
            print(f"{pair.upper()}: skipped -- {error}")

    if not summaries:
        raise ValueError("No FX pairs produced valid walk-forward results.")

    print("\n=== FX validation comparison ===")
    print("Pair     | OOS rows | Balanced accuracy | Always-UP | Previous-day | Beats both")
    print("---------|----------|--------------------|-----------|--------------|-----------")
    for summary in summaries:
        print(
            f"{summary['ticker']:8} | {summary['predictions']:8} | "
            f"{summary['balanced_accuracy']:18.2f}% | "
            f"{summary['always_up']:9.2f}% | "
            f"{summary['previous_day']:12.2f}% | "
            f"{'YES' if summary['beats_both'] else 'NO':10}"
        )

    validated = save_fx_validation(summaries, years, horizon, test_window)
    print(f"\nValidated pairs: {len(validated)}/{len(summaries)}")
    print(f"Validated: {', '.join(validated) if validated else 'none'}")
    return summaries

def analyze_ticker(ticker: str, days: int, news_dir: str='data/news', news_years: int=5, news_cutoff_hour: int=16, auto_historical_news: bool=False, news_cache_hours: float=6, refresh_news: bool=False, show_headlines: bool=False, asset_type: str|None=None):
    """Run live analysis and never substitute an unvalidated asset-class model."""
    ticker = ticker.upper()
    if auto_historical_news:
        try:
            ensure_news_data(ticker, news_dir, news_years, news_cutoff_hour)
        except ValueError:
            pass

    context = fetch_wikidata_context(ticker)
    hist = fetch_price_data(ticker, days)
    technical_score, technical_explanation = compute_technical_signal(hist)
    headlines = fetch_news(ticker, news_dir=news_dir, cache_hours=news_cache_hours, refresh=refresh_news)
    news_score, matched_headlines = score_news_sentiment(headlines)
    macro_score, macro_explanation = fetch_macro_signal(os.getenv('FRED_API_KEY') or '')
    if asset_type is None:
        asset_type = 'currency' if ticker.endswith('=X') else 'equity'

    model_result = None
    global_model_result = None
    if asset_type == 'equity':
        global_model_result = predict_live_global_direction(
            ticker,
            years=max(news_years, 10),
            horizon=days_to_horizon(days),
        )
        model_result = global_model_result
        if model_result is None:
            model_result = predict_live_direction(
                ticker,
                news_years,
                days_to_horizon(days),
                news_dir,
            )
    elif asset_type == 'currency':
        model_result = predict_live_fx_direction(ticker, news_years, days_to_horizon(days))

    closes = list(hist['Close'])
    latest_price = closes[-1]
    momentum_pct = (latest_price - closes[0]) / closes[0] * 100

    if technical_score > 0:
        trend = 'Bullish'
    elif technical_score < 0:
        trend = 'Bearish'
    else:
        trend = 'Neutral'

    if macro_explanation.startswith('Supportive'):
        macro = 'Supportive'
    elif macro_explanation.startswith('Restrictive'):
        macro = 'Restrictive'
    elif macro_explanation.startswith('Mixed'):
        macro = 'Mixed'
    else:
        macro = 'Unavailable'

    print(f'\n{ticker}')
    if context and context.get('company'):
        print(f"Company: {context['company']}")
    print(f'Price: {latest_price:.2f}')
    print(f'Trend: {trend} | Momentum: {momentum_pct:+.2f}%')
    print(f'Recent news: {len(headlines)} headlines')
    print(f'Macro: {macro}')
    print(f'Macro context: {macro_explanation}')

    if model_result:
        probability_up = model_result['probability_up']
        probability_down = 1 - probability_up
        if global_model_result is not None:
            print(f"Prediction: {model_result['direction']} | Global Cross-Stock ML")
            print(f"Probability of UP: {probability_up:.0%}")
            print(f"Probability of DOWN: {probability_down:.0%}")
            print(
                f"Global model context: {model_result.get('country', 'Unknown')} | "
                f"{model_result.get('region', 'Unknown')} | "
                f"Benchmark: {model_result.get('benchmark', '^GSPC')}"
            )
            if model_result.get('model_test_balanced_accuracy') is not None:
                print(
                    f"Global model test balanced accuracy: "
                    f"{model_result['model_test_balanced_accuracy']:.2f}%"
                )
            print(f"Global model training rows: {model_result.get('training_rows', 'unknown'):,}")
        else:
            print(f"Prediction: {model_result['direction']} | Technical + Macro ML")
            print(f"Probability of UP: {probability_up:.0%}")
            print(f"Probability of DOWN: {probability_down:.0%}")
            print(f"Macro features used: {', '.join(model_result.get('macro_features_used', []))}")
    elif asset_type == 'currency':
        print('Prediction: UNAVAILABLE | Model: no validated FX model')
    else:
        heuristic_lean, heuristic_confidence = make_prediction(technical_score, news_score)
        print('Prediction: UNAVAILABLE | Model: unavailable')
        print(f'Heuristic lean: {heuristic_lean} | Confidence: {heuristic_confidence}')
        print('Heuristic is a fallback signal only; it is not the validated ML prediction.')

    if show_headlines and headlines:
        context_matches = find_context_matches(headlines, context) if context else []
        if context_matches:
            print('\nContext-matched headlines:')
            for headline in context_matches:
                print(f'- {headline}')
        print('\nRecent headlines:')
        for headline in headlines:
            print(f'- {headline}')

def days_to_horizon(days: int) -> int:
    """Keep the live forecast horizon fixed at five trading days, matching the validated research target."""
    return 5

def test_asset_resolver():
    currency_cases = [
        ("Algerian currency", "DZD=X"),
        ("Tunisian currency", "TND=X"),
        ("British currency", "GBP=X"),
        ("Japanese currency", "JPY=X"),
        ("Swiss currency", "CHF=X"),
        ("Canadian currency", "CAD=X"),
        ("Australian currency", "AUD=X"),
        ("New Zealand currency", "NZD=X"),
        ("Norwegian currency", "NOK=X"),
        ("Swedish currency", "SEK=X"),
        ("Danish currency", "DKK=X"),
        ("Finnish currency", "EUR=X"),
        ("Polish currency", "PLN=X"),
        ("Czech currency", "CZK=X"),
        ("Hungarian currency", "HUF=X"),
        ("Romanian currency", "RON=X"),
        ("Bulgarian currency", "BGN=X"),
        ("Croatian currency", "EUR=X"),
    ]
    stock_cases = [
        ("Apple stock", "AAPL"),
        ("Microsoft stock", "MSFT"),
        ("Amazon stock", "AMZN"),
        ("NVIDIA stock", "NVDA"),
        ("Alphabet stock", "GOOGL"),
        ("Meta stock", "META"),
        ("Tesla stock", "TSLA"),
        ("Berkshire Hathaway stock", "BRK-B"),
        ("JPMorgan stock", "JPM"),
        ("Coca-Cola stock", "KO"),
    ]
    failures = []
    currency_pass = 0
    stock_pass = 0

    print("\n=== ASSET RESOLVER TEST ===")
    print("\nCurrencies")
    for query, expected in currency_cases:
        try:
            _, results = resolve_asset_query(query)
            symbols: list[str] = [str(item.get("symbol")) for item in results if item.get("symbol") is not None]
            ok = expected in symbols
            print(f"{query:28} -> {', '.join(symbols) if symbols else 'NONE':12} {'PASS' if ok else 'FAIL'}")
            if ok:
                currency_pass += 1
            else:
                failures.append((query, expected, symbols))
        except Exception as error:
            print(f"{query:28} -> ERROR       FAIL")
            failures.append((query, expected, str(error)))

    print("\nStocks")
    for query, expected in stock_cases:
        try:
            _, results = resolve_asset_query(query)
            symbols: list[str] = [str(item.get("symbol")) for item in results if item.get("symbol") is not None]
            ok = expected in symbols
            print(f"{query:28} -> {', '.join(symbols) if symbols else 'NONE':12} {'PASS' if ok else 'FAIL'}")
            if ok:
                stock_pass += 1
            else:
                failures.append((query, expected, symbols))
        except Exception as error:
            print(f"{query:28} -> ERROR       FAIL")
            failures.append((query, expected, str(error)))

    print(f"\nCurrency tests: {currency_pass}/{len(currency_cases)}")
    print(f"Stock tests: {stock_pass}/{len(stock_cases)}")
    print(f"Total: {currency_pass + stock_pass}/{len(currency_cases) + len(stock_cases)}")
    if failures:
        print("Failures:")
        for query, expected, actual in failures:
            print(f"- {query}: expected {expected}, got {actual}")
    else:
        print("All resolver tests passed.")


# ============================================================================
# GLOBAL CROSS-STOCK LEARNING SYSTEM
# ============================================================================
# This section turns the program from a single-stock learner into a global
# multi-market training system. It keeps the existing single-stock research
# tools above, but adds:
#   1. A diversified global equity universe.
#   2. Global market / FX / commodity context.
#   3. A cross-stock training panel.
#   4. Chronological train/validation/test evaluation.
#   5. A scalable SGD logistic model for large panels.
#   6. Live prediction using the trained global model.
#
# The model deliberately uses normalized/relative features instead of raw
# stock prices, so AAPL, SAP, Toyota, Reliance, etc. can be learned together.
# ============================================================================

import pickle
import numpy as np

GLOBAL_V2_DIR = os.path.join("data", "global")
GLOBAL_V2_STOCK_DIR = os.path.join(GLOBAL_V2_DIR, "stocks")
GLOBAL_V2_CONTEXT_DIR = os.path.join(GLOBAL_V2_DIR, "context")
GLOBAL_V2_UNIVERSE = os.path.join(GLOBAL_V2_DIR, "universe.csv")
GLOBAL_V2_PANEL = os.path.join(GLOBAL_V2_DIR, "training_panel.csv.gz")
GLOBAL_V2_MODEL = os.path.join(GLOBAL_V2_DIR, "global_model.pkl")

GLOBAL_INDEX_SOURCES = [
    {
        "name": "S&P 500",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "country": "USA",
        "region": "North America",
        "benchmark": "^GSPC",
        "suffix": "",
        "columns": ["Symbol", "Ticker"],
    },
    {
        "name": "FTSE 100",
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "country": "United Kingdom",
        "region": "Europe",
        "benchmark": "^FTSE",
        "suffix": ".L",
        "columns": ["Ticker", "EPIC", "Symbol"],
    },
    {
        "name": "DAX",
        "url": "https://en.wikipedia.org/wiki/DAX",
        "country": "Germany",
        "region": "Europe",
        "benchmark": "^GDAXI",
        "suffix": ".DE",
        "columns": ["Ticker", "Ticker symbol", "Symbol"],
    },
    {
        "name": "CAC 40",
        "url": "https://en.wikipedia.org/wiki/CAC_40",
        "country": "France",
        "region": "Europe",
        "benchmark": "^FCHI",
        "suffix": ".PA",
        "columns": ["Ticker", "Ticker symbol", "Symbol"],
    },
    {
        "name": "Nikkei 225",
        "url": "https://en.wikipedia.org/wiki/Nikkei_225",
        "country": "Japan",
        "region": "Asia-Pacific",
        "benchmark": "^N225",
        "suffix": ".T",
        "columns": ["Code", "Ticker", "Symbol"],
    },
    {
        "name": "Hang Seng",
        "url": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "country": "Hong Kong",
        "region": "Asia-Pacific",
        "benchmark": "^HSI",
        "suffix": ".HK",
        "columns": ["Ticker", "Ticker code", "Code", "Symbol"],
    },
    {
        "name": "ASX 200",
        "url": "https://en.wikipedia.org/wiki/S%26P/ASX_200",
        "country": "Australia",
        "region": "Asia-Pacific",
        "benchmark": "^AXJO",
        "suffix": ".AX",
        "columns": ["Code", "Ticker", "Symbol"],
    },
    {
        "name": "NIFTY 50",
        "url": "https://en.wikipedia.org/wiki/NIFTY_50",
        "country": "India",
        "region": "Asia-Pacific",
        "benchmark": "^NSEI",
        "suffix": ".NS",
        "columns": ["Symbol", "Ticker", "Code"],
    },
    {
        "name": "S&P/TSX 60",
        "url": "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        "country": "Canada",
        "region": "North America",
        "benchmark": "^GSPTSE",
        "suffix": ".TO",
        "columns": ["Symbol", "Ticker", "Code"],
    },
    {
        "name": "IBEX 35",
        "url": "https://en.wikipedia.org/wiki/IBEX_35",
        "country": "Spain",
        "region": "Europe",
        "benchmark": "^IBEX",
        "suffix": ".MC",
        "columns": ["Ticker", "Symbol", "Code"],
    },
    {
        "name": "AEX",
        "url": "https://en.wikipedia.org/wiki/AEX_index",
        "country": "Netherlands",
        "region": "Europe",
        "benchmark": "^AEX",
        "suffix": ".AS",
        "columns": ["Ticker", "Symbol", "Code"],
    },
]

GLOBAL_CONTEXT_SYMBOLS = {
    "^GSPC": "us_sp500",
    "^IXIC": "us_nasdaq",
    "^VIX": "us_vix",
    "^FTSE": "uk_ftse",
    "^GDAXI": "germany_dax",
    "^FCHI": "france_cac",
    "^N225": "japan_nikkei",
    "^HSI": "hong_kong_hsi",
    "^AXJO": "australia_asx",
    "^NSEI": "india_nifty",
    "^GSPTSE": "canada_tsx",
    "^IBEX": "spain_ibex",
    "^AEX": "netherlands_aex",
    "GC=F": "gold",
    "CL=F": "wti_oil",
    "BZ=F": "brent_oil",
    "EURUSD=X": "eur_usd",
    "GBPUSD=X": "gbp_usd",
    "JPY=X": "usd_jpy",
    "CNY=X": "usd_cny",
    "AUDUSD=X": "aud_usd",
    "CADUSD=X": "cad_usd",
    "CHFUSD=X": "chf_usd",
}

GLOBAL_FALLBACK_UNIVERSE = [
    ("AAPL", "USA", "North America", "^GSPC"),
    ("MSFT", "USA", "North America", "^GSPC"),
    ("NVDA", "USA", "North America", "^GSPC"),
    ("AMZN", "USA", "North America", "^GSPC"),
    ("GOOGL", "USA", "North America", "^GSPC"),
    ("META", "USA", "North America", "^GSPC"),
    ("TSLA", "USA", "North America", "^GSPC"),
    ("JPM", "USA", "North America", "^GSPC"),
    ("V", "USA", "North America", "^GSPC"),
    ("WMT", "USA", "North America", "^GSPC"),
    ("JNJ", "USA", "North America", "^GSPC"),
    ("XOM", "USA", "North America", "^GSPC"),
    ("AVGO", "USA", "North America", "^GSPC"),
    ("ORCL", "USA", "North America", "^GSPC"),
    ("NFLX", "USA", "North America", "^GSPC"),
    ("LLY", "USA", "North America", "^GSPC"),
    ("SAP.DE", "Germany", "Europe", "^GDAXI"),
    ("SIE.DE", "Germany", "Europe", "^GDAXI"),
    ("ALV.DE", "Germany", "Europe", "^GDAXI"),
    ("AIR.PA", "France", "Europe", "^FCHI"),
    ("OR.PA", "France", "Europe", "^FCHI"),
    ("MC.PA", "France", "Europe", "^FCHI"),
    ("ASML.AS", "Netherlands", "Europe", "^AEX"),
    ("NESN.SW", "Switzerland", "Europe", "^STOXX"),
    ("AZN.L", "United Kingdom", "Europe", "^FTSE"),
    ("SHEL.L", "United Kingdom", "Europe", "^FTSE"),
    ("HSBA.L", "United Kingdom", "Europe", "^FTSE"),
    ("7203.T", "Japan", "Asia-Pacific", "^N225"),
    ("6758.T", "Japan", "Asia-Pacific", "^N225"),
    ("9984.T", "Japan", "Asia-Pacific", "^N225"),
    ("005930.KS", "South Korea", "Asia-Pacific", "^KS11"),
    ("0700.HK", "Hong Kong", "Asia-Pacific", "^HSI"),
    ("0005.HK", "Hong Kong", "Asia-Pacific", "^HSI"),
    ("BHP.AX", "Australia", "Asia-Pacific", "^AXJO"),
    ("CBA.AX", "Australia", "Asia-Pacific", "^AXJO"),
    ("RELIANCE.NS", "India", "Asia-Pacific", "^NSEI"),
    ("TCS.NS", "India", "Asia-Pacific", "^NSEI"),
    ("RY.TO", "Canada", "North America", "^GSPTSE"),
    ("SHOP.TO", "Canada", "North America", "^GSPTSE"),
]


def _flatten_table_columns(table):
    if isinstance(table.columns, pd.MultiIndex):
        table = table.copy()
        table.columns = [
            " ".join(str(part) for part in col if str(part).lower() != "nan").strip()
            for col in table.columns
        ]
    else:
        table = table.copy()
        table.columns = [str(col).strip() for col in table.columns]
    return table


def _find_symbol_column(table, candidates):
    lowered = {str(col).strip().lower(): col for col in table.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for col in table.columns:
        name = str(col).lower()
        if any(candidate.lower() in name for candidate in candidates):
            return col
    return None


def _normalize_global_symbol(raw_symbol, suffix):
    value = str(raw_symbol).strip()
    value = re.sub(r"\[[^\]]+\]", "", value)
    value = value.replace("\xa0", " ").strip()
    value = re.sub(r"\s+", "", value)
    if not value or value.lower() in {"nan", "none", "n/a", "-"}:
        return None

    # Wikipedia's US tickers often use BRK.B / BF.B; Yahoo uses BRK-B / BF-B.
    if suffix == "":
        value = value.replace(".", "-")
    # Numeric Asian codes need their exchange-specific zero padding.
    if suffix in {".T", ".HK"} and value.isdigit():
        value = value.zfill(4)

    # Avoid accepting index labels or obvious non-tickers.
    if len(value) > 18 or any(ch in value for ch in (" ", "/", "\\", ";", ",")):
        return None

    if suffix and "." not in value:
        value = value + suffix
    return value.upper()


def _scrape_global_index_source(source):
    request = Request(
        source["url"],
        headers={
            "User-Agent": "Mozilla/5.0 stock-predictor/1.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")

    tables = pd.read_html(StringIO(html))
    best = []
    for table in tables:
        table = _flatten_table_columns(table)
        column = _find_symbol_column(table, source["columns"])
        if column is None:
            continue
        symbols = []
        for raw in table[column].tolist():
            symbol = _normalize_global_symbol(raw, source["suffix"])
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        if len(symbols) > len(best):
            best = symbols

    return [
        {
            "symbol": symbol,
            "country": source["country"],
            "region": source["region"],
            "benchmark": source["benchmark"],
            "index": source["name"],
        }
        for symbol in best
    ]


def load_global_stock_universe(limit=500, refresh=False):
    """Build a diversified current global equity universe from major indices."""
    os.makedirs(GLOBAL_V2_DIR, exist_ok=True)

    if os.path.exists(GLOBAL_V2_UNIVERSE) and not refresh:
        cached = pd.read_csv(GLOBAL_V2_UNIVERSE)
        if not cached.empty and len(cached) >= min(limit, 50):
            cached = cached.drop_duplicates("symbol").head(limit)
            return cached

    source_results = []
    failures = []
    for source in GLOBAL_INDEX_SOURCES:
        try:
            rows = _scrape_global_index_source(source)
            if rows:
                print(f"[UNIVERSE] {source['name']}: {len(rows)} symbols")
                source_results.append(rows)
            else:
                failures.append(source["name"])
                print(f"[UNIVERSE] {source['name']}: no symbols found")
        except Exception as error:
            failures.append(source["name"])
            print(f"[UNIVERSE] {source['name']}: failed ({error})")

    selected = []
    seen = set()
    # Round-robin across indices so a 500-stock universe does not become
    # overwhelmingly US-centric simply because S&P 500 is the first source.
    while len(selected) < limit and any(source_results):
        progressed = False
        for rows in list(source_results):
            while rows:
                row = rows.pop(0)
                symbol = row["symbol"]
                if symbol in seen:
                    continue
                seen.add(symbol)
                selected.append(row)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break

    if len(selected) < min(limit, 50):
        print("[UNIVERSE] Using built-in fallback symbols for missing coverage.")
        for symbol, country, region, benchmark in GLOBAL_FALLBACK_UNIVERSE:
            if symbol not in seen and len(selected) < limit:
                selected.append(
                    {
                        "symbol": symbol,
                        "country": country,
                        "region": region,
                        "benchmark": benchmark,
                        "index": "fallback",
                    }
                )
                seen.add(symbol)

    universe = pd.DataFrame(selected).drop_duplicates("symbol").head(limit)
    if universe.empty:
        raise ValueError("Could not construct a global equity universe.")

    universe.to_csv(GLOBAL_V2_UNIVERSE, index=False)
    print(f"[UNIVERSE] Saved {len(universe)} global equities to {GLOBAL_V2_UNIVERSE}")
    if failures:
        print("[UNIVERSE] Sources needing attention:", ", ".join(failures))
    print("[UNIVERSE] Countries represented:", universe["country"].nunique())
    print("[UNIVERSE] Regions represented:", universe["region"].nunique())
    return universe


def _safe_global_filename(symbol):
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", str(symbol))


def _download_yahoo_chunk(symbols, start, end):
    return yf.download(
        tickers=symbols,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )


def download_global_market_data(
    start="2015-01-01",
    end=None,
    limit=500,
    refresh_universe=False,
    chunk_size=25,
):
    """Download a diversified global stock universe plus global context."""
    end = end or date.today().isoformat()
    universe = load_global_stock_universe(limit=limit, refresh=refresh_universe)
    os.makedirs(GLOBAL_V2_STOCK_DIR, exist_ok=True)
    os.makedirs(GLOBAL_V2_CONTEXT_DIR, exist_ok=True)

    print(f"\n=== GLOBAL DATA DOWNLOAD ===")
    print(f"Universe: {len(universe)} equities")
    print(f"Period: {start} to {end}")
    print(f"Stocks directory: {GLOBAL_V2_STOCK_DIR}")

    downloaded = 0
    failed = []

    for chunk_start in range(0, len(universe), chunk_size):
        chunk = universe.iloc[chunk_start:chunk_start + chunk_size]
        symbols = chunk["symbol"].tolist()
        print(
            f"\nDownloading stock batch "
            f"{chunk_start + 1}-{chunk_start + len(symbols)} / {len(universe)}"
        )
        try:
            data = _download_yahoo_chunk(symbols, start, end)
        except Exception as error:
            print(f"  Batch failed: {error}")
            failed.extend(symbols)
            continue

        if data is None or data.empty:
            failed.extend(symbols)
            continue

        for symbol in symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    available = set(data.columns.get_level_values(0))
                    if symbol not in available:
                        failed.append(symbol)
                        continue
                    frame = data[symbol].dropna(how="all")
                else:
                    frame = data.dropna(how="all") if len(symbols) == 1 else pd.DataFrame()

                if frame.empty or "Close" not in frame.columns:
                    failed.append(symbol)
                    continue

                path = os.path.join(
                    GLOBAL_V2_STOCK_DIR,
                    f"{_safe_global_filename(symbol)}.csv",
                )
                frame.to_csv(path)
                downloaded += 1
                print(f"  [OK] {symbol}: {len(frame)} rows")
            except Exception as error:
                failed.append(symbol)
                print(f"  [FAILED] {symbol}: {error}")

    print(f"\nStock files downloaded: {downloaded}/{len(universe)}")
    if failed:
        print(f"Stock symbols without usable files: {len(set(failed))}")

    # Global context is shared by every company; it is downloaded once.
    print("\nDownloading global market / FX / commodity context...")
    for symbol in GLOBAL_CONTEXT_SYMBOLS:
        try:
            history = cached_ticker(symbol).history(
                start=start,
                end=end,
                auto_adjust=False,
            )
            if history.empty:
                print(f"  [FAILED] context {symbol}")
                continue
            path = os.path.join(
                GLOBAL_V2_CONTEXT_DIR,
                f"{_safe_global_filename(symbol)}.csv",
            )
            history.to_csv(path)
            print(f"  [OK] context {symbol}: {len(history)} rows")
        except Exception as error:
            print(f"  [FAILED] context {symbol}: {error}")

    print(f"\nUniverse metadata: {GLOBAL_V2_UNIVERSE}")
    print(f"Stock data: {GLOBAL_V2_STOCK_DIR}")
    print(f"Global context: {GLOBAL_V2_CONTEXT_DIR}")
    return universe


def _local_trading_date(values):
    """Parse Yahoo timestamps to the exchange's LOCAL calendar date.

    Yahoo CSVs store stamps like '2020-03-02 00:00:00+01:00'. Letting pandas
    convert them to UTC turns midnight Berlin into 23:00 of the PREVIOUS day,
    which shifted every European/Asian series one day into the future.
    Taking the first 10 characters keeps the exchange's own date.
    """
    text = pd.Series(values).astype(str).str.strip().str[:10]
    return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")


def _load_global_context_frame(start=None):
    """Create one date-indexed table of global market, FX and commodity returns.

    Prices are aligned on the combined calendar and forward-filled FIRST, and
    returns are computed afterwards. That way a market that is closed on a
    given day shows a 0 return, instead of repeating yesterday's return
    (which is what ffill() after pct_change() did).
    """
    closes = {}

    for symbol, prefix in GLOBAL_CONTEXT_SYMBOLS.items():
        path = os.path.join(
            GLOBAL_V2_CONTEXT_DIR,
            f"{_safe_global_filename(symbol)}.csv",
        )
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_csv(path)
            date_column = next(
                (col for col in raw.columns if str(col).lower() in {"date", "datetime"}),
                raw.columns[0],
            )
            close_column = "Close" if "Close" in raw.columns else raw.columns[-1]
            frame = pd.DataFrame(
                {
                    "date": _local_trading_date(raw[date_column]).to_numpy(),
                    "close": pd.to_numeric(raw[close_column], errors="coerce"),
                }
            ).dropna()
            if frame.empty:
                continue
            frame = frame.drop_duplicates("date").set_index("date").sort_index()
            closes[prefix] = frame["close"]
        except Exception as error:
            print(f"[CONTEXT] Could not process {symbol}: {error}")

    if not closes:
        return pd.DataFrame()
    prices = pd.DataFrame(closes).sort_index().ffill()
    columns = {}
    for prefix in prices.columns:
        for n in (1, 5, 20):
            columns[f"{prefix}_return_{n}"] = prices[prefix].pct_change(n, fill_method=None)
    context = pd.DataFrame(columns, index=prices.index)
    if start is not None:
        context = context.loc[context.index >= pd.Timestamp(start)]
    return context


def _read_global_stock_file(path):
    raw = pd.read_csv(path)
    date_column = next(
        (col for col in raw.columns if str(col).lower() in {"date", "datetime"}),
        raw.columns[0],
    )
    required = [column for column in ("Close", "Volume", "High", "Low") if column in raw.columns]
    if "Close" not in required:
        return pd.DataFrame()
    frame = raw[[date_column] + required].copy()
    frame["date"] = _local_trading_date(frame.pop(date_column)).to_numpy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "Close"]).drop_duplicates("date").sort_values("date")


def _global_stock_features(frame, horizon):
    """Create scale-free features that can be learned jointly across companies."""
    data = frame.copy().set_index("date")
    close = data["Close"]
    volume = data["Volume"] if "Volume" in data else pd.Series(index=data.index, dtype=float)

    data["return_1"] = close.pct_change(1)
    data["return_5"] = close.pct_change(5)
    data["return_20"] = close.pct_change(20)
    data["return_60"] = close.pct_change(60)

    for window in (5, 20, 60):
        data[f"sma_{window}_ratio"] = close / close.rolling(window).mean() - 1

    data["ema_12_ratio"] = close / close.ewm(span=12, adjust=False).mean() - 1
    data["ema_26_ratio"] = close / close.ewm(span=26, adjust=False).mean() - 1

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    data["rsi_14"] = 100 - 100 / (1 + rs)

    data["volatility_20"] = data["return_1"].rolling(20).std()
    data["volatility_60"] = data["return_1"].rolling(60).std()

    if "High" in data and "Low" in data:
        data["range_1"] = (data["High"] - data["Low"]) / close.replace(0, np.nan)
    else:
        data["range_1"] = np.nan

    data["volume_change_1"] = volume.pct_change()
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std().replace(0, np.nan)
    data["volume_z_20"] = (volume - volume_mean) / volume_std

    rolling_high = close.rolling(60).max()
    data["drawdown_60"] = close / rolling_high - 1

    data["future_return"] = close.shift(-horizon) / close - 1
    data["target"] = (data["future_return"] > 0).astype(int)

    return data.replace([np.inf, -np.inf], np.nan)


GLOBAL_STOCK_FEATURES = [
    "return_1",
    "return_5",
    "return_20",
    "return_60",
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_60_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "rsi_14",
    "volatility_20",
    "volatility_60",
    "range_1",
    "volume_change_1",
    "volume_z_20",
    "drawdown_60",
]

GLOBAL_CONTEXT_FEATURES = [
    column
    for prefix in GLOBAL_CONTEXT_SYMBOLS.values()
    for column in (
        f"{prefix}_return_1",
        f"{prefix}_return_5",
        f"{prefix}_return_20",
    )
]

GLOBAL_MODEL_VERSION = 1


def _attach_global_features(
    stock_features,
    metadata,
    context,
    macro_cache_dir="data/macro",
):
    data = stock_features.copy()

    benchmark = metadata.get("benchmark", "^GSPC")
    benchmark_prefix = GLOBAL_CONTEXT_SYMBOLS.get(benchmark)
    data["benchmark_return_20"] = np.nan
    data["benchmark_return_1"] = np.nan

    if not context.empty:
        # The stock's OWN benchmark closes at the same time as the stock, so
        # its same-day value is known at the close and stays unlagged.
        if benchmark_prefix:
            for n in (1, 20):
                column = f"{benchmark_prefix}_return_{n}"
                if column in context:
                    data[f"benchmark_return_{n}"] = context[column].reindex(data.index)

        # Every other market may close AFTER this stock on the same date
        # (e.g. the S&P 500 closes after Tokyo and Frankfurt). Use the
        # previous day's values so no row sees a close that happens later.
        lagged = context.shift(1)
        data = data.join(lagged, how="left")

    data["relative_return_20"] = data["return_20"] - data["benchmark_return_20"]

    # Add the existing point-in-time FRED macro block when cached.
    fred_by_feature = {}
    for feature_name, series_id in FRED_SERIES.items():
        values = load_cached_fred_series(series_id, macro_cache_dir)
        if values:
            fred_by_feature[feature_name] = values

    if len(fred_by_feature) == len(FRED_SERIES):
        data = _add_macro_columns(data, fred_by_feature)

    return data


def build_global_training_dataset(
    horizon=5,
    start="2015-01-01",
    end=None,
    max_stocks=500,
    rebuild=False,
):
    """Build a large cross-stock panel one ticker at a time.

    The output is a compressed CSV so the complete panel does not need to live
    in RAM at once. Each row represents one company on one trading date.
    """
    end = end or date.today().isoformat()
    os.makedirs(GLOBAL_V2_DIR, exist_ok=True)

    if not os.path.exists(GLOBAL_V2_UNIVERSE):
        universe = load_global_stock_universe(max_stocks)
    else:
        universe = pd.read_csv(GLOBAL_V2_UNIVERSE).head(max_stocks)

    context = _load_global_context_frame(start)
    if context.empty:
        print("[GLOBAL PANEL] Warning: no global context files found.")
    else:
        print(f"[GLOBAL PANEL] Global context columns: {len(context.columns)}")

    # If the user has already downloaded the FRED block, include it.
    macro_complete = all(
        os.path.exists(os.path.join("data/macro", f"{series_id}.csv"))
        for series_id in FRED_SERIES.values()
    )
    if not macro_complete:
        print("[GLOBAL PANEL] Warning: complete FRED macro cache not found.")
        print("[GLOBAL PANEL] Run --download-macro if you want the full macro block.")

    if rebuild and os.path.exists(GLOBAL_V2_PANEL):
        os.remove(GLOBAL_V2_PANEL)

    if os.path.exists(GLOBAL_V2_PANEL) and not rebuild:
        print(f"[GLOBAL PANEL] Existing panel found: {GLOBAL_V2_PANEL}")
        print("[GLOBAL PANEL] Skipping rebuild to prevent duplicate rows.")
        metadata_path = os.path.join(GLOBAL_V2_DIR, "panel_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as file:
                return json.load(file)
        return {
            "model_version": GLOBAL_MODEL_VERSION,
            "panel": GLOBAL_V2_PANEL,
            "horizon": horizon,
        }

    panel_exists = False
    rows_written = 0
    stocks_used = 0
    feature_names = GLOBAL_STOCK_FEATURES + [
        "benchmark_return_1",
        "benchmark_return_20",
        "relative_return_20",
    ]
    if not context.empty:
        feature_names += [column for column in context.columns if column not in feature_names]
    if macro_complete:
        feature_names += [
            column for column in MACRO_FEATURE_NAMES
            if column not in feature_names
        ]

    # On a fresh build, write the header from the first usable stock.
    first_write = not panel_exists

    for _, metadata_row in universe.iterrows():
        symbol = str(metadata_row["symbol"])
        path = os.path.join(GLOBAL_V2_STOCK_DIR, f"{_safe_global_filename(symbol)}.csv")
        if not os.path.exists(path):
            continue

        try:
            history = _read_global_stock_file(path)
            if history.empty:
                continue
            history = history[
                (history["date"] >= pd.Timestamp(start))
                & (history["date"] <= pd.Timestamp(end))
            ]
            if len(history) < max(120, horizon + 61):
                continue

            features = _global_stock_features(history, horizon)
            features = _attach_global_features(
                features,
                metadata_row.to_dict(),
                context,
            )

            features = features.reset_index()
            features["ticker"] = symbol
            features["country"] = str(metadata_row.get("country", ""))
            features["region"] = str(metadata_row.get("region", ""))
            features["benchmark"] = str(metadata_row.get("benchmark", "^GSPC"))
            features = features.replace([np.inf, -np.inf], np.nan)

            required = [
                "date",
                "ticker",
                "country",
                "region",
                "benchmark",
                "future_return",
                "target",
            ] + feature_names
            available = [column for column in required if column in features.columns]
            features = features[available]
            features = features.dropna(
                subset=[column for column in feature_names if column in features.columns]
                + ["future_return"]
            )
            if features.empty:
                continue

            # Ensure a deterministic column order.
            features = features[
                ["date", "ticker", "country", "region", "benchmark", "future_return", "target"]
                + [column for column in feature_names if column in features.columns]
            ]

            features.to_csv(
                GLOBAL_V2_PANEL,
                mode="a",
                header=first_write,
                index=False,
                compression="gzip",
            )
            first_write = False
            rows_written += len(features)
            stocks_used += 1
            print(
                f"[GLOBAL PANEL] {symbol}: {len(features):,} rows | "
                f"cumulative {rows_written:,}"
            )
        except Exception as error:
            print(f"[GLOBAL PANEL] {symbol}: skipped ({error})")

    if first_write:
        raise ValueError(
            "No training rows were created. Download global stock data first."
        )

    metadata = {
        "model_version": GLOBAL_MODEL_VERSION,
        "horizon": horizon,
        "start": start,
        "end": end,
        "stocks_requested": int(len(universe)),
        "stocks_used": int(stocks_used),
        "rows_written": int(rows_written),
        "feature_names": feature_names,
        "panel": GLOBAL_V2_PANEL,
    }
    with open(os.path.join(GLOBAL_V2_DIR, "panel_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n=== GLOBAL PANEL COMPLETE ===")
    print(f"Stocks used: {stocks_used}")
    print(f"Rows written this run: {rows_written:,}")
    print(f"Panel: {GLOBAL_V2_PANEL}")
    return metadata


def _global_panel_dates(panel_path=GLOBAL_V2_PANEL):
    min_date = None
    max_date = None
    for chunk in pd.read_csv(
        panel_path,
        compression="gzip",
        usecols=["date"],
        parse_dates=["date"],
        chunksize=100_000,
    ):
        if chunk.empty:
            continue
        chunk_min = chunk["date"].min()
        chunk_max = chunk["date"].max()
        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)
    if min_date is None or max_date is None:
        raise ValueError("Global training panel contains no dates.")
    return min_date, max_date


def _fit_global_scaler(panel_path, feature_names, train_cutoff):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    seen = 0
    for chunk in pd.read_csv(
        panel_path,
        compression="gzip",
        usecols=["date"] + feature_names,
        parse_dates=["date"],
        chunksize=50_000,
    ):
        train = chunk[chunk["date"] < train_cutoff]
        if train.empty:
            continue
        values = train[feature_names].to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1)
        if finite.any():
            scaler.partial_fit(values[finite])
            seen += int(finite.sum())
    if seen == 0:
        raise ValueError("No valid training rows available for the global scaler.")
    return scaler, seen


def _fit_global_classifier(
    panel_path,
    feature_names,
    scaler,
    train_end,
    train_start=None,
):
    from sklearn.linear_model import SGDClassifier

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=42,
        class_weight="balanced",
    )
    classes = np.array([0, 1], dtype=int)
    batches = 0
    rows = 0

    for chunk in pd.read_csv(
        panel_path,
        compression="gzip",
        usecols=["date", "target"] + feature_names,
        parse_dates=["date"],
        chunksize=50_000,
    ):
        mask = chunk["date"] < train_end
        if train_start is not None:
            mask &= chunk["date"] >= train_start
        train = chunk.loc[mask]
        if train.empty:
            continue

        values = train[feature_names].to_numpy(dtype=float)
        target = train["target"].to_numpy(dtype=int)
        finite = np.isfinite(values).all(axis=1) & np.isfinite(target)
        if not finite.any():
            continue

        classifier.partial_fit(
            scaler.transform(values[finite]),
            target[finite],
            classes=classes if batches == 0 else None,
        )
        batches += 1
        rows += int(finite.sum())

    if batches == 0:
        raise ValueError("No valid batches available to fit the global classifier.")
    return classifier, rows


def _evaluate_global_classifier(
    panel_path,
    feature_names,
    scaler,
    classifier,
    start_date,
    end_date,
):
    actual = []
    predicted = []
    probabilities = []

    for chunk in pd.read_csv(
        panel_path,
        compression="gzip",
        usecols=["date", "target"] + feature_names,
        parse_dates=["date"],
        chunksize=50_000,
    ):
        mask = chunk["date"] >= start_date
        if end_date is not None:
            mask &= chunk["date"] < end_date
        frame = chunk.loc[mask]
        if frame.empty:
            continue

        values = frame[feature_names].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=int)
        finite = np.isfinite(values).all(axis=1) & np.isfinite(target)
        if not finite.any():
            continue

        transformed = scaler.transform(values[finite])
        predicted.extend(classifier.predict(transformed))
        probabilities.extend(classifier.predict_proba(transformed)[:, 1])
        actual.extend(target[finite])

    if not actual:
        raise ValueError("No evaluation rows found for the requested period.")

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    probabilities = np.asarray(probabilities)
    always_up = np.ones_like(actual)
    previous_placeholder = None

    return {
        "rows": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted) * 100),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted) * 100),
        "precision": float(precision_score(actual, predicted, zero_division=0) * 100),
        "recall": float(recall_score(actual, predicted, zero_division=0) * 100),
        "always_up": float(accuracy_score(actual, always_up) * 100),
        "mean_probability_up": float(probabilities.mean()),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
    }


def train_global_model(
    horizon=5,
    train_fraction=0.60,
    validation_fraction=0.20,
):
    """Train and evaluate the cross-stock model using chronological splits."""
    if not os.path.exists(GLOBAL_V2_PANEL):
        raise ValueError(
            f"Global training panel not found: {GLOBAL_V2_PANEL}. "
            "Run --build-global-dataset first."
        )

    with open(os.path.join(GLOBAL_V2_DIR, "panel_metadata.json"), encoding="utf-8") as file:
        panel_metadata = json.load(file)

    feature_names = panel_metadata["feature_names"]
    min_date, max_date = _global_panel_dates()
    total_days = (max_date - min_date).days
    train_cutoff = min_date + timedelta(days=total_days * train_fraction)
    validation_cutoff = min_date + timedelta(
        days=total_days * (train_fraction + validation_fraction)
    )

    print("\n=== GLOBAL CROSS-STOCK MODEL ===")
    print(f"Panel period: {min_date.date()} to {max_date.date()}")
    print(f"Train: before {train_cutoff.date()}")
    print(f"Validation: {train_cutoff.date()} to {validation_cutoff.date()}")
    print(f"Test: {validation_cutoff.date()} onward")
    print(f"Features: {len(feature_names)}")

    scaler, train_rows = _fit_global_scaler(
        GLOBAL_V2_PANEL,
        feature_names,
        train_cutoff,
    )
    classifier, fitted_rows = _fit_global_classifier(
        GLOBAL_V2_PANEL,
        feature_names,
        scaler,
        train_cutoff,
    )

    validation_metrics = _evaluate_global_classifier(
        GLOBAL_V2_PANEL,
        feature_names,
        scaler,
        classifier,
        train_cutoff,
        validation_cutoff,
    )

    # Refit from train + validation for the final test evaluation.
    final_scaler, development_rows = _fit_global_scaler(
        GLOBAL_V2_PANEL,
        feature_names,
        validation_cutoff,
    )
    final_classifier, development_fit_rows = _fit_global_classifier(
        GLOBAL_V2_PANEL,
        feature_names,
        final_scaler,
        validation_cutoff,
    )

    test_metrics = _evaluate_global_classifier(
        GLOBAL_V2_PANEL,
        feature_names,
        final_scaler,
        final_classifier,
        validation_cutoff,
        None,
    )

    payload = {
        "model_version": GLOBAL_MODEL_VERSION,
        "horizon": horizon,
        "feature_names": feature_names,
        "scaler": final_scaler,
        "classifier": final_classifier,
        "train_cutoff": train_cutoff.isoformat(),
        "validation_cutoff": validation_cutoff.isoformat(),
        "panel_start": min_date.isoformat(),
        "panel_end": max_date.isoformat(),
        "development_rows": int(development_fit_rows),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    with open(GLOBAL_V2_MODEL, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nTraining rows: {train_rows:,}")
    print(f"Validation rows: {validation_metrics['rows']:,}")
    print(
        f"Validation balanced accuracy: "
        f"{validation_metrics['balanced_accuracy']:.2f}%"
    )
    print(f"Test rows: {test_metrics['rows']:,}")
    print(
        f"Final test balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.2f}%"
    )
    print(f"Final test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Always-UP baseline: {test_metrics['always_up']:.2f}%")
    print(f"Model saved to: {GLOBAL_V2_MODEL}")

    return payload


def _load_global_model():
    if not os.path.exists(GLOBAL_V2_MODEL):
        return None
    with open(GLOBAL_V2_MODEL, "rb") as file:
        return pickle.load(file)


def predict_live_global_direction(ticker, years=10, horizon=5):
    """Use the trained cross-stock model to predict the requested equity."""
    payload = _load_global_model()
    if payload is None:
        return None

    ticker = ticker.upper()
    feature_names = payload["feature_names"]

    metadata = {}
    if os.path.exists(GLOBAL_V2_UNIVERSE):
        universe = pd.read_csv(GLOBAL_V2_UNIVERSE)
        match = universe[universe["symbol"].str.upper() == ticker]
        if not match.empty:
            metadata = match.iloc[0].to_dict()

    if not metadata:
        metadata = {
            "symbol": ticker,
            "country": "Unknown",
            "region": "Unknown",
            "benchmark": "^GSPC",
        }

    start = date.today() - timedelta(days=max(years, 10) * 365)
    try:
        history = cached_ticker(ticker).history(
            start=start.isoformat(),
            end=(date.today() + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
    except Exception:
        return None

    if history.empty or "Close" not in history.columns:
        return None

    frame = history.reset_index()
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
    frame = frame.dropna(subset=["date", "Close"]).drop_duplicates("date")
    frame = frame.sort_values("date")

    if len(frame) < 120:
        return None

    features = _global_stock_features(frame, horizon)
    context = _load_global_context_frame(start)
    features = _attach_global_features(features, metadata, context)
    features = features.replace([np.inf, -np.inf], np.nan)

    latest = features.iloc[[-1]].copy()
    if not all(name in latest.columns for name in feature_names):
        return None
    if latest[feature_names].isna().any(axis=None):
        return None

    scaler = payload["scaler"]
    classifier = payload["classifier"]
    transformed = scaler.transform(latest[feature_names].to_numpy(dtype=float))
    prediction = int(classifier.predict(transformed)[0])
    probability_up = float(classifier.predict_proba(transformed)[0, 1])

    return {
        "direction": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "features_used": feature_names,
        "training_rows": payload.get("development_rows"),
        "model_type": "global cross-stock SGD logistic",
        "country": metadata.get("country"),
        "region": metadata.get("region"),
        "benchmark": metadata.get("benchmark"),
        "model_test_balanced_accuracy": payload.get("test_metrics", {}).get("balanced_accuracy"),
    }


def global_data_status():
    """Print a concise status report before a long global-data run."""
    print("\n=== GLOBAL DATA STATUS ===")
    if os.path.exists(GLOBAL_V2_UNIVERSE):
        universe = pd.read_csv(GLOBAL_V2_UNIVERSE)
        print(f"Universe: {len(universe)} stocks")
        print(f"Countries: {universe['country'].nunique()}")
        print(f"Regions: {universe['region'].nunique()}")
    else:
        print("Universe: not created")

    if os.path.isdir(GLOBAL_V2_STOCK_DIR):
        stock_files = [
            name for name in os.listdir(GLOBAL_V2_STOCK_DIR)
            if name.endswith(".csv")
        ]
        print(f"Stock files: {len(stock_files)}")
    else:
        print("Stock files: 0")

    if os.path.isdir(GLOBAL_V2_CONTEXT_DIR):
        context_files = [
            name for name in os.listdir(GLOBAL_V2_CONTEXT_DIR)
            if name.endswith(".csv")
        ]
        print(f"Global context files: {len(context_files)}")
    else:
        print("Global context files: 0")

    print(f"Training panel: {'YES' if os.path.exists(GLOBAL_V2_PANEL) else 'NO'}")
    print(f"Trained global model: {'YES' if os.path.exists(GLOBAL_V2_MODEL) else 'NO'}")




def _normalize_news_text(value: str) -> str:
    """Normalize whitespace and Unicode punctuation without changing the meaning of a headline."""
    import unicodedata
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _news_raw_files(news_dir: str):
    """Return only raw ticker CSVs; never treat derived news files as raw input."""
    if not os.path.isdir(news_dir):
        return []
    excluded_suffixes = (
        "_processed.csv", "_daily.csv", "_gemini.csv", "_gemini_daily.csv"
    )
    return sorted(
        os.path.join(news_dir, name)
        for name in os.listdir(news_dir)
        if name.lower().endswith(".csv")
        and not name.lower().endswith(excluded_suffixes)
        and not name.lower().startswith("raw_")
    )


def _prune_gemini_cache_to_processed(ticker: str, news_dir: str):
    """Keep existing Gemini classifications only for headlines that survived news cleaning."""
    gemini_path = os.path.join(news_dir, f"{ticker.upper()}_gemini.csv")
    processed_path = os.path.join(news_dir, f"{ticker.upper()}_processed.csv")
    if not os.path.exists(gemini_path) or not os.path.exists(processed_path):
        return 0

    valid_keys = set()
    with open(processed_path, newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            valid_keys.add((
                row.get("published_at", "").strip(),
                _normalize_news_text(row.get("headline", "")).casefold(),
                _normalize_news_text(row.get("source", "")).casefold(),
            ))

    kept = []
    with open(gemini_path, newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            key = (
                row.get("published_at", "").strip(),
                _normalize_news_text(row.get("headline", "")).casefold(),
                _normalize_news_text(row.get("source", "")).casefold(),
            )
            if key in valid_keys:
                row["headline"] = _normalize_news_text(row.get("headline", ""))
                row["source"] = _normalize_news_text(row.get("source", ""))
                kept.append(row)

    fieldnames = ["ticker", "published_at", "headline", "source", "date", "sentiment", "relevance", "event_type", "confidence"]
    with open(gemini_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(kept, key=lambda row: row.get("published_at", "")))
    return len(kept)


def _rebuild_gemini_daily_cache(ticker: str, news_dir: str):
    """Rebuild the daily Gemini aggregate from the already classified headlines; no API calls."""
    gemini_path = os.path.join(news_dir, f"{ticker.upper()}_gemini.csv")
    daily_path = os.path.join(news_dir, f"{ticker.upper()}_gemini_daily.csv")
    if not os.path.exists(gemini_path):
        return False

    sentiment_values = {"positive": 1, "negative": -1, "neutral": 0}
    daily = {}
    with open(gemini_path, newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            day = (row.get("date") or "").strip()
            if not day:
                continue
            try:
                relevance = max(0.0, min(1.0, float(row.get("relevance") or 0)))
                confidence = max(0.0, min(1.0, float(row.get("confidence") or 0)))
            except ValueError:
                continue
            values = daily.setdefault(day, {
                "gemini_sentiment_score": 0.0,
                "gemini_relevance_sum": 0.0,
                "gemini_confidence_sum": 0.0,
                "gemini_news_count": 0,
            })
            values["gemini_sentiment_score"] += sentiment_values.get((row.get("sentiment") or "neutral").lower(), 0) * relevance
            values["gemini_relevance_sum"] += relevance
            values["gemini_confidence_sum"] += confidence
            values["gemini_news_count"] += 1

    with open(daily_path, "w", newline="", encoding="utf-8") as output_file:
        fields = ["date", "gemini_sentiment_score", "gemini_relevance", "gemini_confidence", "gemini_news_count"]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for day in sorted(daily):
            values = daily[day]
            count = values["gemini_news_count"]
            writer.writerow({
                "date": day,
                "gemini_sentiment_score": values["gemini_sentiment_score"],
                "gemini_relevance": values["gemini_relevance_sum"] / count if count else 0.0,
                "gemini_confidence": values["gemini_confidence_sum"] / count if count else 0.0,
                "gemini_news_count": count,
            })
    return True


def clean_news_data(news_dir: str = "data/news", cutoff_hour: int = 16):
    """Clean all raw historical news files while preserving originals in a backup folder.

    Cleaning is deliberately conservative: malformed rows, missing headlines/timestamps,
    invalid timestamps, and exact duplicate observations are removed. We do not remove
    legitimate headlines merely because they look similar, and we do not apply the
    market-time cutoff to the raw archive. The cutoff is applied when creating features.
    """
    if not os.path.isdir(news_dir):
        raise ValueError(f"News directory not found: {news_dir}")
    if not 0 <= cutoff_hour <= 23:
        raise ValueError("Cutoff hour must be between 0 and 23.")

    raw_files = _news_raw_files(news_dir)
    if not raw_files:
        print("\n=== NEWS DATA CLEAN ===")
        print("No raw ticker news CSV files found.")
        return

    backup_dir = os.path.join(news_dir, "raw_backup")
    os.makedirs(backup_dir, exist_ok=True)
    timezone = ZoneInfo("America/New_York")
    total_stats = Counter()

    print("\n=== NEWS DATA CLEAN ===")
    print(f"Raw files found: {len(raw_files)}")
    print(f"Backup: {backup_dir}")

    for input_path in raw_files:
        ticker = os.path.splitext(os.path.basename(input_path))[0].upper()
        backup_path = os.path.join(backup_dir, os.path.basename(input_path))
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(input_path, backup_path)

        cleaned = {}
        stats = Counter()
        with open(input_path, newline="", encoding="utf-8-sig", errors="replace") as input_file:
            reader = csv.DictReader(input_file)
            if not reader.fieldnames:
                print(f"[SKIP] {ticker}: no CSV header")
                continue
            for row in reader:
                stats["rows_read"] += 1
                published_text = _normalize_news_text(row.get("published_at", ""))
                headline = _normalize_news_text(row.get("headline", ""))
                source = _normalize_news_text(row.get("source", ""))
                if not published_text or not headline:
                    stats["missing_required"] += 1
                    continue
                parsed = None
                for fmt in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        parsed = datetime.strptime(published_text, fmt)
                        break
                    except ValueError:
                        pass
                if parsed is None:
                    stats["invalid_timestamp"] += 1
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    parsed = parsed.astimezone(ZoneInfo("UTC"))
                canonical_timestamp = parsed.strftime("%Y%m%dT%H%M%S")
                # Exact observation identity: timestamp + normalized headline + normalized source.
                key = (canonical_timestamp, headline.casefold(), source.casefold())
                if key in cleaned:
                    stats["duplicates_removed"] += 1
                    continue
                cleaned[key] = {
                    "ticker": ticker,
                    "published_at": canonical_timestamp,
                    "headline": headline,
                    "source": source,
                }

        with open(input_path, "w", newline="", encoding="utf-8") as output_file:
            fields = ["ticker", "published_at", "headline", "source"]
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(cleaned.values(), key=lambda row: row["published_at"]))

        # Rebuild the deterministic daily control from the cleaned raw archive.
        process_historical_news(ticker, news_dir, cutoff_hour)
        gemini_kept = _prune_gemini_cache_to_processed(ticker, news_dir)
        _rebuild_gemini_daily_cache(ticker, news_dir)

        stats["rows_written"] = len(cleaned)
        stats["processed_rows"] = sum(1 for _ in csv.DictReader(open(os.path.join(news_dir, f"{ticker}_processed.csv"), encoding="utf-8")))
        stats["gemini_rows_kept"] = gemini_kept
        for key, value in stats.items():
            total_stats[key] += value
        print(
            f"[OK] {ticker}: raw {stats['rows_read']} -> {stats['rows_written']} | "
            f"duplicates={stats['duplicates_removed']} | missing={stats['missing_required']} | "
            f"bad_timestamp={stats['invalid_timestamp']} | processed={stats['processed_rows']}"
        )

    print("\n=== NEWS CLEAN COMPLETE ===")
    print(f"Raw rows read: {total_stats['rows_read']:,}")
    print(f"Clean raw rows kept: {total_stats['rows_written']:,}")
    print(f"Exact duplicate rows removed: {total_stats['duplicates_removed']:,}")
    print(f"Rows missing timestamp/headline removed: {total_stats['missing_required']:,}")
    print(f"Invalid timestamps removed: {total_stats['invalid_timestamp']:,}")
    print(f"Backup of original raw files: {backup_dir}")
    print("Derived *_processed.csv and *_daily.csv files were rebuilt from the cleaned raw archive.")
    print("Existing Gemini classifications were only pruned/re-aggregated; no Gemini API calls were made.")


def validate_news_data(news_dir: str = "data/news"):
    """Audit raw and derived news files without modifying them."""
    raw_files = _news_raw_files(news_dir)
    print("\n=== NEWS DATA VALIDATION ===")
    print(f"Raw ticker files: {len(raw_files)}")
    totals = Counter()
    for input_path in raw_files:
        ticker = os.path.splitext(os.path.basename(input_path))[0].upper()
        rows = 0
        duplicates = 0
        seen = set()
        invalid = 0
        missing = 0
        with open(input_path, newline="", encoding="utf-8-sig", errors="replace") as input_file:
            for row in csv.DictReader(input_file):
                rows += 1
                published = _normalize_news_text(row.get("published_at", ""))
                headline = _normalize_news_text(row.get("headline", ""))
                source = _normalize_news_text(row.get("source", ""))
                if not published or not headline:
                    missing += 1
                    continue
                try:
                    datetime.strptime(published, "%Y%m%dT%H%M%S")
                except ValueError:
                    invalid += 1
                    continue
                key = (published, headline.casefold(), source.casefold())
                if key in seen:
                    duplicates += 1
                seen.add(key)
        totals.update(rows=rows, duplicates=duplicates, invalid=invalid, missing=missing)
        processed = os.path.exists(os.path.join(news_dir, f"{ticker}_processed.csv"))
        daily = os.path.exists(os.path.join(news_dir, f"{ticker}_daily.csv"))
        print(f"{ticker:12} rows={rows:,} | dup={duplicates} | missing={missing} | bad_timestamp={invalid} | processed={'YES' if processed else 'NO'} | daily={'YES' if daily else 'NO'}")
    print("\nTotals")
    print(f"Rows: {totals['rows']:,}")
    print(f"Duplicates: {totals['duplicates']:,}")
    print(f"Missing timestamp/headline: {totals['missing']:,}")
    print(f"Invalid timestamps: {totals['invalid']:,}")

def main():
    """Parse CLI arguments and dispatch to the requested research or live-analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Research backend for stock, FX, ETF, index, and other Yahoo Finance asset analysis."
    )
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker/symbol for direct analysis, e.g. AAPL or GBP=X")
    parser.add_argument("--query", type=str, help="Resolve and analyze an asset from a natural-language query")
    parser.add_argument("--tickers", nargs="+", help="Run the multi-stock walk-forward comparison for these tickers")
    parser.add_argument("--tickers-news", nargs="+", help="Tickers for multi-stock news comparisons; also used with --walk-forward-news-macro")
    parser.add_argument("--tickers-three-way", nargs="+", help="Compare technical-only, rule-news, and Gemini-news walk-forward results")
    parser.add_argument("--walk-forward-gemini", action="store_true", help="Run the walk-forward model using cached Gemini news features")
    parser.add_argument("--walk-forward-fx", action="store_true", help="Run the FX-specific chronological walk-forward experiment")
    parser.add_argument("--validate-fx", action="store_true", help="Run the FX validation comparison across benchmark pairs")
    parser.add_argument("--test-asset-resolver", action="store_true")
    parser.add_argument("--download-news", action="store_true", help="Download historical Alpha Vantage news for the ticker")
    parser.add_argument("--download-news-yearly", action="store_true", help="Download and merge one historical-news request per calendar year")
    parser.add_argument("--process-news", action="store_true", help="Process cached historical news into daily rule-based features")
    parser.add_argument("--clean-news-data", action="store_true", help="Clean all raw historical news files, rebuild derived news features, and preserve originals in data/news/raw_backup")
    parser.add_argument("--validate-news-data", action="store_true", help="Audit raw historical news files without modifying them")
    parser.add_argument("--process-news-gemini", action="store_true", help="Classify uncached historical headlines with Gemini and build daily Gemini features")
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite", help="Gemini model used for structured headline classification")
    parser.add_argument("--gemini-limit", type=int, default=100, help="Maximum new Gemini classifications; 0 means all uncached headlines")
    parser.add_argument("--gemini-batch-size", type=int, default=10, help="Headlines per Gemini API request")
    parser.add_argument("--gemini-request-delay", type=float, default=4.2, help="Seconds between Gemini batch requests")
    parser.add_argument("--gemini-timeout", type=int, default=60, help="Maximum seconds allowed for one Gemini API request")
    parser.add_argument("--news-cutoff-hour", type=int, default=16, help="New York local cutoff hour for daily historical-news aggregation")
    parser.add_argument("--news-start", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=date.today() - timedelta(days=5 * 365), help="Historical news start date in YYYY-MM-DD format")
    parser.add_argument("--news-end", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=date.today(), help="Historical news end date in YYYY-MM-DD format")
    parser.add_argument("--news-dir", default="data/news", help="Directory containing historical and live news caches")
    parser.add_argument("--news-cache-hours", type=float, default=6, help="Freshness window for cached live Yahoo headlines")
    parser.add_argument("--refresh-news", action="store_true", help="Ignore the live news cache and fetch fresh Yahoo headlines")
    parser.add_argument("--show-headlines", action="store_true", help="Print recent headlines after the compact analysis")
    parser.add_argument("--days", type=int, default=60, help="Number of recent trading days shown in live analysis")
    parser.add_argument("--backtest", action="store_true", help="Run the technical-rule backtest")
    parser.add_argument("--logistic", action="store_true", help="Run the chronological train/validation/test Logistic Regression backtest")
    parser.add_argument("--walk-forward", action="store_true", help="Run the expanding-window technical Logistic Regression evaluation")
    parser.add_argument("--walk-forward-news", action="store_true", help="Run the expanding-window technical-plus-rule-news evaluation")
    parser.add_argument("--walk-forward-macro", action="store_true", help="Run the separate technical-plus-frozen-macro walk-forward experiment")
    parser.add_argument("--walk-forward-news-macro", action="store_true", help="Run the combined technical + rule-news + frozen-macro walk-forward experiment")
    parser.add_argument("--download-macro", action="store_true", help="Download and cache all configured FRED macro indicators for reproducible macro experiments")
    parser.add_argument("--macro-start", default="2015-01-01", help="Earliest macro date to cache for the final historical matrix (YYYY-MM-DD)")
    parser.add_argument("--macro-years", type=int, default=12, help="Fallback number of years of macro history when --macro-start is not changed")
    parser.add_argument("--refresh-macro", action="store_true", help="Refresh all configured macro caches from the requested macro start")
    parser.add_argument("--download-global-data", action="store_true", help="Download a diversified global equity universe plus global market/FX/commodity context")
    parser.add_argument("--global-limit", type=int, default=500, help="Maximum number of global equities to download")
    parser.add_argument("--refresh-global-universe", action="store_true", help="Rebuild the global universe from current major-index constituents")
    parser.add_argument("--global-start", default="2015-01-01", help="Start date for global historical data")
    parser.add_argument("--global-end", default=date.today().isoformat(), help="End date for global historical data")
    parser.add_argument("--build-global-dataset", action="store_true", help="Build the cross-stock training panel from downloaded global data")
    parser.add_argument("--rebuild-global-dataset", action="store_true", help="Rebuild the global training panel from scratch")
    parser.add_argument("--train-global", action="store_true", help="Train and evaluate the global cross-stock model chronologically")
    parser.add_argument("--global-status", action="store_true", help="Show the current global-data/model status")
    parser.add_argument("--test-window", type=int, default=63, help="Trading days in each walk-forward test window")
    parser.add_argument("--years", type=int, default=5, help="Years of history used for backtesting")
    parser.add_argument("--horizon", type=int, default=5, help="Trading days ahead used to define the target")
    args = parser.parse_args()
    try:
        if args.test_asset_resolver:
            test_asset_resolver()
        elif args.query:
            analyze_asset_query(args.query, args.days, args.news_dir, args.years, args.news_cutoff_hour, args.news_cache_hours, args.refresh_news, args.show_headlines)
        elif args.global_status:
            global_data_status()
        elif args.download_global_data:
            download_global_market_data(
                args.global_start,
                args.global_end,
                args.global_limit,
                args.refresh_global_universe,
            )
        elif args.build_global_dataset:
            build_global_training_dataset(
                horizon=args.horizon,
                start=args.global_start,
                end=args.global_end,
                max_stocks=args.global_limit,
                rebuild=args.rebuild_global_dataset,
            )
        elif args.rebuild_global_dataset:
            build_global_training_dataset(
                horizon=args.horizon,
                start=args.global_start,
                end=args.global_end,
                max_stocks=args.global_limit,
                rebuild=True,
            )
        elif args.train_global:
            train_global_model(
                horizon=args.horizon,
            )
        elif args.download_macro:
            status = ensure_macro_data(years=args.macro_years, macro_start=args.macro_start, refresh=args.refresh_macro)
            if status == 'unavailable':
                raise ValueError('Could not fetch macro data -- set FRED_API_KEY (free sign-up at https://fred.stlouisfed.org/docs/api/api_key.html) and try again.')
            print(f'Macro data cache status: {status}')
            if status == 'incomplete':
                print('Some macro series are still missing. Set FRED_API_KEY and rerun --download-macro.')
        elif args.walk_forward_macro:
            walk_forward_macro_model(args.ticker, args.years, args.horizon, args.test_window)
        elif args.walk_forward_news_macro:
            if args.tickers_news:
                backtest_multiple_news_macro_tickers(
                    args.tickers_news,
                    args.years,
                    args.horizon,
                    args.test_window,
                    args.news_dir,
                )
            else:
                walk_forward_news_macro_model(
                    args.ticker,
                    args.years,
                    args.horizon,
                    args.test_window,
                    args.news_dir,
                )
        elif args.clean_news_data:
            clean_news_data(args.news_dir, args.news_cutoff_hour)
        elif args.validate_news_data:
            validate_news_data(args.news_dir)
        elif args.process_news_gemini:
            classify_news_with_gemini(args.ticker, args.news_dir, args.gemini_model, args.gemini_limit, args.gemini_batch_size, args.gemini_request_delay, args.gemini_timeout * 1000)
        elif args.process_news:
            process_historical_news(args.ticker, args.news_dir, args.news_cutoff_hour)
        elif args.download_news_yearly:
            download_historical_news_yearly(args.ticker, args.news_start, args.news_end, args.news_dir)
        elif args.download_news:
            download_historical_news(args.ticker, args.news_start, args.news_end, args.news_dir)
        elif args.tickers_news:
            backtest_multiple_news_tickers(args.tickers_news, args.years, args.horizon, args.test_window, args.news_dir)
        elif args.tickers_three_way:
            backtest_three_way_comparison(args.tickers_three_way, args.years, args.horizon, args.test_window, args.news_dir)
        elif args.tickers:
            backtest_multiple_tickers(args.tickers, args.years, args.horizon, args.test_window)
        elif args.validate_fx:
            backtest_multiple_fx_pairs(["GBP=X", "EURUSD=X", "JPY=X", "CHF=X", "CAD=X", "AUD=X", "NZD=X", "SEK=X", "NOK=X"], args.years, args.horizon, args.test_window)
        elif args.walk_forward_fx:
            walk_forward_fx_model(args.ticker, args.years, args.horizon, args.test_window)
        elif args.walk_forward_gemini:
            walk_forward_gemini_model(args.ticker, args.years, args.horizon, args.test_window, args.news_dir)
        elif args.walk_forward_news:
            walk_forward_news_model(args.ticker, args.years, args.horizon, args.test_window, args.news_dir)
        elif args.walk_forward:
            walk_forward_logistic_model(args.ticker, args.years, args.horizon, args.test_window)
        elif args.logistic:
            backtest_logistic_model(args.ticker, args.years, args.horizon)
        elif args.backtest:
            backtest_technical_rule(args.ticker, args.years, args.horizon)
        else:
            analyze_ticker(args.ticker, args.days, args.news_dir, args.years, args.news_cutoff_hour, auto_historical_news=False, news_cache_hours=args.news_cache_hours, refresh_news=args.refresh_news, show_headlines=args.show_headlines)
    except ValueError as e:
        print(f'Error: {e}')
        sys.exit(1)
if __name__ == '__main__':
    main()