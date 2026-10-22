import importlib.util
import inspect
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "stock_predictor.py"

spec = importlib.util.spec_from_file_location("stock_predictor", SCRIPT)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {SCRIPT}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

STOCKS = [
    ("AAPL", "Apple stock"),
    ("MSFT", "Microsoft stock"),
    ("AMZN", "Amazon stock"),
    ("GOOGL", "Alphabet stock"),
    ("NVDA", "NVIDIA stock"),
    ("META", "Meta stock"),
    ("TSLA", "Tesla stock"),
    ("JPM", "JPMorgan stock"),
    ("KO", "Coca-Cola stock"),
    ("BRK-B", "Berkshire Hathaway stock"),
]

FX = [
    ("GBP=X", "British currency"),
    ("EURUSD=X", "Euro currency"),
    ("CHF=X", "Swiss currency"),
    ("CAD=X", "Canadian currency"),
    ("AUD=X", "Australian currency"),
    ("NZD=X", "New Zealand currency"),
    ("SEK=X", "Swedish currency"),
    ("NOK=X", "Norwegian currency"),
    ("JPY=X", "Japanese currency"),
]


def valid_number(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def call_news(ticker):
    fn = mod.fetch_news
    sig = inspect.signature(fn)
    kwargs = {}

    if "limit" in sig.parameters:
        kwargs["limit"] = 8

    if "news_dir" in sig.parameters:
        kwargs["news_dir"] = "data/news"

    if "cache_hours" in sig.parameters:
        kwargs["cache_hours"] = 6

    return fn(ticker, **kwargs)


def resolver_matches(ticker, symbols, kind):
    symbols = {str(symbol).upper() for symbol in symbols}

    if kind == "stock":
        accepted = {
            "GOOGL": {"GOOGL", "GOOG"},
        }.get(ticker, {ticker})

        return bool(symbols & accepted)

    if ticker == "EURUSD=X":
        accepted = {
            "EURUSD=X",
            "EUR=X",
            "EURUSD=X",
        }
        return bool(symbols & accepted)

    return ticker in symbols

def test_asset(ticker, query, kind):
    checks = []

    try:
        _, assets = mod.resolve_asset_query(query)

        symbols = [
            str(item.get("symbol")).upper()
            for item in assets
            if item.get("symbol") is not None
        ]

        resolver_ok = resolver_matches(ticker, symbols, kind)

        if not resolver_ok:
            print(f"    Resolver returned: {symbols}")

        checks.append(("resolver", resolver_ok))

    except Exception:
        checks.append(("resolver", False))

    try:
        hist = mod.fetch_price_data(ticker, 60)
        close = hist["Close"].iloc[-1]

        checks.append(
            (
                "price",
                valid_number(close) and float(close) > 0
            )
        )

        mod.compute_technical_signal(hist)
        checks.append(("technical", True))

    except Exception:
        checks.append(("price/technical", False))

    try:
        call_news(ticker)
        checks.append(("news", True))

    except Exception:
        checks.append(("news", False))

    try:
        if kind == "stock":

            result = mod.predict_live_direction(
                ticker,
                5,
                5,
                "data/news",
            )

            checks.append(
                (
                    "model",
                    result is not None
                    and result.get("direction") in {"UP", "DOWN"}
                    and valid_number(result.get("probability_up")),
                )
            )

        else:

            validated = mod.load_validated_fx_pairs(5, 5)

            result = mod.predict_live_fx_direction(
                ticker,
                5,
                5,
            )

            if ticker.upper() in validated:

                checks.append(
                    (
                        "model",
                        result is not None
                        and result.get("direction") in {"UP", "DOWN"}
                        and valid_number(result.get("probability_up")),
                    )
                )

            else:

                checks.append(
                    ("model", result is None)
                )

    except Exception:
        checks.append(("model", False))

    ok = all(
        value
        for _, value in checks
    )

    failed = ", ".join(
        name
        for name, value in checks
        if not value
    )

    return ok, failed or "PASS"

def main():

    print("=== EXTENDED LIVE PIPELINE TEST ===")

    passed = 0
    total = len(STOCKS) + len(FX)

    print("\nStocks")

    for ticker, query in STOCKS:

        ok, detail = test_asset(
            ticker,
            query,
            "stock",
        )

        print(
            f"{ticker:10} "
            f"{'PASS' if ok else 'FAIL'}"
            + ("" if ok else f" | {detail}")
        )

        passed += int(ok)

    print("\nFX")

    for ticker, query in FX:

        ok, detail = test_asset(
            ticker,
            query,
            "fx",
        )

        print(
            f"{ticker:10} "
            f"{'PASS' if ok else 'FAIL'}"
            + ("" if ok else f" | {detail}")
        )

        passed += int(ok)

    print(f"\nPassed: {passed}/{total}")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()