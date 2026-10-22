"""Everything the web interface reads: rankings, reasons, prices, names, watchlist.

Nothing here trains or predicts: the nightly run (run_daily.bat -> predict_today.py)
writes predictions/ranking_<model>_<date>.csv and reasons_<model>_<date>.csv,
and this module only reads the newest ones.
"""
from __future__ import annotations

import difflib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # C:\stock predictor
PRED_DIR = ROOT / "predictions"
GLOBAL_DIR = ROOT / "data" / "global"
LIVE_STOCKS = ROOT / "data" / "live" / "stocks"
HIST_STOCKS = GLOBAL_DIR / "stocks"
APP_DIR = ROOT / "data" / "app"
WATCHLIST = APP_DIR / "watchlist.json"
NOTIFICATIONS = APP_DIR / "notifications.json"

# Which model's rankings to show, best first (falls back if the first has no file yet).
MODEL_TAGS = ["h20_rank_main", "h20_rank_v4", "h20_rank_v3", "h20_rank", "h20"]
HORIZON = 20
UNVALIDATED_MARKETS = {"Hong Kong", "Japan"}   # Japan: until a model trained on it is tested

# Average 20-day return of each fifth of stocks MINUS the average of all stocks, on the
# test period (May 2024 - Sep 2026) of the 1,081-stock ranking model. Used to tilt the
# middle of the projected range. Quintile 5 = top 20%.
QUINTILE_TILT = {1: -0.0050, 2: -0.0041, 3: -0.0028, 4: -0.0004, 5: +0.0123}

FRIENDLY = {
    "return_1": "price move on the last day", "return_5": "price change over 1 week",
    "return_20": "price change over 1 month", "return_60": "price change over 3 months",
    "sma_5_ratio": "price vs its 1-week average", "sma_20_ratio": "price vs its 1-month average",
    "sma_60_ratio": "price vs its 3-month average", "rsi_14": "RSI (overbought / oversold)",
    "volatility_20": "volatility over 1 month", "volatility_60": "volatility over 3 months",
    "range_1": "size of the last day's price range", "volume_change_1": "change in trading volume",
    "volume_z_20": "unusual trading volume", "drawdown_60": "distance below its 3-month high",
    "earn_days_since": "days since its last earnings report",
    "earn_surprise_last": "last earnings surprise (vs analysts)",
    "earn_surprise_avg4": "average earnings surprise (last 4 reports)",
    "earn_reaction_2d": "price reaction to its last earnings",
    "fund_revenue_growth": "revenue growth vs a year ago", "fund_profit_growth": "profit growth vs a year ago",
    "fund_gross_margin": "gross margin", "fund_operating_margin": "operating margin",
    "fund_roe": "return on equity", "fund_debt_to_assets": "debt compared with assets",
    "fund_current_ratio": "short-term assets vs short-term debts",
    "fund_days_since_filing": "days since its last SEC filing",
    "benchmark_return_1": "its market index on the last day",
    "benchmark_return_20": "its market index over 1 month",
    "relative_return_20": "stock vs its market index over 1 month",
}
PCT_FEATURES = re.compile(r"(return|growth|margin|roe|ratio$|drawdown|volatility|reaction)")


# ------------------------------------------------------------ files -------
def _newest(pattern: str) -> Path | None:
    files = sorted(PRED_DIR.glob(pattern), key=lambda p: p.stem[-10:])
    return files[-1] if files else None


def latest_ranking_file() -> tuple[Path | None, str]:
    for tag in MODEL_TAGS:
        f = _newest(f"ranking_{tag}_????-??-??.csv")
        if f:
            return f, tag
    return None, ""


def previous_ranking_file(tag: str, before: Path) -> Path | None:
    files = sorted(PRED_DIR.glob(f"ranking_{tag}_????-??-??.csv"), key=lambda p: p.stem[-10:])
    files = [f for f in files if f.stem[-10:] < before.stem[-10:]]
    return files[-1] if files else None


def load_ranking(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["date", "country", "ticker", "sector", "prob", "rank_in_market",
                                     "stocks_in_market", "quintile", "missing_features_pct"])
    return pd.read_csv(path)


# ---------------------------------------------------------- universe ------
_universe_cache: dict = {}


def universe() -> pd.DataFrame:
    """symbol, country, index, sector, industry, name (cached until the files change)."""
    files = [GLOBAL_DIR / "universe.csv", GLOBAL_DIR / "sectors.csv", GLOBAL_DIR / "names.csv"]
    key = tuple(f.stat().st_mtime if f.exists() else 0 for f in files)
    if _universe_cache.get("key") == key:
        return _universe_cache["df"]
    u = pd.read_csv(files[0])
    if files[1].exists():
        u = u.merge(pd.read_csv(files[1]), on="symbol", how="left")
    if files[2].exists():
        u = u.merge(pd.read_csv(files[2]), on="symbol", how="left")
    for c in ("sector", "industry", "name"):
        if c not in u:
            u[c] = ""
        u[c] = u[c].fillna("")
    u["name"] = np.where(u["name"] == "", u["symbol"], u["name"])
    _universe_cache.update(key=key, df=u)
    return u


# ------------------------------------------------------------ search ------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def search(query: str, limit: int = 10) -> list[dict]:
    """Smart search on ticker and company name: exact, prefix, word, substring, then typos."""
    q = _norm(query)
    if not q:
        return []
    u = universe()
    ranking = load_ranking(latest_ranking_file()[0]).set_index("ticker")
    scored = []
    for row in u.itertuples():
        t, name = str(row.symbol), str(row.name)
        tl, nl = t.lower(), _norm(name)
        base = tl.split(".")[0]
        if q in (tl, base):
            s = 100
        elif tl.startswith(q) or base.startswith(q):
            s = 90
        elif nl.startswith(q):
            s = 85
        elif any(w.startswith(q) for w in nl.split()):
            s = 75
        elif q in nl:
            s = 65
        else:
            # typos: compare with each word of the company name (preferred) and with the ticker
            words = [w for w in nl.split() if len(w) > 1 and w[0] == q[0]]
            r_name = max((difflib.SequenceMatcher(None, q, w).ratio() for w in words), default=0)
            r_tick = difflib.SequenceMatcher(None, q, base).ratio() if len(q) <= 6 and base[:1] == q[:1] else 0
            s = max(60 * r_name if r_name >= 0.8 else 0, 40 * r_tick if r_tick >= 0.8 else 0)
        if s:
            info = ranking.loc[t] if t in ranking.index else None
            scored.append({"ticker": t, "name": name, "country": row.country, "score": s,
                           "quintile": int(info["quintile"]) if info is not None and pd.notna(info["quintile"]) else None})
    scored.sort(key=lambda d: (-d["score"], d["ticker"]))
    return scored[:limit]


# ------------------------------------------------------------ prices ------
def price_history(ticker: str, days: int = 260) -> pd.Series:
    safe = re.sub(r"[^A-Za-z0-9_.=-]", "_", ticker)
    for folder in (LIVE_STOCKS, HIST_STOCKS):
        f = folder / f"{safe}.csv"
        if f.exists():
            df = pd.read_csv(f)
            date_col = df.columns[0]
            s = pd.Series(pd.to_numeric(df["Close"], errors="coerce").to_numpy(),
                          index=pd.to_datetime(df[date_col].astype(str).str[:10], errors="coerce"))
            s = s[s.index.notna()].dropna()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s):
                return s.iloc[-days:]
    return pd.Series(dtype=float)


def projection(close: pd.Series, quintile) -> dict | None:
    """Middle line + 68% and 95% ranges for the next HORIZON trading days.
    Width = the stock's own daily volatility (last 60 days); the middle line leans
    up or down by the average extra return of its model quintile (market assumed flat)."""
    if len(close) < 30:
        return None
    logret = np.log(close).diff().dropna().iloc[-60:]
    sigma = float(logret.std())
    tilt = QUINTILE_TILT.get(int(quintile), 0.0) if quintile is not None and not pd.isna(quintile) else 0.0
    last_date, last = close.index[-1], float(close.iloc[-1])
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=HORIZON)
    t = np.arange(1, HORIZON + 1)
    mid = last * np.exp(tilt * t / HORIZON)
    spread = sigma * np.sqrt(t)
    r = lambda a: [round(float(x), 4) for x in a]
    return {"dates": [d.strftime("%Y-%m-%d") for d in dates], "mid": r(mid),
            "low68": r(mid * np.exp(-spread)), "high68": r(mid * np.exp(spread)),
            "low95": r(mid * np.exp(-1.96 * spread)), "high95": r(mid * np.exp(1.96 * spread)),
            "sigma_daily": sigma, "tilt_pct": tilt * 100,
            "end_mid": float(mid[-1]), "end_low68": float(mid[-1] * math.exp(-spread[-1])),
            "end_high68": float(mid[-1] * math.exp(spread[-1]))}


# ----------------------------------------------------------- reasons ------
def friendly_label(feature: str, fallback: str = "") -> str:
    f = feature.replace("_lag1", "")
    if f.endswith("_sector_xrank"):
        base = f[:-len("_sector_xrank")]
        return f"{FRIENDLY.get(base, base.replace('_', ' '))} (rank vs its sector)"
    if f.endswith("_xrank"):
        base = f[:-len("_xrank")]
        return f"{FRIENDLY.get(base, base.replace('_', ' '))} (rank vs its market)"
    if f in FRIENDLY:
        return FRIENDLY[f]
    for prefix, text in (("world_sector_", "its sector worldwide: "), ("rel_sector_", "stock minus its sector: "),
                         ("sector_", "its sector in its market: "), ("rel_", "stock minus its market: "),
                         ("peer_", "rest of its market: ")):
        if f.startswith(prefix):
            base = f[len(prefix):]
            return text + FRIENDLY.get(base, base.replace("_", " "))
    return fallback or f.replace("_", " ")


def format_value(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    v = float(value)
    if feature.endswith("_xrank"):
        return f"top {max(1, round((1 - v) * 100))}%" if v >= 0.5 else f"bottom {max(1, round(v * 100))}%"
    if "days" in feature:
        return f"{v:.0f} days"
    if "surprise" in feature:
        return f"{v:+.1f}%"
    if "volatility" in feature:
        return f"{v * 100:.1f}% a day"
    if feature in ("fund_gross_margin", "fund_operating_margin", "fund_debt_to_assets"):
        return f"{v * 100:.0f}%"
    if feature == "fund_current_ratio":
        return f"{v:.2f}x"
    if feature in ("rsi_14", "volume_z_20"):
        return f"{v:.1f}"
    if PCT_FEATURES.search(feature):
        return f"{v * 100:+.1f}%" if abs(v) < 20 else f"{v:.2f}"
    return f"{v:.3g}"


def reasons(ticker: str, ranking_file: Path | None) -> dict:
    if ranking_file is None:
        return {"up": [], "down": [], "available": False}
    f = ranking_file.with_name(ranking_file.name.replace("ranking", "reasons", 1))
    if not f.exists():
        return {"up": [], "down": [], "available": False}
    r = pd.read_csv(f)
    r = r[r["ticker"] == ticker]
    out = {"available": True}
    for kind in ("up", "down"):
        out[kind] = [{"label": friendly_label(x.feature, x.label), "value": format_value(x.feature, x.value),
                      "effect": float(x.effect)} for x in r[r["kind"] == kind].itertuples()]
    return out


# ------------------------------------------------------------ summary -----
def verdict(q, pct) -> tuple[str, str]:
    q = int(q) if q is not None and not pd.isna(q) else 3
    return {5: ("Likely to beat its market", "strong-up"), 4: ("Leaning to beat its market", "up"),
            3: ("No clear edge", "neutral"), 2: ("Leaning to lag its market", "down"),
            1: ("Likely to lag its market", "strong-down")}[q]


def stock_page(ticker: str) -> dict | None:
    u = universe()
    meta = u[u["symbol"] == ticker]
    if meta.empty:
        return None
    m = meta.iloc[0]
    rfile, tag = latest_ranking_file()
    ranking = load_ranking(rfile)
    row = ranking[ranking["ticker"] == ticker]
    info = row.iloc[0].to_dict() if len(row) else None
    close = price_history(ticker)
    page = {"ticker": ticker, "name": m["name"], "country": m["country"], "index": m.get("index", ""),
            "sector": m["sector"], "industry": m["industry"], "model_tag": tag,
            "ranking_date": rfile.stem[-10:] if rfile else None, "info": info,
            "unvalidated": m["country"] in UNVALIDATED_MARKETS, "horizon": HORIZON,
            "tracked": ticker in watchlist()}
    if info:
        n = int(info["stocks_in_market"])
        better_than = 100 * (1 - (int(info["rank_in_market"]) - 1) / max(n - 1, 1))
        page["better_than"] = round(better_than)
        page["verdict"], page["verdict_class"] = verdict(info["quintile"], better_than)
    page["reasons"] = reasons(ticker, rfile)
    page["history"] = {"dates": [d.strftime("%Y-%m-%d") for d in close.index[-130:]],
                       "close": [round(float(x), 4) for x in close.iloc[-130:]]}
    page["price"] = price_info(ticker, m["country"], close)
    page["buy"] = where_to_buy(ticker, m["country"])
    page["last_price"] = float(close.iloc[-1]) if len(close) else None
    page["price_date"] = close.index[-1].strftime("%Y-%m-%d") if len(close) else None
    page["projection"] = projection(close, info["quintile"] if info else None)
    page["related"] = related(ticker, m, ranking, u)
    return page


def related(ticker: str, m, ranking: pd.DataFrame, u: pd.DataFrame, n: int = 8) -> list[dict]:
    """Same industry (same market first, then worldwide), with their current ranking."""
    if not m["industry"]:
        return []
    same = u[(u["industry"] == m["industry"]) & (u["symbol"] != ticker)]
    if same.empty:
        return []
    r = ranking.set_index("ticker")
    out = []
    for row in same.itertuples():
        info = r.loc[row.symbol] if row.symbol in r.index else None
        out.append({"ticker": row.symbol, "name": row.name, "country": row.country,
                    "same_market": row.country == m["country"],
                    "quintile": int(info["quintile"]) if info is not None and pd.notna(info["quintile"]) else None,
                    "rank": f"{int(info['rank_in_market'])}/{int(info['stocks_in_market'])}" if info is not None else "-"})
    out.sort(key=lambda d: (not d["same_market"], -(d["quintile"] or 0)))
    return out[:n]


def top_picks(n: int = 5) -> tuple[list[dict], str | None, str]:
    rfile, tag = latest_ranking_file()
    ranking = load_ranking(rfile)
    names = universe().set_index("symbol")["name"].to_dict()
    markets = []
    for country, g in ranking.groupby("country"):
        if country in UNVALIDATED_MARKETS:
            continue
        g = g.sort_values("rank_in_market")
        markets.append({"country": country, "date": str(g["date"].iloc[0]), "stocks": len(g),
                        "top": [{"ticker": t, "name": names.get(t, t)} for t in g["ticker"].head(n)],
                        "bottom": [{"ticker": t, "name": names.get(t, t)} for t in g["ticker"].tail(n)[::-1]]})
    markets.sort(key=lambda d: -d["stocks"])
    return markets, (rfile.stem[-10:] if rfile else None), tag


# --------------------------------------------------- watchlist / alerts ---
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def watchlist() -> list[str]:
    return _read_json(WATCHLIST, {"tickers": []}).get("tickers", [])


def set_tracked(ticker: str, on: bool) -> list[str]:
    data = _read_json(WATCHLIST, {"tickers": []})
    tickers = [t for t in data.get("tickers", []) if t != ticker]
    if on:
        tickers.append(ticker)
    data["tickers"] = tickers
    _write_json(WATCHLIST, data)
    return tickers


def notifications() -> list[dict]:
    return _read_json(NOTIFICATIONS, [])


def add_notifications(items: list[dict]) -> None:
    data = notifications() + items
    _write_json(NOTIFICATIONS, data[-200:])


def mark_all_read() -> None:
    data = notifications()
    for d in data:
        d["read"] = True
    _write_json(NOTIFICATIONS, data)


def tracked_rows() -> list[dict]:
    rfile, _ = latest_ranking_file()
    r = load_ranking(rfile).set_index("ticker")
    names = universe().set_index("symbol")["name"].to_dict()
    out = []
    for t in watchlist():
        info = r.loc[t] if t in r.index else None
        out.append({"ticker": t, "name": names.get(t, t),
                    "country": info["country"] if info is not None else "",
                    "quintile": int(info["quintile"]) if info is not None and pd.notna(info["quintile"]) else None,
                    "rank": f"{int(info['rank_in_market'])}/{int(info['stocks_in_market'])}" if info is not None else "-",
                    "verdict": verdict(info["quintile"], 0)[0] if info is not None else "No ranking yet",
                    "verdict_class": verdict(info["quintile"], 0)[1] if info is not None else "neutral"})
    return out


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ------------------------------------------------------- full ranking -----
def market_ranking(country: str, sector: str = "") -> dict:
    """Every stock of one market, best first, optionally only one sector."""
    rfile, tag = latest_ranking_file()
    ranking = load_ranking(rfile)
    countries = sorted(ranking["country"].unique(), key=lambda c: -int((ranking["country"] == c).sum()))
    g = ranking[ranking["country"] == country].sort_values("rank_in_market")
    sectors = sorted(s for s in g["sector"].dropna().unique() if s and s != "Unknown")
    total = len(g)
    if sector:
        g = g[g["sector"] == sector]
    u = universe().set_index("symbol")
    rows = []
    for r in g.itertuples():
        n = int(r.stocks_in_market)
        better = round(100 * (1 - (int(r.rank_in_market) - 1) / max(n - 1, 1)))
        text, cls = verdict(r.quintile, better)
        rows.append({"rank": int(r.rank_in_market), "ticker": r.ticker,
                     "name": u["name"].get(r.ticker, r.ticker) if r.ticker in u.index else r.ticker,
                     "sector": r.sector, "industry": u["industry"].get(r.ticker, "") if r.ticker in u.index else "",
                     "quintile": int(r.quintile) if pd.notna(r.quintile) else None,
                     "better": better, "verdict": text, "verdict_class": cls})
    return {"country": country, "countries": countries, "sectors": sectors, "sector": sector, "rows": rows,
            "total": total, "date": str(g["date"].iloc[0]) if len(g) else None,
            "ranking_date": rfile.stem[-10:] if rfile else None, "tag": tag,
            "unvalidated": country in UNVALIDATED_MARKETS}


# ---------------------------------------------------------- industries ----
ALIASES = {"oil": "oil & gas", "petrol": "oil & gas", "bank": "banks", "chip": "semiconductor",
           "chips": "semiconductor", "mining": "mining", "miner": "mining", "miners": "mining",
           "pharma": "drug manufacturers", "car": "auto manufacturers", "cars": "auto manufacturers",
           "ai": "semiconductor", "defense": "aerospace & defense", "defence": "aerospace & defense"}


def industries_matching(query: str) -> list[dict]:
    """Industries (and sectors) whose name matches the query, e.g. 'gold' -> Gold (26 stocks)."""
    q = _norm(query)
    if len(q) < 3 and q not in ALIASES:
        return []
    target = " ".join(_norm(ALIASES.get(q, q)).split())
    u = universe()
    out = []
    for kind in ("industry", "sector"):
        counts = u[u[kind] != ""].groupby(kind).size()
        for name, n in counts.items():
            nl = " ".join(_norm(name).split())
            if nl.startswith(target) or any(w.startswith(target) for w in nl.split()) or target in nl:
                out.append({"kind": kind, "name": name, "count": int(n)})
    out.sort(key=lambda d: (d["kind"] != "industry", -d["count"]))
    return out[:6]


def group_page(kind: str, name: str) -> dict:
    """All stocks of one industry/sector, across markets, best positioned first.
    Each stock's rank is within ITS OWN market (that's what the model compares)."""
    u = universe()
    members = u[u[kind] == name]
    rfile, tag = latest_ranking_file()
    r = load_ranking(rfile).set_index("ticker")
    rows = []
    for m in members.itertuples():
        info = r.loc[m.symbol] if m.symbol in r.index else None
        if info is not None:
            n = int(info["stocks_in_market"])
            better = round(100 * (1 - (int(info["rank_in_market"]) - 1) / max(n - 1, 1)))
            text, cls = verdict(info["quintile"], better)
            rank = f"{int(info['rank_in_market'])} of {n}"
            q = int(info["quintile"]) if pd.notna(info["quintile"]) else None
        else:
            better, text, cls, rank, q = -1, "No ranking yet", "neutral", "-", None
        rows.append({"ticker": m.symbol, "name": m.name, "country": m.country, "industry": m.industry,
                     "rank": rank, "better": better, "quintile": q, "verdict": text, "verdict_class": cls,
                     "unvalidated": m.country in UNVALIDATED_MARKETS})
    rows.sort(key=lambda d: (d["unvalidated"], -d["better"]))
    return {"kind": kind, "name": name, "rows": rows, "ranking_date": rfile.stem[-10:] if rfile else None,
            "tag": tag, "ranked": sum(1 for d in rows if d["better"] >= 0)}


# ------------------------------------------------- price & where to buy ---
CURRENCY = {"Tunisia": "TND", "Japan": "JPY", "USA": "USD", "Canada": "CAD", "United Kingdom": "GBX", "Australia": "AUD", "Germany": "EUR",
            "France": "EUR", "Spain": "EUR", "Netherlands": "EUR", "India": "INR", "Hong Kong": "HKD"}
CURRENCY_NOTE = {"GBX": "pence (100 pence = 1 GBP)"}
EXCHANGE = {".T": "Tokyo Stock Exchange (TSE)", ".TO": "Toronto Stock Exchange (TSX)", ".L": "London Stock Exchange (LSE)", ".AX": "Australian Securities Exchange (ASX)",
            ".DE": "Xetra (Frankfurt)", ".PA": "Euronext Paris", ".MC": "Bolsa de Madrid", ".AS": "Euronext Amsterdam",
            ".NS": "National Stock Exchange of India (NSE)", ".HK": "Hong Kong Stock Exchange (HKEX)"}

# Brokers that buy REAL shares on most of these exchanges and that BrokerChooser lists as
# available to residents of Tunisia (checked Oct 2026). Edit this list for another country.
BROKERS = [
    {"name": "Interactive Brokers", "url": "https://www.interactivebrokers.com",
     "note": "Widest access: US, Canada, UK, Europe, Japan, Australia, Hong Kong. No minimum deposit."},
    {"name": "MEXEM", "url": "https://www.mexem.com",
     "note": "Uses Interactive Brokers' markets with a simpler account. No minimum deposit."},
    {"name": "Zacks Trade", "url": "https://zackstrade.com",
     "note": "US and many international markets. $250 minimum deposit."},
]


def price_info(ticker: str, country: str, close: pd.Series) -> dict | None:
    if len(close) < 2:
        return None
    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
    year = close[close.index >= close.index[-1] - pd.Timedelta(days=365)]
    month = close.iloc[-22] if len(close) > 22 else close.iloc[0]
    cur = CURRENCY.get(country, "")
    return {"last": last, "currency": cur, "currency_note": CURRENCY_NOTE.get(cur, ""),
            "date": close.index[-1].strftime("%Y-%m-%d"),
            "day_pct": (last / prev - 1) * 100, "month_pct": (last / float(month) - 1) * 100,
            "low_52w": float(year.min()), "high_52w": float(year.max())}


# For stocks listed on the Bourse de Tunis (country "Tunisia"): Tunisian brokers instead.
# The official list of licensed brokers is kept up to date by the BVMT itself.
TUNISIAN_BROKERS = [
    {"name": "Bourse de Tunis: licensed brokers", "url": "https://www.bvmt.com.tn",
     "note": "Official list of the intermédiaires en bourse who can buy Tunisian shares for you."},
]
TUNISIAN_EXCHANGE = "Bourse de Tunis (BVMT)"


def where_to_buy(ticker: str, country: str) -> dict:
    if country == "Tunisia" or ticker.upper().endswith(".TN"):
        symbol = ticker.split(".")[0]
        return {"exchange": TUNISIAN_EXCHANGE, "symbol": symbol, "brokers": TUNISIAN_BROKERS,
                "yahoo": f"https://finance.yahoo.com/quote/{ticker}", "restricted": False, "tunisian": True}
    suffix = "." + ticker.split(".")[-1] if "." in ticker else ""
    exchange = EXCHANGE.get(suffix, "NYSE or Nasdaq (USA)")
    symbol = ticker[: -len(suffix)] if suffix else ticker
    if suffix in (".TO", ".L"):
        symbol = symbol.replace("-", ".")          # TECK-B.TO is "TECK.B" at most brokers
    if suffix == ".HK":
        symbol = str(int(symbol)) if symbol.isdigit() else symbol
    return {"exchange": exchange, "symbol": symbol, "brokers": BROKERS,
            "yahoo": f"https://finance.yahoo.com/quote/{ticker}",
            "restricted": country == "India", "tunisian": False}


def model_status() -> dict:
    """Training date of the main model and the result of the last monthly retrain."""
    out = {}
    meta = ROOT / "lightgbm_model_h20_rank_main.json"
    if meta.exists():
        out["trained_up_to"] = _read_json(meta, {}).get("trained_up_to")
    st = _read_json(PRED_DIR / "monthly_status.json", {})
    if st:
        out["last_retrain"] = st.get("time")
        out["last_result"] = st.get("result")
    return out
