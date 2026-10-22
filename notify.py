"""Nightly notifications for tracked stocks (run by run_daily.bat after the predictions).

For every tracked stock it compares the newest ranking with the last state it saw:
  * moved to another fifth of its market (e.g. "leaning to beat" -> "likely to beat"), or
  * its position in the market changed by 15 points or more,
then adds a notification in the app and shows a Windows pop-up (needs: pip install plyer).

Usage:  python app/notify.py
"""

import pandas as pd

import data

STATE = data.APP_DIR / "notify_state.json"
BIG_MOVE = 15          # percentage points of the market


def popup(title: str, message: str) -> None:
    try:
        from plyer import notification
        notification.notify(title=title, message=message[:250], app_name="Stock Agent", timeout=15)
    except Exception as error:      # no plyer, or no desktop session: the in-app list still works
        print(f"(No Windows pop-up: {error})")


def main():
    tickers = data.watchlist()
    rfile, tag = data.latest_ranking_file()
    if not tickers or rfile is None:
        print("Nothing to check (no tracked stocks or no ranking yet).")
        return
    ranking = data.load_ranking(rfile).set_index("ticker")
    state = data._read_json(STATE, {})
    names = data.universe().set_index("symbol")["name"].to_dict()
    new = []
    for t in tickers:
        if t not in ranking.index:
            continue
        r = ranking.loc[t]
        n = int(r["stocks_in_market"])
        pct = round(100 * (1 - (int(r["rank_in_market"]) - 1) / max(n - 1, 1)))
        q = int(r["quintile"]) if pd.notna(r["quintile"]) else 3
        before = state.get(t)
        if before and before.get("date") != rfile.stem[-10:]:
            text, _ = data.verdict(q, pct)
            if q != before["quintile"]:
                arrow = "up" if q > before["quintile"] else "down"
                new.append({"ticker": t, "message": f"moved {arrow}: now '{text}' (rank {int(r['rank_in_market'])}/{n}, "
                                                    f"was quintile {before['quintile']}, now {q})"})
            elif abs(pct - before["pct"]) >= BIG_MOVE:
                new.append({"ticker": t, "message": f"rank changed a lot: better than {pct}% of its market "
                                                    f"(was {before['pct']}%)"})
        state[t] = {"date": rfile.stem[-10:], "quintile": q, "pct": pct}

    stamp = data.now_str()
    for item in new:
        item.update(time=stamp, read=False)
    if new:
        data.add_notifications(new)
        lines = [f"{d['ticker']} ({names.get(d['ticker'], d['ticker'])}): {d['message']}" for d in new]
        popup(f"Stock Agent: {len(new)} tracked stock(s) changed", "\n".join(lines))
    data._write_json(STATE, state)
    print(f"Checked {len(tickers)} tracked stocks against {rfile.name}: {len(new)} notification(s).")
    for d in new:
        print(f"  {d['ticker']}: {d['message']}")


if __name__ == "__main__":
    main()
