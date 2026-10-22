"""Stock agent - web interface (Flask).

Start it from the project folder:
    python app/app.py
then open http://127.0.0.1:5000 in your browser.

It only READS what the nightly run produces (rankings, reasons, prices),
so it opens instantly and never retrains or re-downloads anything.
"""
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

import data

app = Flask(__name__)


@app.context_processor
def common():
    unread = sum(1 for n in data.notifications() if not n.get("read"))
    return {"unread": unread}


@app.template_filter("money")
def money(v):
    return "-" if v is None else f"{v:,.2f}"


@app.route("/")
def index():
    markets, ranking_date, tag = data.top_picks()
    return render_template("index.html", markets=markets, ranking_date=ranking_date, tag=tag,
                           tracked=data.tracked_rows(), status=data.model_status())


@app.route("/search")
def search_page():
    q = request.args.get("q", "")
    results = data.search(q, limit=30)
    groups = data.industries_matching(q)
    if not groups and (len(results) == 1 or (results and results[0]["score"] == 100)):
        return redirect(url_for("stock", ticker=results[0]["ticker"]))
    return render_template("search.html", q=q, results=results, groups=groups)


@app.route("/stock/<ticker>")
def stock(ticker):
    page = data.stock_page(ticker)
    if page is None:
        hits = data.search(ticker, limit=1)
        if hits:
            return redirect(url_for("stock", ticker=hits[0]["ticker"]))
        abort(404)
    return render_template("stock.html", p=page)


@app.route("/ranking")
@app.route("/ranking/<country>")
def ranking(country=None):
    if country is None:
        markets, _, _ = data.top_picks()
        country = markets[0]["country"] if markets else "USA"
    page = data.market_ranking(country, request.args.get("sector", ""))
    return render_template("ranking.html", r=page, hl=request.args.get("hl", ""))


@app.route("/industry/<path:name>")
def industry(name):
    return render_template("group.html", g=data.group_page("industry", name))


@app.route("/sector/<path:name>")
def sector(name):
    return render_template("group.html", g=data.group_page("sector", name))


@app.route("/tracked")
def tracked():
    notes = list(reversed(data.notifications()))
    page = render_template("tracked.html", rows=data.tracked_rows(), notes=notes)
    data.mark_all_read()
    return page


# ------------------------------------------------------------- API ------
@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    return jsonify({"stocks": data.search(q, limit=8), "groups": data.industries_matching(q)[:3]})


@app.route("/api/track/<ticker>", methods=["POST", "DELETE"])
def api_track(ticker):
    tickers = data.set_tracked(ticker, request.method == "POST")
    return jsonify({"tracked": ticker in tickers, "count": len(tickers)})


@app.errorhandler(404)
def not_found(_):
    return render_template("search.html", q="", results=[], not_found=True), 404


if __name__ == "__main__":
    print("Stock agent interface: http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)
