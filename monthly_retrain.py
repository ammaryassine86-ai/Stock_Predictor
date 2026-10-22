"""Monthly automatic retraining of the main 20-day ranking model.

Run once a month by run_monthly.bat (Windows Task Scheduler). It:
  1. refreshes all price history, macro data, earnings and SEC filings,
  2. rebuilds the panel, matrix, 20-day target and split,
  3. trains a CANDIDATE ranking model (tuned on validation, walk-forward test),
  4. checks the candidate, and only if it passes replaces the main model
     (lightgbm_model_h20_rank_main.txt). The previous main model is archived
     in models/archive/ first, so nothing is ever lost.
If any step fails, the main model is left untouched and the reason is logged.
Everything is written to predictions/monthly_log.txt and the result of the last
run to predictions/monthly_status.json (shown in the web interface).

Usage:
  python monthly_retrain.py                      # the full monthly job
  python monthly_retrain.py --adopt lightgbm_model_h20_rank_v4.txt --results results_v4_rank.json
        # make an already-trained model the main model (used once to start)
  python monthly_retrain.py --check-only         # train + check, never switch
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
MAIN = ROOT / "lightgbm_model_h20_rank_main.txt"
MAIN_RESULTS = ROOT / "models" / "main_results.json"
ARCHIVE = ROOT / "models" / "archive"
CAND = ROOT / "lightgbm_model_h20_rank_candidate.txt"
CAND_RESULTS = ROOT / "results_candidate_rank.json"
STATUS = ROOT / "predictions" / "monthly_status.json"
H20 = "data/final/final_feature_matrix_h20.csv.gz"
SPLIT = "data/split_monthly"

# A candidate must pass ALL of these on its own walk-forward test (never-seen months) ...
MIN_WITHIN_DAY_AUC = 50.2       # random = 50
MIN_TOP_MINUS_BOTTOM = 0.0      # % per 20 days, top 20% minus bottom 20%
MIN_T_STAT = 0.5                # averaged over all start days
# ... and must not be clearly worse than the current main model on the (mostly shared) test period.
MAX_DROP_TOP_MINUS_BOTTOM = 0.40   # percentage points per 20 days
MAX_DROP_T_STAT = 0.8

STEPS = [
    ("Download price history", [PY, "stock_predictor.py", "--download-global-data", "--global-limit", "2000"]),
    ("Refresh macro data (FRED)", [PY, "stock_predictor.py", "--download-macro", "--refresh-macro"]),
    ("Refresh earnings", [PY, "download_earnings.py", "--due"]),
    ("Refresh SEC fundamentals", [PY, "download_fundamentals.py", "--refresh"]),
    ("Rebuild panel", [PY, "stock_predictor.py", "--rebuild-global-dataset", "--global-limit", "2000"]),
    ("Build feature matrix", [PY, "final_matrix_builder.py", "--rebuild-final-matrix"]),
    ("20-day target", [PY, "make_horizon.py", "--horizon", "20"]),
    ("Train / validation / test split", [PY, "make_split.py", "--matrix", H20, "--out", SPLIT, "--horizon", "20"]),
    ("Train candidate model", [PY, "run_lightgbm.py", "--data-dir", SPLIT, "--horizon", "20", "--objective", "rank",
                               "--output", str(CAND_RESULTS), "--predictions", "pred_candidate_rank.csv.gz",
                               "--importance", "importance_candidate_rank.csv", "--model-out", str(CAND)]),
]
NOT_FATAL = {"Refresh macro data (FRED)", "Refresh earnings", "Refresh SEC fundamentals"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}", flush=True)


def write_status(**fields) -> None:
    STATUS.parent.mkdir(exist_ok=True)
    fields["time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    STATUS.write_text(json.dumps(fields, indent=2), encoding="utf-8")


def test_metrics(results_file: Path) -> dict:
    r = json.loads(results_file.read_text(encoding="utf-8"))
    return r["test_walk_forward"]


def check_candidate() -> tuple[bool, str]:
    if not CAND.exists() or not CAND_RESULTS.exists():
        return False, "candidate model or its results file is missing"
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(CAND))
        meta = json.loads(CAND.with_suffix(".json").read_text(encoding="utf-8"))
        if booster.num_feature() != len(meta["features"]) or len(meta["features"]) < 50:
            return False, f"candidate has {booster.num_feature()} features, metadata lists {len(meta['features'])}"
    except Exception as error:
        return False, f"candidate model can't be loaded: {error}"
    new = test_metrics(CAND_RESULTS)
    msg = (f"candidate test: within-day AUC {new['within_day_auc']:.2f}, "
           f"top-bottom {new['top_minus_bottom_pct']:+.3f}%/20d, t {new['t_stat']:+.2f}")
    if new["within_day_auc"] < MIN_WITHIN_DAY_AUC:
        return False, msg + f" -> within-day AUC below {MIN_WITHIN_DAY_AUC}"
    if new["top_minus_bottom_pct"] <= MIN_TOP_MINUS_BOTTOM:
        return False, msg + " -> top 20% did not beat bottom 20%"
    if new["t_stat"] < MIN_T_STAT:
        return False, msg + f" -> t below {MIN_T_STAT}"
    if MAIN_RESULTS.exists():
        old = test_metrics(MAIN_RESULTS)
        msg += (f" | current main: top-bottom {old['top_minus_bottom_pct']:+.3f}%, t {old['t_stat']:+.2f}")
        if new["top_minus_bottom_pct"] < old["top_minus_bottom_pct"] - MAX_DROP_TOP_MINUS_BOTTOM:
            return False, msg + " -> clearly worse than the current main model"
        if new["t_stat"] < old["t_stat"] - MAX_DROP_T_STAT:
            return False, msg + " -> much less reliable than the current main model"
    return True, msg


def promote(model: Path, results: Path) -> None:
    """Archive the current main model, then make `model` the main model."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    if MAIN.exists():
        for src in (MAIN, MAIN.with_suffix(".json"), MAIN_RESULTS):
            if src.exists():
                shutil.copy2(src, ARCHIVE / f"{stamp}_{src.name}")
    shutil.copy2(model, MAIN)
    shutil.copy2(model.with_suffix(".json"), MAIN.with_suffix(".json"))
    MAIN_RESULTS.parent.mkdir(exist_ok=True)
    shutil.copy2(results, MAIN_RESULTS)
    # keep the 6 most recent archived models
    olds = sorted({p.name[:15] for p in ARCHIVE.iterdir()})        # "YYYY-MM-DD_HHMM" prefix
    for stale in olds[:-6]:
        for p in ARCHIVE.glob(f"{stale}_*"):
            p.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adopt", help="Make this already-trained model file the main model.")
    p.add_argument("--results", help="Its results JSON (with --adopt).")
    p.add_argument("--check-only", action="store_true", help="Train and check, but never replace the main model.")
    args = p.parse_args()

    if args.adopt:
        model, results = ROOT / args.adopt, ROOT / (args.results or "")
        if not model.exists() or not model.with_suffix(".json").exists() or not results.is_file():
            raise SystemExit("Need the model .txt, its .json next to it, and --results <results json>.")
        promote(model, results)
        meta = json.loads(MAIN.with_suffix(".json").read_text(encoding="utf-8"))
        log(f"Main model is now {model.name} (trained up to {meta['trained_up_to']}).")
        write_status(result="adopted", model=model.name, trained_up_to=meta["trained_up_to"])
        return

    t0 = time.time()
    log("=== MONTHLY RETRAIN START ===")
    write_status(result="running", step="starting")
    for name, cmd in STEPS:
        log(f"--- {name}")
        write_status(result="running", step=name)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            if name in NOT_FATAL:
                log(f"{name} failed (exit {rc}); continuing with the data already on disk.")
                continue
            log(f"FAILED at '{name}' (exit {rc}). The main model was NOT changed.")
            write_status(result="failed", step=name, kept="current main model")
            sys.exit(1)

    ok, msg = check_candidate()
    log(msg)
    if not ok:
        log("Candidate REJECTED. The main model was NOT changed.")
        write_status(result="rejected", reason=msg, kept="current main model")
    elif args.check_only:
        log("Candidate passed (check-only: main model not changed).")
        write_status(result="passed (not switched)", reason=msg)
    else:
        promote(CAND, CAND_RESULTS)
        meta = json.loads(MAIN.with_suffix(".json").read_text(encoding="utf-8"))
        log(f"Candidate ACCEPTED: main model now trained up to {meta['trained_up_to']}.")
        write_status(result="switched", reason=msg, trained_up_to=meta["trained_up_to"])
    log(f"=== MONTHLY RETRAIN DONE in {(time.time() - t0) / 3600:.1f} h ===")


if __name__ == "__main__":
    main()
