#!/usr/bin/env python3
"""
Refit the probability calibrators WITHOUT retraining the models.

Why: train_eval.py fit isotonic regression on a *subsampled* validation set (only a fraction of
negatives kept). At the very top of the score range almost all negatives were dropped, so the
highest-scoring cells landed on a step at probability 1.0. Ranking (tiers, capture@k) is
unaffected, but displayed probabilities and expected-ignition totals were inflated.

This script:
  1. loads each saved model (model.json) and the FULL validation years (no negative subsampling),
  2. builds score bins that each contain at least --min-pos real positives,
  3. fits a monotone (isotonic) curve through the bin averages, which is smooth and cannot
     jump to 1.0 on a handful of events,
  4. checks old vs new calibration on the validation years (in-sample) and the TEST years
     (out-of-sample), and
  5. writes the new calibrator.pkl (the old one is kept as calibrator_isotonic_old.pkl).

Usage (from the project root, same folder layout as train_eval.py)
    python3 refit_calibration.py --features features --model-dir model_out
    python3 refit_calibration.py --features features --model-dir model_out --dry-run   # report only
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

TARGETS = ["y_any", "y_human", "y_natural", "y_escaped"]


def sigmoid(m):
    return 1.0 / (1.0 + np.exp(-m))


def load_models(model_dir, wanted):
    models = {}
    for t in wanted:
        d = Path(model_dir) / t
        if not (d / "model.json").exists():
            print(f"[skip] {t}: no model.json in {d}")
            continue
        info = json.loads((d / "model_info.json").read_text())
        b = xgb.Booster()
        b.load_model(str(d / "model.json"))
        models[t] = dict(booster=b, cols=info["feature_cols"], dir=d)
    return models


def score_years(feat_dir, years, models):
    """Return {target: (raw_prob float32, y int8, dates)} over the given years."""
    out = {t: dict(raw=[], y=[]) for t in models}
    need = sorted({c for m in models.values() for c in m["cols"]} | set(models))
    for yr in years:
        f = Path(feat_dir) / f"features_{yr}.parquet"
        df = pd.read_parquet(f, columns=need)
        print(f"  scored {yr}: {len(df):,} rows")
        for t, m in models.items():
            dm = xgb.DMatrix(df[m["cols"]], feature_names=m["cols"])
            out[t]["raw"].append(sigmoid(m["booster"].predict(dm, output_margin=True)).astype(np.float32))
            out[t]["y"].append(df[t].values.astype(np.int8))
        del df
    return {t: (np.concatenate(v["raw"]), np.concatenate(v["y"])) for t, v in out.items()}


def fit_binned_isotonic(raw, y, min_pos):
    """Bins of >= min_pos positives each (ordered by score), isotonic through the bin averages."""
    order = np.argsort(-raw, kind="stable")
    r, yy = raw[order].astype(np.float64), y[order].astype(np.float64)
    cp = np.cumsum(yy)
    bin_id = (np.maximum(cp - 1, 0) // min_pos).astype(np.int64)
    # a short last bin (few positives) is merged into the one above it
    last = bin_id.max()
    if last > 0 and (yy[bin_id == last].sum() < 0.5 * min_pos):
        bin_id[bin_id == last] = last - 1
    nb = bin_id.max() + 1
    n = np.bincount(bin_id, minlength=nb).astype(np.float64)
    pos = np.bincount(bin_id, weights=yy, minlength=nb)
    mean_raw = np.bincount(bin_id, weights=r, minlength=nb) / n
    rate = pos / n
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(mean_raw, rate, sample_weight=n)
    table = pd.DataFrame({"mean_raw_prob": mean_raw, "observed_rate": rate, "n": n.astype(np.int64),
                          "positives": pos.astype(np.int64)}).sort_values("mean_raw_prob").reset_index(drop=True)
    return iso, table


def check(label, raw, y, cal):
    p = np.clip(cal.predict(raw), 0, 1)
    obs = float(y.sum())
    row = {"calibrator": label, "expected": float(p.sum()), "observed": obs,
           "expected/observed": float(p.sum() / obs) if obs else np.nan,
           "max_p": float(p.max()), "n_cells_p>=0.5": int((p >= 0.5).sum()),
           "brier": float(np.mean((p - y) ** 2))}
    # reliability inside the top slices of the ranking
    order = np.argsort(-raw, kind="stable")
    for frac, name in [(1e-4, "top0.01%"), (1e-3, "top0.1%"), (1e-2, "top1%"), (1e-1, "top10%")]:
        k = max(int(len(raw) * frac), 1)
        idx = order[:k]
        row[f"{name} pred"] = float(p[idx].mean())
        row[f"{name} obs"] = float(y[idx].mean())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--model-dir", default="model_out")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--min-pos", type=int, default=40, help="min real positives per calibration bin")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write calibrator.pkl")
    args = ap.parse_args()

    feat_dir = Path(args.features)
    if not (feat_dir / "features_meta.json").exists():
        sys.exit(f"No features_meta.json in {feat_dir}. Pass --features <folder holding features_<year>.parquet>.")
    meta = json.loads((feat_dir / "features_meta.json").read_text())
    models = load_models(args.model_dir, args.targets)
    if not models:
        sys.exit("No models found.")

    print(f"Scoring validation years {meta['val_years']} (full, no subsampling)...")
    val = score_years(feat_dir, meta["val_years"], models)
    print(f"Scoring test years {meta['test_years']} (out-of-sample check)...")
    test = score_years(feat_dir, meta["test_years"], models)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.5g}")
    for t, m in models.items():
        raw_v, y_v = val[t]
        raw_t, y_t = test[t]
        npos = int(y_v.sum())
        print(f"\n=================== {t} ===================")
        print(f"validation: {len(y_v):,} rows, {npos:,} positives | test: {len(y_t):,} rows, {int(y_t.sum()):,} positives")
        if npos < 60:
            print("  Too few validation positives to refit; keeping the existing calibrator.")
            continue
        min_pos = int(max(10, min(args.min_pos, npos // 12)))
        new, table = fit_binned_isotonic(raw_v, y_v, min_pos)
        with open(m["dir"] / "calibrator.pkl", "rb") as f:
            old_path = m["dir"] / "calibrator_isotonic_old.pkl"
            old = pickle.load(f)
        rows = []
        for split, (rr, yy) in [("validation (in-sample)", (raw_v, y_v)), ("TEST (out-of-sample)", (raw_t, y_t))]:
            for label, cal in [("old isotonic", old), ("new binned", new)]:
                r = check(label, rr, yy, cal)
                r["split"] = split
                rows.append(r)
        rep = pd.DataFrame(rows).set_index(["split", "calibrator"])
        print(f"bins: {len(table)} (>= {min_pos} positives each)")
        print(rep[["expected", "observed", "expected/observed", "max_p", "n_cells_p>=0.5", "brier"]].to_string())
        print("\nReliability in the highest-ranked cell-days (predicted vs observed rate):")
        print(rep[[c for c in rep.columns if c.startswith("top")]].to_string())
        if not args.dry_run:
            if not old_path.exists():
                shutil.copy(m["dir"] / "calibrator.pkl", old_path)
            with open(m["dir"] / "calibrator.pkl", "wb") as f:
                pickle.dump(new, f)
            table.to_csv(m["dir"] / "calibration_refit_bins.csv", index=False)
            rep.reset_index().to_csv(m["dir"] / "calibration_refit_report.csv", index=False)
            print(f"-> wrote {m['dir'] / 'calibrator.pkl'} (old kept as {old_path.name})")
    if args.dry_run:
        print("\nDry run: nothing written. Rerun without --dry-run to save.")


if __name__ == "__main__":
    main()