#!/usr/bin/env python3
"""
Train the ignition-risk model and run the backtest.

Reads the per-year parquet files from build_features.py, trains XGBoost on the TRAIN
years (GPU if available, otherwise CPU), early-stops on a VALIDATION sample, calibrates
probabilities on validation, and then scores the held-out TEST years against baselines.

Headline metric: "capture@k"
    Each day, flag the top k% of cells by risk. Capture@k is the share of that day's
    real ignitions that fell inside the flagged cells, summed over all test days.
    Random flagging captures k%. Higher is better.

Baselines scored on exactly the same test rows:
    random      : expected capture = k
    erc, bi     : official fire-danger indices alone (higher = riskier)
    climatology : the cell's past ignition rate x a seasonal factor (the "hot-spot map")

Ties are handled fairly: if many cells share a score, capture is the expected value under
a random tie-break, so baselines with many ties are not penalised.

Usage
-----
    python train_eval.py --features features --out model_out
    python train_eval.py --features features --out model_out --targets y_any y_human y_natural y_escaped
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

KS = [0.01, 0.02, 0.05, 0.10, 0.20]
CURVE_KS = np.round(np.arange(0.005, 0.2001, 0.005), 4)


# ----------------------------------------------------------------------------- data
def load_split(feat_dir, years):
    parts = [pd.read_parquet(Path(feat_dir) / f"features_{y}.parquet") for y in years]
    return pd.concat(parts, ignore_index=True)


def subsample_eval(df, target, rate, seed=0):
    """Keep all positives and `rate` of negatives; weight negatives 1/rate (unbiased)."""
    rng = np.random.default_rng(seed)
    pos = df[target].values == 1
    keep = pos | (rng.random(len(df)) < rate)
    out = df[keep].copy()
    out["weight"] = np.where(out[target].values == 1, 1.0, 1.0 / rate).astype(np.float32)
    return out


# ----------------------------------------------------------------------------- metrics
def capture_at_k(df, score, target, ks):
    """Daily top-k% capture with fair tie handling. Returns {k: capture_fraction}."""
    y = df[target].values
    dates = df["date"].values
    groups = pd.Series(np.arange(len(df))).groupby(dates).indices
    tot = {k: 0.0 for k in ks}
    n_pos_total = 0
    for _, idx in groups.items():
        yi = y[idx]
        if not yi.any():
            continue
        s = score[idx]
        n = len(idx)
        srt = np.sort(s)
        sp = s[yi == 1]
        left = np.searchsorted(srt, sp, side="left")
        right = np.searchsorted(srt, sp, side="right")
        n_gt = n - right
        n_eq = np.maximum(right - left, 1)
        for k in ks:
            tot[k] += float(np.clip((k * n - n_gt) / n_eq, 0.0, 1.0).sum())
        n_pos_total += len(sp)
    return {k: (tot[k] / n_pos_total if n_pos_total else float("nan")) for k in ks}, n_pos_total


def evaluate(df, scores, target):
    """scores: dict name -> array. Returns metrics DataFrame and capture-curve DataFrame."""
    y = df[target].values
    rows, curve = [], []
    for name, s in scores.items():
        s = np.nan_to_num(np.asarray(s, dtype=np.float64))
        cap, npos = capture_at_k(df, s, target, sorted(set(KS) | set(CURVE_KS)))
        row = {"method": name, "n_test_ignitions": npos}
        if name == "random":
            row["pr_auc"] = float(y.mean())
            row["roc_auc"] = 0.5
        else:
            row["pr_auc"] = float(average_precision_score(y, s))
            row["roc_auc"] = float(roc_auc_score(y, s))
        for k in KS:
            row[f"capture@{int(k * 100)}%"] = cap[k]
        rows.append(row)
        for k in CURVE_KS:
            curve.append({"method": name, "k": float(k), "capture": cap[k]})
    return pd.DataFrame(rows), pd.DataFrame(curve)


# ----------------------------------------------------------------------------- training
def pick_device(requested):
    """XGBoost silently falls back to CPU when no GPU exists, so check for one ourselves."""
    if requested == "cpu":
        return "cpu"
    import shutil
    import subprocess
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and "GPU" in r.stdout:
                print("GPU found:", r.stdout.strip().splitlines()[0])
                return "cuda"
        except Exception:
            pass
    print("No usable GPU found (nvidia-smi missing or failing); training on CPU.")
    return "cpu"


def fit_model(Xtr, ytr, wtr, Xva, yva, wva, params, device):
    def make(dev):
        return xgb.XGBClassifier(device=dev, tree_method="hist", eval_metric="aucpr",
                                 early_stopping_rounds=60, n_jobs=-1, **params)
    try:
        m = make(device)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
        return m, device
    except xgb.core.XGBoostError as e:
        if device == "cpu":
            raise
        print(f"  GPU training failed ({str(e).splitlines()[0][:120]}); falling back to CPU.")
        m = make("cpu")
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
        return m, "cpu"


def seasonal_factor(train, target):
    """Weighted ignition rate per month relative to the all-year rate."""
    w = train["weight"].values
    y = train[target].values
    m = train["month_next"].values.astype(int)
    overall = (w * y).sum() / w.sum()
    fac = {}
    for mo in range(1, 13):
        sel = m == mo
        fac[mo] = ((w[sel] * y[sel]).sum() / w[sel].sum() / overall) if sel.any() and overall > 0 else 1.0
    return fac


def run_target(target, args, meta, feature_cols, train_all, val_all, test_all, out_root):
    out = out_root / target
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=================== target: {target} ===================")
    ntr_pos = int(train_all[target].sum())
    print(f"train rows {len(train_all):,} (positives {ntr_pos:,}) | "
          f"val rows {len(val_all):,} | test rows {len(test_all):,} "
          f"(positives {int(test_all[target].sum()):,})")
    if ntr_pos < 50 or test_all[target].sum() < 10:
        print("  Too few positives to train/evaluate this target reliably; skipping.")
        return None

    val_s = subsample_eval(val_all, target, args.val_neg_rate)
    params = dict(n_estimators=args.n_estimators, learning_rate=args.lr, max_depth=args.max_depth,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1.0,
                  random_state=args.seed)
    t0 = time.time()
    model, dev = fit_model(train_all[feature_cols], train_all[target].values, train_all["weight"].values,
                           val_s[feature_cols], val_s[target].values, val_s["weight"].values,
                           params, args.device)
    print(f"trained on {dev} in {time.time() - t0:.0f}s; best iteration {model.best_iteration}")

    # --- calibration on the validation sample
    raw_val = model.predict_proba(val_s[feature_cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_val, val_s[target].values, sample_weight=val_s["weight"].values)

    # --- score the test years
    Xte = test_all[feature_cols]
    margin = model.predict(Xte, output_margin=True)            # monotone in risk; used for ranking
    prob_raw = model.predict_proba(Xte)[:, 1]
    prob_cal = iso.predict(prob_raw)

    fac = seasonal_factor(train_all, target)
    prior_col = {"y_human": "cell_prior_human", "y_natural": "cell_prior_natural"}.get(target, "cell_prior_any")
    clim = test_all[prior_col].values * test_all["month_next"].astype(int).map(fac).values
    scores = {
        "random": np.zeros(len(test_all)),
        "erc": test_all["erc"].values,
        "bi": test_all["bi"].values,
        "climatology": clim,
        "model": margin,
    }
    metrics, curve = evaluate(test_all, scores, target)
    print("\nTest-years backtest (flag top k% of cells each day; capture = share of ignitions inside):")
    show = metrics.copy()
    for c in [c for c in show.columns if c.startswith("capture")]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    show["pr_auc"] = show["pr_auc"].map(lambda v: f"{v:.5f}")
    show["roc_auc"] = show["roc_auc"].map(lambda v: f"{v:.3f}")
    print(show.to_string(index=False))
    base = metrics.set_index("method")
    for k in (5, 10):
        c = f"capture@{k}%"
        best_base = max(base.loc[b, c] for b in ("erc", "bi", "climatology"))
        print(f"  capture@{k}%: model {base.loc['model', c]:.1%} | best baseline {best_base:.1%} | "
              f"lift over random {base.loc['model', c] / (k / 100):.1f}x")

    # --- calibration table (test): predicted vs observed ignition rate by risk decile
    q = pd.qcut(pd.Series(prob_cal).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"decile": q, "pred": prob_cal, "obs": test_all[target].values}) \
        .groupby("decile").agg(mean_predicted=("pred", "mean"), observed_rate=("obs", "mean"), rows=("obs", "size"))
    cal.to_csv(out / "calibration.csv")
    print("\nCalibration by risk decile (test):")
    print(cal.assign(mean_predicted=cal.mean_predicted.map("{:.6f}".format),
                     observed_rate=cal.observed_rate.map("{:.6f}".format)).to_string())

    # --- importance
    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(feature_cols).fillna(0)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.rename("gain_share").to_csv(out / "feature_importance.csv")
    print("\nTop features by gain share:\n", imp.head(12).round(3).to_string())

    # --- save artifacts
    model.get_booster().save_model(str(out / "model.json"))
    with open(out / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    metrics.to_csv(out / "metrics.csv", index=False)
    curve.to_csv(out / "capture_curve.csv", index=False)
    (out / "model_info.json").write_text(json.dumps({
        "target": target, "feature_cols": feature_cols, "device": dev, "best_iteration": int(model.best_iteration),
        "train_years": meta["train_years"], "val_years": meta["val_years"], "test_years": meta["test_years"],
        "seasonal_factor": fac, "params": params,
        "note": "Probabilities: model.predict_proba then calibrator.pkl. Ranking: raw model output.",
    }, indent=2, default=str))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="folder from build_features.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="+", default=["y_any"],
                    choices=["y_any", "y_human", "y_natural", "y_escaped"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--val-neg-rate", type=float, default=0.10,
                    help="fraction of validation negatives used for early stopping/calibration")
    ap.add_argument("--min-train-year", type=int, default=None,
                    help="skip earliest training years (cell priors are noisy in the first years)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat_dir = Path(args.features)
    meta = json.loads((feat_dir / "features_meta.json").read_text())
    feature_cols = meta["feature_cols"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_years = [y for y in meta["train_years"] if args.min_train_year is None or y >= args.min_train_year]
    print(f"Loading train {train_years[:1]}..{train_years[-1:]}, val {meta['val_years']}, test {meta['test_years']}")
    train_all = load_split(feat_dir, train_years)
    val_all = load_split(feat_dir, meta["val_years"])
    test_all = load_split(feat_dir, meta["test_years"])
    if len(train_all) == 0 or len(test_all) == 0:
        sys.exit("Empty train or test set; check features_meta.json years.")

    args.device = pick_device(args.device)
    results = {}
    for t in args.targets:
        results[t] = run_target(t, args, meta, feature_cols, train_all, val_all, test_all, out_root)
    print(f"\nSaved everything under {out_root}/<target>/  (model.json, calibrator.pkl, metrics.csv, "
          f"capture_curve.csv, calibration.csv, feature_importance.csv, model_info.json)")


if __name__ == "__main__":
    main()
