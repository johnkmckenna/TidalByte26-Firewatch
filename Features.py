#!/usr/bin/env python3
"""
Build the model-ready table: weather features for every cell-day, joined to the
next-day ignition labels from build_cellday_labels.py.

Inputs
------
--meta       grid.json                      (from build_cellday_labels.py)
--positives  cellday_positives.parquet      (from build_cellday_labels.py)
--cache      gridmet_cache/                 (from download_gridmet.py)

Output (one file per year, so memory stays small)
------
<out>/features_<year>.parquet   rows = (feature_date, cell); columns = features, labels, weight, split
<out>/features_meta.json        feature column list, split years, negative sampling rate

Row meaning
-----------
A row says: "on `date`, this cell had these conditions (and this history); was a fire
discovered there on date + 1?"  Labels: y_any, y_human, y_natural, y_escaped.

Splits are by YEAR (never random). Defaults: the last 3 complete years are test, the 2
before are validation, everything earlier is train. Override with --val-years/--test-years.

Negative sampling (train years only)
------------------------------------
Every positive row is kept; negatives are kept with probability --neg-rate and given
weight 1/neg_rate so weighted metrics and probabilities stay unbiased. Validation and
test years keep EVERY row (weight 1), so backtest numbers are honest.

Features use only information available at prediction time:
  * weather of day D and trailing windows ending on D
  * calendar of day D+1 (is tomorrow a weekend / holiday)  -- known in advance
  * cell_prior_*: that cell's past ignition rate from YEARS BEFORE the row's year only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

BASE_VARS = ["erc", "bi", "fm100", "fm1000", "vs", "vpd", "tmmx", "rmin", "pr"]


# --------------------------------------------------------------------------- helpers
def load_cache(cache, var, year):
    p = Path(cache) / f"{var}_{year}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["data"], z["dates"].astype("datetime64[D]"), z["lat_idx"], z["lon_idx"]


def roll_sum(a, w):
    """Trailing window sum along axis 0, window includes the current day (partial at the start)."""
    c = np.cumsum(a, axis=0, dtype=np.float64)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out.astype(np.float32)


def roll_mean(a, w):
    denom = np.minimum(w, np.arange(a.shape[0]) + 1).astype(np.float32)[:, None, None]
    return roll_sum(np.nan_to_num(a), w) / denom


def roll_max(a, w):
    out = a.copy()
    for k in range(1, w):
        out[k:] = np.fmax(out[k:], a[:-k])
    return out


def dry_day_count(pr, thresh=1.0, cap=60):
    """Consecutive days (ending today) with less than `thresh` mm of rain, capped."""
    out = np.empty_like(pr, dtype=np.float32)
    cnt = np.zeros(pr.shape[1:], dtype=np.float32)
    for t in range(pr.shape[0]):
        wet = np.nan_to_num(pr[t]) >= thresh
        cnt = np.where(wet, 0.0, np.minimum(cnt + 1, cap)).astype(np.float32)
        out[t] = cnt
    return out


def load_year_stack(cache, year, variables):
    """Dict var -> (T,H,W) for one year, plus shared dates and cell index arrays."""
    arrs, ref = {}, None
    for v in variables:
        r = load_cache(cache, v, year)
        if r is None:
            return None
        data, dates, li, lj = r
        if ref is None:
            ref = (dates, li, lj)
        else:
            if not (np.array_equal(dates, ref[0]) and np.array_equal(li, ref[1]) and np.array_equal(lj, ref[2])):
                sys.exit(f"Variable {v} year {year} has different dates/cells than the others.")
        arrs[v] = data
    return arrs, ref[0], ref[1], ref[2]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--positives", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-rate", type=float, default=0.03, help="fraction of train negatives kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-years", nargs="+", type=int)
    ap.add_argument("--test-years", nargs="+", type=int)
    ap.add_argument("--min-year", type=int, default=None, help="first year to build (default: first fire year)")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pos = pd.read_parquet(args.positives)

    first_label = pd.Timestamp(meta["first_label_date"])
    last_label = pd.Timestamp(meta["last_label_date"])
    first_year = args.min_year or first_label.year
    last_complete = last_label.year if last_label >= pd.Timestamp(year=last_label.year, month=12, day=15) \
        else last_label.year - 1
    years = list(range(first_year, last_complete + 1))
    if last_complete < last_label.year:
        print(f"NOTE: {last_label.year} looks partial in the fire data (ends {last_label.date()}); "
              f"it is skipped, because missing fire reports would look like false negatives.")

    test_years = args.test_years or years[-3:]
    val_years = args.val_years or [y for y in years if y < min(test_years)][-2:]
    train_years = [y for y in years if y < min(val_years + test_years)]
    split_of = {**{y: "train" for y in train_years}, **{y: "val" for y in val_years},
                **{y: "test" for y in test_years}}
    print(f"Years {years[0]}-{years[-1]} | train {train_years[:1]}..{train_years[-1:]} "
          f"| val {val_years} | test {test_years}")

    H = W = None
    # ---- static: elevation
    elev = None
    ep = Path(args.cache) / "elevation.npz"
    if ep.exists():
        z = np.load(ep)
        elev = (z["data"].astype(np.float32), z["lat_idx"], z["lon_idx"])

    # ---- prior ignition counts by cell and year (from the label table)
    pos = pos.copy()
    pos["year"] = pd.to_datetime(pos["label_date"]).dt.year
    hol_cal = USFederalHolidayCalendar()

    feature_cols = None
    summary = []
    for Y in years:
        stack = load_year_stack(args.cache, Y, BASE_VARS)
        if stack is None:
            sys.exit(f"Missing gridMET cache for year {Y}. Run download_gridmet.py (check for 'missing'/'failed').")
        cur, dates_y, lat_idx, lon_idx = stack
        T_y, H, W = cur["erc"].shape
        HW = H * W

        # previous-year tail (31 days) for trailing windows
        prev = load_year_stack(args.cache, Y - 1, BASE_VARS)
        PAD = 31
        if prev is not None and np.array_equal(prev[2], lat_idx) and np.array_equal(prev[3], lon_idx):
            full = {v: np.concatenate([prev[0][v][-PAD:], cur[v]], axis=0) for v in BASE_VARS}
            pad = min(PAD, prev[0]["erc"].shape[0])
        else:
            print(f"  (no gridMET for {Y - 1}; trailing windows are partial at the start of {Y})")
            full = {v: cur[v] for v in BASE_VARS}
            pad = 0

        # ---- derived weather features, computed on the padded series then trimmed to year Y
        d = {}
        d["pr_7d"] = roll_sum(np.nan_to_num(full["pr"]), 7)
        d["pr_30d"] = roll_sum(np.nan_to_num(full["pr"]), 30)
        d["dry_days"] = dry_day_count(full["pr"])
        erc_chg = np.full_like(full["erc"], np.nan)
        erc_chg[7:] = full["erc"][7:] - full["erc"][:-7]
        d["erc_chg_7d"] = erc_chg
        d["vpd_mean_3d"] = roll_mean(full["vpd"], 3)
        d["tmmx_mean_7d"] = roll_mean(full["tmmx"], 7)
        d["vs_max_3d"] = roll_max(full["vs"], 3)
        feats = {v: full[v][pad:] for v in BASE_VARS}
        feats.update({k: a[pad:] for k, a in d.items()})

        # ---- which days of the year are usable rows
        win_start = max(pd.Timestamp(year=first_label.year, month=1, day=1), pd.Timestamp(year=Y, month=1, day=1))
        win_end = last_label - pd.Timedelta(days=1)
        day_ts = pd.to_datetime(dates_y)
        ok_day = (day_ts >= win_start) & (day_ts <= win_end) & (day_ts.year == Y)

        # ---- calendar of the LABEL day (tomorrow)
        label_ts = day_ts + pd.Timedelta(days=1)
        hol = set(hol_cal.holidays(start=label_ts.min(), end=label_ts.max()))
        cal = {
            "month_next": label_ts.month.values.astype(np.float32),
            "doy_sin": np.sin(2 * np.pi * label_ts.dayofyear.values / 366.0).astype(np.float32),
            "doy_cos": np.cos(2 * np.pi * label_ts.dayofyear.values / 366.0).astype(np.float32),
            "dow_next": label_ts.dayofweek.values.astype(np.float32),
            "is_weekend_next": (label_ts.dayofweek.values >= 5).astype(np.float32),
            "is_holiday_next": np.array([t in hol for t in label_ts], dtype=np.float32),
            "is_july4_window": ((label_ts.month.values == 7) & np.isin(label_ts.day.values, [3, 4, 5])).astype(np.float32),
        }

        # ---- labels for these days
        lab = {k: np.zeros((T_y, H, W), dtype=np.int8) for k in ["y_any", "y_human", "y_natural", "y_escaped"]}
        pf = pos[pd.to_datetime(pos["feature_date"]).dt.year == Y]
        if len(pf):
            t_idx = np.searchsorted(dates_y, pd.to_datetime(pf["feature_date"]).values.astype("datetime64[D]"))
            i_idx = np.searchsorted(lat_idx, pf["lat_idx"].values)
            j_idx = np.searchsorted(lon_idx, pf["lon_idx"].values)
            good = (t_idx < T_y) & (i_idx < H) & (j_idx < W)
            good &= (dates_y[np.minimum(t_idx, T_y - 1)] == pd.to_datetime(pf["feature_date"]).values.astype("datetime64[D]"))
            good &= (lat_idx[np.minimum(i_idx, H - 1)] == pf["lat_idx"].values)
            good &= (lon_idx[np.minimum(j_idx, W - 1)] == pf["lon_idx"].values)
            for col in lab:
                m = good & (pf[col].values > 0)
                lab[col][t_idx[m], i_idx[m], j_idx[m]] = 1
            if (~good).any():
                print(f"  year {Y}: {(~good).sum()} fire cell-days fell outside the weather grid/dates")

        # ---- cell priors: past ignition rate per cell, years strictly before Y
        prior = {}
        past = pos[pos["year"] < Y]
        days_before = max((pd.Timestamp(year=Y, month=1, day=1) - pd.Timestamp(year=first_label.year, month=1, day=1)).days, 1)
        for name, col in [("cell_prior_any", "n_fires"), ("cell_prior_human", "n_human"),
                          ("cell_prior_natural", "n_natural")]:
            grid = np.zeros((H, W), dtype=np.float32)
            if len(past):
                ii = np.searchsorted(lat_idx, past["lat_idx"].values)
                jj = np.searchsorted(lon_idx, past["lon_idx"].values)
                okk = (ii < H) & (jj < W)
                np.add.at(grid, (ii[okk], jj[okk]), past[col].values[okk].astype(np.float32))
            prior[name] = (grid / days_before).reshape(-1)

        # ---- row selection
        split = split_of.get(Y, "train")
        n_full = int(ok_day.sum()) * HW
        t_ok = np.flatnonzero(ok_day)
        y_any_flat = lab["y_any"][t_ok].reshape(-1)
        rng = np.random.default_rng(args.seed + Y)
        if split == "train":
            keep = (y_any_flat == 1) | (rng.random(y_any_flat.shape[0]) < args.neg_rate)
            weight = np.where(y_any_flat == 1, 1.0, 1.0 / args.neg_rate).astype(np.float32)
        else:
            keep = np.ones(y_any_flat.shape[0], dtype=bool)
            weight = np.ones(y_any_flat.shape[0], dtype=np.float32)
        sel = np.flatnonzero(keep)
        t_loc = sel // HW                 # index into t_ok
        c = sel % HW                      # flat cell index
        t_abs = t_ok[t_loc]
        ci, cj = c // W, c % W

        cols = {
            "date": dates_y[t_abs],
            "lat_idx": lat_idx[ci].astype(np.int16),
            "lon_idx": lon_idx[cj].astype(np.int16),
            "cell_lat": (meta["lat0"] + lat_idx[ci] * meta["step_deg"]).astype(np.float32),
            "cell_lon": (meta["lon0"] + lon_idx[cj] * meta["step_deg"]).astype(np.float32),
        }
        for k, a in feats.items():
            cols[k] = a[t_ok].reshape(-1)[sel]
        for k, a in cal.items():
            cols[k] = a[t_abs]
        for k, a in prior.items():
            cols[k] = a[c]
        if elev is not None:
            e_arr, e_li, e_lj = elev
            ei = np.searchsorted(e_li, lat_idx[ci])
            ej = np.searchsorted(e_lj, lon_idx[cj])
            ei, ej = np.minimum(ei, e_arr.shape[0] - 1), np.minimum(ej, e_arr.shape[1] - 1)
            cols["elevation"] = e_arr[ei, ej]
        for k in lab:
            cols[k] = lab[k][t_ok].reshape(-1)[sel]
        cols["weight"] = weight[sel]
        cols["split"] = np.full(len(sel), split)

        df = pd.DataFrame(cols)
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(out / f"features_{Y}.parquet", index=False)

        if feature_cols is None:
            skip = {"date", "lat_idx", "lon_idx", "y_any", "y_human", "y_natural", "y_escaped", "weight", "split"}
            feature_cols = [c_ for c_ in df.columns if c_ not in skip]
        summary.append((Y, split, n_full, int(y_any_flat.sum()), len(df)))
        print(f"{Y} [{split:5s}] full rows {n_full:>10,}  positives {int(y_any_flat.sum()):>6,}  "
              f"kept rows {len(df):>9,}", flush=True)

    (out / "features_meta.json").write_text(json.dumps({
        "feature_cols": feature_cols,
        "label_cols": ["y_any", "y_human", "y_natural", "y_escaped"],
        "train_years": train_years, "val_years": val_years, "test_years": test_years,
        "neg_rate_train": args.neg_rate,
        "note": "Train negatives are subsampled; use the `weight` column. Val/test keep all rows.",
    }, indent=2))
    tot_full = sum(s[2] for s in summary)
    tot_pos = sum(s[3] for s in summary)
    print(f"\nTotal full cell-days {tot_full:,}; positives {tot_pos:,}; base rate {tot_pos / max(tot_full, 1):.5%}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Wrote {len(summary)} parquet files and features_meta.json to {out}")


if __name__ == "__main__":
    main()