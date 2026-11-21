"""
Build the model's feature rows for ONE date, from the gridMET cache.

This reproduces build_features.py exactly (same formulas, same 31-day padding from the
previous year, same calendar and cell-history rules), so a row built here for a past date
equals the training row for that date. Tested against build_features output.

The only intended difference: for dates after the fire records end, the cell-history
denominator stops at the last fire-record date, so years with no records yet don't dilute it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

BASE_VARS = ["erc", "bi", "fm100", "fm1000", "vs", "vpd", "tmmx", "rmin", "pr"]
PAD = 31


class DataNotAvailable(Exception):
    """Raised when the gridMET cache does not cover the requested date."""


# ---- identical helpers to build_features.py -------------------------------------------
def roll_sum(a, w):
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
    out = np.empty_like(pr, dtype=np.float32)
    cnt = np.zeros(pr.shape[1:], dtype=np.float32)
    for t in range(pr.shape[0]):
        wet = np.nan_to_num(pr[t]) >= thresh
        cnt = np.where(wet, 0.0, np.minimum(cnt + 1, cap)).astype(np.float32)
        out[t] = cnt
    return out


def load_cache(cache, var, year):
    p = Path(cache) / f"{var}_{year}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["data"], z["dates"].astype("datetime64[D]"), z["lat_idx"], z["lon_idx"]


def load_year_stack(cache, year, variables=BASE_VARS):
    arrs, ref = {}, None
    for v in variables:
        r = load_cache(cache, v, year)
        if r is None:
            return None
        data, dates, li, lj = r
        if ref is None:
            ref = (dates, li, lj)
        elif not (np.array_equal(dates, ref[0]) and np.array_equal(li, ref[1]) and np.array_equal(lj, ref[2])):
            # Live current-year files can differ by a day between variables; trim to the common days.
            common = np.intersect1d(dates, ref[0])
            if not (np.array_equal(li, ref[1]) and np.array_equal(lj, ref[2])) or len(common) == 0:
                raise DataNotAvailable(f"{v}_{year} has a different grid than the other variables.")
            keep_new = np.isin(dates, common)
            keep_old = np.isin(ref[0], common)
            arrs = {k: a[keep_old] for k, a in arrs.items()}
            data = data[keep_new]
            ref = (common, li, lj)
        arrs[v] = data
    return arrs, ref[0], ref[1], ref[2]


def latest_weather_date(cache, year):
    """Latest day present in ALL variables' cache for `year` (None if the year is missing)."""
    try:
        s = load_year_stack(cache, year)
    except DataNotAvailable:
        return None
    return None if s is None else pd.Timestamp(s[1].max())


# ---- feature builder --------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, cache_dir, grid_meta_path, positives_path):
        self.cache = Path(cache_dir)
        self.meta = json.loads(Path(grid_meta_path).read_text())
        pos = pd.read_parquet(positives_path)
        pos["label_date"] = pd.to_datetime(pos["label_date"])
        pos["year"] = pos["label_date"].dt.year
        self.pos = pos
        self.first_label = pd.Timestamp(self.meta["first_label_date"])
        self.last_label = pd.Timestamp(self.meta["last_label_date"])
        self.hol_cal = USFederalHolidayCalendar()
        self.elev = None
        ep = self.cache / "elevation.npz"
        if ep.exists():
            z = np.load(ep)
            self.elev = (z["data"].astype(np.float32), z["lat_idx"], z["lon_idx"])
        self._prior_cache = {}

    def available_range(self, year):
        return latest_weather_date(self.cache, year)

    def _priors(self, year, lat_idx, lon_idx):
        key = (year, lat_idx.tobytes(), lon_idx.tobytes())
        if key in self._prior_cache:
            return self._prior_cache[key]
        H, W = len(lat_idx), len(lon_idx)
        past = self.pos[self.pos["year"] < year]
        # Training used (Jan 1 of year) - (Jan 1 of first fire year). Past the end of the fire
        # records, stop the clock at the last record so empty years don't dilute the rate.
        end = min(pd.Timestamp(year=year, month=1, day=1), self.last_label + pd.Timedelta(days=1))
        days_before = max((end - pd.Timestamp(year=self.first_label.year, month=1, day=1)).days, 1)
        out = {}
        for name, col in [("cell_prior_any", "n_fires"), ("cell_prior_human", "n_human"),
                          ("cell_prior_natural", "n_natural")]:
            grid = np.zeros((H, W), dtype=np.float32)
            if len(past):
                ii = np.searchsorted(lat_idx, past["lat_idx"].values)
                jj = np.searchsorted(lon_idx, past["lon_idx"].values)
                ok = (ii < H) & (jj < W)
                ok &= lat_idx[np.minimum(ii, H - 1)] == past["lat_idx"].values
                ok &= lon_idx[np.minimum(jj, W - 1)] == past["lon_idx"].values
                np.add.at(grid, (ii[ok], jj[ok]), past[col].values[ok].astype(np.float32))
            out[name] = (grid / days_before).reshape(-1)
        self._prior_cache[key] = out
        return out

    def build(self, feature_date):
        """DataFrame with one row per cell for `feature_date` (predicting ignitions on the next day)."""
        D = pd.Timestamp(feature_date).normalize()
        Y = D.year
        cur = load_year_stack(self.cache, Y)
        if cur is None:
            raise DataNotAvailable(f"No gridMET cache for {Y}. Refresh weather for {Y} first.")
        arrs, dates_y, lat_idx, lon_idx = cur
        d64 = np.datetime64(D.date(), "D")
        hit = np.flatnonzero(dates_y == d64)
        if len(hit) == 0:
            raise DataNotAvailable(f"gridMET cache for {Y} ends {pd.Timestamp(dates_y.max()).date()}; "
                                   f"{D.date()} is not available yet.")
        t = int(hit[0])
        H, W = len(lat_idx), len(lon_idx)

        prev = load_year_stack(self.cache, Y - 1)
        if prev is not None and np.array_equal(prev[2], lat_idx) and np.array_equal(prev[3], lon_idx):
            pad = min(PAD, prev[0]["erc"].shape[0])
            full = {v: np.concatenate([prev[0][v][-PAD:], arrs[v][: t + 1]], axis=0) for v in BASE_VARS}
        else:
            pad = 0
            full = {v: arrs[v][: t + 1] for v in BASE_VARS}

        last = -1  # the requested day is the last row of the (causal) series
        f = {v: full[v][last] for v in BASE_VARS}
        f["pr_7d"] = roll_sum(np.nan_to_num(full["pr"]), 7)[last]
        f["pr_30d"] = roll_sum(np.nan_to_num(full["pr"]), 30)[last]
        f["dry_days"] = dry_day_count(full["pr"])[last]
        f["erc_chg_7d"] = (full["erc"][last] - full["erc"][last - 7]) if full["erc"].shape[0] > 7 \
            else np.full((H, W), np.nan, dtype=np.float32)
        f["vpd_mean_3d"] = roll_mean(full["vpd"], 3)[last]
        f["tmmx_mean_7d"] = roll_mean(full["tmmx"], 7)[last]
        f["vs_max_3d"] = roll_max(full["vs"], 3)[last]

        label = D + pd.Timedelta(days=1)
        is_hol = len(self.hol_cal.holidays(start=label, end=label)) > 0
        cal = {
            "month_next": float(label.month),
            "doy_sin": float(np.float32(np.sin(2 * np.pi * label.dayofyear / 366.0))),
            "doy_cos": float(np.float32(np.cos(2 * np.pi * label.dayofyear / 366.0))),
            "dow_next": float(label.dayofweek),
            "is_weekend_next": float(label.dayofweek >= 5),
            "is_holiday_next": float(is_hol),
            "is_july4_window": float(label.month == 7 and label.day in (3, 4, 5)),
        }

        ci, cj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ci, cj = ci.ravel(), cj.ravel()
        cols = {
            "date": np.full(H * W, np.datetime64(D.date(), "ns")),
            "lat_idx": lat_idx[ci].astype(np.int16),
            "lon_idx": lon_idx[cj].astype(np.int16),
            "cell_lat": (self.meta["lat0"] + lat_idx[ci] * self.meta["step_deg"]).astype(np.float32),
            "cell_lon": (self.meta["lon0"] + lon_idx[cj] * self.meta["step_deg"]).astype(np.float32),
        }
        for k, a in f.items():
            cols[k] = np.asarray(a, dtype=np.float32).reshape(-1)
        for k, v in cal.items():
            cols[k] = np.full(H * W, v, dtype=np.float32)
        cols.update(self._priors(Y, lat_idx, lon_idx))
        if self.elev is not None:
            e_arr, e_li, e_lj = self.elev
            ei = np.minimum(np.searchsorted(e_li, lat_idx[ci]), e_arr.shape[0] - 1)
            ej = np.minimum(np.searchsorted(e_lj, lon_idx[cj]), e_arr.shape[1] - 1)
            cols["elevation"] = e_arr[ei, ej]
        df = pd.DataFrame(cols)
        df["date"] = pd.to_datetime(df["date"])
        return df
