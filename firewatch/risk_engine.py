"""
Inference layer: score every cell for tomorrow and explain each score.

    engine = RiskEngine()                       # loads models, calibrators, features builder
    result = engine.forecast("2023-07-03")      # weather date D -> risk for D+1
    result.grid                                 # DataFrame: one row per cell, probabilities + tier
    engine.explain(result, lat_idx, lon_idx)    # top drivers for one cell

Outputs per cell
    p_ignition          calibrated P(any fire discovered tomorrow)          [y_any model]
    p_large_fire        calibrated P(a fire that reaches 10+ acres starts)   [y_escaped model]
    p_escape_if_ignited p_large_fire / p_ignition, an approximate conditional (two separate models)
    risk_percentile     rank of this cell among all cells today (1.0 = riskiest)
    tier                Extreme (top 1%), Very High (1-5%), High (5-10%), Elevated (10-25%), Normal
Drivers come from XGBoost's exact per-feature contributions (pred_contribs; they sum to the
model's raw score), translated into plain language with the feature's actual value.
"""
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config
from .featurize import DataNotAvailable, FeatureBuilder  # noqa: F401  (re-exported)

TIERS = [(0.99, "Extreme"), (0.95, "Very High"), (0.90, "High"), (0.75, "Elevated"), (0.0, "Normal")]

# plain-language names and value formatting for drivers
FEATURE_TEXT = {
    "erc": ("Energy release component (fuel dryness index)", lambda v: f"{v:.0f}"),
    "bi": ("Burning index", lambda v: f"{v:.0f}"),
    "fm100": ("100-hour dead fuel moisture", lambda v: f"{v:.1f}%"),
    "fm1000": ("1000-hour dead fuel moisture", lambda v: f"{v:.1f}%"),
    "vs": ("Wind speed", lambda v: f"{v * 2.237:.0f} mph"),
    "vs_max_3d": ("Peak wind, last 3 days", lambda v: f"{v * 2.237:.0f} mph"),
    "vpd": ("Vapor pressure deficit (air dryness)", lambda v: f"{v:.2f} kPa"),
    "vpd_mean_3d": ("Air dryness, 3-day mean", lambda v: f"{v:.2f} kPa"),
    "tmmx": ("Max temperature", lambda v: f"{(v - 273.15) * 9 / 5 + 32:.0f}°F"),
    "tmmx_mean_7d": ("Max temperature, 7-day mean", lambda v: f"{(v - 273.15) * 9 / 5 + 32:.0f}°F"),
    "rmin": ("Minimum relative humidity", lambda v: f"{v:.0f}%"),
    "pr": ("Rain today", lambda v: f"{v:.1f} mm"),
    "pr_7d": ("Rain, last 7 days", lambda v: f"{v:.1f} mm"),
    "pr_30d": ("Rain, last 30 days", lambda v: f"{v:.1f} mm"),
    "dry_days": ("Consecutive dry days", lambda v: f"{v:.0f} days"),
    "erc_chg_7d": ("Fuel dryness change over 7 days", lambda v: f"{v:+.0f}"),
    "cell_prior_any": ("Fire history in this cell", lambda v: f"{v * 365:.2f} fires/yr"),
    "cell_prior_human": ("Human-caused fire history in this cell", lambda v: f"{v * 365:.2f} fires/yr"),
    "cell_prior_natural": ("Lightning fire history in this cell", lambda v: f"{v * 365:.2f} fires/yr"),
    "elevation": ("Elevation", lambda v: f"{v * 3.281:,.0f} ft"),
    "is_holiday_next": ("Tomorrow is a federal holiday", lambda v: "yes" if v else "no"),
    "is_weekend_next": ("Tomorrow is a weekend day", lambda v: "yes" if v else "no"),
    "is_july4_window": ("July 3-5 window", lambda v: "yes" if v else "no"),
    "month_next": ("Month", lambda v: f"{int(v)}"),
    "doy_sin": ("Time of year", lambda v: ""),
    "doy_cos": ("Time of year", lambda v: ""),
    "dow_next": ("Day of week", lambda v: f"{int(v)}"),
    "cell_lat": ("Location (latitude)", lambda v: f"{v:.2f}"),
    "cell_lon": ("Location (longitude)", lambda v: f"{v:.2f}"),
}


# features that describe the place, not today's weather
BASELINE_FEATURES = {"cell_prior_any", "cell_prior_human", "cell_prior_natural", "elevation", "cell_lat", "cell_lon"}
HIDE_FEATURES = {"cell_lat", "cell_lon"}      # location proxies: kept in the model, never shown to dispatchers


def tier_of(pct):
    for cut, name in TIERS:
        if pct >= cut:
            return name
    return "Normal"


@dataclass
class TargetModel:
    name: str
    booster: xgb.Booster
    calibrator: object
    info: dict
    feature_cols: list

    def raw(self, X):
        dm = xgb.DMatrix(X[self.feature_cols], feature_names=self.feature_cols)
        margin = self.booster.predict(dm, output_margin=True)
        prob = 1.0 / (1.0 + np.exp(-margin))
        return margin, prob

    def calibrated(self, prob):
        return np.clip(self.calibrator.predict(prob), 0.0, 1.0)

    def contributions(self, X):
        dm = xgb.DMatrix(X[self.feature_cols], feature_names=self.feature_cols)
        return self.booster.predict(dm, pred_contribs=True)   # (n, F+1); last column is the bias


@dataclass
class Forecast:
    feature_date: pd.Timestamp
    forecast_date: pd.Timestamp
    grid: pd.DataFrame
    features: pd.DataFrame = field(repr=False)
    summary: dict = field(default_factory=dict)


class RiskEngine:
    def __init__(self, model_dir=None, cache_dir=None, grid_meta=None, positives=None):
        self.model_dir = Path(model_dir or config.MODEL_DIR)
        self.builder = FeatureBuilder(cache_dir or config.CACHE_DIR, grid_meta or config.GRID_META,
                                      positives or config.POSITIVES)
        self.models = {}
        for t in ["y_any", "y_escaped", "y_human", "y_natural"]:
            d = self.model_dir / t
            if not (d / "model.json").exists():
                continue
            info = json.loads((d / "model_info.json").read_text())
            b = xgb.Booster()
            b.load_model(str(d / "model.json"))
            with open(d / "calibrator.pkl", "rb") as f:
                cal = pickle.load(f)
            self.models[t] = TargetModel(t, b, cal, info, info["feature_cols"])
        if "y_any" not in self.models:
            raise FileNotFoundError(f"No y_any model found in {self.model_dir}")

    # ------------------------------------------------------------------ forecasting
    def forecast(self, feature_date):
        X = self.builder.build(feature_date)
        any_m = self.models["y_any"]
        margin, prob = any_m.raw(X)
        grid = X[["lat_idx", "lon_idx", "cell_lat", "cell_lon"]].copy()
        grid["score"] = margin
        grid["p_ignition"] = any_m.calibrated(prob)
        grid["risk_percentile"] = pd.Series(margin).rank(pct=True, method="average").values
        grid["tier"] = [tier_of(p) for p in grid["risk_percentile"].values]
        if "y_escaped" in self.models:
            _, pe = self.models["y_escaped"].raw(X)
            grid["p_large_fire"] = self.models["y_escaped"].calibrated(pe)
            grid["p_escape_if_ignited"] = np.clip(
                grid["p_large_fire"] / np.maximum(grid["p_ignition"], 1e-9), 0, 1)
            grid.loc[grid["p_ignition"] <= 0, "p_escape_if_ignited"] = np.nan
        for t, col in [("y_human", "p_human"), ("y_natural", "p_natural")]:
            if t in self.models:
                _, p = self.models[t].raw(X)
                grid[col] = self.models[t].calibrated(p)

        D = pd.Timestamp(feature_date).normalize()
        # "unusual for this place today": today's probability vs this cell's normal for this month
        fac = any_m.info.get("seasonal_factor", {})
        m = int((D + pd.Timedelta(days=1)).month)
        f_m = float(fac.get(str(m), fac.get(m, 1.0)))
        normal = X["cell_prior_any"].values.astype(float) * f_m
        eps = 0.5 * float(np.mean(normal)) if np.mean(normal) > 0 else 1e-4   # shrink cells with little history
        grid["p_normal"] = normal
        grid["normal_eps"] = eps
        grid["risk_vs_normal"] = (grid["p_ignition"].values + eps) / (normal + eps)
        # only cells with meaningful risk today compete on "unusual" (top 25% of p_ignition)
        meaningful = grid["p_ignition"] >= grid["p_ignition"].quantile(0.75)
        grid["unusual_percentile"] = np.where(meaningful, grid["risk_vs_normal"].rank(pct=True).values, 0.0)
        if "p_large_fire" in grid:
            grid["large_fire_percentile"] = grid["p_large_fire"].rank(pct=True, method="average").values
        fc = Forecast(D, D + pd.Timedelta(days=1), grid, X)
        fc.summary = self._summary(fc)
        return fc

    def _summary(self, fc):
        g, X = fc.grid, fc.features
        exp_ign = float(g["p_ignition"].sum())
        # climatology expectation: cell history x seasonal factor for tomorrow's month
        fac = self.models["y_any"].info.get("seasonal_factor", {})
        month = int(fc.forecast_date.month)
        f_m = float(fac.get(str(month), fac.get(month, 1.0)))
        clim = float((X["cell_prior_any"] * f_m).sum())
        out = {
            "feature_date": str(fc.feature_date.date()),
            "forecast_date": str(fc.forecast_date.date()),
            "n_cells": int(len(g)),
            "expected_ignitions_statewide": round(exp_ign, 2),
            "climatology_expected_ignitions": round(clim, 2),
            "ratio_vs_normal": round(exp_ign / clim, 2) if clim > 0 else None,
            "tier_counts": g["tier"].value_counts().to_dict(),
            "p_ignition_max": float(g["p_ignition"].max()),
        }
        if "p_large_fire" in g:
            out["expected_large_fires_statewide"] = round(float(g["p_large_fire"].sum()), 3)
        out["data_quality"] = self.data_quality(X)
        return out

    # ------------------------------------------------------------------ QA
    @staticmethod
    def data_quality(X):
        """Physical-range and missing-value checks on the weather inputs."""
        ranges = {"erc": (0, 150), "bi": (0, 300), "fm100": (0, 60), "fm1000": (0, 60), "vs": (0, 40),
                  "vpd": (0, 10), "tmmx": (220, 335), "rmin": (0, 100), "pr": (0, 300)}
        issues = []
        for c, (lo, hi) in ranges.items():
            if c not in X:
                continue
            v = X[c]
            nan = float(v.isna().mean())
            bad = float(((v < lo) | (v > hi)).mean())
            if nan > 0.01:
                issues.append(f"{c}: {nan:.1%} missing")
            if bad > 0.001:
                issues.append(f"{c}: {bad:.1%} outside physical range [{lo}, {hi}]")
        return {"status": "ok" if not issues else "warning", "issues": issues}

    # ------------------------------------------------------------------ explanations
    def explain(self, fc, lat_idx, lon_idx, target="y_any", top=5, groups=None, hide=()):
        """Top drivers for one cell. Each driver has group "conditions" (weather, fuel, calendar) or
        "baseline" (this cell's fire history, elevation, location). `groups` filters, `hide` drops features."""
        m = self.models[target]
        X = fc.features
        row = X[(X["lat_idx"] == lat_idx) & (X["lon_idx"] == lon_idx)]
        if row.empty:
            raise KeyError(f"cell ({lat_idx}, {lon_idx}) is not in the forecast grid")
        contrib = m.contributions(row)[0]
        fc_cols = m.feature_cols
        # merge the time-of-year features
        items = {}
        for name, c in zip(fc_cols, contrib[:-1]):
            key = "time_of_year" if name in ("doy_sin", "doy_cos", "month_next") else name
            items[key] = items.get(key, 0.0) + float(c)
        ranked = sorted(items.items(), key=lambda kv: abs(kv[1]), reverse=True)
        out = []
        for name, c in ranked:
            grp = "baseline" if name in BASELINE_FEATURES else "conditions"
            if name in hide or (groups and grp not in groups):
                continue
            if name == "time_of_year":
                label, val = "Time of year", fc.forecast_date.strftime("%b %d")
            else:
                label, fmt = FEATURE_TEXT.get(name, (name, lambda v: f"{v:.3g}"))
                val = fmt(float(row[name].iloc[0]))
            out.append({"feature": name, "label": label, "value": val, "group": grp,
                        "effect": "raises risk" if c > 0 else "lowers risk", "contribution": round(c, 3)})
            if len(out) >= top:
                break
        return out

    def explain_split(self, fc, lat_idx, lon_idx, n_cond=3, n_base=2):
        """Dispatcher-friendly drivers: today's conditions vs the cell's long-term baseline.
        Latitude/longitude are shown to nobody: they only stand in for location."""
        return {"conditions": self.explain(fc, lat_idx, lon_idx, top=n_cond, groups=("conditions",)),
                "baseline": self.explain(fc, lat_idx, lon_idx, top=n_base, groups=("baseline",), hide=HIDE_FEATURES)}

    # ------------------------------------------------------------------ zones
    VIEWS = {
        "ignitions": ("risk_percentile", "most ignitions expected"),
        "unusual": ("unusual_percentile", "unusual for this place today"),
        "large_fire": ("large_fire_percentile", "highest large-fire potential"),
    }

    def zones(self, fc, min_zones=8, link=1, percentiles=(0.99, 0.97, 0.95, 0.90), view="ignitions",
              max_cells=25):
        """Group adjacent high-risk cells into watch zones for one view (see VIEWS).
        Zones larger than `max_cells` are split so a metro area cannot win on size alone. Zones are ranked by
        average per-cell value of the view's metric with a small bonus for size: mean * (1 + 0.25 ln n)."""
        g = fc.grid
        col, why = self.VIEWS[view]
        if col not in g:
            return []
        for pct in percentiles:
            cand = g[g[col] >= pct].reset_index(drop=True)
            zs = self._cluster(fc, cand, link, max_cells)
            if len(zs) >= min_zones:
                break
        metric = {"ignitions": "mean_p_ignition", "unusual": "risk_vs_normal", "large_fire": "mean_p_large_fire"}[view]
        for z in zs:
            z["rank_score"] = float(z.get(metric) or 0) * (1 + 0.25 * np.log(z["n_cells"]))
            z["view"], z["why_listed"], z["cutoff_percentile"] = view, why, pct
        zs.sort(key=lambda z: -z["rank_score"])
        for i, z in enumerate(zs, 1):
            z["zone_id"] = f"{view[0].upper()}{i}"
        return zs

    @staticmethod
    def _split(idx, cand, max_cells):
        """Recursively bisect a cluster along its longer side until each piece has <= max_cells cells."""
        if len(idx) <= max_cells:
            return [idx]
        sub = cand.iloc[idx]
        axis = "lat_idx" if np.ptp(sub.lat_idx.values) >= np.ptp(sub.lon_idx.values) else "lon_idx"
        order = np.argsort(sub[axis].values, kind="stable")
        half = len(idx) // 2
        a, b = [idx[i] for i in order[:half]], [idx[i] for i in order[half:]]
        return RiskEngine._split(a, cand, max_cells) + RiskEngine._split(b, cand, max_cells)

    def _cluster(self, fc, cand, link, max_cells=25):
        n = len(cand)
        if n == 0:
            return []
        pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cand.lat_idx, cand.lon_idx))}
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for (a, b), i in pos.items():
            for da in range(-link, link + 1):
                for db in range(-link, link + 1):
                    j = pos.get((a + da, b + db))
                    if j is not None and j != i:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[ri] = rj
        comp = {}
        for i in range(n):
            comp.setdefault(find(i), []).append(i)
        pieces = []
        for idx in comp.values():
            pieces += self._split(idx, cand, max_cells)
        comp = {k: v for k, v in enumerate(pieces)}
        has_esc = "p_large_fire" in cand
        zones = []
        for idx in comp.values():
            sub = cand.iloc[idx]
            anchor = sub.loc[sub["score"].idxmax()]
            lat_c = float(np.average(sub.cell_lat, weights=sub.p_ignition + 1e-12))
            lon_c = float(np.average(sub.cell_lon, weights=sub.p_ignition + 1e-12))
            cell_km2 = (111.32 / 24) * (111.32 * np.cos(np.radians(lat_c)) / 24)
            z = {
                "anchor_lat_idx": int(anchor.lat_idx), "anchor_lon_idx": int(anchor.lon_idx),
                "lat": round(float(anchor.cell_lat), 4), "lon": round(float(anchor.cell_lon), 4),
                "centroid_lat": round(lat_c, 4), "centroid_lon": round(lon_c, 4),
                "n_cells": int(len(sub)), "area_km2": round(float(len(sub) * cell_km2), 1),
                "bbox": {"lat_min": round(float(sub.cell_lat.min()), 4), "lat_max": round(float(sub.cell_lat.max()), 4),
                         "lon_min": round(float(sub.cell_lon.min()), 4), "lon_max": round(float(sub.cell_lon.max()), 4)},
                "tier": anchor.tier, "risk_percentile": round(float(sub.risk_percentile.max()), 4),
                "expected_ignitions": round(float(sub.p_ignition.sum()), 4),
                "p_ignition_max": round(float(sub.p_ignition.max()), 5),
                "mean_p_ignition": round(float(sub.p_ignition.mean()), 5),
                "normal_expected_ignitions": round(float(sub.p_normal.sum()), 4) if "p_normal" in sub else None,
                "risk_vs_normal": round(float((sub.p_ignition.sum() + len(sub) * sub.normal_eps.iloc[0]) /
                                              (sub.p_normal.sum() + len(sub) * sub.normal_eps.iloc[0])), 2)
                if "p_normal" in sub else None,
                "members": [[int(a), int(b)] for a, b in zip(sub.lat_idx, sub.lon_idx)],
            }
            if has_esc:
                z["expected_large_fires"] = round(float(sub.p_large_fire.sum()), 5)
                z["mean_p_large_fire"] = float(sub.p_large_fire.mean())
                z["p_escape_if_ignited"] = round(float(np.clip(sub.p_large_fire.sum() /
                                                               max(sub.p_ignition.sum(), 1e-9), 0, 1)), 4)
            z.update(self.explain_split(fc, z["anchor_lat_idx"], z["anchor_lon_idx"]))
            zones.append(z)
        return zones

    def watch_zones(self, fc, n_per_view=(("ignitions", 5), ("unusual", 4), ("large_fire", 4)), max_overlap=0.3):
        """Candidate zones from all three views, de-duplicated (a zone overlapping an already chosen zone
        by more than `max_overlap` of its cells is merged into it as an extra reason). IDs Z1..Zn."""
        chosen = []
        for view, n in n_per_view:
            for z in self.zones(fc, view=view)[: n * 2]:
                mem = {tuple(m) for m in z["members"]}
                dup = next((c for c in chosen if len(mem & c["_mem"]) > max_overlap * len(mem)), None)
                if dup is not None:
                    if z["why_listed"] not in dup["listed_for"]:
                        dup["listed_for"].append(z["why_listed"])
                    continue
                if sum(1 for c in chosen if c["view"] == view) >= n:
                    continue
                z["_mem"], z["listed_for"] = mem, [z["why_listed"]]
                chosen.append(z)
        # interleave views so the top of the list covers all three (ignitions, unusual, large fire, ...)
        by = {v: [z for z in chosen if z["view"] == v] for v, _ in n_per_view}
        order = []
        while any(by.values()):
            for v, _ in n_per_view:
                if by[v]:
                    order.append(by[v].pop(0))
        chosen = order
        for i, z in enumerate(chosen, 1):
            z.pop("_mem")
            z["view_rank_id"], z["zone_id"] = z["zone_id"], f"Z{i}"
        return chosen

    @staticmethod
    def zone_reason(z):
        def fmt(ds):
            return ", ".join(f"{d['label']} {d['value']}".strip() + (" (raises risk)" if d["effect"] == "raises risk"
                             else " (lowers risk)") for d in ds)
        return f"Today: {fmt(z['conditions'])}. Baseline: {fmt(z['baseline'])}."

    def top_cells(self, fc, n=10, explain=True):
        g = fc.grid.sort_values("score", ascending=False).head(n)
        rows = []
        for _, r in g.iterrows():
            d = {"lat_idx": int(r.lat_idx), "lon_idx": int(r.lon_idx),
                 "lat": round(float(r.cell_lat), 4), "lon": round(float(r.cell_lon), 4),
                 "tier": r.tier, "risk_percentile": round(float(r.risk_percentile), 4),
                 "p_ignition": round(float(r.p_ignition), 5)}
            if "p_large_fire" in r:
                d["p_large_fire"] = round(float(r.p_large_fire), 6)
                d["p_escape_if_ignited"] = None if pd.isna(r.p_escape_if_ignited) \
                    else round(float(r.p_escape_if_ignited), 4)
            if explain:
                d["drivers"] = self.explain(fc, int(r.lat_idx), int(r.lon_idx))
            rows.append(d)
        return rows

    def cell_record(self, fc, lat_idx, lon_idx):
        g = fc.grid
        r = g[(g["lat_idx"] == lat_idx) & (g["lon_idx"] == lon_idx)]
        if r.empty:
            return None
        r = r.iloc[0]
        d = {k: (round(float(r[k]), 6) if isinstance(r[k], (float, np.floating)) else
                 (int(r[k]) if isinstance(r[k], (int, np.integer)) else r[k]))
             for k in r.index if k != "score"}
        d["drivers"] = self.explain(fc, lat_idx, lon_idx)
        if "y_escaped" in self.models:
            d["large_fire_drivers"] = self.explain(fc, lat_idx, lon_idx, target="y_escaped", top=3)
        return d
