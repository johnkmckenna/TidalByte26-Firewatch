"""
Live data: refresh recent gridMET days into the cache, and pull NASA FIRMS hotspots.

gridMET  current-year files grow daily (data lag ~1-2 days). refresh_gridmet_year() re-reads
         one year for the study box through OPeNDAP and overwrites the cached .npz.
FIRMS    Area API, CSV:  /api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{DATE}
         DAY_RANGE is 1-5 days. NRT sources cover recent data; *_SP (standard) for older dates.
"""
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

GRIDMET_VARS = ["erc", "bi", "fm100", "fm1000", "vs", "vpd", "tmmx", "rmin", "pr"]
URL_DAP = "http://thredds.northwestknowledge.net/thredds/dodsC/MET/{var}/{var}_{year}.nc#fillmismatch"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}"
NRT_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
SP_SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]


# ------------------------------------------------------------------------------ gridMET
def _box_index_range(meta):
    s, n, w, e = meta["bbox_south_north_west_east"]
    lat_all = meta["lat0"] + np.arange(meta["n_lat"]) * meta["step_deg"]
    lon_all = meta["lon0"] + np.arange(meta["n_lon"]) * meta["step_deg"]
    ii = np.where((lat_all >= s) & (lat_all <= n))[0]
    jj = np.where((lon_all >= w) & (lon_all <= e))[0]
    return int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())


def _subset(ds, meta):
    i0, i1, j0, j1 = _box_index_range(meta)
    latn = next(n for n in ("lat", "latitude") if n in ds.variables)
    lonn = next(n for n in ("lon", "longitude") if n in ds.variables)
    li_all = np.rint((ds[latn].values - meta["lat0"]) / meta["step_deg"]).astype(int)
    lj_all = np.rint((ds[lonn].values - meta["lon0"]) / meta["step_deg"]).astype(int)
    pi = np.where((li_all >= i0) & (li_all <= i1))[0]
    pj = np.where((lj_all >= j0) & (lj_all <= j1))[0]
    name = [v for v in ds.data_vars if latn in ds[v].dims and lonn in ds[v].dims][0]
    da = ds[name].isel({latn: slice(pi.min(), pi.max() + 1), lonn: slice(pj.min(), pj.max() + 1)}).load()
    tdim = [d for d in da.dims if d not in (latn, lonn)][0]
    arr = da.transpose(tdim, latn, lonn).values
    li, lj = li_all[pi.min():pi.max() + 1], lj_all[pj.min():pj.max() + 1]
    oi, oj = np.argsort(li), np.argsort(lj)
    return arr[:, oi][:, :, oj].astype(np.float32), np.asarray(ds[tdim].values).astype("datetime64[D]"), li[oi], lj[oj], name


def refresh_gridmet_year(year, cache_dir=None, grid_meta=None, variables=GRIDMET_VARS, retries=3):
    """Re-download one year of gridMET for the study box (overwrites the cache). Returns status dict."""
    import xarray as xr
    cache = Path(cache_dir or config.CACHE_DIR)
    meta = json.loads(Path(grid_meta or config.GRID_META).read_text())
    cache.mkdir(parents=True, exist_ok=True)
    done, failed, last_dates = [], {}, {}
    for var in variables:
        err = None
        for attempt in range(retries):
            try:
                ds = xr.open_dataset(URL_DAP.format(var=var, year=year), engine="netcdf4")
                arr, dates, li, lj, name = _subset(ds, meta)
                ds.close()
                tmp = cache / f"{var}_{year}.tmp.npz"
                np.savez_compressed(tmp, data=arr, dates=dates.astype("int64"), lat_idx=li, lon_idx=lj, ncvar=name)
                tmp.replace(cache / f"{var}_{year}.npz")
                done.append(var)
                last_dates[var] = str(pd.Timestamp(dates.max()).date())
                err = None
                break
            except Exception as e:  # network errors are common on this server; retry
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        if err:
            failed[var] = err[:200]
    return {"year": year, "refreshed": done, "failed": failed,
            "latest_date_by_variable": last_dates,
            "latest_common_date": min(last_dates.values()) if last_dates and not failed else None}


# ------------------------------------------------------------------------------ FIRMS
def _norm_confidence(series):
    def one(v):
        s = str(v).strip().lower()
        if s in ("l", "low"):
            return "low"
        if s in ("n", "nominal"):
            return "nominal"
        if s in ("h", "high"):
            return "high"
        try:
            x = float(s)
            return "low" if x < 30 else ("nominal" if x < 80 else "high")
        except ValueError:
            return "unknown"
    return series.map(one)


def parse_firms_csv(text, source):
    if not text.strip() or text.lstrip().lower().startswith(("invalid", "error", "<")):
        raise RuntimeError(f"FIRMS returned no CSV for {source}: {text[:160]!r}")
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    out = pd.DataFrame({
        "latitude": df["latitude"].astype(float),
        "longitude": df["longitude"].astype(float),
        "acq_date": pd.to_datetime(df["acq_date"]),
        "acq_time": df.get("acq_time", pd.Series(0, index=df.index)).astype(str).str.zfill(4),
        "confidence": _norm_confidence(df["confidence"]) if "confidence" in df else "unknown",
        "frp": df.get("frp", pd.Series(np.nan, index=df.index)).astype(float),
        "daynight": df.get("daynight", pd.Series("", index=df.index)),
        "source": source,
    })
    return out


def fetch_firms(bbox, day_range=2, date=None, archive=False, map_key=None, fetcher=None):
    """Hotspots in bbox (south, north, west, east). `fetcher(url)->text` can be injected for tests."""
    key = map_key or config.FIRMS_MAP_KEY
    if not key and fetcher is None:
        raise RuntimeError("FIRMS_MAP_KEY is not set (free key: firms.modaps.eosdis.nasa.gov/api/map_key).")
    s, n, w, e = bbox
    area = f"{w},{s},{e},{n}"
    day_range = int(min(max(day_range, 1), 5))
    frames, errors = [], {}
    for src in (SP_SOURCES if archive else NRT_SOURCES):
        url = FIRMS_URL.format(key=key, source=src, area=area, days=day_range)
        if date is not None:
            url += f"/{pd.Timestamp(date).date()}"
        try:
            if fetcher is not None:
                text = fetcher(url)
            else:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "firewatch"}),
                                            timeout=60) as r:
                    text = r.read().decode("utf-8", "replace")
            frames.append(parse_firms_csv(text, src))
        except urllib.error.HTTPError as ex:      # keep the server's explanation (it says why a 400 happened)
            try:
                body = ex.read().decode("utf-8", "replace").strip().replace("\n", " ")[:200]
            except Exception:
                body = ""
            errors[src] = f"HTTPError {ex.code}: {body or ex.reason}"
        except Exception as ex:
            errors[src] = f"{type(ex).__name__}: {str(ex)[:160]}"
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) \
        else pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight", "source"])
    return df, errors


def find_new_ignitions(hotspots, meta, target_date, lookback_days=3, radius_cells=2, min_confidence="nominal"):
    """
    Cells with a hotspot on target_date and NO hotspot within `radius_cells` (~4 km per cell)
    during the previous `lookback_days`. Returns one row per detected cell.
    """
    if hotspots is None or len(hotspots) == 0:
        return pd.DataFrame()
    h = hotspots.copy()
    rank = {"low": 0, "unknown": 0, "nominal": 1, "high": 2}
    floor = rank.get(min_confidence, 1)
    h["lat_idx"] = np.rint((h["latitude"] - meta["lat0"]) / meta["step_deg"]).astype(int)
    h["lon_idx"] = np.rint((h["longitude"] - meta["lon0"]) / meta["step_deg"]).astype(int)
    T = pd.Timestamp(target_date).normalize()
    today = h[(h["acq_date"] == T) & (h["confidence"].map(rank).fillna(0) >= floor)]
    before = h[(h["acq_date"] < T) & (h["acq_date"] >= T - pd.Timedelta(days=lookback_days))]
    if today.empty:
        return pd.DataFrame()
    agg = today.groupby(["lat_idx", "lon_idx"]).agg(
        n_detections=("frp", "size"), max_frp=("frp", "max"), first_time=("acq_time", "min"),
        confidence=("confidence", lambda s: "high" if (s == "high").any() else "nominal"),
        latitude=("latitude", "mean"), longitude=("longitude", "mean"),
        sources=("source", lambda s: ",".join(sorted(set(s))))).reset_index()
    prior = before[["lat_idx", "lon_idx"]].drop_duplicates().values
    def isolated(i, j):
        if len(prior) == 0:
            return True
        return not ((np.abs(prior[:, 0] - i) <= radius_cells) & (np.abs(prior[:, 1] - j) <= radius_cells)).any()
    agg["new_ignition"] = [isolated(i, j) for i, j in zip(agg.lat_idx, agg.lon_idx)]
    return agg
