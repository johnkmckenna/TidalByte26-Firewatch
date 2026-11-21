"""
Download gridMET daily weather for the study box and cache it as small .npz files.

For every (variable, year) it fetches the gridMET NetCDF, keeps only the cells in the
study box (from grid.json written by build_cellday_labels.py), and writes
    <cache>/<var>_<year>.npz   with arrays: data (days, lat, lon), dates, lat_idx, lon_idx
Existing files are skipped, so you can stop and re-run it safely.

Sources
-------
--source opendap  (default) reads only the box through NOAA/UI's THREDDS OPeNDAP server.
                  Small transfers, but it can be slow or flaky; failures are retried.
--source http     downloads each whole-CONUS yearly file, subsets it, then deletes it.
                  More reliable, much more data (each file is hundreds of MB).
--source local    reads files already on disk, named <var>_<year>.nc, from --local-dir.

Example
-------
    python download_gridmet.py --meta out/grid.json --cache gridmet_cache
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_VARS = ["erc", "bi", "fm100", "fm1000", "vs", "vpd", "tmmx", "rmin", "pr"]
URL_HTTP = "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
URL_DAP = "http://thredds.northwestknowledge.net/thredds/dodsC/MET/{var}/{var}_{year}.nc#fillmismatch"
URL_ELEV = "https://climate.northwestknowledge.net/METDATA/data/metdata_elevationdata.nc"


def box_index_range(meta):
    """Integer cell-index ranges (lat_idx, lon_idx) of the study box, same rule as the label script."""
    s, n, w, e = meta["bbox_south_north_west_east"]
    lat_all = meta["lat0"] + np.arange(meta["n_lat"]) * meta["step_deg"]
    lon_all = meta["lon0"] + np.arange(meta["n_lon"]) * meta["step_deg"]
    ii = np.where((lat_all >= s) & (lat_all <= n))[0]
    jj = np.where((lon_all >= w) & (lon_all <= e))[0]
    return int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())


def find_names(ds):
    lat = next((n for n in ("lat", "latitude") if n in ds.variables), None)
    lon = next((n for n in ("lon", "longitude") if n in ds.variables), None)
    if lat is None or lon is None:
        raise RuntimeError(f"Could not find lat/lon coordinates; variables: {list(ds.variables)}")
    return lat, lon


def contiguous_slice(idx, lo, hi):
    """Positions in the coordinate array whose cell index is within [lo, hi], as one slice."""
    pos = np.where((idx >= lo) & (idx <= hi))[0]
    if len(pos) == 0:
        raise RuntimeError(f"No cells in index range {lo}..{hi}; coordinates span {idx.min()}..{idx.max()}")
    return slice(int(pos.min()), int(pos.max()) + 1)


def subset_dataset(ds, meta, want_dims=3):
    """Return (data, dates, lat_idx, lon_idx) for the study box, cells sorted by ascending index."""
    i0, i1, j0, j1 = box_index_range(meta)
    latn, lonn = find_names(ds)
    step, lat0, lon0 = meta["step_deg"], meta["lat0"], meta["lon0"]
    lat_idx_all = np.rint((ds[latn].values - lat0) / step).astype(int)
    lon_idx_all = np.rint((ds[lonn].values - lon0) / step).astype(int)
    sl_lat = contiguous_slice(lat_idx_all, i0, i1)
    sl_lon = contiguous_slice(lon_idx_all, j0, j1)

    cands = [v for v in ds.data_vars if ds[v].ndim >= 2 and latn in ds[v].dims and lonn in ds[v].dims]
    if not cands:
        raise RuntimeError(f"No gridded data variable found; data_vars: {list(ds.data_vars)}")
    name = cands[0]
    da = ds[name]
    tdim = next((d for d in da.dims if d not in (latn, lonn)), None)
    da = da.isel({latn: sl_lat, lonn: sl_lon}).load()    # xarray applies scale_factor/offset here
    lat_idx = lat_idx_all[sl_lat]
    lon_idx = lon_idx_all[sl_lon]
    if tdim is None:
        arr = da.transpose(latn, lonn).values[None]
        dates = np.array(["1970-01-01"], dtype="datetime64[D]")
    else:
        arr = da.transpose(tdim, latn, lonn).values
        dates = np.asarray(ds[tdim].values).astype("datetime64[D]")
    # sort cells so index arrays are ascending, whatever order the file used
    oi, oj = np.argsort(lat_idx), np.argsort(lon_idx)
    arr = arr[:, oi][:, :, oj].astype(np.float32)
    return arr, dates, lat_idx[oi], lon_idx[oj], name


def open_source(source, var, year, local_dir, cache):
    """Open one gridMET file. Returns (dataset, temp_path_to_delete_or_None)."""
    if source == "opendap":
        return xr.open_dataset(URL_DAP.format(var=var, year=year), engine="netcdf4"), None
    if source == "local":
        p = Path(local_dir) / f"{var}_{year}.nc"
        if not p.exists():
            raise FileNotFoundError(p)
        return xr.open_dataset(p), None
    tmp = Path(cache) / f"_tmp_{var}_{year}.nc"
    req = urllib.request.Request(URL_HTTP.format(var=var, year=year), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return xr.open_dataset(tmp), tmp


def fetch_one(job):
    var, year, source, local_dir, cache, meta, retries = job
    out = Path(cache) / f"{var}_{year}.npz"
    if out.exists():
        return var, year, "cached", None
    last = None
    for attempt in range(1, retries + 1):
        tmp = None
        try:
            ds, tmp = open_source(source, var, year, local_dir, cache)
            arr, dates, lat_idx, lon_idx, name = subset_dataset(ds, meta)
            ds.close()
            np.savez_compressed(out, data=arr, dates=dates.astype("datetime64[D]").astype("int64"),
                                lat_idx=lat_idx, lon_idx=lon_idx, ncvar=name)
            return var, year, "ok", None
        except FileNotFoundError as e:
            return var, year, "missing", str(e)
        except Exception as e:  # network hiccups, server errors
            last = f"{type(e).__name__}: {e}"
            if "404" in last or "Not Found" in last:
                return var, year, "missing", last
            time.sleep(2 * attempt)
        finally:
            if tmp is not None and Path(tmp).exists():
                Path(tmp).unlink()
    return var, year, "failed", last


def fetch_elevation(meta, cache, source, local_dir):
    out = Path(cache) / "elevation.npz"
    if out.exists():
        return "cached"
    tmp = None
    try:
        if source == "local":
            p = Path(local_dir) / "metdata_elevationdata.nc"
            if not p.exists():
                return "skipped (no local file)"
            ds = xr.open_dataset(p)
        else:
            tmp = Path(cache) / "_tmp_elevation.nc"
            req = urllib.request.Request(URL_ELEV, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            ds = xr.open_dataset(tmp)
        arr, _, lat_idx, lon_idx, name = subset_dataset(ds, meta)
        ds.close()
        np.savez_compressed(out, data=arr[0].astype(np.float32), lat_idx=lat_idx, lon_idx=lon_idx, ncvar=name)
        return f"ok (variable {name!r})"
    except Exception as e:
        return f"skipped ({type(e).__name__}: {e}); elevation is optional"
    finally:
        if tmp is not None and Path(tmp).exists():
            Path(tmp).unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="grid.json from build_cellday_labels.py")
    ap.add_argument("--cache", required=True, help="output directory for the .npz files")
    ap.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="default: (first fire year - 1) .. last fire year")
    ap.add_argument("--source", choices=["opendap", "http", "local"], default="opendap")
    ap.add_argument("--local-dir", default=".", help="for --source local: folder with <var>_<year>.nc")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--no-elevation", action="store_true")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    first = int(meta["first_label_date"][:4]) - 1       # one extra year for trailing-window features
    last = int(meta["last_label_date"][:4])
    if args.years:
        first, last = args.years
    last = min(last, date.today().year)
    years = list(range(first, last + 1))
    jobs = [(v, y, args.source, args.local_dir, str(cache), meta, args.retries)
            for y in years for v in args.vars]
    print(f"{len(jobs)} files: {len(args.vars)} variables x years {first}-{last}, source={args.source}")

    done = {"ok": 0, "cached": 0, "missing": 0, "failed": 0}
    problems = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            var, year, status, msg = f.result()
            done[status] += 1
            if status in ("missing", "failed"):
                problems.append((var, year, status, msg))
            if k % 10 == 0 or status in ("missing", "failed"):
                print(f"[{k}/{len(jobs)}] {var}_{year}: {status}" + (f"  ({msg})" if msg else ""), flush=True)

    if not args.no_elevation:
        print("Elevation:", fetch_elevation(meta, cache, args.source, args.local_dir))

    print("\nSummary:", done)
    if problems:
        print("Problems (re-run to retry the 'failed' ones; 'missing' means the server has no such file):")
        for p in sorted(problems):
            print("  ", p)
        if any(p[2] == "failed" for p in problems) and args.source == "opendap":
            print("Many failures with opendap? Try:  --source http")
        sys.exit(1)


if __name__ == "__main__":
    main()