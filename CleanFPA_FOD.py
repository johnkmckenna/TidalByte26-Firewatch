#!/usr/bin/env python3
"""
Build the sparse cell-day label table from the FPA-FOD SQLite file.

What it produces
----------------
cellday_positives.parquet : one row per (grid cell, discovery date) that had >= 1 fire.
    Every cell-day NOT in this table is a negative (no fire discovered) by definition.
    Negatives are created later, when features are joined, so we never store the
    giant cell x day cross-product.
grid.json                 : the grid definition, so the gridMET feature builder uses
                            exactly the same cell indexing.

Grid
----
gridMET: 1/24 degree (~4 km) cells, centers from 25.0667N to 49.4N and
124.7667W to 67.0583W. A fire belongs to the cell whose center is nearest.
Cells are identified by integer indices (lat_idx, lon_idx) computed from
coordinates, NOT from array position, so it does not matter whether a gridMET
file stores latitude ascending or descending.

Labels
------
Rows are keyed by the fire DISCOVERY date. To predict "tomorrow", features from
day D are paired with fires discovered on D+1. The column `feature_date` is
already label_date - 1 day, so the feature builder can join on it directly.

Usage
-----
    python build_cellday_labels.py --db FPA_FOD_20170508.sqlite --out out_dir
    python build_cellday_labels.py --db ... --out out_dir --state CO \
        --bbox 37.0 41.0 -109.05 -102.05

If a column name differs in your edition, run once: the script prints the
columns it found and tells you which ones it could not find.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- gridMET grid constants (cell centers) ---------------------------------
STEP = 1.0 / 24.0
LAT0 = 25.066666666666666   # southernmost cell center
LON0 = -124.76666666666667  # westernmost cell center
N_LAT = 585
N_LON = 1386

# Colorado is almost exactly a lat/lon rectangle, so a bounding box works.
# (south, north, west, east). Use a polygon mask for non-rectangular states.
DEFAULT_BBOX = {"CO": (37.0, 41.0, -109.05, -102.05)}

# ---- column names (override here if your edition differs) ------------------
COL_LAT = "LATITUDE"
COL_LON = "LONGITUDE"
COL_DATE = "DISCOVERY_DATE"
COL_DOY = "DISCOVERY_DOY"        # optional, used only for a sanity check
COL_YEAR = "FIRE_YEAR"
COL_STATE = "STATE"
COL_SIZE = "FIRE_SIZE"
COL_SIZE_CLASS = "FIRE_SIZE_CLASS"
COL_CAUSE_CLASS = "NWCG_CAUSE_CLASSIFICATION"  # optional
COL_CAUSE_DESC = "STAT_CAUSE_DESCR"            # optional fallback


def latlon_to_idx(lat, lon):
    """Nearest gridMET cell center -> integer (lat_idx, lon_idx)."""
    lat_idx = np.rint((np.asarray(lat) - LAT0) / STEP).astype(int)
    lon_idx = np.rint((np.asarray(lon) - LON0) / STEP).astype(int)
    return lat_idx, lon_idx


def idx_to_latlon(lat_idx, lon_idx):
    return LAT0 + np.asarray(lat_idx) * STEP, LON0 + np.asarray(lon_idx) * STEP


def find_table(con, forced=None):
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
    print("Tables in file:", tables)
    if forced:
        return forced
    # Newer editions carry spatial-index virtual tables (st_spindex__*) that need a
    # SQLite module Python does not have; inspecting them raises an error. Prefer the
    # table literally named "Fires", and skip any table that cannot be inspected.
    for t in tables:
        if t.lower() == "fires":
            return t
    for t in tables:
        if t.lower().startswith(("st_", "sqlite_")):
            continue
        try:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        except sqlite3.OperationalError:
            continue
        if COL_LAT in cols and COL_LON in cols:
            return t
    sys.exit("No table with LATITUDE/LONGITUDE found. Pass --table NAME.")


def parse_dates(series):
    """DISCOVERY_DATE is a Julian day number in the 5th-edition SQLite; text elsewhere."""
    if pd.api.types.is_numeric_dtype(series):
        med = float(series.dropna().median())
        if 2_000_000 < med < 3_000_000:      # Julian day number
            return pd.to_datetime(series, unit="D", origin="julian").dt.floor("D")
        if 1e11 < med < 1e14:                # epoch milliseconds
            return pd.to_datetime(series, unit="ms", errors="coerce").dt.floor("D")
        if 1e8 < med < 1e11:                 # epoch seconds
            return pd.to_datetime(series, unit="s", errors="coerce").dt.floor("D")
        sys.exit(f"Numeric dates with median {med} are in an unrecognised format; handle manually.")
    # Text dates (ISO or MM/DD/YYYY). A numeric string column is converted first.
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.9:
        return parse_dates(num)
    return pd.to_datetime(series, errors="coerce").dt.floor("D")


def classify_cause(df):
    """Return 'human' / 'natural' / 'unknown' for each fire."""
    if COL_CAUSE_CLASS in df.columns:
        s = df[COL_CAUSE_CLASS].astype(str).str.lower()
        out = np.where(s.str.contains("human"), "human",
              np.where(s.str.contains("natural"), "natural", "unknown"))
        return pd.Series(out, index=df.index)
    if COL_CAUSE_DESC in df.columns:
        s = df[COL_CAUSE_DESC].astype(str).str.lower()
        out = np.where(s.str.contains("lightning"), "natural",
              np.where(s.str.contains("missing|undefined|unknown|miscellaneous"), "unknown", "human"))
        print("NOTE: no NWCG cause classification column; derived cause from", COL_CAUSE_DESC,
              "(lightning=natural, missing/undefined/miscellaneous=unknown, all else=human).")
        return pd.Series(out, index=df.index)
    print("WARNING: no cause column found; all fires labelled 'unknown'.")
    return pd.Series("unknown", index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to the FPA-FOD .sqlite file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--table", default=None, help="table name (auto-detected if omitted)")
    ap.add_argument("--state", default="CO", help="two-letter state to keep (default CO)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "N", "W", "E"),
                    help="south north west east in degrees; default is the state's box")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    db = Path(args.db)
    if not db.is_file():
        near = sorted(p.name for p in (db.parent if str(db.parent) else Path(".")).glob("*.sqlite*"))
        sys.exit(f"File not found: {db}\n"
                 f"Check the filename (ls -la). .sqlite-like files in that folder: {near or 'none'}\n"
                 f"If you only have a .zip, unzip it first.")
    if db.stat().st_size == 0:
        sys.exit(f"{db} is 0 bytes (an empty file, possibly created by an earlier failed run). "
                 f"Delete it and point --db at the real .sqlite file.")
    # Open read-only so SQLite can never create an empty database by accident.
    con = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    table = find_table(con, args.table)
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
    print(f"Using table {table!r} with {len(cols)} columns:\n  ", ", ".join(cols))

    required = [COL_LAT, COL_LON, COL_DATE]
    missing = [c for c in required if c not in cols]
    if missing:
        sys.exit(f"Missing required columns {missing}. Edit the COL_* constants at the top.")

    want = [c for c in [COL_LAT, COL_LON, COL_DATE, COL_DOY, COL_YEAR, COL_STATE, COL_SIZE,
                        COL_SIZE_CLASS, COL_CAUSE_CLASS, COL_CAUSE_DESC] if c in cols]
    sql = f'SELECT {", ".join(chr(34) + c + chr(34) for c in want)} FROM "{table}"'
    df = pd.read_sql(sql, con)
    n_all = len(df)
    print(f"\nLoaded {n_all:,} fire records.")

    # ---- state filter ------------------------------------------------------
    if COL_STATE in df.columns and args.state:
        df = df[df[COL_STATE] == args.state].copy()
        print(f"After STATE == {args.state}: {len(df):,}")

    # ---- dates -------------------------------------------------------------
    df["label_date"] = parse_dates(df[COL_DATE])
    n_bad = df["label_date"].isna().sum()
    if n_bad:
        print(f"Dropping {n_bad:,} rows with unparseable dates.")
        df = df[df["label_date"].notna()].copy()

    # Sanity check: Julian date vs. FIRE_YEAR / DISCOVERY_DOY, when both exist.
    if COL_YEAR in df.columns and COL_DOY in df.columns:
        chk = df[[COL_YEAR, COL_DOY]].apply(pd.to_numeric, errors="coerce")
        chk["label_date"] = df["label_date"]
        chk = chk.dropna()
        doy_date = pd.to_datetime(chk[COL_YEAR].astype(int).astype(str) + "-01-01") + \
                   pd.to_timedelta(chk[COL_DOY].astype(int) - 1, unit="D")
        mismatch = int((doy_date != chk["label_date"]).sum())
        print(f"Date check vs FIRE_YEAR+DISCOVERY_DOY: {mismatch:,} of {len(chk):,} rows disagree "
              f"({mismatch / max(len(chk), 1):.2%}). Near 0% is good; ~100% means an off-by-one or wrong epoch.")

    df["feature_date"] = df["label_date"] - pd.Timedelta(days=1)

    # ---- grid cells --------------------------------------------------------
    south, north, west, east = args.bbox if args.bbox else DEFAULT_BBOX.get(args.state, (None,) * 4)
    if south is None:
        sys.exit(f"No default bbox for {args.state}; pass --bbox S N W E (or add a polygon mask).")

    lat_idx, lon_idx = latlon_to_idx(df[COL_LAT].values, df[COL_LON].values)
    df["lat_idx"], df["lon_idx"] = lat_idx, lon_idx
    clat, clon = idx_to_latlon(lat_idx, lon_idx)
    inside = (clat >= south) & (clat <= north) & (clon >= west) & (clon <= east) \
             & (lat_idx >= 0) & (lat_idx < N_LAT) & (lon_idx >= 0) & (lon_idx < N_LON)
    print(f"Fires whose cell center is inside the box: {inside.sum():,} of {len(df):,} "
          f"(dropped {(~inside).sum():,}).")
    df = df[inside].copy()

    # Number of cells in the study box (for the base rate).
    lat_all = LAT0 + np.arange(N_LAT) * STEP
    lon_all = LON0 + np.arange(N_LON) * STEP
    n_cells = int(((lat_all >= south) & (lat_all <= north)).sum()
                  * ((lon_all >= west) & (lon_all <= east)).sum())

    # ---- per-fire flags ----------------------------------------------------
    df["cause"] = classify_cause(df)
    df["is_human"] = (df["cause"] == "human").astype(int)
    df["is_natural"] = (df["cause"] == "natural").astype(int)
    df["is_unknown"] = (df["cause"] == "unknown").astype(int)
    if COL_SIZE_CLASS in df.columns:
        df["is_escaped"] = df[COL_SIZE_CLASS].astype(str).str.upper().isin(list("CDEFG")).astype(int)
    else:
        df["is_escaped"] = 0
        print("WARNING: no FIRE_SIZE_CLASS; escaped-fire label (class C+) set to 0.")
    df["acres"] = df[COL_SIZE] if COL_SIZE in df.columns else np.nan

    # ---- aggregate to cell-days -------------------------------------------
    g = df.groupby(["lat_idx", "lon_idx", "label_date", "feature_date"], as_index=False).agg(
        n_fires=("cause", "size"),
        n_human=("is_human", "sum"),
        n_natural=("is_natural", "sum"),
        n_unknown=("is_unknown", "sum"),
        n_escaped=("is_escaped", "sum"),
        max_acres=("acres", "max"),
    )
    # Binary labels. y_human / y_natural treat a day with only the OTHER cause as negative.
    g["y_any"] = (g["n_fires"] > 0).astype(np.int8)
    g["y_human"] = (g["n_human"] > 0).astype(np.int8)
    g["y_natural"] = (g["n_natural"] > 0).astype(np.int8)
    g["y_escaped"] = (g["n_escaped"] > 0).astype(np.int8)
    g["cell_lat"], g["cell_lon"] = idx_to_latlon(g["lat_idx"], g["lon_idx"])
    g["year"] = g["label_date"].dt.year

    g.to_parquet(out / "cellday_positives.parquet", index=False)

    d0, d1 = df["label_date"].min(), df["label_date"].max()
    n_days = (d1 - d0).days + 1
    meta = {
        "step_deg": STEP, "lat0": LAT0, "lon0": LON0, "n_lat": N_LAT, "n_lon": N_LON,
        "state": args.state, "bbox_south_north_west_east": [south, north, west, east],
        "n_cells_in_box": n_cells, "first_label_date": str(d0.date()), "last_label_date": str(d1.date()),
        "n_days": n_days, "n_positive_cell_days": int(len(g)), "n_fires": int(len(df)),
        "note": "feature_date = label_date - 1. Cell-days absent from the parquet are negatives.",
    }
    (out / "grid.json").write_text(json.dumps(meta, indent=2))

    # ---- report ------------------------------------------------------------
    total_cd = n_cells * n_days
    print("\n================ SUMMARY ================")
    print(f"Study box cells: {n_cells:,}   days: {n_days:,}   cell-days: {total_cd:,}")
    print(f"Fires kept: {len(df):,}   positive cell-days: {len(g):,}   "
          f"base rate: {len(g) / total_cd:.5%}  (1 in {total_cd / max(len(g), 1):,.0f})")
    multi = (g['n_fires'] > 1).sum()
    print(f"Cell-days with 2+ fires: {multi:,} ({multi / max(len(g), 1):.1%})")
    print("\nFires by cause:\n", df["cause"].value_counts().to_string())
    if COL_SIZE_CLASS in df.columns:
        print("\nFires by size class:\n", df[COL_SIZE_CLASS].value_counts().sort_index().to_string())
    print("\nPositive cell-days by year:\n", g.groupby("year").size().to_string())
    print(f"\nWrote {out / 'cellday_positives.parquet'} and {out / 'grid.json'}")


if __name__ == "__main__":
    main()