#!/usr/bin/env python3
"""
Download GeoNames "cities1000" (every populated place with 1,000+ people, free, CC-BY) and keep one state.
Agent 2 uses it to name the nearest towns for each watch zone.

    python3 build_places.py                 # Colorado -> out/places.csv
    python3 build_places.py --state CA --out out/places_ca.csv
"""
import argparse
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://download.geonames.org/export/dump/cities1000.zip"
COLS = ["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude", "feature_class",
        "feature_code", "country", "cc2", "admin1", "admin2", "admin3", "admin4", "population", "elevation",
        "dem", "timezone", "modified"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="CO")
    ap.add_argument("--out", default="out/places.csv")
    ap.add_argument("--min-pop", type=int, default=1000)
    ap.add_argument("--zip", help="use an already-downloaded cities1000.zip instead of downloading")
    args = ap.parse_args()

    if args.zip:
        raw = Path(args.zip).read_bytes()
    else:
        print("Downloading", URL)
        raw = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "firewatch"}), timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    rows = []
    with z.open("cities1000.txt") as f:
        for rec in csv.reader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(rec) < len(COLS) - 1:
                continue
            d = dict(zip(COLS, rec))
            if d["country"] != "US" or d["admin1"] != args.state:
                continue
            pop = int(d["population"] or 0)
            if pop < args.min_pop:
                continue
            rows.append({"name": d["name"], "latitude": float(d["latitude"]), "longitude": float(d["longitude"]),
                         "population": pop})
    rows.sort(key=lambda r: -r["population"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "latitude", "longitude", "population"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} places in {args.state} to {out}")
    print("Largest:", ", ".join(f"{r['name']} ({r['population']:,})" for r in rows[:5]))


if __name__ == "__main__":
    main()