"""
Town lookup for watch zones: nearest towns, towns inside a zone, and a plain location label
("9 km NE of Pueblo"). Uses out/places.csv from build_places.py (GeoNames, 1,000+ people).
If that file is missing, a small built-in list of larger Colorado towns is used (approximate
coordinates) and the result says so.
"""
import csv
import math
from pathlib import Path

from . import config

# fallback only: larger Colorado places, approximate coordinates (GeoNames file is preferred)
FALLBACK_CO = [
    ("Denver", 39.7392, -104.9903, 715522), ("Colorado Springs", 38.8339, -104.8214, 478961),
    ("Aurora", 39.7294, -104.8319, 386261), ("Fort Collins", 40.5853, -105.0844, 169810),
    ("Lakewood", 39.7047, -105.0814, 155984), ("Thornton", 39.8680, -104.9719, 141867),
    ("Arvada", 39.8028, -105.0875, 124402), ("Westminster", 39.8367, -105.0372, 116317),
    ("Pueblo", 38.2544, -104.6091, 111876), ("Greeley", 40.4233, -104.7091, 108795),
    ("Boulder", 40.0150, -105.2705, 105673), ("Longmont", 40.1672, -105.1019, 100758),
    ("Loveland", 40.3978, -105.0750, 76378), ("Grand Junction", 39.0639, -108.5506, 65560),
    ("Castle Rock", 39.3722, -104.8561, 74000), ("Broomfield", 39.9205, -105.0867, 74112),
    ("Parker", 39.5186, -104.7614, 58512), ("Littleton", 39.6133, -105.0166, 45652),
    ("Brighton", 39.9853, -104.8206, 40083), ("Windsor", 40.4775, -104.9014, 32716),
    ("Fruita", 39.1589, -108.7290, 13395), ("Montrose", 38.4783, -107.8762, 20291),
    ("Durango", 37.2753, -107.8801, 19071), ("Cañon City", 38.4409, -105.2425, 17141),
    ("Fountain", 38.6822, -104.7008, 29802), ("Evans", 40.3764, -104.6922, 22165),
    ("Sterling", 40.6255, -103.2077, 13735), ("Delta", 38.7422, -108.0690, 9035),
    ("Cortez", 37.3489, -108.5859, 8766), ("Glenwood Springs", 39.5505, -107.3248, 9963),
    ("Steamboat Springs", 40.4850, -106.8317, 13224), ("Alamosa", 37.4695, -105.8700, 9806),
    ("Trinidad", 37.1695, -104.5005, 8329), ("Gunnison", 38.5458, -106.9253, 6560),
    ("Salida", 38.5347, -105.9989, 5926), ("Craig", 40.5153, -107.5464, 9060),
    ("Rifle", 39.5347, -107.7831, 10437), ("La Junta", 37.9850, -103.5438, 7322),
    ("Lamar", 38.0872, -102.6208, 7687), ("Pagosa Springs", 37.2694, -107.0098, 1571),
]
DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_dir(lat1, lon1, lat2, lon2):
    """Compass direction of point 2 as seen from point 1."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    deg = (math.degrees(math.atan2(x, y)) + 360) % 360
    return DIRS[int((deg + 22.5) // 45) % 8]


class Places:
    def __init__(self, path=None):
        self.path = Path(path or config.PLACES)
        self.rows, self.source = [], None
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.rows.append({"name": r["name"], "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                                      "population": int(float(r.get("population") or 0))})
            self.source = f"GeoNames ({self.path.name}, {len(self.rows)} places)"
        if not self.rows:
            self.rows = [{"name": n, "lat": a, "lon": b, "population": p} for n, a, b, p in FALLBACK_CO]
            self.source = "built-in fallback list (approximate coordinates; run build_places.py)"
        self.names = sorted({r["name"] for r in self.rows}, key=len, reverse=True)

    def near(self, lat, lon, n=3, max_km=60.0, min_pop=0):
        out = []
        for r in self.rows:
            if r["population"] < min_pop:
                continue
            d = haversine_km(r["lat"], r["lon"], lat, lon)
            if d <= max_km:
                out.append({"name": r["name"], "population": r["population"], "distance_km": round(d, 1),
                            "zone_is_to_the": bearing_dir(r["lat"], r["lon"], lat, lon)})
        out.sort(key=lambda x: x["distance_km"])
        return out[:n]

    def inside(self, bbox, pad_deg=0.02, n=6):
        out = [r for r in self.rows
               if bbox["lat_min"] - pad_deg <= r["lat"] <= bbox["lat_max"] + pad_deg
               and bbox["lon_min"] - pad_deg <= r["lon"] <= bbox["lon_max"] + pad_deg]
        out.sort(key=lambda r: -r["population"])
        return [{"name": r["name"], "population": r["population"]} for r in out[:n]]

    def describe_zone(self, z):
        lat, lon = z["centroid_lat"], z["centroid_lon"]
        inside = self.inside(z["bbox"])
        nearest = self.near(lat, lon, n=3)
        big = self.near(lat, lon, n=1, max_km=80, min_pop=50000) or self.near(lat, lon, n=1, max_km=80, min_pop=10000)
        if inside and len(inside) >= 2 and z.get("area_km2", 0) > 300:
            label = f"{inside[0]['name']} area (includes {', '.join(t['name'] for t in inside[1:3])})"
        elif nearest and nearest[0]["distance_km"] < 5:
            close = [t for t in nearest if t["distance_km"] < 5]
            label = f"{max(close, key=lambda t: t['population'])['name']} area"   # biggest town within 5 km
        elif nearest:
            t = nearest[0]
            label = f"{t['distance_km']:.0f} km {t['zone_is_to_the']} of {t['name']}"
        else:
            label = f"remote area near {lat:.3f}, {lon:.3f}"
        if big and (not nearest or big[0]["name"] != nearest[0]["name"]) and "area" not in label:
            label += f" ({big[0]['distance_km']:.0f} km {big[0]['zone_is_to_the']} of {big[0]['name']})"
        return {"location_label": label, "towns_in_zone": inside, "nearest_towns": nearest,
                "nearest_larger_town": big[0] if big else None}

    def mentioned(self, text):
        """Place names (from the whole state list) that appear in a piece of text."""
        found, low = [], f" {text} "
        for name in self.names:
            if len(name) < 4:
                continue
            i = low.find(name)
            if i > 0 and not low[i - 1].isalpha() and not low[i + len(name)].isalpha():
                found.append(name)
                low = low.replace(name, " " * len(name))
        return found
