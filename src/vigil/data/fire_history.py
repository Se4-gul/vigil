"""Download Victoria's fire history records (bushfires and planned burns).

Source: DEECA "Fire History Records of Fires across Victoria", served over WFS
from opendata.maps.vic.gov.au. Licence CC BY 4.0.

The full layer is several GB of polygons, mostly small fires. To keep it
manageable this script writes two files:

  fire_history/records.csv      every record: type, season, start date, cause,
                                area, and the centre of its bounding box
  fire_history/large_fires.geojsonl
                                full polygons for fires of at least
                                --min-area hectares (one GeoJSON feature per
                                line), used for burn history and fire replays

Usage:
  python -m vigil.data.fire_history                 # full download, ~20 min
  python -m vigil.data.fire_history --max-pages 2   # quick test
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request

from vigil.config import raw_dir

WFS = "https://opendata.maps.vic.gov.au/geoserver/wfs"
LAYER = "open-data-platform:fire_history"
FIELDS = ["firekey", "firetype", "season", "start_date", "cause", "area_ha", "region"]
PAGE_SIZE = 2000


def fetch_page(start: int, min_season: int) -> dict:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": LAYER,
        "outputFormat": "application/json",
        "count": PAGE_SIZE,
        "startIndex": start,
        # A stable sort order is needed for paging to be reliable.
        "sortBy": "globalid",
        "CQL_FILTER": f"season>={min_season}",
        "propertyName": ",".join(FIELDS + ["geom"]),
    }
    url = WFS + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                return json.load(resp)
        except Exception as err:  # network hiccups: back off and retry
            wait = 2 ** attempt * 5
            print(f"  page {start}: {err}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch page starting at {start}")


def bbox_centre(geometry: dict) -> tuple[float, float]:
    """Centre of a (Multi)Polygon's bounding box, as (lon, lat)."""
    xs, ys = [], []

    def walk(coords):
        if isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for c in coords:
                walk(c)

    walk(geometry["coordinates"])
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-season", type=int, default=1980,
                        help="earliest fire season to keep (default 1980, so "
                             "'years since last burnt' has history before 2000)")
    parser.add_argument("--min-area", type=float, default=50,
                        help="keep full polygons for fires at least this many ha")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="stop after this many pages (for testing)")
    args = parser.parse_args()

    out = raw_dir("fire_history")
    records_path = out / "records.csv"
    large_path = out / "large_fires.geojsonl"

    n_records = n_large = 0
    start = page = 0
    with open(records_path, "w", newline="") as rec_f, open(large_path, "w") as big_f:
        writer = csv.writer(rec_f)
        writer.writerow(FIELDS + ["lon", "lat"])
        while True:
            data = fetch_page(start, args.min_season)
            features = data.get("features", [])
            if page == 0:
                print(f"{data.get('numberMatched', '?')} records to download")
            for feat in features:
                props = feat["properties"]
                geom = feat.get("geometry")
                lon = lat = ""
                if geom:
                    lon, lat = (round(v, 5) for v in bbox_centre(geom))
                writer.writerow([props.get(f) for f in FIELDS] + [lon, lat])
                n_records += 1
                if geom and (props.get("area_ha") or 0) >= args.min_area:
                    big_f.write(json.dumps({"type": "Feature", "properties": props,
                                            "geometry": geom}) + "\n")
                    n_large += 1
            page += 1
            print(f"  page {page}: {n_records} records, {n_large} large fires")
            if len(features) < PAGE_SIZE:
                break
            if args.max_pages and page >= args.max_pages:
                print("Stopping early (--max-pages)")
                break
            start += PAGE_SIZE

    print(f"Done. {records_path}\n      {large_path}")


if __name__ == "__main__":
    main()
