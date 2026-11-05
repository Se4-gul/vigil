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

One fire (firekey) is usually split into many polygon pieces, one record each.
Sum or union pieces per firekey; don't drop duplicates.

Paging: each page asks for records with globalid after the last one seen
("keyset" paging). Paging by startIndex makes the server skip N rows first,
which slows down as N grows and hits the database's statement timeout past
~400k rows; that silently truncated an earlier download at 414,000 of 714,383.

Usage:
  python -m vigil.data.fire_history                 # full download, ~20 min
  python -m vigil.data.fire_history --max-pages 2   # quick test
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request

from vigil.config import raw_dir

WFS = "https://opendata.maps.vic.gov.au/geoserver/wfs"
LAYER = "open-data-platform:fire_history"
FIELDS = ["globalid", "firekey", "firetype", "season", "start_date", "cause",
          "area_ha", "region"]
PAGE_SIZE = 2000


def request(params: dict, as_json: bool = True):
    base = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": LAYER}
    url = WFS + "?" + urllib.parse.urlencode(base | params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                return json.load(resp) if as_json else resp.read().decode()
        except Exception as err:  # network hiccups: back off and retry
            wait = 2 ** attempt * 5
            print(f"  request failed: {err}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after 5 tries: {url}")


def count_matched(cql: str) -> int:
    xml = request({"resultType": "hits", "CQL_FILTER": cql}, as_json=False)
    return int(re.search(r'numberMatched="(\d+)"', xml).group(1))


def fetch_page(cql: str, after: str | None) -> list[dict]:
    if after is not None:
        cql = f"{cql} AND globalid > '{after}'"
    data = request({
        "outputFormat": "application/json",
        "count": PAGE_SIZE,
        "sortBy": "globalid",  # must match the keyset comparison
        "CQL_FILTER": cql,
        "propertyName": ",".join(FIELDS + ["geom"]),
    })
    return data.get("features", [])


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
    records_tmp = records_path.with_suffix(".tmp")
    large_tmp = large_path.with_suffix(".tmp")

    cql = f"season>={args.min_season}"
    expected = count_matched(cql)
    print(f"{expected} records to download")

    seen = set()
    n_large = page = 0
    after = None
    with open(records_tmp, "w", newline="") as rec_f, open(large_tmp, "w") as big_f:
        writer = csv.writer(rec_f)
        writer.writerow(FIELDS + ["lon", "lat"])
        while True:
            features = fetch_page(cql, after)
            for feat in features:
                props = feat["properties"]
                gid = props["globalid"]
                if gid in seen:
                    raise RuntimeError(f"globalid {gid} returned twice; paging is unsafe")
                seen.add(gid)
                geom = feat.get("geometry")
                lon = lat = ""
                if geom:
                    lon, lat = (round(v, 5) for v in bbox_centre(geom))
                writer.writerow([props.get(f) for f in FIELDS] + [lon, lat])
                if geom and (props.get("area_ha") or 0) >= args.min_area:
                    big_f.write(json.dumps({"type": "Feature", "properties": props,
                                            "geometry": geom}) + "\n")
                    n_large += 1
            page += 1
            print(f"  page {page}: {len(seen)} records, {n_large} large fires")
            if len(features) < PAGE_SIZE:
                break
            if args.max_pages and page >= args.max_pages:
                print("Stopping early (--max-pages); not replacing existing files")
                return
            after = features[-1]["properties"]["globalid"]

    if len(seen) != expected:
        raise RuntimeError(f"Got {len(seen)} records but the server matched {expected}; "
                           f"partial files left as {records_tmp.name} / {large_tmp.name}")
    records_tmp.rename(records_path)  # only keep complete files
    large_tmp.rename(large_path)
    print(f"Done: {len(seen)} records (matches server).\n  {records_path}\n  {large_path}")


if __name__ == "__main__":
    main()
