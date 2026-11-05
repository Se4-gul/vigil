"""Download terrain elevation tiles covering Victoria.

Source: Copernicus DEM GLO-90 (90 m), public on Amazon S3. Free to use with
attribution (see data/README.md).

Tiles are 1 x 1 degree. Ocean tiles don't exist and are skipped. About 40
tiles, ~250 MB total. Slope and aspect are derived later in processing.

Output: dem/<tile>.tif

Usage:
  python -m vigil.data.dem
"""

import math
import urllib.error
import urllib.request

from vigil.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, raw_dir

URL = "https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"


def tile_name(lat: int, lon: int) -> str:
    # Tiles are named after their south-west corner, e.g. S38_00_E145_00
    # covers latitude -38 to -37 and longitude 145 to 146.
    ns = "S" if lat < 0 else "N"
    return f"Copernicus_DSM_COG_30_{ns}{abs(lat):02d}_00_E{lon:03d}_00_DEM"


def main():
    out_dir = raw_dir("dem")
    lats = range(math.floor(LAT_MIN), math.ceil(LAT_MAX))
    lons = range(math.floor(LON_MIN), math.ceil(LON_MAX))
    got = skipped = 0
    for lat in lats:
        for lon in lons:
            name = tile_name(lat, lon)
            path = out_dir / f"{name}.tif"
            if path.exists():
                got += 1
                continue
            try:
                urllib.request.urlretrieve(URL.format(name=name), path)
                got += 1
                print(f"  {name}")
            except urllib.error.HTTPError as err:
                if err.code in (403, 404):  # no tile here (ocean)
                    skipped += 1
                else:
                    raise
    print(f"Done: {got} tiles, {skipped} ocean tiles skipped")


if __name__ == "__main__":
    main()
