"""Download MODIS MCD64A1 (burned area, monthly, 500 m) tiles covering our bounds.

Source: NASA LP DAAC, MCD64A1 Collection 6.1. One HDF4 file per MODIS tile per
month, with a "Burn Date" layer giving the day of year each 500 m pixel burnt
(0 = unburnt, -1 = unmapped, -2 = water) and its uncertainty in days. The same
product everywhere, from November 2000, so it covers Vic, NSW and SA alike.

Needs a free NASA Earthdata login in ~/.netrc:
  machine urs.earthdata.nasa.gov login <user> password <password>
(chmod 600 ~/.netrc). Searching works without it; downloading doesn't.

Our bounds fall in sinusoidal tiles h28v12 (a sliver of the SW corner),
h29v12 and h30v12.

Output: mcd64a1/MCD64A1.A<year><doy>.h<HH>v<VV>.061.<stamp>.hdf

Usage:
  python -m vigil.data.mcd64a1                  # 2012-2025
  python -m vigil.data.mcd64a1 --years 2020     # quick test
"""

import argparse

import earthaccess

from vigil.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, raw_dir

TILES = {"h28v12", "h29v12", "h30v12"}
YEARS = range(2012, 2026)


def search(years) -> list:
    results = earthaccess.search_data(
        short_name="MCD64A1", version="061",
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
        temporal=(f"{min(years)}-01-01", f"{max(years)}-12-31"),
    )
    # The bbox search can return neighbouring tiles; keep ours only.
    return [g for g in results if any(t in g["umm"]["GranuleUR"] for t in TILES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()

    granules = search(args.years)
    print(f"{len(granules)} tile-months found")
    if args.search_only:
        for g in granules[:5]:
            print(" ", g["umm"]["GranuleUR"])
        return

    earthaccess.login(strategy="netrc")
    out = raw_dir("mcd64a1")
    have = {p.stem for p in out.glob("*.hdf")}
    todo = [g for g in granules if g["umm"]["GranuleUR"] not in have]
    print(f"{len(todo)} to download ({len(granules) - len(todo)} already here)")
    # Download into a staging folder and move each file across only once the
    # whole batch has finished, so a crash never leaves a partial .hdf in out/.
    staging = out / ".partial"
    staging.mkdir(exist_ok=True)
    for f in staging.iterdir():  # leftovers from a crashed run may be partial
        f.unlink()
    files =earthaccess.download(todo, str(staging), threads=4)
    for f in staging.glob("*.hdf"):
        f.rename(out / f.name)
    print(f"Downloaded {len(files)} files to {out}")


if __name__ == "__main__":
    main()
