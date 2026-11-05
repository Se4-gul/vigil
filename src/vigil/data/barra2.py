"""Download hourly near-surface weather from BARRA-C2, cropped to Victoria.

Source: Bureau of Meteorology BARRA2 reanalysis, BARRA-C2 (AUST-04, ~4 km),
public on Amazon S3 (no account). Licence CC BY 4.0.

BARRA2's monthly files cover all of Australia (~750 MB each). This script
reads them over S3 and only pulls the Victorian slice, about 40 seconds per
variable-month. Only fire-season months (October to April) are downloaded;
2012-2025 DEECA records put 2.3% of bushfire area outside that window.

Variables (hourly, instantaneous on the hour, times in UTC):
  uas, vas  10 m eastward / northward wind (m/s)
  tas       2 m air temperature (K)
  hurs      2 m relative humidity (%)

Output: barra2/<variable>/<YYYYMM>.nc

Usage:
  python -m vigil.data.barra2                            # all variables and months
  python -m vigil.data.barra2 --years 2019 --vars uas    # quick test
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import s3fs
import xarray as xr

from vigil.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, raw_dir

BASE = ("bom-opendata-climate/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/"
        "historical/hres/BARRA-C2/v1/1hr")

VARIABLES = ["uas", "vas", "tas", "hurs"]
YEARS = range(2012, 2026)
FIRE_MONTHS = [1, 2, 3, 4, 10, 11, 12]

fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "ap-southeast-2"})


def source_path(var: str, year: int, month: int) -> str:
    # Files sit under a version folder (e.g. v20250528), so look it up.
    ym = f"{year}{month:02d}"
    matches = fs.glob(f"{BASE}/{var}/*/{var}_*_1hr_{ym}-{ym}.nc")
    if len(matches) != 1:
        raise FileNotFoundError(f"expected 1 file for {var} {ym}, found {len(matches)}")
    return matches[0]


def download(var: str, year: int, month: int) -> None:
    out_dir = raw_dir("barra2") / var
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{year}{month:02d}.nc"
    label = f"{var} {year}-{month:02d}"
    if out_path.exists():
        print(f"  {label}: already downloaded")
        return

    t0 = time.time()
    # A small read block matters: 1 MB blocks are ~4x faster than 4 MB here,
    # because the Victorian slice only touches a few chunks of each block.
    with fs.open(source_path(var, year, month), block_size=2**20) as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        sub = ds[var].sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        sub = sub.astype("float32").load()
    tmp = out_path.with_suffix(".tmp")
    sub.to_netcdf(tmp, engine="h5netcdf",
                  encoding={var: {"zlib": True, "complevel": 4}})
    tmp.rename(out_path)  # only keep complete files
    print(f"  {label}: {sub.shape} in {time.time() - t0:.0f}s")


def download_with_retry(job) -> None:
    for attempt in range(3):
        try:
            download(*job)
            return
        except Exception as err:
            print(f"  {job}: {err}; retry {attempt + 1}/3")
            time.sleep(10)
    print(f"  {job}: FAILED after 3 tries")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    parser.add_argument("--months", type=int, nargs="+", default=FIRE_MONTHS)
    parser.add_argument("--vars", nargs="+", default=VARIABLES)
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel downloads (processes: h5py reads hold a global lock)")
    args = parser.parse_args()

    jobs = [(v, y, m) for v in args.vars for y in args.years for m in args.months]
    print(f"{len(jobs)} variable-months to fetch")
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(download_with_retry, jobs))


if __name__ == "__main__":
    main()
