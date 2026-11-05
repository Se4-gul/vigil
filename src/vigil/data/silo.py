"""Download daily gridded weather from SILO, cropped to Victoria.

Source: SILO (Queensland Government), 0.05 degree (~5 km) daily grids on
Amazon S3. Licence CC BY 4.0.

SILO's annual files cover all of Australia (~410 MB each). This script reads
them over HTTP and only pulls the Victorian slice, so each variable-year is a
few MB and takes about 30 seconds.

Output: silo/<variable>/<year>.nc

Usage:
  python -m vigil.data.silo                          # all variables and years
  python -m vigil.data.silo --years 2019 --vars max_temp   # quick test
"""

import argparse
import time

import fsspec
import xarray as xr

from vigil.config import (END_YEAR, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN,
                          START_YEAR, raw_dir)

URL = "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/{var}/{year}.{var}.nc"

# max_temp, rh_tmax and daily_rain feed the Forest Fire Danger Index;
# min_temp and vp_deficit are extra dryness signals for the model.
VARIABLES = ["max_temp", "min_temp", "rh_tmax", "daily_rain", "vp_deficit"]


def download(var: str, year: int) -> None:
    out_dir = raw_dir("silo") / var
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{year}.nc"
    if out_path.exists():
        print(f"  {var} {year}: already downloaded")
        return

    t0 = time.time()
    url = URL.format(var=var, year=year)
    with fsspec.open(url, block_size=2**22) as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        sub = ds[var].sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        sub = sub.astype("float32").load()
    sub.to_netcdf(out_path.with_suffix(".tmp"),
                  encoding={var: {"zlib": True, "complevel": 4}})
    out_path.with_suffix(".tmp").rename(out_path)  # only keep complete files
    print(f"  {var} {year}: {sub.shape} in {time.time() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(START_YEAR, END_YEAR + 1)))
    parser.add_argument("--vars", nargs="+", default=VARIABLES)
    args = parser.parse_args()

    for var in args.vars:
        for year in args.years:
            for attempt in range(3):
                try:
                    download(var, year)
                    break
                except Exception as err:
                    print(f"  {var} {year}: {err}; retry {attempt + 1}/3")
                    time.sleep(10)


if __name__ == "__main__":
    main()
