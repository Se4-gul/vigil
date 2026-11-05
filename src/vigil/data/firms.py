"""Download VIIRS 375 m active fire detections for Victoria from NASA FIRMS.

Source: NASA FIRMS area API, Standard Processing (archive-quality) products:
  VIIRS_SNPP_SP     Suomi-NPP, from 2012-01-20 (used for the fire maps)
  VIIRS_NOAA20_SP   NOAA-20, from 2018 (kept for checks; using it would
                    change the detection rate partway through training data)

Needs a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
Put it in .env at the project root as FIRMS_MAP_KEY=... (git ignores .env),
or set the FIRMS_MAP_KEY environment variable.

The API returns at most 5 days per request, so a year is ~73 requests. The
limit is 5000 requests per 10 minutes, well above what this uses.

Output: firms/<source>/<year>.csv (one row per detection, times in UTC)

Usage:
  python -m vigil.data.firms                                  # both sources, 2012-2025
  python -m vigil.data.firms --years 2019 --sources VIIRS_SNPP_SP   # quick test
"""

import argparse
import io
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from vigil.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, ROOT, raw_dir

API = "https://firms.modaps.eosdis.nasa.gov/api"
SOURCES = {"VIIRS_SNPP_SP": date(2012, 1, 20), "VIIRS_NOAA20_SP": date(2018, 1, 1)}
YEARS = range(2012, 2026)
DAYS_PER_REQUEST = 5
BBOX = f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"  # west,south,east,north


def map_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FIRMS_MAP_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit("No FIRMS_MAP_KEY: get one at "
                         "https://firms.modaps.eosdis.nasa.gov/api/map_key/ "
                         "and add FIRMS_MAP_KEY=... to .env")
    return key


def available_until(key: str) -> dict[str, date]:
    """Last date each source has archive data for (SP lags NRT by months)."""
    r = requests.get(f"{API}/data_availability/csv/{key}/all", timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return {row.data_id: date.fromisoformat(row.max_date) for row in df.itertuples()}


def fetch(key: str, source: str, start: date) -> pd.DataFrame:
    url = f"{API}/area/csv/{key}/{source}/{BBOX}/{DAYS_PER_REQUEST}/{start}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            if not r.text.startswith("latitude"):  # errors come back as plain text
                raise ValueError(r.text.strip()[:200])
            return pd.read_csv(io.StringIO(r.text))
        except Exception as err:
            print(f"    {source} {start}: {err}; retry {attempt + 1}/3")
            time.sleep(10)
    raise RuntimeError(f"{source} {start}: failed after 3 tries")


def download(key: str, source: str, year: int, last: date) -> None:
    out_dir = raw_dir("firms") / source
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{year}.csv"
    if out_path.exists():
        print(f"  {source} {year}: already downloaded")
        return

    start = max(date(year, 1, 1), SOURCES[source])
    end = min(date(year, 12, 31), last)
    if start > end:
        print(f"  {source} {year}: no archive data")
        return
    if end < date(year, 12, 31):
        # Don't save a partial year as if it were finished.
        print(f"  {source} {year}: archive only reaches {last}; skipping until complete")
        return

    t0 = time.time()
    parts, day = [], start
    while day <= end:
        parts.append(fetch(key, source, day))
        day += timedelta(days=DAYS_PER_REQUEST)
    df = pd.concat(parts, ignore_index=True)
    df = df[pd.to_datetime(df.acq_date).dt.year == year]  # last window can spill over
    df = df.drop_duplicates()

    tmp = out_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(out_path)  # only keep complete files
    print(f"  {source} {year}: {len(df):,} detections in {time.time() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    args = parser.parse_args()

    key = map_key()
    last = available_until(key)
    for source in args.sources:
        for year in args.years:
            download(key, source, year, last[source])


if __name__ == "__main__":
    main()
