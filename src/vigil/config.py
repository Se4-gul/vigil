"""Shared settings: study area, years and data paths."""

from pathlib import Path

# Victoria plus a small margin, in degrees (WGS84).
LON_MIN, LON_MAX = 140.8, 150.1
LAT_MIN, LAT_MAX = -39.3, -33.8

# Calendar years of weather to download. Fire seasons are labelled by the
# year they end in, so season 2001 runs from mid-2000 to mid-2001.
START_YEAR = 2000
END_YEAR = 2025

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def raw_dir(name: str) -> Path:
    """Return data/raw/<name>, creating it if needed."""
    path = RAW / name
    path.mkdir(parents=True, exist_ok=True)
    return path

# Modelling grid: VicGrid94 (EPSG:3111, metres). Resolution is still an open
# decision (see CLAUDE.md); 375 m matches the VIIRS pixel and is provisional.
GRID_CRS = 3111
GRID_RES_M = 375

# Fire days run 06:00-06:00 local time in Victoria (FIRMS times are UTC), so
# the ~02:00 night overpass counts toward the day whose fire it shows. With
# plain calendar dates, documented arrivals (Mallacoota, Sarsfield) came out
# a day late.
LOCAL_TZ = "Australia/Melbourne"
FIRE_DAY_START_HOUR = 6


def fire_day(utc_times):
    """UTC timestamps (pandas Series) -> fire day (midnight timestamps)."""
    import pandas as pd
    local = utc_times.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return (local - pd.Timedelta(hours=FIRE_DAY_START_HOUR)).dt.normalize()

# Season cross-validation: 5 folds of whole seasons, balanced on windows with
# new burn; 2013, 2014 and 2019 in different folds. Re-balanced on 2026-11-05
# after the event-tagging fix (217-245 windows with new burn per fold). 2020 = test.
CV_FOLDS = {
    0: [2013],
    1: [2014],
    2: [2015, 2019, 2021, 2022],
    3: [2016, 2018, 2024],
    4: [2017, 2023, 2025],
}
TEST_SEASONS = [2020]
