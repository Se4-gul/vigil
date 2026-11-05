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
