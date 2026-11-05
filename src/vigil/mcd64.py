"""MCD64A1 burned area on the modelling grid, one layer per fire season.

For each season (July-June, labelled by the year it ends) and grid cell: the
first date MCD64A1 says the cell burnt that season, and the product's own
uncertainty in days. Tiles are mosaicked and resampled (nearest) from the
463 m MODIS sinusoidal grid onto our grid.

MCD64A1 is the same product over Vic, NSW and SA, so it gives each fire's
final outline consistently across borders. Its burn dates are rough (often
+-several days), so daily progression comes from VIIRS detection times instead
(see labels.py); MCD64A1 supplies the outline.

Output: data/processed/mcd64_burn.nc
  burn_day     days since 2000-01-01 of first burn in the season (0 = not burnt)
  uncertainty  MCD64A1 burn date uncertainty, days

Usage:
  python -m vigil.mcd64
"""

import glob
import re
from datetime import date, timedelta

import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from vigil.burn_history import grid
from vigil.config import GRID_CRS, GRID_RES_M, PROCESSED, RAW

SINU = CRS.from_proj4("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +units=m")
SEASONS = range(2003, 2026)   # 2003 on: enough for a 10-year time-since-burnt cap
EPOCH = date(2000, 1, 1)
OUT = PROCESSED / "mcd64_burn.nc"


def read_tile(path: str):
    sd = SD(path, SDC.READ)
    bd = sd.select("Burn Date").get().astype(np.int32)
    unc = sd.select("Burn Date Uncertainty").get().astype(np.int16)
    meta = sd.attributes()["StructMetadata.0"]
    ul = [float(v) for v in re.search(r"UpperLeftPointMtrs=\(([^)]*)\)", meta).group(1).split(",")]
    lr = [float(v) for v in re.search(r"LowerRightMtrs=\(([^)]*)\)", meta).group(1).split(",")]
    res = (lr[0] - ul[0]) / bd.shape[1]
    return bd, unc, from_origin(ul[0], ul[1], res, res)


def month_files() -> dict[tuple[int, int], list[str]]:
    """(year, month) -> tile files. File names carry the month's first day-of-year."""
    out: dict = {}
    for f in glob.glob(str(RAW / "mcd64a1" / "*.hdf")):
        m = re.search(r"\.A(\d{4})(\d{3})\.", f)
        d = date(int(m.group(1)), 1, 1) + timedelta(int(m.group(2)) - 1)
        out.setdefault((d.year, d.month), []).append(f)
    return out


def build() -> None:
    transform, height, width = grid()
    files = month_files()
    burn = np.zeros((len(SEASONS), height, width), dtype=np.int32)
    unc = np.zeros((len(SEASONS), height, width), dtype=np.int16)
    for i, season in enumerate(SEASONS):
        months = [(season - 1, m) for m in range(7, 13)] + [(season, m) for m in range(1, 7)]
        for year, month in months:
            for f in files.get((year, month), []):
                bd, u, tr = read_tile(f)
                # Day of year -> days since EPOCH, so seasons crossing New Year work.
                days = np.where(bd > 0, (date(year, 1, 1) - EPOCH).days + bd - 1, 0)
                dst_d = np.zeros((height, width), dtype=np.int32)
                dst_u = np.zeros((height, width), dtype=np.int16)
                kw = dict(src_transform=tr, src_crs=SINU, dst_transform=transform,
                          dst_crs=CRS.from_epsg(GRID_CRS), resampling=Resampling.nearest,
                          src_nodata=0, dst_nodata=0)
                reproject(days.astype(np.int32), dst_d, **kw)
                reproject(np.where(bd > 0, u, 0).astype(np.int16), dst_u, **kw)
                new = (dst_d > 0) & ((burn[i] == 0) | (dst_d < burn[i]))   # keep the first burn
                burn[i][new] = dst_d[new]
                unc[i][new] = dst_u[new]
        print(f"  season {season}: {(burn[i] > 0).sum() * GRID_RES_M**2 / 1e4 / 1e3:,.0f} kha burnt",
              flush=True)

    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ds = xr.Dataset({"burn_day": (("season", "y", "x"), burn),
                     "uncertainty": (("season", "y", "x"), unc)},
                    coords={"season": list(SEASONS), "y": ys, "x": xs},
                    attrs={"crs": f"EPSG:{GRID_CRS}", "transform": list(transform)[:6],
                           "burn_day": "days since 2000-01-01; 0 = not burnt"})
    tmp = OUT.with_suffix(".tmp")
    ds.to_netcdf(tmp, engine="h5netcdf",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    tmp.rename(OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    build()
