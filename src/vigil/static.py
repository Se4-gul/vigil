"""Static input layers on the 375 m grid: terrain and fuel.

Terrain is computed at full DEM resolution and then summarised to 375 m
(never average elevation first, see CLAUDE.md). The DEM is resampled to a
93.75 m grid in VicGrid (4 x 4 sub-cells per grid cell) so gradients are in
metres and in grid axes.
  elevation       mean elevation (m)
  grad_x, grad_y  mean terrain gradient (dz/dx east, dz/dy north; m/m). Aspect
                  kept as a vector: it rotates with the window under
                  augmentation, and its length carries steepness.
  slope_gt5/15/25 fraction of sub-cells steeper than 5, 15, 25 degrees.
                  Thresholds from the land slope histogram (62% of land is
                  under 2 deg): 5/15/25 are exceeded by ~24%, 7%, 1.5% of land;
                  10-degree steps match spread roughly doubling per 10 deg
                  upslope.
Fuel: CSIRO BFC classes, grouped and summarised as the fraction of each
375 m cell in each group (nearest-resampled to 93.75 m first).

Output: data/processed/static.nc

Usage:
  python -m vigil.static
"""

import glob

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

from vigil.burn_history import grid
from vigil.config import GRID_CRS, PROCESSED, RAW

SUB = 4
SLOPE_THRESHOLDS = [5, 15, 25]
FUEL_GROUPS = {
    "fuel_forest": [110, 120, 210, 220, 230],
    "fuel_plantation": [310, 321, 322, 323, 324, 330],
    "fuel_woodland": [411, 421, 422, 423, 424, 431, 432, 433, 434],
    "fuel_shrubland": [510, 520, 531, 532, 533],
    "fuel_grassland": [610, 620, 631, 632, 633],
    "fuel_crop": [640, 700],
    "fuel_wetland": [800],
    "fuel_wui": [920, 930, 940],
    "fuel_nonburnable": [0, 910, 950, 960],
}
OUT = PROCESSED / "static.nc"


def blocks(a: np.ndarray, H: int, W: int) -> np.ndarray:
    """Mean over SUB x SUB sub-cells."""
    return a.reshape(H, SUB, W, SUB).mean(axis=(1, 3))


def build() -> None:
    tr, H, W = grid()
    fine_tr = tr * tr.scale(1 / SUB, 1 / SUB)
    dst_crs = CRS.from_epsg(GRID_CRS)

    srcs = [rasterio.open(f) for f in glob.glob(str(RAW / "dem" / "*.tif"))]
    mos, mtr = merge(srcs, nodata=-9999)
    dem = np.full((H * SUB, W * SUB), -9999, np.float32)
    reproject(mos[0], dem, src_transform=mtr, src_crs=srcs[0].crs, src_nodata=-9999,
              dst_transform=fine_tr, dst_crs=dst_crs, dst_nodata=-9999,
              resampling=Resampling.bilinear)
    z = np.where(dem > -9999, dem, 0).astype(np.float32)   # sea = 0 m
    res = abs(fine_tr.a)
    d_row, d_col = np.gradient(z, res)
    gx, gy = d_col, -d_row                                   # rows run south
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    layers = {"elevation": blocks(z, H, W), "grad_x": blocks(gx, H, W),
              "grad_y": blocks(gy, H, W)}
    for t in SLOPE_THRESHOLDS:
        layers[f"slope_gt{t}"] = blocks((slope > t).astype(np.float32), H, W)
    del z, gx, gy, slope, d_row, d_col

    with rasterio.open(RAW / "fuel" / "bfc_fuel_types_crop.tif") as f:
        fuel = np.zeros((H * SUB, W * SUB), np.int32)
        reproject(f.read(1), fuel, src_transform=f.transform, src_crs=f.crs,
                  dst_transform=fine_tr, dst_crs=dst_crs, src_nodata=None,
                  resampling=Resampling.nearest)
    for name, codes in FUEL_GROUPS.items():
        layers[name] = blocks(np.isin(fuel, codes).astype(np.float32), H, W)

    ys = tr.f + tr.e * (np.arange(H) + 0.5)
    xs = tr.c + tr.a * (np.arange(W) + 0.5)
    ds = xr.Dataset({k: (("y", "x"), v.astype(np.float32)) for k, v in layers.items()},
                    coords={"y": ys, "x": xs}, attrs={"crs": f"EPSG:{GRID_CRS}"})
    tmp = OUT.with_suffix(".tmp")
    ds.to_netcdf(tmp, engine="h5netcdf",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    tmp.rename(OUT)
    total = sum(layers[k] for k in FUEL_GROUPS)
    print(f"-> {OUT}; fuel fractions sum to 1: {np.allclose(total, 1, atol=1e-4)}")


if __name__ == "__main__":
    build()
