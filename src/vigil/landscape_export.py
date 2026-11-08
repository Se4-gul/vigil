"""Export the Landscape page data (app/landscape.html).

Years since the land last burnt, as at 1 October of year Y (before season
Y+1's fires), from MCD64A1 only, so it's the same source on both sides of the
border. Same definition as the model's time-since-burnt input: for season
Y+1, years = (Y+1) - season of the last burn, capped at 10 ("10+"). MCD64A1
seasons run July-June, so burns in July-September of Y count from next year.

  app/data/landscape/tsb_<Y>.png   one 8-bit greyscale PNG per October,
                                   Web Mercator: value = years since burnt
                                   (1-9), 10 = 10+ or never, 0 = outside data
  app/data/landscape/fuel.png      CSIRO fuel type, 4 groups (1 forest/
                                   woodland, 2 shrub/heath, 3 grass/crop/
                                   wetland, 4 non-burnable/urban; 0 = sea)
  app/data/landscape/meta.json     corners, years, share of Victoria burnt
                                   in the last 5 years per October

Usage: python -m vigil.landscape_export
"""

import json

import geopandas as gpd
import numpy as np
import xarray as xr
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from vigil.burn_history import edge_distance_km, grid
from vigil.config import GRID_CRS, PROCESSED, RAW, ROOT
from vigil.static import FUEL_GROUPS
from vigil.web_export import rasterio_bounds

OUT = ROOT / "app" / "data" / "landscape"
YEARS = range(2013, 2026)          # October 2013 ... October 2025
CAP = 10
FUEL4 = {"fuel_forest": 1, "fuel_plantation": 1, "fuel_woodland": 1, "fuel_shrubland": 2,
         "fuel_grassland": 3, "fuel_crop": 3, "fuel_wetland": 3, "fuel_wui": 4, "fuel_nonburnable": 4}


def main():
    tr, H, W = grid()
    burn = xr.open_dataset(PROCESSED / "mcd64_burn.nc", engine="h5netcdf").burn_day
    seasons = burn.season.values
    burnt = {int(s): burn.sel(season=s).values > 0 for s in seasons}

    vic = gpd.read_file(RAW / "abs_boundary" / "vic_ste_2021.geojson").to_crs(GRID_CRS)
    vic_mask = rasterize(vic.geometry, out_shape=(H, W), transform=tr, fill=0, default_value=1).astype(bool)

    # One nearest-neighbour index map, Web Mercator <- grid cell, reused for every layer.
    src_crs, dst_crs = CRS.from_epsg(GRID_CRS), CRS.from_epsg(3857)
    dst_tr, dw, dh = calculate_default_transform(src_crs, dst_crs, W, H, *rasterio_bounds(tr, (H, W)),
                                                 resolution=470)
    idx = np.zeros((dh, dw), np.int32)
    reproject(np.arange(1, H * W + 1, dtype=np.int32).reshape(H, W), idx, src_transform=tr, src_crs=src_crs,
              dst_transform=dst_tr, dst_crs=dst_crs, resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    inside, flat = idx > 0, np.maximum(idx - 1, 0)
    to_merc = lambda a: np.where(inside, a.ravel()[flat], 0).astype(np.uint8)

    OUT.mkdir(parents=True, exist_ok=True)
    share5 = {}
    for Y in YEARS:
        years = np.full((H, W), CAP, np.uint8)
        for s in sorted(x for x in burnt if Y + 1 - CAP < x <= Y):   # oldest first, newest overwrites
            years[burnt[s]] = Y + 1 - s
        share5[Y] = round(float((years[vic_mask] <= 5).mean()), 4)
        Image.fromarray(to_merc(years), "L").save(OUT / f"tsb_{Y}.png", optimize=True)

    st = xr.open_dataset(PROCESSED / "static.nc", engine="h5netcdf")
    names = list(FUEL_GROUPS)
    frac = np.nan_to_num(np.stack([st[k].values for k in names]))
    group = np.array([FUEL4[k] for k in names], np.uint8)[frac.argmax(0)]
    group[(st.elevation.values <= 0) | ~np.isfinite(st.elevation.values)] = 0      # sea
    group[edge_distance_km() < 0] = 0      # the fuel map stops at the study box; outside it is no data, not "non-burnable"
    Image.fromarray(to_merc(group), "L").save(OUT / "fuel.png", optimize=True)

    merc_to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
    x0, y1 = dst_tr.c, dst_tr.f
    x1, y0 = x0 + dw * dst_tr.a, y1 + dh * dst_tr.e
    corners = [list(merc_to_ll.transform(x, y)) for x, y in ((x0, y1), (x1, y1), (x1, y0), (x0, y0))]
    fuel_share = {k: round(float((group[vic_mask] == v).mean()), 4)
                  for k, v in (("forest/woodland", 1), ("shrub/heath", 2), ("grass/crop", 3), ("non-burnable", 4))}
    meta = {"corners": corners, "years": list(YEARS), "share_vic_last5": share5, "fuel_share_vic": fuel_share,
            "size": [dw, dh]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"{len(YEARS)} Octobers + fuel -> {OUT} ({size:.1f} MB, {dw}x{dh} px)")
    print("share of Victoria burnt in the last 5 years:", share5)
    print("fuel groups (share of Victoria):", fuel_share)


if __name__ == "__main__":
    main()
