"""Export fire replays for the static website (app/).

For each labelled fire of a season (Oct-Apr scope, at least MIN_CELLS cells,
burning on at least MIN_DAYS days, and not clipped by the study box, i.e. its
MCD64A1 burnt patch doesn't extend past the box edge):
  app/data/fire_<event>.png  burn day per cell, reprojected to Web Mercator
                             (EPSG:3857) so it overlays the basemap exactly.
                             R + 256*G = days since the fire's first burn day,
                             A = 255 where burnt.
  app/data/fires.json        id, label, corner coordinates, dates, daily area.
Labels come from the nearest town in a small gazetteer; the largest fire in the
east is "East Gippsland".

Usage:
  python -m vigil.web_export            # season 2020 (2019-20)
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from vigil.burn_history import edge_distance_km, grid
from vigil.config import GRID_CRS, PROCESSED, ROOT
from vigil.labels import mcd_season, patches

OUT = ROOT / "app" / "data"
EPOCH = pd.Timestamp("2000-01-01")
MIN_CELLS = 300
MIN_DAYS = 3
TOWNS = {
    "Mallacoota": (-37.553, 149.754), "Buchan": (-37.497, 148.176), "Orbost": (-37.708, 148.456),
    "Bairnsdale": (-37.823, 147.610), "Omeo": (-37.098, 147.594), "Bright": (-36.730, 146.960),
    "Corryong": (-36.197, 147.902), "Mansfield": (-37.052, 146.088), "Walhalla": (-37.940, 146.450),
    "Licola": (-37.630, 146.620), "Cann River": (-37.567, 149.150), "Bendoc": (-37.147, 148.886),
    "Tumbarumba": (-35.777, 148.011), "Bombala": (-36.912, 149.237), "Cooma": (-36.235, 149.125),
    "Jindabyne": (-36.416, 148.622), "Khancoban": (-36.217, 148.128), "Eden": (-37.064, 149.903),
    "Batemans Bay": (-35.708, 150.175), "Braidwood": (-35.441, 149.799), "Tumut": (-35.300, 148.224),
    "Halls Gap": (-37.137, 142.519), "Hamilton": (-37.744, 142.022), "Portland": (-38.343, 141.604),
    "Horsham": (-36.712, 142.199), "Ballarat": (-37.562, 143.850), "Bendigo": (-36.757, 144.279),
    "Mildura": (-34.185, 142.163), "Wodonga": (-36.121, 146.888), "Kinglake": (-37.522, 145.348),
    "Moe": (-38.177, 146.262), "Yarram": (-38.565, 146.677), "Swifts Creek": (-37.260, 147.722),
}


def build(season: int = 2020) -> None:
    tr, H, W = grid()
    L = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf").sel(season=season)
    ev, bd = L.event.values, L.burn_day.values
    lo = (pd.Timestamp(f"{season - 1}-10-01") - EPOCH).days
    hi = (pd.Timestamp(f"{season}-04-30") - EPOCH).days
    inscope = (ev >= 0) & (bd >= lo) & (bd <= hi)
    ids, counts = np.unique(ev[inscope], return_counts=True)
    to_ll = Transformer.from_crs(GRID_CRS, 4326, always_xy=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("fire_*.png"):
        old.unlink()
    edge = edge_distance_km()
    patch = patches(mcd_season(season) > 0)
    clipped_patches = set(np.unique(patch[edge < 0])) - {0}
    fires, skipped = [], []
    for e, n in sorted(zip(ids, counts), key=lambda t: -t[1]):
        if n < MIN_CELLS:
            break
        r, c = np.nonzero(inscope & (ev == e))
        n_days = len(np.unique(bd[r, c]))
        clipped = bool(set(np.unique(patch[r, c])) & clipped_patches)
        if n_days < MIN_DAYS or clipped:
            skipped.append((int(e), int(n), n_days, "clipped" if clipped else "short"))
            continue
        pad = 8
        r0, r1 = max(r.min() - pad, 0), min(r.max() + pad + 1, H)
        c0, c1 = max(c.min() - pad, 0), min(c.max() + pad + 1, W)
        days = np.where(inscope[r0:r1, c0:c1] & (ev[r0:r1, c0:c1] == e), bd[r0:r1, c0:c1], 0)
        first, last = int(days[days > 0].min()), int(days[days > 0].max())
        off = np.where(days > 0, days - first + 1, 0).astype(np.int32)     # 0 = unburnt
        src_tr = from_origin(tr.c + c0 * tr.a, tr.f + r0 * tr.e, tr.a, -tr.e)
        src_crs, dst_crs = CRS.from_epsg(GRID_CRS), CRS.from_epsg(3857)
        dst_tr, dw, dh = calculate_default_transform(src_crs, dst_crs, off.shape[1], off.shape[0],
                                                     *rasterio_bounds(src_tr, off.shape),
                                                     resolution=470)
        dst = np.zeros((dh, dw), np.int32)
        reproject(off, dst, src_transform=src_tr, src_crs=src_crs, dst_transform=dst_tr,
                  dst_crs=dst_crs, resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        img = np.zeros((dh, dw, 4), np.uint8)
        d0 = np.maximum(dst - 1, 0)
        img[..., 0], img[..., 1] = d0 % 256, d0 // 256
        img[..., 3] = np.where(dst > 0, 255, 0)
        Image.fromarray(img).save(OUT / f"fire_{e}.png", optimize=True)
        merc_to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
        x0, y1 = dst_tr.c, dst_tr.f
        x1, y0 = x0 + dw * dst_tr.a, y1 + dh * dst_tr.e
        corners = [list(merc_to_ll.transform(x, y)) for x, y in ((x0, y1), (x1, y1), (x1, y0), (x0, y0))]
        cx, cy = to_ll.transform(tr.c + tr.a * (c.mean() + 0.5), tr.f + tr.e * (r.mean() + 0.5))
        town = min(TOWNS, key=lambda t: (TOWNS[t][0] - cy) ** 2 + (TOWNS[t][1] - cx) ** 2)
        daily = np.bincount(off[off > 0] - 1, minlength=last - first + 1)
        fires.append({"id": int(e), "town": town, "cells": int(n),
                      "kha": round(n * 0.375 ** 2 * 100 / 1e3, 1),
                      "first": (EPOCH + pd.Timedelta(days=first)).strftime("%Y-%m-%d"),
                      "days": last - first + 1, "corners": corners, "center": [cx, cy],
                      "daily_kha": [round(v * 0.375 ** 2 * 100 / 1e3, 2) for v in daily]})
    # Labels: the biggest fire east of 147.5E is "East Gippsland"; others "near <town>".
    eg = max((f for f in fires if f["center"][0] > 147.5), key=lambda f: f["cells"])
    for f in fires:
        f["label"] = "East Gippsland" if f is eg else f"Near {f['town']}"
        f["default"] = f is eg
    fires.sort(key=lambda f: (not f["default"], -f["cells"]))
    (OUT / "fires.json").write_text(json.dumps({"season": f"{season - 1}–{str(season)[2:]}", "fires": fires}))
    size = sum(p.stat().st_size for p in OUT.glob("*")) / 1e6
    print(f"{len(fires)} fires -> {OUT} ({size:.1f} MB)")
    print("  skipped (event, cells, burn days, why):", skipped)
    for f in fires[:12]:
        print(f"  {f['label']:<22} {f['kha']:>8} kha  {f['first']} +{f['days']} d  (event {f['id']})")


def rasterio_bounds(transform, shape):
    h, w = shape
    left, top = transform.c, transform.f
    return left, top + h * transform.e, left + w * transform.a, top


if __name__ == "__main__":
    build()
