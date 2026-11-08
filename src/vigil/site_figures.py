"""Static figures for the website's text pages (Findings).

  app/img/findings_burn_days.jpg   burn day per cell for the 2019-20 East
                                   Gippsland fire (event 26793), on a quiet
                                   hillshade in the site's palette. The colour
                                   key is drawn in HTML next to it.

Usage: python -m vigil.site_figures
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

from vigil.config import PROCESSED, ROOT

OUT = ROOT / "app" / "img"
EPOCH = pd.Timestamp("2000-01-01")
EVENT, SEASON = 26793, 2020
# Burn-day bands (first day of each), so the two big runs (30-31 Dec, 4 Jan) read as steps.
BANDS = [("2019-11-01", "before 29 Dec"), ("2019-12-29", "29 to 30 Dec"), ("2019-12-31", "31 Dec"),
         ("2020-01-01", "1 to 3 Jan"), ("2020-01-04", "4 Jan"), ("2020-01-05", "after 4 Jan")]
LAND, WATER = (38, 41, 45), (15, 18, 21)        # --land, --water in site.css
EARLY, LATE = "#6e3b2a", "#f2d39c"              # dark copper -> pale amber (palette C)
UPSCALE = 3


def _oklab(rgb):
    c = np.asarray(rgb, float) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l, m, s = (np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                         [0.0883024619, 0.2817188376, 0.6299787005]]) @ c) ** (1 / 3)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def _rgb(lab):
    L, a, b = lab
    l, m, s = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, \
              (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    c = np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                  -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                  -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(c * 255, 0, 255)


def ramp(t):
    """t in [0, 1] -> RGB, interpolated in OKLab."""
    hexrgb = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    a, b = _oklab(hexrgb(EARLY)), _oklab(hexrgb(LATE))
    lab = a[:, None] + (b - a)[:, None] * np.ravel(t)[None, :]
    return _rgb(lab).T.reshape(*np.shape(t), 3)


def burn_days():
    lab = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf").sel(season=SEASON)
    ev, bd = lab.event.values, lab.burn_day.values
    elev = xr.open_dataset(PROCESSED / "static.nc", engine="h5netcdf").elevation.values
    ys, xs = np.where(ev == EVENT)
    pad = 24
    r0, r1, c0, c1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    e = np.nan_to_num(elev[r0:r1, c0:c1], nan=0.0)
    mine = ev[r0:r1, c0:c1] == EVENT
    day = bd[r0:r1, c0:c1].astype(float)

    # Quiet hillshade: light from the north-west, 375 m cells, land a step above water.
    gy, gx = np.gradient(e, 375.0)
    slope, aspect = np.arctan(np.hypot(gx, gy) * 2), np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(40)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = (shade - np.sin(alt)) * 22                        # +-~20 levels around the land colour
    img = np.where((e > 0)[..., None], np.array(LAND) + shade[..., None], np.array(WATER))

    starts = np.array([(pd.Timestamp(d) - EPOCH).days for d, _ in BANDS])
    band = np.clip(np.searchsorted(starts, day, side="right") - 1, 0, len(BANDS) - 1)
    steps = ramp(np.linspace(0, 1, len(BANDS)))
    fire = steps[band]
    img = np.where(mine[..., None], fire * 0.92 + img * 0.08, img)

    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
    im = im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.BILINEAR)     # nothing finer than 375 m implied
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / "findings_burn_days.jpg", quality=88, optimize=True)

    # Key for the HTML: one swatch per band, with its share of the fire's cells.
    share = np.bincount(band[mine], minlength=len(BANDS)) / mine.sum()
    key = {"bands": [{"label": lab, "colour": "#%02x%02x%02x" % tuple(int(v) for v in steps[i]),
                      "share": round(float(share[i]), 3)} for i, (_, lab) in enumerate(BANDS)],
           "size": [im.width, im.height], "cells": int(mine.sum())}
    print(json.dumps(key))


if __name__ == "__main__":
    burn_days()
