"""Turn a window (event, fire day D, centre) into model tensors.

Inputs, 32 channels, 128 x 128 at 375 m:
  fire       active_D (this fire's cells burning on D), own_burnt (this fire's
             cells burnt up to and including D)
  history    tslb: years since last burnt (MCD64A1 earlier seasons + FIRMS this
             season by other fires up to D-1), / 10
  terrain    elevation, grad_x, grad_y (vector), slope_gt5/15/25
  fuel       9 fuel-group fractions
  dryness    KBDI and McArthur drought factor on D (from SILO, dryness.py)
  weather    for D and D+1 (D+1 = the day being predicted; assumes a perfect
             forecast): ws_max, u_peak, v_peak (vector), t_max, rh_min, and
             peak McArthur FFDI (with that day's drought factor)
Target: this fire's cells newly burnt on D+1.
Loss mask: 0 where the label isn't trustworthy: the uncertain mask
(0 < uncertain_day <= D+1), other modelling fires burning by D+1 in the
window (their spread isn't this sample's target), and in training, new burn
more than 24 km from this fire's burnt area on D (new ignitions).

Winds come as true east/north components; they're rotated into grid axes
(VicGrid north differs from true north by up to ~3 deg). Augmentation (random
90-degree rotations and flips) rotates every vector channel with the image.

Normalisation: mean/std of continuous channels from the training folds only
(`norm_stats`), per CV fold and for the final all-seasons model.

Usage:
  python -m vigil.dataset stats     # write data/processed/norm_stats.json
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, map_coordinates

from vigil.burn_history import CAP_YEARS, grid, tslb
from vigil.config import CV_FOLDS, PROCESSED
from vigil.static import FUEL_GROUPS, SLOPE_THRESHOLDS
from vigil.windows import SIZE, window_origin

EPOCH = pd.Timestamp("2000-01-01")
WEATHER = ["ws_max", "u_peak", "v_peak", "t_max", "rh_min"]
FFDI_TERMS = "danger_max"   # stored in weather_daily; combined with DF into peak FFDI
TERRAIN = ["elevation", "grad_x", "grad_y"] + [f"slope_gt{t}" for t in SLOPE_THRESHOLDS]
DRYNESS = ["kbdi", "df"]
CHANNELS = (["active_D", "own_burnt", "tslb"] + TERRAIN + list(FUEL_GROUPS)
            + [f"{v}_D" for v in DRYNESS]
            + [f"{v}_D" for v in WEATHER] + [f"{v}_D1" for v in WEATHER]
            + ["ffdi_D", "ffdi_D1"])
VECTORS = [("grad_x", "grad_y"), ("u_peak_D", "v_peak_D"), ("u_peak_D1", "v_peak_D1")]
# Channels left as they are (fractions, masks, tslb already scaled to 0-1).
NO_NORM = {"active_D", "own_burnt", "tslb"} | {f"slope_gt{t}" for t in SLOPE_THRESHOLDS} \
          | set(FUEL_GROUPS)
STATS = PROCESSED / "norm_stats.json"


class Data:
    """Holds the grids in memory and cuts samples."""

    def __init__(self):
        self.transform, self.H, self.W = grid()
        self.labels = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf").load()
        st = xr.open_dataset(PROCESSED / "static.nc", engine="h5netcdf")
        self.static = {k: st[k].values for k in TERRAIN + list(FUEL_GROUPS)}
        # Held in memory (~1.9 GB): reading the compressed file per sample is ~0.7 s.
        self.weather = xr.open_dataset(PROCESSED / "weather_daily.nc", engine="h5netcdf")
        self.wx = {v: self.weather[v].values for v in WEATHER + [FFDI_TERMS]}
        self.weather_days = {int((pd.Timestamp(d) - EPOCH).days): i
                             for i, d in enumerate(self.weather.day.values)}
        dry = xr.open_dataset(PROCESSED / "dryness_daily.nc", engine="h5netcdf")
        self.dry = {v: dry[v].values for v in DRYNESS}
        self.dry_days = {int((pd.Timestamp(d) - EPOCH).days): i for i, d in enumerate(dry.day.values)}
        self.dry_lat, self.dry_lon = dry.lat.values, dry.lon.values
        self._grid_geometry()

    def _grid_geometry(self):
        """Fractional BARRA indices of every grid cell, and the angle from grid
        north to true north (clockwise), for rotating winds into grid axes."""
        tr = self.transform
        xs = tr.c + tr.a * (np.arange(self.W) + 0.5)
        ys = tr.f + tr.e * (np.arange(self.H) + 0.5)
        X, Y = np.meshgrid(xs, ys)
        to_ll = Transformer.from_crs(3111, 4326, always_xy=True)
        lon, lat = to_ll.transform(X, Y)
        blat, blon = self.weather.lat.values, self.weather.lon.values
        self.wrow = ((lat - blat[0]) / (blat[1] - blat[0])).astype(np.float32)
        self.wcol = ((lon - blon[0]) / (blon[1] - blon[0])).astype(np.float32)
        to_grid = Transformer.from_crs(4326, 3111, always_xy=True)
        x2, y2 = to_grid.transform(lon, lat + 0.01)      # step true-north
        self.gamma = np.arctan2(x2 - X, y2 - Y).astype(np.float32)
        self.drow = ((lat - self.dry_lat[0]) / (self.dry_lat[1] - self.dry_lat[0])).astype(np.float32)
        self.dcol = ((lon - self.dry_lon[0]) / (self.dry_lon[1] - self.dry_lon[0])).astype(np.float32)

    def _far_from_fire(self, L, e: int, D: int, r0: int, c0: int) -> np.ndarray:
        """Cells more than FAR_KM from fire e's burnt area on D. Uses a margin of
        FAR_KM around the window, so burnt area just outside it counts."""
        pad = int(np.ceil(FAR_KM / 0.375)) + 1
        R0, C0 = max(r0 - pad, 0), max(c0 - pad, 0)
        R1, C1 = min(r0 + SIZE + pad, self.H), min(c0 + SIZE + pad, self.W)
        ev, bd = L.event.values[R0:R1, C0:C1], L.burn_day.values[R0:R1, C0:C1]
        burnt = (ev == e) & (bd > 0) & (bd <= D)
        if not burnt.any():
            return np.ones((SIZE, SIZE), bool)
        full = np.zeros((SIZE + 2 * pad, SIZE + 2 * pad), bool)
        full[R0 - (r0 - pad):R1 - (r0 - pad), C0 - (c0 - pad):C1 - (c0 - pad)] = burnt
        return _far(full, pad)

    def has_weather(self, day: int) -> bool:
        return day in self.weather_days and day + 1 in self.weather_days

    def _weather(self, day: int, sl: tuple) -> dict:
        i = self.weather_days[day]
        rr, cc = self.wrow[sl], self.wcol[sl]
        out = {v: map_coordinates(self.wx[v][i], [rr, cc], order=1, mode="nearest")
               for v in WEATHER + [FFDI_TERMS]}
        g = self.gamma[sl]
        u, v = out["u_peak"], out["v_peak"]
        out["u_peak"] = u * np.cos(g) + v * np.sin(g)
        out["v_peak"] = -u * np.sin(g) + v * np.cos(g)
        return out

    def region(self, season: int, event: int, D: int, r0: int, r1: int, c0: int, c1: int):
        """Inputs, target and label mask for any grid rectangle (whole-fire
        evaluation uses this; windows are the 128 x 128 case).
        Returns (X [C, h, w], target [h, w], mask [h, w] bool)."""
        sl = (slice(r0, r1), slice(c0, c1))
        L = self.labels.sel(season=season)
        ev, bd, un = L.event.values[sl], L.burn_day.values[sl], L.uncertain_day.values[sl]
        own = ev == event
        ch = {"active_D": (own & (bd == D)), "own_burnt": (own & (bd > 0) & (bd <= D))}
        ch["tslb"] = tslb(int(season), EPOCH + pd.Timedelta(days=D), event)[sl] / CAP_YEARS
        for k, a in self.static.items():
            ch[k] = a[sl]
        rr, cc = self.drow[sl], self.dcol[sl]
        df = {}
        for day in (D, D + 1):
            i = self.dry_days[day]
            df[day] = map_coordinates(self.dry["df"][i], [rr, cc], order=1, mode="nearest")
            if day == D:
                ch["df_D"] = df[day]
                ch["kbdi_D"] = map_coordinates(self.dry["kbdi"][i], [rr, cc], order=1, mode="nearest")
        for suffix, day in [("_D", D), ("_D1", D + 1)]:
            wx = self._weather(day, sl)
            for k in WEATHER:
                ch[k + suffix] = wx[k]
            # Peak McArthur FFDI: 2 exp(-0.45 + 0.987 ln DF + max hourly weather terms).
            ch["ffdi" + suffix] = 2 * np.exp(-0.45 + 0.987 * np.log(np.maximum(df[day], 0.1))
                                             + wx[FFDI_TERMS])
        X = np.stack([np.asarray(ch[k], dtype=np.float32) for k in CHANNELS])
        y = (own & (bd == D + 1)).astype(np.float32)
        other = (ev >= 0) & ~own & (bd > 0) & (bd <= D + 1)
        mask = ~(((un > 0) & (un <= D + 1)) | other)
        return X, y, mask

    def sample(self, win: pd.Series, rng: np.random.Generator | None = None, augment=False,
               mask_far: bool = True):
        """(X [C, 128, 128], target [128, 128], loss mask [128, 128]).

        mask_far: also mask new-burn cells more than FAR_KM from this fire's
        burnt area on D (new ignitions the inputs can't predict). On for
        training; evaluation keeps them and scores them in the >24 km band."""
        r0, c0 = window_origin(win.centre_row, win.centre_col, (self.H, self.W), rng)
        D, e = int(win.day), int(win.event)
        X, y, mask = self.region(int(win.season), e, D, r0, r0 + SIZE, c0, c0 + SIZE)
        if mask_far and y.any():
            mask &= ~(y.astype(bool) & self._far_from_fire(self.labels.sel(season=win.season),
                                                           e, D, r0, c0))
        mask = mask.astype(np.float32)
        if augment and rng is not None:
            X, y, mask = augment_sample(X, y, mask, rng)
        return X, y, mask


FAR_KM = 24


def _far(own_burnt_padded: np.ndarray, pad: int) -> np.ndarray:
    d = distance_transform_edt(~own_burnt_padded) * 0.375
    return d[pad:pad + SIZE, pad:pad + SIZE] > FAR_KM


def augment_sample(X, y, mask, rng):
    """Random rotation by k x 90 deg and optional left-right flip; vector
    channels are rotated/flipped so they still point the right way."""
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    X, y, mask = rotate(X, y, mask, k, flip)
    return X, y, mask


def rotate(X, y, mask, k: int, flip: bool):
    X = np.rot90(X, k, axes=(1, 2)).copy()
    y = np.rot90(y, k).copy()
    mask = np.rot90(mask, k).copy()
    for cx, cy in VECTORS:
        i, j = CHANNELS.index(cx), CHANNELS.index(cy)
        u, v = X[i].copy(), X[j].copy()
        for _ in range(k):                 # rot90 is counter-clockwise: (u, v) -> (-v, u)
            u, v = -v, u
        X[i], X[j] = u, v
    if flip:
        X = X[:, :, ::-1].copy()
        y = y[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        for cx, _ in VECTORS:
            X[CHANNELS.index(cx)] *= -1     # left-right flip negates the east component
    return X, y, mask


def norm_stats(data: Data, windows: pd.DataFrame, n: int = 400, seed: int = 0) -> dict:
    """Per-channel mean/std over a sample of windows (training folds only)."""
    pick = windows.sample(min(n, len(windows)), random_state=seed)
    acc = {k: [] for k in CHANNELS if k not in NO_NORM}
    for _, w in pick.iterrows():
        X, _, _ = data.sample(w)
        for k in acc:
            acc[k].append(X[CHANNELS.index(k)][::4, ::4].ravel())
    out = {}
    for k, v in acc.items():
        a = np.concatenate(v)
        # Vector components: centre at 0 and share one scale, so directions survive.
        out[k] = {"mean": float(a.mean()), "std": float(a.std() + 1e-6)}
    for cx, cy in VECTORS:
        s = float(np.sqrt((out[cx]["std"] ** 2 + out[cy]["std"] ** 2) / 2))
        out[cx] = {"mean": 0.0, "std": s}
        out[cy] = {"mean": 0.0, "std": s}
    return out


def normalise(X: np.ndarray, stats: dict) -> np.ndarray:
    X = X.copy()
    for k, s in stats.items():
        i = CHANNELS.index(k)
        X[i] = (X[i] - s["mean"]) / s["std"]
    return X


def write_stats() -> None:
    data = Data()
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    w = w[[data.has_weather(int(d)) for d in w.day]]
    train = w[w.fold != "test"]
    stats = {f"fold_{f}": norm_stats(data, train[train.fold != str(f)]) for f in CV_FOLDS}
    stats["all"] = norm_stats(data, train)
    tmp = STATS.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=1))
    tmp.rename(STATS)
    print(f"-> {STATS}")


if __name__ == "__main__":
    write_stats()
