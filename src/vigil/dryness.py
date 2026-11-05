"""Daily KBDI and McArthur drought factor from SILO rain and max temperature.

Computed the same way for every year from SILO 2000-2025 (5 km), starting
from KBDI = 0 on 1 Jan 2000; the 11+ years before the first fire season we
use (2013) are ample spin-up.

KBDI (Keetch-Byram drought index, metric form as used for Australian FFDI):
  each day, rain first: the first 5.08 mm of each run of rain days is lost to
  interception; the rest lowers KBDI. Then evapotranspiration raises it:
    ET = (203.2 - KBDI) * (0.968 exp(0.0875 T + 1.5552) - 8.30)
         / (1 + 10.88 exp(-0.001736 R)) * 1e-3
  T = max temperature (deg C), R = mean annual rainfall at the cell (mm,
  2000-2025). KBDI is in mm, 0-203.2.
Drought factor (Griffiths 1999):
    DF = 10.5 (1 - exp(-(KBDI + 30) / 40)) (41 x^2 + x) / (40 x^2 + x + 1), <= 10
  x from recent rain: for each rain event in the last 20 days (consecutive
  days with > 2 mm), x = N^1.3 / (N^1.3 + P - 2), with P the event total and
  N the days since its wettest day (0.8 in place of N on that day itself); the
  smallest x (strongest damping) is used, 1 if no event, then capped by the
  KBDI limit (1 / (1 + 0.1135 KBDI) below KBDI 20, else 75 / (270.525 - 1.267 KBDI)).

SILO's rain for date t is the 24 h to 9 am on t, so values for day D are
known on the morning of D (plus D's max temperature, which the D weather
inputs already use).

Output: data/processed/dryness_daily.nc (SILO grid), from 1 Jul 2011.

Usage:
  python -m vigil.dryness
"""

import numpy as np
import pandas as pd
import xarray as xr

from vigil.config import PROCESSED, RAW

OUT = PROCESSED / "dryness_daily.nc"
KEEP_FROM = "2011-07-01"


def load(var: str) -> xr.DataArray:
    parts = [xr.open_dataset(RAW / "silo" / var / f"{y}.nc", engine="h5netcdf")[var]
             for y in range(2000, 2026)]
    return xr.concat(parts, dim="time").astype("float32")


def drought_factor(kbdi: np.ndarray, window: np.ndarray) -> np.ndarray:
    """window: rain of the last 20 days, oldest first (20, cells)."""
    n_days = window.shape[0]
    x_min = np.ones(kbdi.shape, np.float32)
    run_sum = np.zeros_like(kbdi)
    peak_val = np.zeros_like(kbdi)
    peak_j = np.zeros(kbdi.shape, np.int32)

    def close(mask):
        nonlocal x_min
        n = (n_days - 1 - peak_j).astype(np.float32)
        n13 = np.where(n == 0, 0.8 ** 1.3, n ** 1.3)
        x = n13 / np.maximum(n13 + run_sum - 2, 1e-6)
        x_min = np.where(mask & (run_sum > 2), np.minimum(x_min, x), x_min)

    for j in range(n_days):
        wet = window[j] > 2
        close(~wet)
        run_sum = np.where(wet, run_sum + window[j], 0)
        new_peak = wet & (window[j] >= peak_val)
        peak_j = np.where(new_peak, j, peak_j)
        peak_val = np.where(wet, np.where(new_peak, window[j], peak_val), 0)
    close(np.ones_like(kbdi, bool))
    xlim = np.where(kbdi < 20, 1 / (1 + 0.1135 * kbdi), 75 / (270.525 - 1.267 * kbdi))
    x = np.minimum(x_min, xlim)
    df = 10.5 * (1 - np.exp(-(kbdi + 30) / 40)) * (41 * x**2 + x) / (40 * x**2 + x + 1)
    return np.clip(df, 0, 10).astype(np.float32)


def build() -> None:
    rain = load("daily_rain")
    tmax = load("max_temp")
    times = pd.to_datetime(rain.time.values)
    R = rain.values.reshape(len(times), -1)
    T = tmax.values.reshape(len(times), -1)
    land = ~np.isnan(R).all(0)
    R = np.nan_to_num(R)
    T = np.nan_to_num(T, nan=20.0)
    annual = R.sum(0) / (len(times) / 365.25)

    cells = R.shape[1]
    kbdi = np.zeros(cells, np.float32)
    event_rain = np.zeros(cells, np.float32)
    window = np.zeros((20, cells), np.float32)
    keep = times >= pd.Timestamp(KEEP_FROM)
    out_k = np.zeros((keep.sum(), cells), np.float32)
    out_d = np.zeros((keep.sum(), cells), np.float32)
    k = 0
    for t in range(len(times)):
        r = R[t]
        # Interception: the first 5.08 mm of each run of rain days is lost.
        before = np.maximum(event_rain - 5.08, 0)
        event_rain = np.where(r > 0, event_rain + r, 0)
        net = np.where(r > 0, np.maximum(event_rain - 5.08, 0) - before, 0)
        kbdi = np.maximum(kbdi - net, 0)
        et = ((203.2 - kbdi) * (0.968 * np.exp(0.0875 * T[t] + 1.5552) - 8.30)
              / (1 + 10.88 * np.exp(-0.001736 * annual)) * 1e-3)
        kbdi = np.clip(kbdi + np.maximum(et, 0), 0, 203.2)
        window = np.roll(window, -1, axis=0)
        window[-1] = r
        if keep[t]:
            out_k[k] = kbdi
            out_d[k] = drought_factor(kbdi, window)
            k += 1
    shape = (keep.sum(),) + rain.shape[1:]
    out_k[:, ~land] = 0
    out_d[:, ~land] = 0
    ds = xr.Dataset({"kbdi": (("day", "lat", "lon"), out_k.reshape(shape)),
                     "df": (("day", "lat", "lon"), out_d.reshape(shape))},
                    coords={"day": times[keep], "lat": rain.lat.values, "lon": rain.lon.values},
                    attrs={"note": "SILO date = rain to 9 am; 0 over the sea"})
    tmp = OUT.with_suffix(".tmp")
    ds.to_netcdf(tmp, engine="h5netcdf",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    tmp.rename(OUT)
    print(f"{ds.sizes['day']} days -> {OUT}")


if __name__ == "__main__":
    build()
