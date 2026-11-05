"""Daily fire-weather summaries from hourly BARRA-C2, per fire day.

Fire days run 06:00-06:00 local (config.FIRE_DAY_START_HOUR). For each fire
day and BARRA cell (4 km):
  ws_max   maximum 10 m wind speed (m/s)
  u_peak   10 m wind components (m/s, eastward / northward) at the hour of
  v_peak   peak fire danger: the hour that maximises the weather terms of the
           McArthur FFDI, -0.0345 RH + 0.0338 T + 0.0234 V (V in km/h), with
           the drought factor held constant
  t_max    maximum 2 m temperature (deg C)
  rh_min   minimum 2 m relative humidity (%), clipped at 100 (BARRA has a
           few values just above 100)
  danger_max  max over hours of FFDI's weather terms (for peak FFDI, with the
           drought factor added in dataset.py)
Only complete fire days are kept (>= 23 hours): BARRA was downloaded for
Oct-Apr, so 1 October (which starts on 30 September UTC) is dropped.

Inputs use day D and day D+1 (the day being predicted). Using D+1's actual
weather assumes a perfect forecast (see limitations in CLAUDE.md).

Output: data/processed/weather_daily.nc on the BARRA grid. dataset.Data
resamples each window onto the 375 m grid (bilinear) and rotates winds into
grid axes.

Usage:
  python -m vigil.weather
"""

import glob

import numpy as np
import pandas as pd
import xarray as xr

from vigil.config import PROCESSED, RAW, fire_day

OUT = PROCESSED / "weather_daily.nc"
VARS = ["ws_max", "u_peak", "v_peak", "t_max", "rh_min", "danger_max"]


def months() -> list[str]:
    return sorted(p.split("/")[-1][:6] for p in glob.glob(str(RAW / "barra2" / "uas" / "*.nc")))


def load_month(ym: str) -> xr.Dataset:
    # Wind is at 10 m and temperature/humidity at 2 m, so drop the scalar
    # `height` coordinate before merging.
    ds = xr.merge([xr.open_dataset(RAW / "barra2" / v / f"{ym}.nc", engine="h5netcdf")[v]
                   .drop_vars("height", errors="ignore")
                   for v in ["uas", "vas", "tas", "hurs"]])
    return ds.load()


def summarise(ds: xr.Dataset) -> xr.Dataset:
    """One fire day of hourly data -> daily summaries."""
    ws = np.hypot(ds.uas, ds.vas)
    rh = ds.hurs.clip(0, 100)
    tc = ds.tas - 273.15
    danger = -0.0345 * rh + 0.0338 * tc + 0.0234 * ws * 3.6
    peak = danger.fillna(-1e9).argmax("time")
    return xr.Dataset({
        "ws_max": ws.max("time"),
        "u_peak": ds.uas.isel(time=peak),
        "v_peak": ds.vas.isel(time=peak),
        "t_max": tc.max("time"),
        "rh_min": rh.min("time"),
        # Max over hours of FFDI's weather terms; peak FFDI for the day is
        # 2 exp(-0.45 + 0.987 ln DF + danger_max) (DF is constant within a day).
        "danger_max": danger.max("time"),
    })


def build() -> None:
    out, buffer = [], None
    for ym in months():
        ds = load_month(ym)
        if buffer is not None:
            ds = xr.concat([buffer, ds], dim="time")
        fd = fire_day(pd.Series(ds.time.values)).values
        days = np.unique(fd)
        # The last fire day may continue into next month: carry it over.
        for day in days[:-1]:
            sel = fd == day
            if sel.sum() >= 23:
                out.append(summarise(ds.isel(time=np.nonzero(sel)[0])).expand_dims(day=[day]))
        buffer = ds.isel(time=np.nonzero(fd == days[-1])[0])
        print(f"  {ym}: {len(out)} days so far", flush=True)
    # Last fire day of the record, if complete.
    if buffer is not None and buffer.sizes["time"] >= 23:
        day = fire_day(pd.Series(buffer.time.values)).values[0]
        out.append(summarise(buffer).expand_dims(day=[day]))
    daily = xr.concat(out, dim="day").astype("float32")
    tmp = OUT.with_suffix(".tmp")
    daily.to_netcdf(tmp, engine="h5netcdf",
                    encoding={v: {"zlib": True, "complevel": 4} for v in VARS})
    tmp.rename(OUT)
    print(f"{daily.sizes['day']} fire days -> {OUT}")


if __name__ == "__main__":
    build()
