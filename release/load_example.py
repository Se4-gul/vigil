"""Load one season, list its fires, and get one day's newly burnt cells.
Needs: pip install xarray h5netcdf pandas pyproj"""
import pandas as pd
import xarray as xr
from pyproj import Transformer

ds = xr.open_dataset("data/vigil_labels_2020.nc")          # dates decode to datetime64, -1 to NaN/NaT
events = pd.read_csv("vigil_events.csv", parse_dates=["first_burn_date", "last_burn_date"])
print(events[events.season == 2020].nlargest(5, "area_ha")[["event_id", "first_burn_date", "area_ha", "flags"]])

day = pd.Timestamp("2019-12-31")                            # a fire day (06:00 to 06:00 local)
new = ds.burn_date == day                                   # cells that first burnt that day
burnt_before = ds.burn_date < day                           # everything burnt by the start of it
scored = ~(ds.uncertain_from <= day)                        # drop unreliable cells before scoring
print(f"{int((new & scored).sum())} newly burnt cells on {day.date()}, "
      f"{int((new & scored).sum()) * 14.0625:.0f} ha")

y, x = map(lambda a: a[0], (new & scored).values.nonzero())  # first scored new cell -> lon/lat
lon, lat = Transformer.from_crs(3111, 4326, always_xy=True).transform(ds.x[x].item(), ds.y[y].item())
print(f"e.g. a cell at {lat:.3f}, {lon:.3f}, event {int(ds.event_id[y, x])}")
