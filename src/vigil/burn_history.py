"""Time since last burnt, on the modelling grid.

The fuel map is static, so it can't know what burnt recently (e.g. Black
Summer, for training seasons 2021-2025). This builds the inputs for a "years
since last burnt" channel, capped at CAP_YEARS ("10+"), in two versions for
an ablation:

  mcd64      previous seasons from MCD64A1 burned area (default; the same
             product over Vic, NSW and SA, from 2003)
  combined   previous seasons from DEECA (Vic), NPWS (NSW), DEW (SA) polygons
             and FIRMS footprints (2012 on)
  firms      previous seasons from FIRMS footprints only

Both add FIRMS burns from the current season up to the day before the
prediction day (fire days run 06:00-06:00 local). FIRMS dates are exact and that's known on the day, so it's not
leakage. Burns by the current fire itself are left out: those go in a separate
"own burnt area so far" channel, so the two channels don't duplicate each
other.

Planned burns count (they remove fuel), even though they aren't training events.
Polygons use season only (DEECA start dates are unreliable).

Outputs (data/processed/):
  burn_history.nc        last_prev_combined, last_prev_firms: per target season,
                         the most recent earlier season each cell burnt (0 = not
                         within the record)
  firms_burn_cells.parquet
                         each FIRMS-burnt cell: season, local fire day, event

Usage:
  python -m vigil.burn_history          # build
Then at window time: tslb(season, day, event, source) -> years, capped.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr
from affine import Affine
from pyproj import Transformer

from vigil.config import (GRID_CRS, GRID_RES_M, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN,
                          PROCESSED, RAW, fire_day)

CAP_YEARS = 10          # forest fuel mostly recovers within ~10-15 years
TARGET_SEASONS = range(2013, 2026)
FIRST_POLYGON_SEASON = TARGET_SEASONS[0] - CAP_YEARS   # older burns all read as "10+"
OUT = PROCESSED / "burn_history.nc"
CELLS = PROCESSED / "firms_burn_cells.parquet"


def grid() -> tuple[Affine, int, int]:
    """Transform and shape of the modelling grid covering the config bounds."""
    t = Transformer.from_crs(4326, GRID_CRS, always_xy=True)
    lons = np.linspace(LON_MIN, LON_MAX, 50)
    lats = np.linspace(LAT_MIN, LAT_MAX, 50)
    edge_lon = np.concatenate([lons, lons, np.full(50, LON_MIN), np.full(50, LON_MAX)])
    edge_lat = np.concatenate([np.full(50, LAT_MIN), np.full(50, LAT_MAX), lats, lats])
    x, y = t.transform(edge_lon, edge_lat)
    x0 = np.floor(x.min() / GRID_RES_M) * GRID_RES_M
    y1 = np.ceil(y.max() / GRID_RES_M) * GRID_RES_M
    width = int(np.ceil((x.max() - x0) / GRID_RES_M))
    height = int(np.ceil((y1 - y.min()) / GRID_RES_M))
    return Affine(GRID_RES_M, 0, x0, 0, -GRID_RES_M, y1), height, width


def season_from_label(s: pd.Series) -> pd.Series:
    """'1970-71 Wildfire' or '2013/2014' -> 1971 / 2014 (season = year it ends)."""
    start = s.str.extract(r"(\d{4})")[0].astype(float)
    return start + 1


def load_polygons() -> gpd.GeoDataFrame:
    """Burnt-area polygons from DEECA, NPWS and DEW: season + geometry."""
    parts = []
    feats = [json.loads(line) for line in open(RAW / "fire_history" / "large_fires.geojsonl")]
    v = gpd.GeoDataFrame.from_features(feats, crs=4326)
    v = v[v.firetype.isin(["Bushfire", "Burn"])]
    parts.append(gpd.GeoDataFrame({"season": v.season.astype(float), "source": "DEECA"},
                                  geometry=v.geometry, crs=4326))

    n = gpd.read_file(RAW / "fire_history_nsw" / "NPWSFireHistory.shp")
    parts.append(gpd.GeoDataFrame({"season": season_from_label(n.Label), "source": "NPWS"},
                                  geometry=n.geometry, crs=n.crs).to_crs(4326))

    s = gpd.read_file(RAW / "fire_history_sa" / "FIREMGT_FireHistory_GDA2020.shp")
    parts.append(gpd.GeoDataFrame({"season": season_from_label(s.FINANCIALY), "source": "DEW"},
                                  geometry=s.geometry, crs=s.crs).to_crs(4326))

    p = pd.concat(parts, ignore_index=True)
    p = p[p.season.between(FIRST_POLYGON_SEASON, TARGET_SEASONS[-1] - 1)]
    p = p.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX].to_crs(GRID_CRS)
    p["geometry"] = p.geometry.make_valid()
    p["season"] = p.season.astype(int)
    return p


def firms_cells(transform: Affine, height: int, width: int) -> pd.DataFrame:
    """One row per FIRMS-burnt cell per season: first local fire day and its event.

    Uses every detection that passed the filters (planned burns and small
    events included: they all remove fuel), plus season 2012 (partial, from
    2012-01-20) for history before the first target season."""
    d = pd.read_parquet(PROCESSED / "detections_tagged.parquet",
                        columns=["x", "y", "time", "season", "removed_by"])
    d = d[d.removed_by.isna() | ((d.removed_by == "season") & (d.season == 2012))]
    ev = pd.read_parquet(PROCESSED / "fire_events.parquet", columns=["x", "y", "time", "event"])
    d = d.merge(ev, on=["x", "y", "time"], how="left")   # season 2012 has no event id
    d["event"] = d.event.fillna(-1).astype(int)

    inv = ~transform
    col, row = inv * (d.x.values, d.y.values)
    d["row"], d["col"] = np.floor(row).astype(int), np.floor(col).astype(int)
    d = d[(d.row >= 0) & (d.row < height) & (d.col >= 0) & (d.col < width)]
    d["fire_day"] = fire_day(d.time)
    cells = (d.sort_values("fire_day")
              .drop_duplicates(["season", "row", "col", "event"])
              [["season", "row", "col", "fire_day", "event"]])
    return cells.reset_index(drop=True)


def build() -> None:
    transform, height, width = grid()
    print(f"Grid: {width} x {height} cells of {GRID_RES_M} m (EPSG:{GRID_CRS})")

    polys = load_polygons()
    print("Polygons per source (seasons "
          f"{FIRST_POLYGON_SEASON}-{TARGET_SEASONS[-1] - 1}):",
          polys.source.value_counts().to_dict())
    cells = firms_cells(transform, height, width)
    print(f"FIRMS burnt cells: {len(cells):,} (cell x season x event)")

    def burnt(season: int, with_polygons: bool) -> np.ndarray:
        m = np.zeros((height, width), dtype=bool)
        if with_polygons:
            ps = polys[polys.season == season]
            if len(ps):
                m |= rasterio.features.rasterize(
                    ((g, 1) for g in ps.geometry), out_shape=(height, width),
                    transform=transform, fill=0, dtype="uint8").astype(bool)
        c = cells[cells.season == season]
        m[c.row.values, c.col.values] = True
        return m

    mcd = xr.open_dataset(PROCESSED / "mcd64_burn.nc", engine="h5netcdf").burn_day

    first = FIRST_POLYGON_SEASON
    last_comb = np.zeros((height, width), dtype=np.int16)
    last_firms = np.zeros((height, width), dtype=np.int16)
    last_mcd = np.zeros((height, width), dtype=np.int16)
    out_comb, out_firms, out_mcd = [], [], []
    for season in range(first, TARGET_SEASONS[-1] + 1):
        if season in TARGET_SEASONS:   # snapshot: burns from earlier seasons only
            out_comb.append(last_comb.copy())
            out_firms.append(last_firms.copy())
            out_mcd.append(last_mcd.copy())
        last_comb[burnt(season, with_polygons=True)] = season
        if season >= 2012:
            last_firms[burnt(season, with_polygons=False)] = season
        last_mcd[mcd.sel(season=season).values > 0] = season

    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    xs = transform.c + GRID_RES_M * (np.arange(width) + 0.5)
    ds = xr.Dataset(
        {"last_prev_combined": (("season", "y", "x"), np.stack(out_comb)),
         "last_prev_firms": (("season", "y", "x"), np.stack(out_firms)),
         "last_prev_mcd64": (("season", "y", "x"), np.stack(out_mcd))},
        coords={"season": list(TARGET_SEASONS), "y": ys, "x": xs},
        attrs={"crs": f"EPSG:{GRID_CRS}", "transform": list(transform)[:6],
               "note": "most recent earlier season burnt; 0 = not in record"})
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    tmp = OUT.with_suffix(".tmp")
    ds.to_netcdf(tmp, engine="h5netcdf", encoding=enc)
    tmp.rename(OUT)
    tmp = CELLS.with_suffix(".tmp")
    cells.to_parquet(tmp, index=False)
    tmp.rename(CELLS)
    print(f"-> {OUT}\n-> {CELLS}")


_cache: dict = {}


def tslb(season: int, day: pd.Timestamp, event: int, source: str = "mcd64") -> np.ndarray:
    """Years since last burnt for a prediction on local fire `day`, capped at
    CAP_YEARS. Earlier seasons from `source`: 'mcd64' (default: MODIS burned
    area, the same product in every state, from 2003), 'combined' (state
    polygons + FIRMS) or 'firms'. This season: FIRMS burns by other events up
    to the day before."""
    if "ds" not in _cache:
        _cache["ds"] = xr.open_dataset(OUT, engine="h5netcdf").load()  # in memory: fast per sample
        _cache["cells"] = pd.read_parquet(CELLS)
    last = _cache["ds"][f"last_prev_{source}"].sel(season=season).values
    years = np.where(last > 0, season - last, CAP_YEARS).astype(np.float32)
    if "by_season" not in _cache:
        _cache["by_season"] = dict(tuple(_cache["cells"].groupby("season")))
    c = _cache["by_season"].get(season, _cache["cells"].iloc[:0])
    c = c[(c.fire_day < day) & (c.event != event)]
    years[c.row.values, c.col.values] = 0
    return np.minimum(years, CAP_YEARS)


if __name__ == "__main__":
    build()
