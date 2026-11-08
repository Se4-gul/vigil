"""Build the public fire-label dataset in release/ (CC BY 4.0).

  release/data/vigil_labels_<season>.nc   one CF-1.8 NetCDF per fire season
                                          (2013-2025), 375 m grid, EPSG:3111:
                                          burn_date, event_id, uncertain_from
  release/vigil_events.csv                one row per modelling event
  release/vigil_splits.csv                season -> CV fold or test

The datasheet, README and loading example in release/ are written by hand;
this script only rebuilds the data files from data/processed/ (labels.nc,
label_events.parquet, fire_events.parquet, windows.parquet, edge_events.csv,
absorbed_fragments.csv).

Usage: python -m vigil.release
"""

import hashlib
import json

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from vigil.config import CV_FOLDS, GRID_CRS, PROCESSED, ROOT, TEST_SEASONS

OUT = ROOT / "release"
VERSION = "1.0"
DOI = "https://doi.org/10.5281/zenodo.24049740"
EPOCH = "2000-01-01"
FILL = -1
CELL_HA = 0.375 ** 2 * 100            # 14.0625 ha per 375 m cell
FIRE_SEASON_MONTHS = {10, 11, 12, 1, 2, 3, 4}
DAYS = f"days since {EPOCH} 00:00:00"


def fold_of(season):
    if season in TEST_SEASONS:
        return "test"
    return next(str(k) for k, v in CV_FOLDS.items() if season in v)


def season_file(lab, season, transform):
    s = lab.sel(season=season)
    bd, ev, ud = (s[k].values.astype(np.int32) for k in ("burn_day", "event", "uncertain_day"))
    crs = CRS.from_epsg(GRID_CRS)
    gt = " ".join(str(v) for v in (transform[2], transform[0], 0.0, transform[5], 0.0, transform[4]))
    common = {"grid_mapping": "crs"}
    ds = xr.Dataset(
        {
            "burn_date": (("y", "x"), np.where(bd > 0, bd, FILL), {
                **common, "long_name": "fire day on which the cell first burnt",
                "units": DAYS, "calendar": "standard",
                "comment": "Fire days run 06:00 to 06:00 Australian Eastern local time and are named by the "
                           "date they start on. A cell burnt on fire day D is 'newly burnt' when predicting "
                           "D from D-1. Missing = did not burn in a modelling event this season."}),
            "event_id": (("y", "x"), np.where(ev >= 0, ev, FILL), {
                **common, "long_name": "fire event the burnt cell belongs to",
                "comment": "Joins to event_id in vigil_events.csv. Missing = not burnt by a modelling event."}),
            "uncertain_from": (("y", "x"), np.where(ud > 0, ud, FILL), {
                **common, "long_name": "date from which the cell's label is unreliable",
                "units": DAYS, "calendar": "standard",
                "comment": "Leave the cell out when scoring a prediction for fire day T if uncertain_from <= T. "
                           "A date of 2000-01-02 means always (within 6 km of the study-area edge or outside "
                           "it). Later dates come from fire detections outside any burned-area outline, cells "
                           "of excluded events (e.g. planned burns) and burned-area patches with no detections. "
                           "Missing = never uncertain."}),
            "crs": ((), np.int32(0), {
                "grid_mapping_name": "transverse_mercator", "crs_wkt": crs.to_wkt(), "spatial_ref": crs.to_wkt(),
                "epsg_code": f"EPSG:{GRID_CRS}", "GeoTransform": gt,
                **{k: v for k, v in crs.to_cf().items() if k not in ("crs_wkt", "grid_mapping_name")}}),
        },
        coords={
            "x": ("x", s.x.values, {"standard_name": "projection_x_coordinate", "long_name": "easting of cell centre",
                                    "units": "m", "axis": "X"}),
            "y": ("y", s.y.values, {"standard_name": "projection_y_coordinate", "long_name": "northing of cell centre",
                                    "units": "m", "axis": "Y"}),
        },
        attrs={
            "title": f"Vigil daily burn-day labels, fire season {season - 1}-{str(season)[2:]}",
            "summary": "Day each 375 m cell first burnt in a bushfire, from MODIS MCD64A1 burned-area outlines "
                       "timed with VIIRS S-NPP 375 m active-fire detections, for Victoria, Australia and the "
                       "adjoining parts of NSW and SA. Built to train and test next-day fire-spread models.",
            "fire_season": f"{season - 1}-{str(season)[2:]} (labelled {season}; 1 July {season - 1} to 30 June {season})",
            "cv_fold": fold_of(season),
            "Conventions": "CF-1.8",
            "license": "CC-BY-4.0",
            "version": VERSION,
            "doi": DOI,
            "creator_name": "Rishwanth Kurra",
            "source": "NASA FIRMS VIIRS S-NPP 375 m (SP); NASA MODIS MCD64A1 C6.1; DEECA Victoria fire history "
                      "(event tagging only)",
            "references": "See DATASHEET.md. Method follows Parks (2014), doi:10.1071/WF13138.",
            "geospatial_crs": f"EPSG:{GRID_CRS} (GDA94 / Vicgrid94)",
            "cell_size_m": 375.0,
            "fire_day": "06:00 to 06:00 local time",
            "code": "https://github.com/Se4-gul/vigil (src/vigil/events.py, src/vigil/labels.py, src/vigil/release.py)",
        },
    )
    enc = {k: {"zlib": True, "complevel": 5, "dtype": "int32", "_FillValue": FILL}
           for k in ("burn_date", "event_id", "uncertain_from")}
    enc["x"] = enc["y"] = {"_FillValue": None}
    path = OUT / "data" / f"vigil_labels_{season}.nc"
    ds.to_netcdf(path, engine="h5netcdf", encoding=enc)
    return bd, ev


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    lab = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf")
    transform = [float(v) for v in lab.attrs["transform"]]
    le = pd.read_parquet(PROCESSED / "label_events.parquet").set_index("event")
    det = pd.read_parquet(PROCESSED / "fire_events.parquet", columns=["event", "deeca_firekey"])
    keys = det.dropna(subset=["deeca_firekey"]).groupby("event").deeca_firekey.agg(lambda s: s.mode().iat[0])
    win = pd.read_parquet(PROCESSED / "windows.parquet")
    with_windows = set(zip(win.season, win.event))
    edge = pd.read_csv(PROCESSED / "edge_events.csv")
    clipped = set(zip(edge.season[edge.patch_past_edge], edge.event[edge.patch_past_edge]))
    absorbed = pd.read_csv(PROCESSED / "absorbed_fragments.csv").groupby(["season", "host"]).size()
    to_ll = Transformer.from_crs(GRID_CRS, 4326, always_xy=True)
    x0, dx, y0, dy = transform[2], transform[0], transform[5], transform[4]

    rows = []
    for season in lab.season.values.tolist():
        bd, ev = season_file(lab, season, transform)
        for e in np.unique(ev[ev >= 0]).tolist():
            r, c = np.nonzero(ev == e)
            d = bd[r, c]
            dates = pd.Timestamp(EPOCH) + pd.to_timedelta(np.unique(d), unit="D")
            lon, lat = to_ll.transform(x0 + dx * (c.mean() + 0.5), y0 + dy * (r.mean() + 0.5))
            flags = []
            if (season, e) in clipped:
                flags.append("edge_clipped")
            if (season, e) not in with_windows:
                flags.append("no_scored_days")
            if len(dates) < 3:
                flags.append("under_3_burn_days")
            if any(m not in FIRE_SEASON_MONTHS for m in dates.month):
                flags.append("burns_outside_oct_apr")
            if (season, e) in absorbed.index:
                flags.append(f"absorbed_{absorbed[(season, e)]}_fragments")
            rows.append({
                "event_id": e, "season": season, "fold": fold_of(season),
                "source": le.loc[e, "source"], "deeca_fire_key": keys.get(e, ""),
                "first_burn_date": dates.min().date(), "last_burn_date": dates.max().date(),
                "burn_days": len(dates), "cells": len(r), "area_ha": round(len(r) * CELL_HA, 1),
                "centroid_lon": round(lon, 4), "centroid_lat": round(lat, 4),
                "detections": int(le.loc[e, "n"]), "flags": ";".join(flags),
            })
    events = pd.DataFrame(rows).sort_values(["season", "first_burn_date", "event_id"])
    events.to_csv(OUT / "vigil_events.csv", index=False)

    splits = [{"season": s, "season_span": f"{s - 1}-{str(s)[2:]}", "fold": fold_of(s),
               "role": "test (run once)" if s in TEST_SEASONS else "cross-validation"}
              for s in lab.season.values.tolist()]
    pd.DataFrame(splits).to_csv(OUT / "vigil_splits.csv", index=False)

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".nc", ".csv"))
    sums = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(events)} events, {len(splits)} seasons -> {OUT} ({size:.1f} MB)")
    print(events["flags"].str.split(";").explode().value_counts().to_string())
    print(json.dumps(events.groupby("fold").event_id.count().to_dict()))


if __name__ == "__main__":
    main()
