"""Daily burn-day maps: MCD64A1 outlines + VIIRS timing (Parks 2014, IJWF).

For each season:
  1. Fire outlines = connected patches (8-neighbour) of MCD64A1 burnt cells.
  2. A patch belongs to the events whose VIIRS detections fall in it or within
     TOUCH_CELLS of it.
  3. Each cell in the patch gets a burn day interpolated from the patch's
     detections (local fire day): either the nearest detection ("nearest") or
     an inverse-distance-weighted mean of the K nearest ("idw"). The cell's
     event is the event of its nearest detection.
Daily labels for prediction day D: burnt so far = burn_day <= D; newly burnt
= burn_day == D + 1.

Timing uses "constrained nearest": the nearest detection dated within the
cell's MCD64A1 date -2..+3 days (plain nearest if none). Fire days run
06:00-06:00 local. See build() for the outputs and the uncertain mask.

Usage:
  python -m vigil.labels evaluate --season 2020
  python -m vigil.labels build
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

from vigil.burn_history import edge_distance_km, grid
from vigil.config import EDGE_MASK_KM, GRID_RES_M, PROCESSED, fire_day

TOUCH_CELLS = 2      # detection within 2 cells (750 m) of a patch belongs to it
IDW_K, IDW_POWER = 8, 2
CANDIDATES = 32                    # constrained: nearest detections considered
WINDOW_BEFORE, WINDOW_AFTER = 2, 3  # constrained: days around MCD64A1's own date
MIN_MCD_RATIO = 0.5                # events need MCD64A1 cells >= half their detected cells
EPOCH = pd.Timestamp("2000-01-01")


def detections(season: int | None = None) -> pd.DataFrame:
    """Kept detections with event, split and local fire day (days since EPOCH)."""
    d = pd.read_parquet(PROCESSED / "fire_events.parquet")
    if season is not None:
        d = d[d.season == season]
    d = d.assign(fire_day=(fire_day(d.time) - EPOCH).dt.days)
    transform, height, width = grid()
    col, row = ~transform * (d.x.values, d.y.values)
    d["row"], d["col"] = np.floor(row).astype(int), np.floor(col).astype(int)
    return d[(d.row >= 0) & (d.row < height) & (d.col >= 0) & (d.col < width)]


def mcd_season(season: int) -> np.ndarray:
    ds = xr.open_dataset(PROCESSED / "mcd64_burn.nc", engine="h5netcdf")
    return ds.burn_day.sel(season=season).values


def patches(burnt: np.ndarray) -> np.ndarray:
    labels, _ = ndimage.label(burnt, structure=np.ones((3, 3)))
    return labels


def interpolate(det: pd.DataFrame, patch: np.ndarray, method: str, mcd: np.ndarray | None = None):
    """Burn day and event per patch cell, from the detections touching it.

    Returns (burn_day, event) arrays (0 / -1 where not assigned) and the patch
    id each detection was matched to (0 = no patch nearby)."""
    # One point per detected cell: its first fire day (and that detection's event).
    det_full = det
    det = det.sort_values("fire_day").drop_duplicates(["row", "col"])
    # Match each detected cell to the patch at or near it.
    near = ndimage.grey_dilation(patch, size=(2 * TOUCH_CELLS + 1,) * 2)
    det_patch = np.where(patch[det.row, det.col] > 0, patch[det.row, det.col],
                         near[det.row, det.col])
    burn_day = np.zeros(patch.shape, dtype=np.int32)
    event = np.full(patch.shape, -1, dtype=np.int64)
    rows, cols = np.nonzero(patch)
    cell_patch = patch[rows, cols]
    order = np.argsort(cell_patch)
    rows, cols, cell_patch = rows[order], cols[order], cell_patch[order]
    starts = np.searchsorted(cell_patch, np.unique(cell_patch))
    bounds = dict(zip(np.unique(cell_patch), zip(starts, np.append(starts[1:], len(cell_patch)))))
    dp = pd.Series(np.arange(len(det)))[det_patch > 0].groupby(det_patch[det_patch > 0])
    for pid, idx in dp:
        a, b = bounds[pid]
        r, c = rows[a:b], cols[a:b]
        dd = det.iloc[idx.values]
        pts = np.stack([dd.row.values, dd.col.values], axis=1).astype(float)
        tree = cKDTree(pts)
        k = min({"idw": IDW_K, "constrained": CANDIDATES}.get(method, 1), len(pts))
        dist, nn = tree.query(np.stack([r, c], axis=1), k=k)
        if k == 1:
            dist, nn = dist[:, None], nn[:, None]
        days = dd.fire_day.values[nn]
        if method == "constrained":
            # Nearest detection dated within the cell's MCD64A1 date -2..+3 days;
            # if none, plain nearest. Stops later burns inside the outline from
            # setting the burn day.
            m = mcd[r, c][:, None]
            ok = (days >= m - WINDOW_BEFORE) & (days <= m + WINDOW_AFTER)
            pick = np.where(ok.any(1), ok.argmax(1), 0)
            val = days[np.arange(len(r)), pick]
            nn = nn[np.arange(len(r)), pick][:, None]
        elif method == "nearest" or k == 1:
            val = days[:, 0]
        else:
            w = 1.0 / np.maximum(dist, 0.5) ** IDW_POWER
            val = np.rint((w * days).sum(1) / w.sum(1)).astype(int)
        burn_day[r, c] = val
        event[r, c] = dd.event.values[nn[:, 0]]
    # Patch id for every original detection (via its cell).
    cell_pid = pd.Series(det_patch, index=pd.MultiIndex.from_arrays([det.row, det.col]))
    full_pid = cell_pid.reindex(pd.MultiIndex.from_arrays([det_full.row, det_full.col])).values
    return burn_day, event, full_pid


def holdout_error(det: pd.DataFrame, patch: np.ndarray, method: str, frac=0.1, seed=0):
    """Hide a random 10% of detection cells, interpolate from the rest, and
    compare with their true fire day."""
    # Hide whole cells (all their detections), truth = the cell's first fire day.
    cells = det.sort_values("fire_day").drop_duplicates(["row", "col"])
    rng = np.random.default_rng(seed)
    hide = rng.random(len(cells)) < frac
    test = cells[hide]
    hidden = pd.MultiIndex.from_arrays([test.row, test.col])
    train = det[~pd.MultiIndex.from_arrays([det.row, det.col]).isin(hidden)]
    burn_day, _, _ = interpolate(train, patch, method)
    pred = burn_day[test.row, test.col]
    ok = pred > 0
    err = pred[ok] - test.fire_day.values[ok]
    return pd.Series({"cells_tested": int(ok.sum()), "MAE_days": np.abs(err).mean(),
                      "within_1_day": (np.abs(err) <= 1).mean(),
                      "within_2_days": (np.abs(err) <= 2).mean(), "median_bias": np.median(err)})


TOWNS = {  # documented arrival (local date), from CSIRO ES22011
    "Mallacoota": (-37.553, 149.754, "2019-12-31"),
    "Buchan": (-37.497, 148.176, "2019-12-30/31"),
    "Sarsfield": (-37.750, 147.722, "2019-12-30/31"),
    "Clifton Creek": (-37.693, 147.655, "2019-12-30/31"),
    "Corryong": (-36.197, 147.902, "2019-12-30/31"),
}


def town_check(burn_day: np.ndarray, radius_cells: int = 8) -> pd.DataFrame:
    from pyproj import Transformer
    transform, _, _ = grid()
    t = Transformer.from_crs(4326, 3111, always_xy=True)
    rows = []
    for name, (lat, lon, known) in TOWNS.items():
        x, y = t.transform(lon, lat)
        c, r = ~transform * (x, y)
        r, c = int(r), int(c)
        win = burn_day[r - radius_cells:r + radius_cells + 1, c - radius_cells:c + radius_cells + 1]
        v = win[win > 0]
        days = pd.to_datetime(EPOCH) + pd.to_timedelta(v, unit="D")
        rows.append({"town": name, "known": known,
                     "burnt_cells_within_3km": len(v),
                     "interp_earliest": days.min().date() if len(v) else None,
                     "interp_median": days[len(days) // 2].date() if len(v) else None})
    return pd.DataFrame(rows)


def evaluate(season: int) -> None:
    mcd = mcd_season(season)
    patch = patches(mcd > 0)
    det = detections(season)
    for method in ["nearest", "idw"]:
        print(f"\n[{method}] hold-out (10% of detection cells):")
        print(holdout_error(det, patch, method).round(3).to_string())
        bd, _, _ = interpolate(det, patch, method)
        print(town_check(bd).to_string(index=False))
    # MCD64A1's own burn date, for comparison.
    mcd_day = np.where(mcd > 0, mcd, 0)
    print("\n[MCD64A1 own burn date]")
    print(town_check(mcd_day).to_string(index=False))


def coverage() -> None:
    """Per split: events with little/no MCD64A1 area, and detections outside outlines."""
    rows = []
    det_all = detections()
    for season, det in det_all.groupby("season"):
        mcd = mcd_season(season)
        patch = patches(mcd > 0)
        _, event_map, det_patch = interpolate(det, patch, "nearest")
        det = det.assign(in_patch=det_patch > 0,
                         in_mcd=mcd[det.row, det.col] > 0)
        ev_cells = pd.Series(event_map[event_map >= 0]).value_counts()
        g = det.groupby("event")
        e = pd.DataFrame({
            "split": g.split.first(), "exclude": g.exclude_reason.first(),
            "n": g.size(), "det_cells": g.apply(lambda x: len(set(zip(x.row, x.col)))),
            "mcd_cells": ev_cells.reindex(g.size().index).fillna(0).astype(int),
            "det_in_mcd": g.in_mcd.sum(), "det_in_patch": g.in_patch.sum()})
        rows.append(e)
    e = pd.concat(rows)
    e = e[e.exclude.isna()]
    e["ratio"] = e.mcd_cells / e.det_cells
    e["kind"] = np.select([e.mcd_cells == 0, e.ratio < 0.5], ["no MCD64A1", "little (<0.5x)"], "ok")
    print("\nModelling events by MCD64A1 coverage (MCD cells assigned vs detected cells):")
    print(pd.crosstab(e.kind, e.split, margins=True).to_string())
    print("\nDetections:")
    print(pd.crosstab(e.kind, e.split, values=e.n, aggfunc="sum", margins=True).to_string())
    ok = e[e.kind != "no MCD64A1"]
    print("\nFor events with an outline: detections outside any MCD64A1 burnt cell "
          "(strict) / not within 750 m of their patch:")
    print(ok.groupby("split").apply(lambda x: pd.Series({
        "detections": x.n.sum(), "outside_strict": 1 - x.det_in_mcd.sum() / x.n.sum(),
        "outside_750m": 1 - x.det_in_patch.sum() / x.n.sum()})).round(3).to_string())
    e.to_csv(PROCESSED / "label_coverage.csv")


OUT = PROCESSED / "labels.nc"
EVENTS_OUT = PROCESSED / "label_events.parquet"


def build() -> None:
    """Final label rasters, one layer per season.

    burn_day       fire day (days since 2000-01-01) a modelling-event cell burnt
                   (MCD64A1 outline, timing from VIIRS, constrained nearest); 0 = no
    event          event id owning the cell; -1 = none
    uncertain_day  cells excluded from the loss from this day on (0 = never):
                   detections outside any MCD64A1 outline, cells of excluded
                   events, and MCD64A1 burnt patches with no detections. On
                   prediction day D, mask cells with 0 < uncertain_day <= D + 1.
                   Edge: cells within EDGE_MASK_KM of the study box (or outside
                   it) get uncertain_day = 1, i.e. always masked, and cells
                   outside the box lose their labels (their burn day would
                   come from detections inside the box).

    Events with no MCD64A1 area, or fewer MCD64A1 cells than half their detected
    cells, are dropped (their cells become uncertain), so train and test labels
    are both outline-based."""
    transform, height, width = grid()
    det_all = detections()
    seasons = sorted(det_all.season.unique())
    per_season, stats = {}, []
    for season in seasons:
        det = det_all[det_all.season == season]
        mcd = mcd_season(season)
        patch = patches(mcd > 0)
        bd, ev, _ = interpolate(det, patch, "constrained", mcd)
        per_season[season] = (mcd, bd, ev, det)
        cells = det.drop_duplicates(["row", "col", "event"])
        stats.append(pd.DataFrame({
            "det_cells": cells.groupby("event").size(),
            "mcd_cells": pd.Series(ev[ev >= 0]).value_counts()}).fillna(0))
        print(f"  interpolated season {season}", flush=True)

    # Event table: earlier exclusions + MCD64A1 coverage.
    st = pd.concat(stats).groupby(level=0).sum()
    g = det_all.groupby("event")
    e = pd.DataFrame({"split": g.split.first(), "source": g.event_source.first(),
                      "n": g.size(), "exclude_reason": g.exclude_reason.first()})
    e = e.join(st).fillna({"det_cells": 0, "mcd_cells": 0})
    e["mcd_ratio"] = e.mcd_cells / e.det_cells.clip(lower=1)
    keep = e.exclude_reason.isna()
    e.loc[keep & (e.mcd_cells == 0), "exclude_reason"] = "no_mcd64"
    e.loc[keep & (e.mcd_cells > 0) & (e.mcd_ratio < MIN_MCD_RATIO), "exclude_reason"] = "little_mcd64"
    modelled = set(e.index[e.exclude_reason.isna()])

    burn = np.zeros((len(seasons), height, width), dtype=np.int32)
    owner = np.full((len(seasons), height, width), -1, dtype=np.int32)
    unc = np.zeros((len(seasons), height, width), dtype=np.int32)
    edge = edge_distance_km()
    outside_box, near_edge = edge < 0, edge < EDGE_MASK_KM
    for i, season in enumerate(seasons):
        mcd, bd, ev, det = per_season[season]
        good = (ev >= 0) & np.isin(ev, list(modelled))
        burn[i][good] = bd[good]
        owner[i][good] = ev[good]
        u = np.zeros((height, width), dtype=np.int32)
        # MCD64A1 cells not labelled: excluded events' cells (their interpolated
        # day) and patches with no detections (MCD64A1's own date).
        other = (mcd > 0) & ~good
        u[other] = np.where(bd[other] > 0, bd[other], mcd[other])
        # Detection cells outside any MCD64A1 outline: first fire day there.
        first = det.sort_values("fire_day").drop_duplicates(["row", "col"])
        outside = mcd[first.row, first.col] == 0
        r, c, fd = first.row.values[outside], first.col.values[outside], first.fire_day.values[outside]
        cur = u[r, c]
        u[r, c] = np.where((cur == 0) | (fd < cur), fd, cur)
        burn[i][outside_box] = 0
        owner[i][outside_box] = -1
        u[near_edge] = 1
        unc[i] = u

    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ds = xr.Dataset({"burn_day": (("season", "y", "x"), burn),
                     "event": (("season", "y", "x"), owner),
                     "uncertain_day": (("season", "y", "x"), unc)},
                    coords={"season": seasons, "y": ys, "x": xs},
                    attrs={"crs": "EPSG:3111", "transform": list(transform)[:6],
                           "days": "days since 2000-01-01, fire days 06:00-06:00 local"})
    tmp = OUT.with_suffix(".tmp")
    ds.to_netcdf(tmp, engine="h5netcdf",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    tmp.rename(OUT)
    tmp = EVENTS_OUT.with_suffix(".tmp")
    e.reset_index().to_parquet(tmp, index=False)
    tmp.rename(EVENTS_OUT)
    print(f"-> {OUT}\n-> {EVENTS_OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--season", type=int, default=2020)
    sub.add_parser("coverage")
    sub.add_parser("build")
    args = parser.parse_args()
    {"evaluate": lambda: evaluate(args.season), "coverage": coverage, "build": build}[args.cmd]()


if __name__ == "__main__":
    main()
