"""Training samples: 128 x 128 windows around each day's active fire.

A sample = (event, fire day D, window centre). It exists when the event has
active fire on D (cells with burn_day == D). The target is the cells newly
burnt on D+1; days where nothing new burns are "no-new-burn" samples.

Large fires get several windows: active cells are grouped into 64 x 64 blocks
(half a window), and each block with active fire gets a window centred on its
active cells. So every active cell starts at least 32 cells (12 km) from its
window's edge, or 16 cells (6 km) after jitter.

Jitter: the loader shifts each training window's centre by a random offset of
up to +-JITTER cells, new every epoch (`window_origin`), so fires aren't always
centred. Held-out seasons and the test set are scored on whole fires, not
windows (the U-Net works on any input size).

No-new-burn days: 70% of training samples vs 34% in test. Their sampling rate
is a hyperparameter tuned in the season cross-validation (keep all / 50% /
25%), not matched to test (that would peek at the test set).

Scope: only fire-season windows (D and D+1 in October-April), the same rule
for every split, including test.

Output: data/processed/windows.parquet

Usage:
  python -m vigil.windows
"""

import numpy as np
import pandas as pd
import xarray as xr

from vigil.config import CV_FOLDS, PROCESSED

SIZE = 128
BLOCK = SIZE // 2
JITTER = 16
TEST_SEASON = 2020
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4]   # scope rule: fire season only
OUT = PROCESSED / "windows.parquet"


def build() -> pd.DataFrame:
    L = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf")
    rows = []
    for season in L.season.values:
        ev = L.event.sel(season=season).values
        bd = L.burn_day.sel(season=season).values
        H, W = ev.shape
        r, c = np.nonzero(ev >= 0)
        cells = pd.DataFrame({"event": ev[r, c], "day": bd[r, c], "row": r, "col": c})
        for (event, day), act in cells.groupby(["event", "day"]):
            ev_cells = cells[cells.event == event]
            new_next = ev_cells[ev_cells.day == day + 1]
            act = act.assign(br=act.row // BLOCK, bc=act.col // BLOCK)
            for _, blk in act.groupby(["br", "bc"]):
                cr = int(np.clip(round(blk.row.mean()), SIZE // 2, H - SIZE // 2))
                cc = int(np.clip(round(blk.col.mean()), SIZE // 2, W - SIZE // 2))
                inside = ((new_next.row - cr).abs() < SIZE // 2) & ((new_next.col - cc).abs() < SIZE // 2)
                rows.append({"season": int(season), "event": int(event), "day": int(day),
                             "centre_row": cr, "centre_col": cc,
                             "active_cells": len(blk),
                             "new_cells_event": len(new_next),
                             "new_cells_window": int(inside.sum())})
    w = pd.DataFrame(rows)
    # Scope: fire season only. D and D+1 must both fall in October-April, in
    # every split (train, CV and test alike). Weather (BARRA) covers Oct-Apr;
    # fires outside it are mostly off-season burns.
    month = lambda days: (pd.Timestamp("2000-01-01") + pd.to_timedelta(days, "D")).dt.month
    in_season = month(w.day).isin(SEASON_MONTHS) & month(w.day + 1).isin(SEASON_MONTHS)
    print(f"Out of scope (D or D+1 outside Oct-Apr): {int((~in_season).sum())} windows dropped")
    w = w[in_season].reset_index(drop=True)
    w["has_new_burn"] = w.new_cells_event > 0
    season_fold = {s: str(f) for f, ss in CV_FOLDS.items() for s in ss}
    w["fold"] = w.season.map(season_fold).fillna("test")
    tmp = OUT.with_suffix(".tmp")
    w.to_parquet(tmp, index=False)
    tmp.rename(OUT)
    return w


def window_origin(centre_row: int, centre_col: int, shape: tuple[int, int],
                  rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Top-left corner of a window, jittered by up to +-JITTER cells if rng given."""
    dr = dc = 0
    if rng is not None:
        dr, dc = rng.integers(-JITTER, JITTER + 1, size=2)
    r0 = int(np.clip(centre_row + dr - SIZE // 2, 0, shape[0] - SIZE))
    c0 = int(np.clip(centre_col + dc - SIZE // 2, 0, shape[1] - SIZE))
    return r0, c0


def keep_no_burn(w: pd.DataFrame, rate: float, seed: int = 0) -> pd.DataFrame:
    """Subsample no-new-burn windows to `rate` (CV hyperparameter: 1, 0.5, 0.25)."""
    rng = np.random.default_rng(seed)
    keep = w.has_new_burn | (rng.random(len(w)) < rate)
    return w[keep]


if __name__ == "__main__":
    w = build()
    print(f"{len(w):,} windows -> {OUT}")
