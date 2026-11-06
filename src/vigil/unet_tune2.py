"""Re-tune the main model (U-Net, no raw weather, + physics; decided
2026-11-06) on CV. Same grid and rule as the first plan: per held-out fold,
fire weight {3, 10, 30} x no-burn rate {1, 0.5, 0.25}, seed 0, the setting
with the best inner-validation F1 is chosen. Every grid run is also scored
on its held-out fold (cheap), so the chosen seed-0 run is reused; seeds 1-2
are then trained at the chosen settings. CV folds only; test untouched.

  grid rows   -> data/processed/unet_tuning_v2.csv
  chosen rows -> unet_results_v2.csv as "Main (tuned)", so the summary
                 compares it with everything else
  settings    -> unet_runs/v2/main_chosen_settings.json

Usage: python -m vigil.unet_tune2
"""

import gc
import json

import pandas as pd

from vigil.config import CV_FOLDS, PROCESSED
from vigil.dataset import Data
from vigil.unet_plan2 import COLUMNS, RESULTS, RUNS, append, fit_and_score, log, summary

TUNING = PROCESSED / "unet_tuning_v2.csv"
GRID = [(fw, r) for fw in (3.0, 10.0, 30.0) for r in (1.0, 0.5, 0.25)]
BASE = "Main grid"
NAME = "Main (tuned)"
KW = {"drop": ("weather",), "physics": True}


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    data = Data()
    folds = sorted(CV_FOLDS)
    grid = pd.read_csv(TUNING) if TUNING.exists() else pd.DataFrame(columns=COLUMNS)
    for f in folds:  # grid, resumable
        for fw, rate in GRID:
            if ((grid.fold == f) & (grid.fire_weight == fw) & (grid.no_burn_rate == rate)).any():
                continue
            fit_and_score(data, f, BASE, "T", 0, fw, rate, **KW, results=TUNING, tag_extra=f"_w{fw:g}_r{rate:g}")
            gc.collect()
        grid = pd.read_csv(TUNING)
    best = grid.loc[grid.groupby("fold").inner_val_F1.idxmax()].set_index("fold")
    chosen = {int(f): [float(r.fire_weight), float(r.no_burn_rate)] for f, r in best.iterrows()}
    (RUNS / "main_chosen_settings.json").write_text(json.dumps({str(k): v for k, v in chosen.items()}))
    log(f"chosen per fold: {chosen}")
    done = set(zip(*[pd.read_csv(RESULTS)[c] for c in ("model", "fold", "seed")]))
    for f in folds:
        if (NAME, f, 0) not in done:
            append(RESULTS, {**best.loc[f].to_dict(), "fold": f, "model": NAME, "step": "G"}, COLUMNS)
    for seed in (1, 2):
        for f in folds:
            if (NAME, f, seed) not in done:
                fit_and_score(data, f, NAME, "G", seed, *chosen[f], **KW)
                gc.collect()
    log("done\n" + summary().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
