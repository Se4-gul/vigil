"""U-Net runs on the fixed data (edge mask, validity channel, fragments
absorbed; 2026-11-06). CV folds only; test untouched.

Per-fold settings (fire weight, no-burn rate) are the ones step 1 of the
first plan chose (unet_runs/chosen_settings.json), not re-tuned: the data
changed very little (windows per fold within 3).

  A  U-Net, seeds 0-2
  B  U-Net + physics, seeds 0-2
  C  wind sensitivity on the seed-0 models of A and B: score the held-out fold
     with every wind vector (D and D+1) rotated by 90 and 180 degrees, the
     image unchanged. If the model uses wind direction, F1 drops and
     predictions move.
  D  no weather, seeds 0-2
  E  no weather + physics, seeds 0-2
  F  wind-aligned windows (each window rotated so D+1 peak wind points up,
     exact angle; flips only), seeds 0-2

Rows go to data/processed/unet_results_v2.csv (one fixed layout), the
sensitivity test to unet_sensitivity.csv.

Usage:
  python -m vigil.unet_plan2 [--steps A B C D E F]
  python -m vigil.unet_plan2 summary
"""

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import torch

from vigil.config import CV_FOLDS, PROCESSED
from vigil.dataset import CHANNELS, VECTORS, Data
from vigil.unet import (PAD, best_threshold, load_channel_store, metrics_from_counts,
                        physics_slice, predict_region, score_fold, train)

RESULTS = PROCESSED / "unet_results_v2.csv"
SENS = PROCESSED / "unet_sensitivity.csv"
SUMMARY = PROCESSED / "unet_summary_v2.csv"
RUNS = PROCESSED / "unet_runs" / "v2"
CHOSEN = PROCESSED / "unet_runs" / "chosen_settings.json"
METRICS = ["F1", "IoU", "precision", "recall", "PR_AUC", "F1 <=6 km", "recall <=6 km",
           "F1 6-24 km", "recall 6-24 km", "F1 >24 km", "recall >24 km"]
COLUMNS = (["step", "model", "fold", "seed", "fire_weight", "no_burn_rate", "physics", "drop", "align",
            "inner_val_fold", "epochs", "threshold", "inner_val_F1", "minutes"]
           + [f"heldout_{m}" for m in METRICS])
WIND = [("u_peak_D", "v_peak_D"), ("u_peak_D1", "v_peak_D1")]


def log(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def memory() -> str:
    """Process RSS and MPS memory, for diagnosing slowdowns or freezes."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes
    mps = torch.mps.driver_allocated_memory() / 2**30 if torch.backends.mps.is_available() else 0
    return f"peak RSS {rss:.1f} GB, MPS {mps:.1f} GB"


def done_runs() -> set:
    if not RESULTS.exists():
        return set()
    r = pd.read_csv(RESULTS)
    return set(zip(r.model, r.fold, r.seed))


def append(path, row: dict, columns=None):
    df = pd.DataFrame([row])
    if columns:
        df = df.reindex(columns=columns)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def fit_and_score(data, fold, name, step, seed, fw, rate, physics=False, drop=(), align=False,
                  results=RESULTS, tag_extra=""):
    t0 = time.time()
    model, stats, keep, inner_val, hist = train(fold, physics=physics, drop=drop, fire_weight=fw,
                                                no_burn_rate=rate, seed=seed, align=align)
    pv, nv = score_fold(model, data, inner_val, stats, keep, physics, align=align)
    thr = best_threshold(pv, nv)
    pt, nt = score_fold(model, data, fold, stats, keep, physics, align=align)
    tag = f"{name.replace(' ', '_').replace('+', 'plus')}{tag_extra}_f{fold}_s{seed}"
    torch.save(model.state_dict(), RUNS / f"{tag}.pt")
    np.savez(RUNS / f"{tag}_counts.npz", pos=pt, neg=nt, pos_val=pv, neg_val=nv)
    row = {"step": step, "model": name, "fold": fold, "seed": seed, "fire_weight": fw,
           "no_burn_rate": rate, "physics": physics, "drop": "+".join(drop), "align": align,
           "inner_val_fold": inner_val, "epochs": len(hist), "threshold": thr / 1000,
           "inner_val_F1": metrics_from_counts(pv, nv, thr)["F1"],
           "minutes": round((time.time() - t0) / 60, 1),
           **{f"heldout_{k}": v for k, v in metrics_from_counts(pt, nt, thr).items()}}
    append(results, row, COLUMNS)
    log(f"{step} {name} seed {seed} fold {fold}: held-out F1 {row['heldout_F1']:.3f} "
        f"({row['minutes']} min, {len(hist)} epochs; {memory()})")
    return row, model, stats, keep, thr


def rotate_wind(deg: float):
    """Input transform: rotate every wind vector by deg (counter-clockwise),
    leaving the image and everything else as is."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)

    def f(X):
        X = X.copy()
        for cx, cy in WIND:
            i, j = CHANNELS.index(cx), CHANNELS.index(cy)
            u, v = X[i].copy(), X[j].copy()
            X[i], X[j] = c * u - s * v, s * u + c * v
        return X
    return f


def prediction_change(model, data, fold, stats, keep, physics, thr, transform):
    """Over the held-out fold's scored cells: mean |change in probability|,
    and the share of predicted-positive cells (at thr) that change label."""
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    fd = w[w.fold == str(fold)].drop_duplicates(["season", "event", "day"])
    fd = fd[[data.has_weather(int(d)) for d in fd.day]]
    ib = CHANNELS.index("own_burnt")
    dp, n, flips, npos, stores = 0.0, 0, 0, 0, {}
    model.eval()
    with torch.no_grad():
        for (season, event), g in fd.groupby(["season", "event"]):
            ev = data.labels.event.sel(season=season).values
            r, c = np.nonzero(ev == event)
            r0, r1 = max(r.min() - PAD, 0), min(r.max() + PAD + 1, data.H)
            c0, c1 = max(c.min() - PAD, 0), min(c.max() + PAD + 1, data.W)
            if physics and season not in stores:
                stores[season] = load_channel_store(int(season))
            for D in g.day:
                X, y, mask = data.region(int(season), int(event), int(D), r0, r1, c0, c1)
                burnt = X[ib] > 0
                if not burnt.any():
                    continue
                phys = (physics_slice(stores[season], int(event), int(D), r0, c0, r1 - r0, c1 - c0)
                        if physics else None)
                p0 = predict_region(model, X, stats, keep, phys)
                p1 = predict_region(model, transform(X), stats, keep, phys)
                v = mask & ~burnt
                dp += np.abs(p1 - p0)[v].sum(); n += v.sum()
                a, b = p0[v] >= thr, p1[v] >= thr
                flips += (a != b).sum(); npos += a.sum()
    return dp / max(n, 1), flips / max(npos, 1)


def sensitivity(data, name, fold, model, stats, keep, physics, thr_bin, base_row):
    for deg in (90, 180):
        pt, nt = score_fold(model, data, fold, stats, keep, physics, transform=rotate_wind(deg))
        m = metrics_from_counts(pt, nt, thr_bin)
        dp, flips = prediction_change(model, data, fold, stats, keep, physics, thr_bin / 1000,
                                      rotate_wind(deg))
        row = {"model": name, "fold": fold, "rotation_deg": deg, "F1": base_row["heldout_F1"],
               "F1_rotated": m["F1"], "PR_AUC": base_row["heldout_PR_AUC"], "PR_AUC_rotated": m["PR_AUC"],
               "mean_abs_dprob": dp, "pred_pos_changed": flips}
        append(SENS, row)
        log(f"C {name} fold {fold} wind +{deg} deg: F1 {row['F1']:.3f} -> {m['F1']:.3f}, "
            f"mean |dp| {dp:.4f}, {flips:.1%} of predicted cells change")


def summary() -> pd.DataFrame:
    """Mean over seeds of the 5-fold pooled metrics; fold sd of the seed-mean
    F1; seed sd of the fold-mean F1; paired difference vs the plain U-Net."""
    r = pd.read_csv(RESULTS)
    cols = ["F1", "IoU", "precision", "recall", "PR_AUC", "F1 <=6 km", "F1 6-24 km"]
    base = r[r.model == "U-Net"].groupby("fold").heldout_F1.mean()
    rows = []
    for model, g in r.groupby("model", sort=False):
        per_seed = g.groupby("seed")[[f"heldout_{c}" for c in cols]].mean()
        per_fold = g.groupby("fold").heldout_F1.mean()
        diff = per_fold - base.reindex(per_fold.index)
        rows.append({"model": model, "seeds": len(per_seed),
                     **{c: per_seed[f"heldout_{c}"].mean() for c in cols},
                     "F1 fold sd": per_fold.std(),
                     "F1 seed sd": per_seed.heldout_F1.std() if len(per_seed) > 1 else np.nan,
                     "vs U-Net": "" if model == "U-Net" else
                     f"{diff.mean():+.3f} ({int((diff > 0).sum())}/{len(diff)} folds)"})
    for model, g in pd.read_csv(PROCESSED / "baseline_cv.csv").groupby("model"):
        rows.append({"model": model, "seeds": 1, **{c: g[c].mean() for c in cols if c in g},
                     "F1 fold sd": g.F1.std()})
    s = pd.DataFrame(rows).sort_values("F1", ascending=False)
    s.to_csv(SUMMARY, index=False)
    return s


STEPS = {
    "A": ("U-Net", {}),
    "B": ("U-Net + physics", {"physics": True}),
    "D": ("U-Net no weather", {"drop": ("weather",)}),
    "E": ("U-Net no weather + physics", {"drop": ("weather",), "physics": True}),
    "F": ("U-Net wind-aligned", {"align": True}),
}


def wait_for_physics():
    """The physics cache is rebuilt by a separate job; wait until its log says
    it's written (VIGIL_REBUILD_LOG), or just check the file if no log is set."""
    import os
    from pathlib import Path
    from vigil.unet import CACHE
    logf = os.environ.get("VIGIL_REBUILD_LOG")
    while True:
        if logf and "physics channel for" in Path(logf).read_text():
            return
        if not logf and (CACHE / "physics.npy").exists():
            return
        log("waiting for the physics cache ...")
        time.sleep(120)


def main(steps):
    RUNS.mkdir(parents=True, exist_ok=True)
    chosen = {int(k): v for k, v in json.loads(CHOSEN.read_text()).items()}
    data = Data()
    folds = sorted(CV_FOLDS)
    done = done_runs()
    for step in steps:
        if step == "C":
            continue
        name, kw = STEPS[step]
        if kw.get("physics"):
            wait_for_physics()
        for seed in (0, 1, 2):
            for f in folds:
                if (name, f, seed) in done:  # resume: finished runs are skipped
                    continue
                row, model, stats, keep, thr = fit_and_score(data, f, name, step, seed, *chosen[f], **kw)
                if seed == 0 and "C" in steps and step in ("A", "B"):
                    sensitivity(data, name, f, model, stats, keep, kw.get("physics", False), thr, row)
                del model
                gc.collect()
    s = summary()
    log("done\n" + s.round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", nargs="?", default="run", choices=["run", "summary"])
    p.add_argument("--steps", nargs="*", default=list("ABCDEF"))
    a = p.parse_args()
    if a.cmd == "summary":
        print(summary().round(3).to_string(index=False))
    else:
        main(a.steps)
