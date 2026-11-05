"""Overnight U-Net run plan (approved 2026-11-05). CV folds only; test untouched.

  1. Tuning, no physics, seed 0: fire weight {3, 10, 30} x no-burn rate
     {1, 0.5, 0.25}; per held-out fold, the setting with the best inner-
     validation F1 (whole fires) is chosen and only it is scored on the
     held-out fold -> "U-Net (no physics)" seed 0.
  1b. Same per-fold settings, seeds 1 and 2.
  2. Physics channel: wait for it, build its cache, check the encoding
     (1 - t/24 if reached, 0 if not; never the raw 255 code), then seeds 0-2.
  3. Ablations (no physics, seed 0): no weather, no fuel, fire channels only.
Every run appends a row to data/processed/unet_results.csv; a summary table
against the baselines goes to data/processed/unet_summary.csv.

Usage:
  python -m vigil.unet_plan
"""

import gc
import json
import time
import traceback

import numpy as np
import pandas as pd
import torch

from vigil.config import CV_FOLDS, PROCESSED
from vigil.dataset import Data
from vigil.physics import CHANNEL_DIR, UNREACHED
from vigil.unet import (CACHE, GROUPS, best_threshold, metrics_from_counts, physics_cache,
                        score_fold, train)

RESULTS = PROCESSED / "unet_results.csv"
SUMMARY = PROCESSED / "unet_summary.csv"
RUNS = PROCESSED / "unet_runs"
GRID = [(fw, r) for fw in (3.0, 10.0, 30.0) for r in (1.0, 0.5, 0.25)]
ABLATIONS = {"no weather": ["weather"], "no fuel": ["fuel"],
             "fire only": [g for g in GROUPS if g != "fire"]}


def log(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def append(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def fit_and_score(data, fold, model_name, step, seed, fw, rate, physics=False, drop=(), score_heldout=True):
    t0 = time.time()
    model, stats, keep, inner_val, hist = train(fold, physics=physics, drop=drop, fire_weight=fw,
                                                no_burn_rate=rate, seed=seed)
    pv, nv = score_fold(model, data, inner_val, stats, keep, physics)
    thr = best_threshold(pv, nv)
    val = metrics_from_counts(pv, nv, thr)
    row = {"step": step, "model": model_name, "fold": fold, "seed": seed, "fire_weight": fw,
           "no_burn_rate": rate, "physics": physics, "drop": "+".join(drop), "inner_val_fold": inner_val,
           "epochs": len(hist), "threshold": thr / 1000, "inner_val_F1": val["F1"],
           "inner_val_PR_AUC": val["PR_AUC"], "minutes": None}
    tag = f"{model_name.replace(' ', '_')}_f{fold}_s{seed}_w{int(fw)}_r{rate}"
    torch.save(model.state_dict(), RUNS / f"{tag}.pt")
    if score_heldout:
        pt, nt = score_fold(model, data, fold, stats, keep, physics)
        np.savez(RUNS / f"{tag}_counts.npz", pos=pt, neg=nt)
        row.update({f"heldout_{k}": v for k, v in metrics_from_counts(pt, nt, thr).items()})
    row["minutes"] = round((time.time() - t0) / 60, 1)
    del model
    gc.collect()
    return row


def check_physics_encoding() -> str | None:
    """None if the cached physics channel is sensible, else a reason."""
    P = np.load(CACHE / "physics.npy", mmap_mode="r")
    sample = np.asarray(P[:: max(len(P) // 200, 1)], np.float32)
    if not np.isfinite(sample).all():
        return "non-finite values"
    if sample.min() < 0 or sample.max() > 1:
        return f"values outside [0, 1]: {sample.min()}..{sample.max()}"
    if np.isclose(sample, UNREACHED).any() or (sample > 1.0001).any():
        return "raw 'not reached' code present"
    share_zero, share_one = (sample == 0).mean(), (sample >= 0.999).mean()
    if share_zero < 0.05:
        return f"too few unreached (0) cells: {share_zero:.1%}"
    # Spot check: the cached slice must equal 1 - q/240 where reached and 0
    # where the raw store says "not reached" (255).
    from vigil.unet import physics_slice
    w = pd.read_parquet(CACHE / "windows.parquet")
    for _, x in w.sample(5, random_state=0).iterrows():
        store = dict(np.load(CHANNEL_DIR / f"{int(x.season)}.npz"))
        key = f"{int(x.event)}_{int(x.day)}"
        if key not in store:
            return f"missing physics entry {key}"
        q, (o0, o1) = store[key], store[key + "_origin"]
        r0, c0 = int(x.r0) - o0, int(x.c0) - o1
        raw = q[max(r0, 0):r0 + 160, max(c0, 0):c0 + 160]
        exp = np.where(raw == UNREACHED, 0, 1 - raw / 240)
        got = np.asarray(P[x.name], np.float32)[max(-r0, 0):max(-r0, 0) + raw.shape[0],
                                                 max(-c0, 0):max(-c0, 0) + raw.shape[1]]
        if not np.allclose(got, exp, atol=2e-3):
            return f"cached physics slice disagrees with the raw store for {key}"
    log(f"physics encoding OK: range {sample.min():.3f}-{sample.max():.3f}, "
        f"{share_zero:.0%} unreached (0), {share_one:.1%} at start/arrival 0 (1)")
    return None


def summary() -> pd.DataFrame:
    r = pd.read_csv(RESULTS)
    r = r[r.heldout_F1.notna()]
    cols = ["F1", "IoU", "precision", "recall", "PR_AUC", "F1 <=6 km", "F1 6-24 km", "F1 >24 km"]
    rows = []
    for model, g in r.groupby("model"):
        per_seed = g.groupby("seed")[[f"heldout_{c}" for c in cols]].mean()   # mean over folds
        row = {"model": model, "seeds": len(per_seed)}
        for c in cols:
            fold_sd = g[g.seed == g.seed.min()][f"heldout_{c}"].std()
            row[c] = per_seed[f"heldout_{c}"].mean()
            row[f"{c} fold sd"] = fold_sd
            row[f"{c} seed sd"] = per_seed[f"heldout_{c}"].std() if len(per_seed) > 1 else np.nan
        rows.append(row)
    base = pd.concat([pd.read_csv(PROCESSED / "baseline_cv.csv"),
                      pd.read_csv(PROCESSED / "pixel_baselines_cv.csv").query("model == 'logreg'"),
                      pd.read_csv(PROCESSED / "physics_cv.csv")])
    for model, g in base.groupby("model"):
        row = {"model": model, "seeds": 1}
        for c in cols:
            row[c] = g[c].mean()
            row[f"{c} fold sd"] = g[c].std()
        rows.append(row)
    s = pd.DataFrame(rows).sort_values("F1", ascending=False)
    s.to_csv(SUMMARY, index=False)
    return s


def main():
    RUNS.mkdir(exist_ok=True)
    data = Data()
    folds = sorted(CV_FOLDS)
    chosen = {}
    # Step 1: tuning (no physics, seed 0)
    for f in folds:
        best = None
        for fw, rate in GRID:
            try:
                row = fit_and_score(data, f, "U-Net tuning", "1-grid", 0, fw, rate, score_heldout=False)
            except Exception:
                log(f"fold {f} w{fw} r{rate} failed:\n{traceback.format_exc()}")
                continue
            append(row)
            log(f"step 1 fold {f} w{fw} r{rate}: inner-val F1 {row['inner_val_F1']:.3f} ({row['minutes']} min)")
            if best is None or row["inner_val_F1"] > best[0]:
                best = (row["inner_val_F1"], fw, rate)
        chosen[f] = best[1:]
        row = fit_and_score(data, f, "U-Net", "1-chosen", 0, *chosen[f])
        append(row)
        log(f"step 1 fold {f} chosen w{chosen[f][0]} r{chosen[f][1]}: held-out F1 {row['heldout_F1']:.3f}")
    (RUNS / "chosen_settings.json").write_text(json.dumps({str(k): v for k, v in chosen.items()}))
    # Step 1b: extra seeds
    for seed in (1, 2):
        for f in folds:
            row = fit_and_score(data, f, "U-Net", "1b-seeds", seed, *chosen[f])
            append(row)
            log(f"step 1b seed {seed} fold {f}: held-out F1 {row['heldout_F1']:.3f}")
    # Step 2: physics channel
    while len(list(CHANNEL_DIR.glob("*.npz"))) < 13 or any(CHANNEL_DIR.glob("*.tmp.npz")):
        log("waiting for the physics channel ...")
        time.sleep(300)
    physics_cache()
    problem = check_physics_encoding()
    if problem:
        log(f"STEP 2 SKIPPED, physics encoding check failed: {problem}")
    else:
        for seed in (0, 1, 2):
            for f in folds:
                row = fit_and_score(data, f, "U-Net + physics", "2-physics", seed, *chosen[f], physics=True)
                append(row)
                log(f"step 2 seed {seed} fold {f}: held-out F1 {row['heldout_F1']:.3f}")
    # Step 3: ablations
    for name, drop in ABLATIONS.items():
        for f in folds:
            row = fit_and_score(data, f, f"U-Net {name}", "3-ablation", 0, *chosen[f], drop=tuple(drop))
            append(row)
            log(f"step 3 {name} fold {f}: held-out F1 {row['heldout_F1']:.3f}")
    s = summary()
    log("done\n" + s[["model", "seeds", "F1", "F1 fold sd", "F1 seed sd", "PR_AUC"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
