"""Test-set evaluation (season 2020), following the protocol in CLAUDE.md
(approved 2026-11-06). Run once; refuses to run if the results exist.

Models: main (U-Net, no raw weather, + physics), plain U-Net, U-Net + physics
(each trained on all 12 non-test seasons, 3 seeds), wind-shifted persistence
from active fire, physics alone. Every setting comes from CV:
  U-Nets   fire weight / no-burn rate = modal per-fold CV choice (tie -> best
           mean CV F1); epochs = median CV early-stop epoch (epochs run -
           patience; runs that hit the 30-epoch cap count as 30); threshold =
           median CV threshold; normalisation = "all" stats.
  baseline modal CV variant and distance cutoff.
  physics  modal CV ROS scale; predicted = reached within 24 h.
Scoring is CV's: whole fires + 24 km margin, edge mask, already-burnt cells
excluded, far cells kept (>24 km band). Counts are kept per fire, so 95% CIs
and paired model differences come from a bootstrap over test fires.

Usage:
  python -m vigil.test_eval                  # the real, one-off run
  python -m vigil.test_eval --dry-run 0      # code check on CV fold 0 (tiny)
"""

import argparse
import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from vigil.config import PROCESSED
from vigil.dataset import CHANNELS, STATS, Data
from vigil.evaluate import BANDS, BIN, N_BINS, PAD, smear, wind_at
from vigil.static import FUEL_GROUPS
from vigil.unet import (CACHE, DEVICE, PROB_BINS, ResUNet, Windows, channel_index, load_channel_store,
                        masked_bce, physics_slice, predict_region)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "results"
FINAL = OUT / "test_results.csv"
WORK = PROCESSED / "test_run"
PATIENCE, MAX_EPOCHS = 6, 30
BOOT = 2000
SEEDS = (0, 1, 2)
FUEL_CATS = ["forest/woodland", "shrub/heath", "grass/crop", "other"]
_GROUP_CAT = {"fuel_forest": 0, "fuel_plantation": 0, "fuel_woodland": 0, "fuel_shrubland": 1,
              "fuel_grassland": 2, "fuel_crop": 2, "fuel_wetland": 3, "fuel_wui": 3, "fuel_nonburnable": 3}
UNETS = {  # name: (results_v2 model name, physics, drop)
    "Main: U-Net no weather + physics": ("Main (tuned)", True, ("weather",)),
    "U-Net": ("U-Net", False, ()),
    "U-Net + physics": ("U-Net + physics", True, ()),
}
BASELINE = "Wind-shifted persistence (active fire)"
PHYSICS = "Physics alone"
BANDS_N = ["<=6 km", "6-24 km", ">24 km"]


def log(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


# ------------------------------------------------------------------ settings from CV
def modal(values, tiebreak):
    """Most common value; ties -> the one with the highest tiebreak(value)."""
    s = pd.Series([tuple(v) if isinstance(v, list) else v for v in values]).value_counts()
    top = list(s[s == s.max()].index)
    return top[0] if len(top) == 1 else max(top, key=tiebreak)


def stand_in(cv_name, available, dry):
    """Dry runs before tuning finishes: stand in the untuned variant for 'Main (tuned)'."""
    if dry and cv_name == "Main (tuned)" and cv_name not in set(available):
        return "U-Net no weather + physics"
    return cv_name


def settings(dry=False) -> dict:
    res = pd.read_csv(PROCESSED / "unet_results_v2.csv")
    out = {}
    for name, (cv_name, physics, drop) in UNETS.items():
        cv_name = stand_in(cv_name, res.model, dry)
        r = res[res.model == cv_name]
        assert len(r) == 15, f"{cv_name}: {len(r)} CV runs, expected 15"
        if cv_name == "Main (tuned)":
            per_fold = json.loads((PROCESSED / "unet_runs/v2/main_chosen_settings.json").read_text())
            grid = pd.read_csv(PROCESSED / "unet_tuning_v2.csv")
            tb = lambda s: grid[(grid.fire_weight == s[0]) & (grid.no_burn_rate == s[1])].heldout_F1.mean()
        else:
            per_fold = json.loads((PROCESSED / "unet_runs/chosen_settings.json").read_text())
            tb = lambda s: r[(r.fire_weight == s[0]) & (r.no_burn_rate == s[1])].heldout_F1.mean()
        fw, rate = modal(list(per_fold.values()), tb)
        best_ep = np.where(r.epochs < MAX_EPOCHS, r.epochs - PATIENCE, MAX_EPOCHS)
        out[name] = {"cv_model": cv_name, "physics": physics, "drop": list(drop), "fire_weight": fw,
                     "no_burn_rate": rate, "epochs": int(np.median(best_ep)),
                     "threshold_bin": int(round(np.median(r.threshold) * PROB_BINS)),
                     "per_fold_settings": per_fold}
    b = pd.read_csv(PROCESSED / "baseline_cv.csv")
    b = b[b.model == "wind_active"]
    variant = modal(b.variant, lambda v: b[b.variant == v].F1.mean())
    k_km = modal(b.k_km, lambda k: b[b.k_km == k].F1.mean())
    out[BASELINE] = {"variant": variant, "beta": float(variant.split("@")[1]),
                     "k_bin": int(round(k_km / (BIN * 0.375))), "k_km": k_km}
    p = pd.read_csv(PROCESSED / "physics_cv.csv")
    out[PHYSICS] = {"scale": float(modal(p.scale, lambda s: p[p.scale == s].F1.mean()))}
    return out


# ------------------------------------------------------------------ training on all CV seasons
def train_all(physics, drop, fire_weight, no_burn_rate, epochs, seed, max_batches=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cw = pd.read_parquet(CACHE / "windows.parquet")
    stats = json.loads(STATS.read_text())["all"]
    keep = channel_index(list(drop))
    model = ResUNet(len(keep) + int(physics)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx, burn = cw.index, cw.has_new_burn.values
    t0 = time.time()
    for ep in range(epochs):
        pick = idx[burn | (rng.random(len(idx)) < no_burn_rate)]
        dl = torch.utils.data.DataLoader(Windows(pick, stats, keep, physics, augment=True, seed=seed + ep),
                                         batch_size=16, shuffle=True)
        model.train()
        tl, nb = 0.0, 0
        for X, y, m in dl:
            X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            loss = masked_bce(model(X), y, m, fire_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
            if max_batches and nb >= max_batches:
                break
        print(f"  epoch {ep}: train {tl / max(nb, 1):.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)
    model.eval()
    return model, stats, keep


# ------------------------------------------------------------------ scoring
def fuel_category(static: dict) -> np.ndarray:
    names = list(FUEL_GROUPS)
    dom = np.argmax(np.nan_to_num(np.stack([static[k] for k in names])), axis=0)
    return np.array([_GROUP_CAT[n] for n in names], np.int8)[dom]


def fire_days(data, fold: str) -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    fd = w[w.fold == fold].drop_duplicates(["season", "event", "day"])
    return fd[[data.has_weather(int(d)) for d in fd.day]][["season", "event", "day"]].reset_index(drop=True)


def region_box(ev, event, H, W):
    r, c = np.nonzero(ev == event)
    return max(r.min() - PAD, 0), min(r.max() + PAD + 1, H), max(c.min() - PAD, 0), min(c.max() + PAD + 1, W)


def add(counts, e, score_bin, pos, valid, band, fuel):
    """counts = (P, N) of shape (event, band, fuel, bin)."""
    nb = counts[0].shape[-1]
    for k in range(3):
        for f in range(len(FUEL_CATS)):
            sel = valid & (band == k) & (fuel == f)
            if sel.any():
                counts[0][e, k, f] += np.bincount(score_bin[sel & pos], minlength=nb)
                counts[1][e, k, f] += np.bincount(score_bin[sel & ~pos], minlength=nb)


def score_models(data, fd, events, models, cfg, max_days=None):
    """One pass over the fire-days for the U-Nets and the baseline."""
    fuel_all = fuel_category(data.static)
    E = len(events)
    counts = {n: (np.zeros((E, 3, 4, PROB_BINS)), np.zeros((E, 3, 4, PROB_BINS))) for n in models}
    counts[BASELINE] = (np.zeros((E, 3, 4, N_BINS)), np.zeros((E, 3, 4, N_BINS)))
    beta, ib, ia = cfg[BASELINE]["beta"], CHANNELS.index("own_burnt"), CHANNELS.index("active_D")
    stores, done = {}, 0
    with torch.no_grad():
        for (season, event), g in fd.groupby(["season", "event"]):
            e = events.index((int(season), int(event)))
            ev = data.labels.event.sel(season=season).values
            r0, r1, c0, c1 = region_box(ev, event, data.H, data.W)
            if season not in stores:
                stores = {season: load_channel_store(int(season))}
            for D in g.day:
                X, y, mask = data.region(int(season), int(event), int(D), r0, r1, c0, c1)
                burnt = X[ib] > 0
                if not burnt.any():
                    continue
                pos, valid = y > 0, mask & ~burnt
                true_d = ndimage.distance_transform_edt(~burnt)
                band = np.digitize(true_d, BANDS, right=True)
                fuel = fuel_all[r0:r1, c0:c1]
                phys = physics_slice(stores[season], int(event), int(D), r0, c0, r1 - r0, c1 - c0)
                for name, (model, stats, keep, physics) in models.items():
                    prob = predict_region(model, X, stats, keep, phys if physics else None)
                    add(counts[name], e, np.minimum((prob * PROB_BINS).astype(int), PROB_BINS - 1),
                        pos, valid, band, fuel)
                active = X[ia] > 0          # wind-shifted persistence from active fire on D
                br, bc = np.nonzero(burnt)
                u, v = wind_at(data, int(D) + 1, br.mean() + r0, bc.mean() + c0)
                eff = ndimage.distance_transform_edt(~smear(active, u, v, beta)) if active.any() else true_d
                b_eff = np.minimum((eff / BIN).astype(int), N_BINS - 1)
                add(counts[BASELINE], e, N_BINS - 1 - b_eff, pos, valid, band, fuel)
                done += 1
                if max_days and done >= max_days:
                    return counts
            log(f"  scored event {event} ({done} fire-days so far)")
    return counts


_CTX, _FUEL = None, None


def _init_physics():
    global _CTX, _FUEL
    from vigil.physics import Ctx
    _CTX = Ctx()
    _FUEL = fuel_category(_CTX.static)


def _physics_event(task):
    from vigil.physics import N_T, T_BIN, prepare, run
    season, event, days, scale = task
    ev, bd, un = _CTX.season(season)
    r0, r1, c0, c1 = region_box(ev, event, _CTX.H, _CTX.W)
    E, B, U = ev[r0:r1, c0:c1], bd[r0:r1, c0:c1], un[r0:r1, c0:c1]
    own, fuel = E == event, _FUEL[r0:r1, c0:c1]
    P, N = np.zeros((3, 4, N_T)), np.zeros((3, 4, N_T))
    for D in days:
        burnt = own & (B > 0) & (B <= D)
        if not burnt.any():
            continue
        pos = own & (B == D + 1)
        other = (E >= 0) & ~own & (B > 0) & (B <= D + 1)
        valid = ~burnt & ~((U > 0) & (U <= D + 1)) & ~other
        band = np.digitize(ndimage.distance_transform_edt(~burnt), BANDS, right=True)
        arr = run(prepare(_CTX, season, event, D, r0, r1, c0, c1, start="active"), scale)
        b = np.where(np.isfinite(arr), np.minimum((arr / T_BIN).astype(int), N_T - 2), N_T - 1)
        add((P[None], N[None]), 0, N_T - 1 - b, pos, valid, band, fuel)   # score 0 = not reached
    return season, event, P, N


def score_physics(fd, events, scale, workers=3, max_tasks=None):
    from vigil.physics import N_T
    P, N = np.zeros((len(events), 3, 4, N_T)), np.zeros((len(events), 3, 4, N_T))
    tasks = [(int(s), int(e), [int(d) for d in g.day], scale) for (s, e), g in fd.groupby(["season", "event"])]
    tasks.sort(key=lambda t: -len(t[2]))
    tasks = tasks[:max_tasks] if max_tasks else tasks
    with ProcessPoolExecutor(workers, initializer=_init_physics) as pool:
        for season, event, p, n in pool.map(_physics_event, tasks):
            i = events.index((season, event))
            P[i], N[i] = p, n
            log(f"  physics event {event} done")
    return P, N


# ------------------------------------------------------------------ metrics and bootstrap
def metrics(pos, neg, thr):
    """pos/neg: per-bin counts (bin = score, higher = more likely); predicted
    positive iff bin >= thr."""
    p, n = pos[::-1], neg[::-1]
    tp, fp, P = np.cumsum(p), np.cumsum(n), p.sum()
    k = len(p) - 1 - thr
    prec, rec = tp[k] / max(tp[k] + fp[k], 1), tp[k] / max(P, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    pc, rc = tp / np.maximum(tp + fp, 1), tp / max(P, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rc])) * pc))
    return {"F1": f1, "IoU": f1 / (2 - f1), "precision": prec, "recall": rec, "PR_AUC": ap}


def full_metrics(P, N, thr):
    """All metrics for counts (band, fuel, bin) summed over the chosen events."""
    out = metrics(P.sum((0, 1)), N.sum((0, 1)), thr)
    for k, bn in enumerate(BANDS_N):
        m = metrics(P[k].sum(0), N[k].sum(0), thr)
        out[f"F1 {bn}"], out[f"recall {bn}"] = m["F1"], m["recall"]
    for f, fn in enumerate(FUEL_CATS):
        m = metrics(P[:, f].sum(0), N[:, f].sum(0), thr)
        out[f"F1 {fn}"], out[f"precision {fn}"], out[f"recall {fn}"] = m["F1"], m["precision"], m["recall"]
        out[f"positives {fn}"] = P[:, f].sum()
    return out


def model_runs(counts, cfg):
    """name -> list of (P, N, thr) per seed (one entry for non-U-Nets)."""
    runs = {}
    for name in UNETS:
        runs[name] = [(*counts[f"{name}|s{s}"], cfg[name]["threshold_bin"]) for s in SEEDS
                      if f"{name}|s{s}" in counts]
    runs[BASELINE] = [(*counts[BASELINE], N_BINS - 1 - cfg[BASELINE]["k_bin"])]
    if PHYSICS in counts:
        runs[PHYSICS] = [(*counts[PHYSICS], 1)]
    return runs


def seed_mean(runs, idx=None, keys=("F1", "PR_AUC")):
    """Overall metrics averaged over seeds, on the events in idx (all if None)."""
    vals = [metrics((P if idx is None else P[idx]).sum((0, 1, 2)),
                    (N if idx is None else N[idx]).sum((0, 1, 2)), t) for P, N, t in runs]
    return {k: float(np.mean([v[k] for v in vals])) for k in keys}


def bootstrap(runs, n_events, rng):
    idx = [rng.integers(0, n_events, n_events) for _ in range(BOOT)]
    boot = {name: np.array([[seed_mean(r, i)[k] for k in ("F1", "PR_AUC")] for i in idx])
            for name, r in runs.items()}
    return boot


def cv_scores(cfg) -> dict:
    s = pd.read_csv(PROCESSED / "unet_summary_v2.csv").set_index("model")
    missing = [c["cv_model"] for n, c in cfg.items() if n in UNETS and c["cv_model"] not in s.index]
    assert not missing, f"CV summary lacks {missing}: run python -m vigil.unet_plan2 summary"
    out = {n: {"CV F1": s.loc[c["cv_model"], "F1"], "CV PR_AUC": s.loc[c["cv_model"], "PR_AUC"],
               "CV F1 fold sd": s.loc[c["cv_model"], "F1 fold sd"]} for n, c in cfg.items() if n in UNETS}
    b = pd.read_csv(PROCESSED / "baseline_cv.csv")
    b = b[b.model == "wind_active"]
    out[BASELINE] = {"CV F1": b.F1.mean(), "CV PR_AUC": b.PR_AUC.mean(), "CV F1 fold sd": b.F1.std()}
    p = pd.read_csv(PROCESSED / "physics_cv.csv")
    out[PHYSICS] = {"CV F1": p.F1.mean(), "CV PR_AUC": p.PR_AUC.mean(), "CV F1 fold sd": p.F1.std()}
    return out


def report(counts, cfg, events, out_dir, tag=""):
    runs = model_runs(counts, cfg)
    rng = np.random.default_rng(0)
    boot = bootstrap(runs, len(events), rng)
    cv = cv_scores(cfg)
    rows = []
    for name, r in runs.items():
        per_seed = [full_metrics(P.sum(0), N.sum(0), t) for P, N, t in r]
        row = {"model": name, "seeds": len(r), **{k: np.mean([m[k] for m in per_seed]) for k in per_seed[0]}}
        row["F1 seed sd"] = np.std([m["F1"] for m in per_seed], ddof=1) if len(r) > 1 else np.nan
        row["F1 95% CI"] = "{:.3f}-{:.3f}".format(*np.percentile(boot[name][:, 0], [2.5, 97.5]))
        row["PR_AUC 95% CI"] = "{:.3f}-{:.3f}".format(*np.percentile(boot[name][:, 1], [2.5, 97.5]))
        rows.append({**row, **cv.get(name, {})})
    res = pd.DataFrame(rows)
    main = next(iter(UNETS))
    diffs = []
    for other in ("U-Net", BASELINE):
        for j, k in enumerate(("F1", "PR_AUC")):
            d = boot[main][:, j] - boot[other][:, j]
            point = res.set_index("model").loc[main, k] - res.set_index("model").loc[other, k]
            cvd = cv[main][f"CV {k}"] - cv[other][f"CV {k}"]
            diffs.append({"comparison": f"main - {other}", "metric": k, "test difference": point,
                          "95% CI low": np.percentile(d, 2.5), "95% CI high": np.percentile(d, 97.5),
                          "share of resamples > 0": (d > 0).mean(), "CV difference": cvd})
    diffs = pd.DataFrame(diffs)
    by_fire = []
    for i, (season, event) in enumerate(events):
        for name, r in runs.items():
            m = seed_mean(r, [i], keys=("F1", "precision", "recall"))
            by_fire.append({"season": season, "event": event, "model": name,
                            "new_burn_cells": float(r[0][0][i].sum()), **m})
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / f"test_results{tag}.csv", index=False)
    diffs.to_csv(out_dir / f"test_differences{tag}.csv", index=False)
    pd.DataFrame(by_fire).to_csv(out_dir / f"test_by_fire{tag}.csv", index=False)
    show = ["model", "F1", "F1 95% CI", "CV F1", "PR_AUC", "CV PR_AUC", "precision", "recall",
            "F1 <=6 km", "F1 6-24 km", "F1 >24 km"] + [f"F1 {f}" for f in FUEL_CATS[:3]]
    print(res[show].round(3).to_string(index=False))
    print(diffs.round(3).to_string(index=False))
    return res, diffs


# ------------------------------------------------------------------ main
def main(dry_fold=None, skip_physics=False):
    dry = dry_fold is not None
    out_dir = WORK / "dry" if dry else OUT
    if not dry and FINAL.exists():
        raise SystemExit(f"{FINAL} exists: the test set has been evaluated already. Not running again.")
    work = WORK / ("dry" if dry else "real")
    work.mkdir(parents=True, exist_ok=True)
    cfg = settings(dry)
    (work / "settings.json").write_text(json.dumps(cfg, indent=1, default=str))
    log("settings:\n" + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_fold_settings"}
                                    for k, v in cfg.items()}, default=str))
    data = Data()
    fd = fire_days(data, str(dry_fold) if dry else "test")
    if dry:
        fd = fd[fd.event.isin(fd.event.drop_duplicates().iloc[:3])]
    events = [(int(s), int(e)) for s, e in fd[["season", "event"]].drop_duplicates().itertuples(index=False)]
    log(f"{len(fd)} fire-days, {len(events)} fires")
    counts_file = work / "counts.npz"
    counts = dict(np.load(counts_file)) if counts_file.exists() else {}
    counts = {k: (counts[k + "|P"], counts[k + "|N"]) for k in {k.rsplit("|", 1)[0] for k in counts}}

    def save():
        np.savez(counts_file, **{f"{k}|{s}": v[i] for k, v in counts.items() for i, s in enumerate("PN")})

    if f"{next(iter(UNETS))}|s{SEEDS[-1]}" not in counts:
        models = {}
        for name, c in UNETS.items():
            s = cfg[name]
            for seed in SEEDS:
                path = work / f"{name.split(':')[0].replace(' ', '_').replace('+', 'plus')}_s{seed}.pt"
                if path.exists():
                    model = ResUNet(len(channel_index(s["drop"])) + int(s["physics"])).to(DEVICE)
                    model.load_state_dict(torch.load(path, map_location=DEVICE)); model.eval()
                    stats, keep = json.loads(STATS.read_text())["all"], channel_index(s["drop"])
                else:
                    log(f"training {name} seed {seed}: w{s['fire_weight']} r{s['no_burn_rate']}, "
                        f"{s['epochs']} epochs")
                    model, stats, keep = train_all(s["physics"], s["drop"], s["fire_weight"], s["no_burn_rate"],
                                                   1 if dry else s["epochs"], seed, max_batches=3 if dry else None)
                    torch.save(model.state_dict(), path)
                models[f"{name}|s{seed}"] = (model, stats, keep, s["physics"])
        log("scoring U-Nets and baseline on whole fires")
        counts.update(score_models(data, fd, events, models, cfg, max_days=6 if dry else None))
        save()
        del models
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    if not skip_physics and PHYSICS not in counts:
        del data
        gc.collect()
        log(f"physics alone at ROS scale {cfg[PHYSICS]['scale']} (CPU, after all GPU work)")
        counts[PHYSICS] = score_physics(fd, events, cfg[PHYSICS]["scale"], max_tasks=1 if dry else None)
        save()
    report(counts, cfg, events, out_dir, tag="_DRY_RUN" if dry else "")
    log("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", type=int, metavar="FOLD")
    p.add_argument("--skip-physics", action="store_true")
    a = p.parse_args()
    main(a.dry_run, a.skip_physics)
