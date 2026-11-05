"""Shared evaluation: season cross-validation scored on whole fires.

Unit: each fire-day transition D -> D+1 in the window table (fire active on D,
D and D+1 in scope with weather), scored over the fire's whole extent plus a
64-cell (24 km) margin, not a window.
Scored cells: everything in that area except cells already burnt by this fire
(they can't newly burn) and the loss mask (uncertain cells, other fires
burning by D+1). New burn far from the fire stays in and is scored in its
distance band.
Positives: this fire's cells newly burnt on D+1.
Bands: distance from the fire's burnt area on D: <=6 km, 6-24 km, >24 km.

Main metric: F1 on newly burnt cells, pooled over all fire-days in the
held-out fold, with the threshold tuned on the other folds. Also PR-AUC
(threshold-free), and F1/recall per band. Reported as mean +- sd over the 5
folds; test (2020) is not touched here.

Baselines rank cells by an effective distance from the fire (smaller = more
likely to burn), so scores are accumulated as counts of positives and
negatives per distance bin, which gives exact F1 at any threshold and PR-AUC.
  persistence         distance from the burnt area on D; predict new burn within k
  persistence_active  distance from active fire on D (cells burning on D)
  wind / wind_active  the same, after smearing the area downwind by beta cells
                      per m/s of D+1 peak wind (at the fire's centroid)

Usage:
  python -m vigil.evaluate baselines
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.ndimage import map_coordinates

from vigil.config import CV_FOLDS, PROCESSED
from vigil.dataset import Data

PAD = 64
BANDS = [16, 64]                      # cells: 6 km, 24 km
BAND_NAMES = ["<=6 km", "6-24 km", ">24 km"]
BIN = 0.5                             # cells
MAX_BIN = 200                         # cells (75 km); beyond goes in the last bin
N_BINS = int(MAX_BIN / BIN) + 1
BETAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]   # wind smear, cells per m/s


def fire_days(data: Data) -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    w = w[w.fold != "test"]
    fd = w.drop_duplicates(["season", "event", "day"])[["season", "event", "day", "fold"]]
    return fd[[data.has_weather(int(d)) for d in fd.day]].reset_index(drop=True)


def wind_at(data: Data, day: int, r: float, c: float) -> tuple[float, float]:
    """D+1 peak wind (grid axes) at a grid cell."""
    i = data.weather_days[day]
    rr, cc = data.wrow[int(r), int(c)], data.wcol[int(r), int(c)]
    u = map_coordinates(data.wx["u_peak"][i], [[rr], [cc]], order=1)[0]
    v = map_coordinates(data.wx["v_peak"][i], [[rr], [cc]], order=1)[0]
    g = data.gamma[int(r), int(c)]
    return u * np.cos(g) + v * np.sin(g), -u * np.sin(g) + v * np.cos(g)


def smear(burnt: np.ndarray, u: float, v: float, beta: float) -> np.ndarray:
    """Burnt area shifted downwind step by step, up to beta * wind speed cells."""
    speed = np.hypot(u, v)
    steps = int(round(beta * speed))
    if steps == 0 or speed == 0:
        return burnt
    du, dv = u / speed, v / speed           # unit downwind; rows run south
    out = burnt.copy()
    for t in range(1, steps + 1):
        out |= ndimage.shift(burnt, (-dv * t, du * t), order=0, cval=False)
    return out


def accumulate(data: Data, models: dict) -> dict:
    """counts[model] = (pos, neg) arrays of shape (fold, band, bin)."""
    fd = fire_days(data)
    folds = sorted(CV_FOLDS)
    counts = {m: (np.zeros((len(folds), 3, N_BINS)), np.zeros((len(folds), 3, N_BINS)))
              for m in models}
    L = data.labels
    for season, fs in fd.groupby("season"):
        ev = L.event.sel(season=season).values
        bd = L.burn_day.sel(season=season).values
        un = L.uncertain_day.sel(season=season).values
        H, W = ev.shape
        for e, fe in fs.groupby("event"):
            r, c = np.nonzero(ev == e)
            r0, r1 = max(r.min() - PAD, 0), min(r.max() + PAD + 1, H)
            c0, c1 = max(c.min() - PAD, 0), min(c.max() + PAD + 1, W)
            E, B, U = ev[r0:r1, c0:c1], bd[r0:r1, c0:c1], un[r0:r1, c0:c1]
            own = E == e
            for _, row in fe.iterrows():
                D, f = int(row.day), int(row.fold)
                burnt = own & (B > 0) & (B <= D)
                if not burnt.any():
                    continue
                pos = own & (B == D + 1)
                other = (E >= 0) & ~own & (B > 0) & (B <= D + 1)
                valid = ~burnt & ~((U > 0) & (U <= D + 1)) & ~other
                true_d = ndimage.distance_transform_edt(~burnt)
                band = np.digitize(true_d, BANDS, right=True)
                active = own & (B == D)
                br, bc = np.nonzero(burnt)
                u, v = wind_at(data, D + 1, br.mean() + r0, bc.mean() + c0)
                for name, fn in models.items():
                    eff = fn(burnt, active, true_d, u, v)
                    b = np.minimum((eff / BIN).astype(int), N_BINS - 1)
                    for k in range(3):
                        sel = valid & (band == k)
                        counts[name][0][f, k] += np.bincount(b[sel & pos], minlength=N_BINS)
                        counts[name][1][f, k] += np.bincount(b[sel & ~pos], minlength=N_BINS)
        print(f"  season {season} done", flush=True)
    return counts


def curve(pos: np.ndarray, neg: np.ndarray):
    """Cumulative TP/FP for thresholds 'eff <= bin'; pos/neg are per-bin counts."""
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    return tp, fp, pos.sum()


def f1_at(pos, neg, k):
    tp, fp, P = curve(pos, neg)
    t, f = tp[k], fp[k]
    prec = t / max(t + f, 1)
    rec = t / max(P, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec


def pr_auc(pos, neg):
    tp, fp, P = curve(pos, neg)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(P, 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def report(counts: dict) -> pd.DataFrame:
    """Held-out fold scores, with the threshold (and model variant) tuned on the
    other folds."""
    folds = range(len(CV_FOLDS))
    families = {}
    for name in counts:
        families.setdefault(name.split("@")[0], []).append(name)
    rows = []
    for fam, variants in families.items():
        for f in folds:
            others = [g for g in folds if g != f]
            best = None
            for name in variants:
                pos, neg = (counts[name][i][others].sum(axis=(0, 1)) for i in (0, 1))
                tp, fp, P = curve(pos, neg)
                f1s = 2 * tp / np.maximum(tp + fp + P, 1)
                k = int(np.argmax(f1s))
                if best is None or f1s[k] > best[0]:
                    best = (f1s[k], name, k)
            _, name, k = best
            pos_b, neg_b = counts[name][0][f], counts[name][1][f]
            pos, neg = pos_b.sum(0), neg_b.sum(0)
            f1, prec, rec = f1_at(pos, neg, k)
            row = {"model": fam, "fold": f, "variant": name, "k_km": round(k * BIN * 0.375, 2),
                   "F1": f1, "IoU": f1 / (2 - f1) if f1 < 2 else 0, "precision": prec,
                   "recall": rec, "PR_AUC": pr_auc(pos, neg)}
            for j, bn in enumerate(BAND_NAMES):
                f1b, _, recb = f1_at(pos_b[j], neg_b[j], k)
                row[f"F1 {bn}"] = f1b
                row[f"recall {bn}"] = recb
            rows.append(row)
    return pd.DataFrame(rows)


def baselines() -> None:
    data = Data()
    edt = ndimage.distance_transform_edt
    models = {"persistence": lambda burnt, active, d, u, v: d,
              "persistence_active": lambda burnt, active, d, u, v: edt(~active) if active.any() else d}
    for beta in BETAS:
        models[f"wind@{beta}"] = (lambda b_: lambda burnt, active, d, u, v:
                                  edt(~smear(burnt, u, v, b_)))(beta)
        models[f"wind_active@{beta}"] = (lambda b_: lambda burnt, active, d, u, v:
                                         edt(~smear(active, u, v, b_)) if active.any() else d)(beta)
    counts = accumulate(data, models)
    np.savez(PROCESSED / "baseline_counts.npz",
             **{f"{m}|{i}": counts[m][i] for m in counts for i in (0, 1)})
    r = report(counts)
    r.to_csv(PROCESSED / "baseline_cv.csv", index=False)
    cols = ["F1", "IoU", "precision", "recall", "PR_AUC"] + [f"F1 {b}" for b in BAND_NAMES] \
           + [f"recall {b}" for b in BAND_NAMES]
    summary = r.groupby("model")[cols].agg(["mean", "std"]).round(3)
    print("\nPer fold:")
    print(r[["model", "fold", "variant", "k_km", "F1", "PR_AUC"]].round(3).to_string(index=False))
    print("\nMean and sd over 5 folds:")
    print(summary.T.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baselines")
    args = parser.parse_args()
    if args.cmd == "baselines":
        baselines()


if __name__ == "__main__":
    main()
