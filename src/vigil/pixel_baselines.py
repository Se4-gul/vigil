"""Huot et al. 2022 pixel baselines: logistic regression and random forest.

Features: each pixel's 32 input channels at itself and its 8 neighbours (3 x 3,
288 features), as in Huot et al. Scored with the same season CV, whole-fire
areas, masks and distance bands as evaluate.py.

Sampling (whole fires are ~60 M cells): every new-burn cell is kept; cells
that don't newly burn are sampled at NEG_RATE and weighted 1/NEG_RATE, which
keeps F1 and PR-AUC unbiased. Training uses fire and no-fire balanced 1:1
(Huot et al.), excluding new burn > 24 km from the fire (as for training
elsewhere). Deviation from Huot: no extra no-fire class weight, since the
decision threshold is tuned anyway.

Tuning, per held-out fold: on the 4 training folds, inner out-of-fold
predictions (train on 3, predict the 4th) choose the RF depth and the
threshold that maximise F1; then refit on all 4 and score the held-out fold.

Usage:
  python -m vigil.pixel_baselines prepare    # build the sampled feature pool
  python -m vigil.pixel_baselines run
  python -m vigil.pixel_baselines rf_check   # shallow-RF overfitting check
"""

import argparse

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vigil.config import PROCESSED
from vigil.dataset import CHANNELS, Data
from vigil.evaluate import BAND_NAMES, BANDS, PAD, fire_days

NEG_RATE = 0.02
FAR_CELLS = 64
POOL = PROCESSED / "pixel_pool.npz"
RF_DEPTHS = [10, 15, 20]


def neighbourhood(X: np.ndarray) -> np.ndarray:
    """(C, h, w) -> (9C, h, w): each channel at the 3 x 3 offsets (edge-padded)."""
    P = np.pad(X, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = X.shape[1:]
    return np.concatenate([P[:, 1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                           for dr in (-1, 0, 1) for dc in (-1, 0, 1)])


def prepare(seed: int = 0) -> None:
    data = Data()
    rng = np.random.default_rng(seed)
    fd = fire_days(data)
    feats, label, weight, fold, band, far = [], [], [], [], [], []
    ib = CHANNELS.index("own_burnt")
    for season, fs in fd.groupby("season"):
        ev = data.labels.event.sel(season=season).values
        for e, fe in fs.groupby("event"):
            r, c = np.nonzero(ev == e)
            r0, r1 = max(r.min() - PAD, 0), min(r.max() + PAD + 1, data.H)
            c0, c1 = max(c.min() - PAD, 0), min(c.max() + PAD + 1, data.W)
            for _, row in fe.iterrows():
                X, y, mask = data.region(int(season), int(e), int(row.day), r0, r1, c0, c1)
                X = np.nan_to_num(X)     # missing outside the study box (always masked there)
                burnt = X[ib] > 0
                if not burnt.any():
                    continue
                valid = mask & ~burnt
                dist = ndimage.distance_transform_edt(~burnt)
                pos = valid & (y > 0)
                neg = valid & (y == 0) & (rng.random(y.shape) < NEG_RATE)
                keep = pos | neg
                F = neighbourhood(X)[:, keep].T.astype(np.float32)
                feats.append(F)
                label.append(pos[keep])
                weight.append(np.where(pos[keep], 1.0, 1 / NEG_RATE).astype(np.float32))
                fold.append(np.full(keep.sum(), int(row.fold), np.int8))
                band.append(np.digitize(dist[keep], BANDS, right=True).astype(np.int8))
                far.append(dist[keep] > FAR_CELLS)
        print(f"  season {season}: {sum(len(l) for l in label):,} rows so far", flush=True)
    tmp = POOL.with_name(POOL.stem + ".tmp.npz")
    np.savez(tmp, X=np.concatenate(feats), y=np.concatenate(label), w=np.concatenate(weight),
             fold=np.concatenate(fold), band=np.concatenate(band), far=np.concatenate(far))
    tmp.rename(POOL)
    print(f"-> {POOL}")


def balanced(idx: np.ndarray, y: np.ndarray, far: np.ndarray, rng) -> np.ndarray:
    """Training rows: positives (not far) and an equal number of negatives."""
    pos = idx[y[idx] & ~far[idx]]
    neg = idx[~y[idx]]
    return np.concatenate([pos, rng.choice(neg, size=min(len(pos), len(neg)), replace=False)])


def make(model: str, depth: int | None):
    if model == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return RandomForestClassifier(n_estimators=100, max_depth=depth, max_features="sqrt",
                                  n_jobs=-1, random_state=0)


def weighted_curve(score, y, w):
    o = np.argsort(-score, kind="stable")
    tp = np.cumsum(w[o] * y[o])
    fp = np.cumsum(w[o] * ~y[o])
    return score[o], tp, fp, (w * y).sum()


def best_threshold(score, y, w):
    s, tp, fp, P = weighted_curve(score, y, w)
    f1 = 2 * tp / np.maximum(tp + fp + P, 1e-9)
    i = int(np.argmax(f1))
    return s[i], f1[i]


def metrics(score, y, w, thr):
    pred = score >= thr
    tp, fp = (w * (pred & y)).sum(), (w * (pred & ~y)).sum()
    P = (w * y).sum()
    prec, rec = tp / max(tp + fp, 1e-9), tp / max(P, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    _, ctp, cfp, _ = weighted_curve(score, y, w)
    p_curve = ctp / np.maximum(ctp + cfp, 1e-9)
    r_curve = ctp / max(P, 1e-9)
    ap = float(np.sum(np.diff(np.concatenate([[0], r_curve])) * p_curve))
    return f1, prec, rec, ap


def run(seed: int = 0) -> None:
    z = np.load(POOL)
    X, y, w, fold, band, far = z["X"], z["y"], z["w"], z["fold"], z["band"], z["far"]
    rng = np.random.default_rng(seed)
    rows = []
    for model, depths in [("logreg", [None]), ("random_forest", RF_DEPTHS)]:
        for f in range(5):
            train_folds = [g for g in range(5) if g != f]
            tr = np.nonzero(np.isin(fold, train_folds))[0]
            # Inner out-of-fold predictions on the training folds.
            best = None
            for depth in depths:
                oof = np.zeros(len(tr), np.float32)
                for g in train_folds:
                    inner = tr[fold[tr] != g]
                    fit_idx = balanced(inner, y, far, rng)
                    m = make(model, depth).fit(X[fit_idx], y[fit_idx])
                    sel = fold[tr] == g
                    oof[sel] = m.predict_proba(X[tr[sel]])[:, 1]
                thr, f1 = best_threshold(oof, y[tr], w[tr])
                if best is None or f1 > best[1]:
                    best = (depth, f1, thr)
            depth, _, thr = best
            fit_idx = balanced(tr, y, far, rng)
            m = make(model, depth).fit(X[fit_idx], y[fit_idx])
            te = np.nonzero(fold == f)[0]
            s = m.predict_proba(X[te])[:, 1]
            f1, prec, rec, ap = metrics(s, y[te], w[te], thr)
            row = {"model": model, "fold": f, "depth": depth, "F1": f1, "IoU": f1 / (2 - f1),
                   "precision": prec, "recall": rec, "PR_AUC": ap}
            for b, name in enumerate(BAND_NAMES):
                sb = band[te] == b
                fb, _, rb, _ = metrics(s[sb], y[te][sb], w[te][sb], thr)
                row[f"F1 {name}"], row[f"recall {name}"] = fb, rb
            rows.append(row)
            print(f"  {model} fold {f}: depth {depth}, F1 {f1:.3f}, PR-AUC {ap:.3f}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(PROCESSED / "pixel_baselines_cv.csv", index=False)
    cols = ["F1", "IoU", "precision", "recall", "PR_AUC"] + [f"F1 {b}" for b in BAND_NAMES] \
           + [f"recall {b}" for b in BAND_NAMES]
    print(r.groupby("model")[cols].agg(["mean", "std"]).round(3).T.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("run")
    sub.add_parser("rf_check")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "rf_check": rf_check}[args.cmd]()



def rf_check(seed: int = 0) -> None:
    """Timeboxed overfitting check: shallow RF grid (depth x min leaf), with
    train / inner out-of-fold / held-out F1 per fold."""
    z = np.load(POOL)
    X, y, w, fold, far = z["X"], z["y"], z["w"], z["fold"], z["far"]
    rng = np.random.default_rng(seed)
    grid_ = [(d, leaf) for d in (4, 8, 12) for leaf in (50, 200)]
    rows = []
    for f in range(5):
        train_folds = [g for g in range(5) if g != f]
        tr = np.nonzero(np.isin(fold, train_folds))[0]
        best = None
        for depth, leaf in grid_:
            oof = np.zeros(len(tr), np.float32)
            for g in train_folds:
                fit_idx = balanced(tr[fold[tr] != g], y, far, rng)
                m = RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
                                           max_features="sqrt", n_jobs=-1, random_state=0)
                m.fit(X[fit_idx], y[fit_idx])
                sel = fold[tr] == g
                oof[sel] = m.predict_proba(X[tr[sel]])[:, 1]
            thr, f1_oof = best_threshold(oof, y[tr], w[tr])
            if best is None or f1_oof > best[2]:
                best = (depth, leaf, f1_oof, thr)
        depth, leaf, f1_oof, thr = best
        fit_idx = balanced(tr, y, far, rng)
        m = RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
                                   max_features="sqrt", n_jobs=-1, random_state=0).fit(X[fit_idx], y[fit_idx])
        f1_train = metrics(m.predict_proba(X[tr])[:, 1], y[tr], w[tr], thr)[0]
        te = np.nonzero(fold == f)[0]
        s = m.predict_proba(X[te])[:, 1]
        f1_te, _, _, ap = metrics(s, y[te], w[te], thr)
        rows.append({"fold": f, "depth": depth, "min_leaf": leaf, "F1_train": f1_train,
                     "F1_inner_oof": f1_oof, "F1_heldout": f1_te, "PR_AUC_heldout": ap})
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(PROCESSED / "rf_check.csv", index=False)
    print(r.round(3).to_string(index=False))
    print(r[["F1_train", "F1_inner_oof", "F1_heldout", "PR_AUC_heldout"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
