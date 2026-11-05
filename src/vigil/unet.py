"""U-Net for next-day spread (Huot et al. 2022 style), optional physics channel.

Model: residual U-Net, 3 down/up levels (128 -> 16 cells), base filters 16-32,
dropout 0.1, one logit per cell. Fully convolutional, so it trains on 128 x 128
windows and predicts whole fires at evaluation.
Loss: weighted binary cross-entropy (pos_weight = fire weight) over the loss
mask only (uncertain cells, other fires and far new ignitions excluded).
Hyperparameters tuned in the season CV: fire weight, no-burn sampling rate.

Inputs: the 32 channels of dataset.py, minus ablated groups, plus the physics
arrival-time channel if --physics (1 - arrival/24 for cells reached within
24 h, 0 otherwise; scale 1.0, precomputed by physics.py).

Training data: windows cached once at 160 x 160 (window + 16-cell margin) in
float16 memmaps, so jitter is a random 128 x 128 crop and rotations/flips are
applied on the fly (vector channels rotated consistently).

Per held-out fold: train on 3 of the other folds, early-stop and pick the
decision threshold on the 4th (inner validation), then score the held-out fold
on whole fires exactly as evaluate.py does (scored cells, distance bands).

Usage:
  python -m vigil.unet cache               # build the window cache (once)
  python -m vigil.unet physics_cache       # add the physics channel to it
  python -m vigil.unet smoke               # a few batches + one whole-fire inference
  python -m vigil.unet train --fold 0 [--physics] [--drop weather fuel] ...
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from vigil.config import CV_FOLDS, PROCESSED
from vigil.dataset import CHANNELS, DRYNESS, STATS, VECTORS, WEATHER, Data, normalise, rotate
from vigil.evaluate import BANDS, PAD
from vigil.static import FUEL_GROUPS, SLOPE_THRESHOLDS
from vigil.windows import SIZE

CACHE = PROCESSED / "window_cache"
MARGIN = 16
BIG = SIZE + 2 * MARGIN
FAR_CELLS = 64
PROB_BINS = 1000
GROUPS = {
    "fire": ["active_D", "own_burnt"],
    "history": ["tslb"],
    "terrain": ["elevation", "grad_x", "grad_y"] + [f"slope_gt{t}" for t in SLOPE_THRESHOLDS],
    "fuel": list(FUEL_GROUPS),
    "weather": [f"{v}_D" for v in DRYNESS] + [f"{v}_{d}" for d in ("D", "D1") for v in WEATHER]
               + ["ffdi_D", "ffdi_D1"],
}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------------------------------------------ cache
def _far_mask(ev, bd, e, D, r0, c0, H, W, size):
    """Cells of a size x size region more than 24 km from fire e's burnt area on D."""
    pad = FAR_CELLS + 1
    R0, C0 = max(r0 - pad, 0), max(c0 - pad, 0)
    R1, C1 = min(r0 + size + pad, H), min(c0 + size + pad, W)
    burnt = (ev[R0:R1, C0:C1] == e) & (bd[R0:R1, C0:C1] > 0) & (bd[R0:R1, C0:C1] <= D)
    if not burnt.any():
        return np.ones((size, size), bool)
    d = distance_transform_edt(~burnt)
    return d[r0 - R0:r0 - R0 + size, c0 - C0:c0 - C0 + size] > FAR_CELLS


def cache() -> None:
    data = Data()
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    w = w[(w.fold != "test") & np.array([data.has_weather(int(d)) for d in w.day])].reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    n = len(w)
    X = np.lib.format.open_memmap(CACHE / "X.npy", "w+", np.float16, (n, len(CHANNELS), BIG, BIG))
    Y = np.lib.format.open_memmap(CACHE / "y.npy", "w+", np.uint8, (n, BIG, BIG))
    M = np.lib.format.open_memmap(CACHE / "mask.npy", "w+", np.uint8, (n, BIG, BIG))
    origins = []
    for i, x in w.iterrows():
        r0 = int(np.clip(x.centre_row - BIG // 2, 0, data.H - BIG))
        c0 = int(np.clip(x.centre_col - BIG // 2, 0, data.W - BIG))
        Xi, yi, mi = data.region(int(x.season), int(x.event), int(x.day), r0, r0 + BIG, c0, c0 + BIG)
        ev, bd = (data.labels.sel(season=int(x.season))[v].values for v in ("event", "burn_day"))
        far = _far_mask(ev, bd, int(x.event), int(x.day), r0, c0, data.H, data.W, BIG)
        mi &= ~((yi > 0) & far)
        X[i], Y[i], M[i] = Xi.astype(np.float16), yi.astype(np.uint8), mi.astype(np.uint8)
        origins.append((r0, c0))
        if i % 200 == 0:
            print(f"  {i}/{n}", flush=True)
    w["r0"], w["c0"] = zip(*origins)
    w.to_parquet(CACHE / "windows.parquet", index=False)
    X.flush(); Y.flush(); M.flush()
    print(f"{n} windows -> {CACHE} ({X.nbytes / 1e9:.1f} GB)")


def load_channel_store(season: int) -> dict:
    from vigil.physics import CHANNEL_DIR
    return dict(np.load(CHANNEL_DIR / f"{season}.npz"))


def physics_slice(store: dict, event: int, D: int, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """Physics feature for a grid rectangle: 1 - arrival/24 if reached, else 0."""
    from vigil.physics import UNREACHED
    out = np.zeros((h, w), np.float32)
    key = f"{event}_{D}"
    if key not in store:
        return out
    q = store[key]
    o0, o1 = store[key + "_origin"]
    a0, b0 = max(r0, o0), max(c0, o1)
    a1, b1 = min(r0 + h, o0 + q.shape[0]), min(c0 + w, o1 + q.shape[1])
    if a1 <= a0 or b1 <= b0:
        return out
    sub = q[a0 - o0:a1 - o0, b0 - o1:b1 - o1].astype(np.float32)
    out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = np.where(sub == UNREACHED, 0, 1 - sub / 240)
    return out


def physics_cache() -> None:
    w = pd.read_parquet(CACHE / "windows.parquet")
    P = np.lib.format.open_memmap(CACHE / "physics.npy", "w+", np.float16, (len(w), BIG, BIG))
    for season, ws in w.groupby("season"):
        store = load_channel_store(int(season))
        for i, x in ws.iterrows():
            P[i] = physics_slice(store, int(x.event), int(x.day), int(x.r0), int(x.c0), BIG, BIG)
    P.flush()
    print(f"physics channel for {len(w)} windows -> {CACHE / 'physics.npy'}")


# ------------------------------------------------------------------ data
def channel_index(drop: list[str]) -> list[int]:
    dropped = {c for g in drop for c in GROUPS[g]}
    return [i for i, c in enumerate(CHANNELS) if c not in dropped]


class Windows(torch.utils.data.Dataset):
    def __init__(self, idx, stats, keep, physics: bool, augment: bool, seed=0):
        self.idx, self.keep, self.physics, self.augment = list(idx), keep, physics, augment
        self.X = np.load(CACHE / "X.npy", mmap_mode="r")
        self.Y = np.load(CACHE / "y.npy", mmap_mode="r")
        self.M = np.load(CACHE / "mask.npy", mmap_mode="r")
        self.P = np.load(CACHE / "physics.npy", mmap_mode="r") if physics else None
        self.stats = stats
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        dr, dc = self.rng.integers(0, 2 * MARGIN + 1, 2) if self.augment else (MARGIN, MARGIN)
        sl = (slice(dr, dr + SIZE), slice(dc, dc + SIZE))
        X = np.asarray(self.X[i][:, sl[0], sl[1]], np.float32)
        y = np.asarray(self.Y[i][sl], np.float32)
        m = np.asarray(self.M[i][sl], np.float32)
        if self.physics:     # append as a scalar channel so it rotates with the rest
            X = np.concatenate([X, np.asarray(self.P[i][sl], np.float32)[None]])
        if self.augment:
            X, y, m = rotate(X, y, m, int(self.rng.integers(4)), bool(self.rng.integers(2)))
        base = normalise(X[:len(CHANNELS)], self.stats)[self.keep]
        X = np.concatenate([base, X[len(CHANNELS):]]) if self.physics else base
        return torch.from_numpy(X.copy()), torch.from_numpy(y.copy()), torch.from_numpy(m.copy())


# ------------------------------------------------------------------ model
class ResBlock(nn.Module):
    def __init__(self, cin, cout, dropout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.drop(h)
        h = self.b2(self.c2(h))
        return F.relu(h + self.skip(x))


class ResUNet(nn.Module):
    def __init__(self, cin, base=16, levels=3, dropout=0.1):
        super().__init__()
        ch = [base * 2 ** i for i in range(levels + 1)]
        self.inc = ResBlock(cin, ch[0], dropout)
        self.down = nn.ModuleList([ResBlock(ch[i], ch[i + 1], dropout) for i in range(levels)])
        self.up = nn.ModuleList([ResBlock(ch[i + 1] + ch[i], ch[i], dropout) for i in reversed(range(levels))])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for u in self.up:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = u(torch.cat([h, s], 1))
        return self.out(h)[:, 0]


def masked_bce(logits, y, m, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                              pos_weight=torch.tensor(pos_weight, device=logits.device))
    return (loss * m).sum() / m.sum().clamp(min=1)


# ------------------------------------------------------------------ whole-fire scoring
def score_fold(model, data: Data, fold: int | str, stats, keep, physics: bool):
    """Counts of positives/negatives per (band, probability bin) on whole fires."""
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    w = w[w.fold == str(fold)]
    fd = w.drop_duplicates(["season", "event", "day"])
    fd = fd[[data.has_weather(int(d)) for d in fd.day]]
    pos = np.zeros((3, PROB_BINS))
    neg = np.zeros((3, PROB_BINS))
    model.eval()
    stores = {}
    ib = CHANNELS.index("own_burnt")
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
                Xn = normalise(X, stats)[keep]
                if physics:
                    Xn = np.concatenate([Xn, physics_slice(stores[season], int(event), int(D),
                                                           r0, c0, r1 - r0, c1 - c0)[None]])
                h, wd = Xn.shape[1:]
                ph, pw = (-h) % 8, (-wd) % 8
                Xt = torch.from_numpy(np.pad(Xn, ((0, 0), (0, ph), (0, pw)), mode="edge")).to(DEVICE)
                prob = torch.sigmoid(model(Xt[None]))[0, :h, :wd].float().cpu().numpy()
                valid = mask & ~burnt
                band = np.digitize(distance_transform_edt(~burnt), BANDS, right=True)
                b = np.minimum((prob * PROB_BINS).astype(int), PROB_BINS - 1)
                for k in range(3):
                    sel = valid & (band == k)
                    pos[k] += np.bincount(b[sel & (y > 0)], minlength=PROB_BINS)
                    neg[k] += np.bincount(b[sel & (y == 0)], minlength=PROB_BINS)
    return pos, neg


def metrics_from_counts(pos, neg, thr_bin):
    """F1 etc. for 'probability bin >= thr_bin', plus PR-AUC; pos/neg (band, bin)."""
    p, n = pos.sum(0)[::-1], neg.sum(0)[::-1]        # high probability first
    tp, fp, P = np.cumsum(p), np.cumsum(n), p.sum()
    k = PROB_BINS - 1 - thr_bin
    prec, rec = tp[k] / max(tp[k] + fp[k], 1), tp[k] / max(P, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    pc, rc = tp / np.maximum(tp + fp, 1), tp / max(P, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rc])) * pc))
    out = {"F1": f1, "IoU": f1 / (2 - f1), "precision": prec, "recall": rec, "PR_AUC": ap}
    for j, name in enumerate(["<=6 km", "6-24 km", ">24 km"]):
        pb, nb = pos[j][::-1], neg[j][::-1]
        t, f_, Pb = pb[:k + 1].sum(), nb[:k + 1].sum(), pb.sum()
        pr, rc_ = t / max(t + f_, 1), t / max(Pb, 1)
        out[f"F1 {name}"] = 2 * pr * rc_ / max(pr + rc_, 1e-12)
        out[f"recall {name}"] = rc_
    return out


def best_threshold(pos, neg):
    p, n = pos.sum(0)[::-1], neg.sum(0)[::-1]
    tp, fp, P = np.cumsum(p), np.cumsum(n), p.sum()
    f1 = 2 * tp / np.maximum(tp + fp + P, 1)
    return PROB_BINS - 1 - int(np.argmax(f1))


# ------------------------------------------------------------------ training
def train(fold: int, physics=False, drop=(), base=16, fire_weight=10.0, no_burn_rate=0.5,
          epochs=30, batch=16, lr=1e-3, patience=6, seed=0, max_batches=None, tag=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cw = pd.read_parquet(CACHE / "windows.parquet")
    train_folds = [f for f in CV_FOLDS if f != fold]
    inner_val = train_folds[(fold + 1) % len(train_folds)]       # fixed, documented choice
    fit_folds = [f for f in train_folds if f != inner_val]
    stats = json.loads(STATS.read_text())[f"fold_{fold}"]
    keep = channel_index(list(drop))
    cin = len(keep) + (1 if physics else 0)
    model = ResUNet(cin, base=base).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fit_idx = cw.index[cw.fold.isin([str(f) for f in fit_folds])]
    val_idx = cw.index[cw.fold == str(inner_val)]
    val = torch.utils.data.DataLoader(Windows(val_idx, stats, keep, physics, augment=False), batch_size=batch)
    best, best_state, bad, log = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        burn = cw.loc[fit_idx].has_new_burn.values
        pick = fit_idx[burn | (rng.random(len(fit_idx)) < no_burn_rate)]
        dl = torch.utils.data.DataLoader(Windows(pick, stats, keep, physics, augment=True, seed=seed + ep),
                                         batch_size=batch, shuffle=True)
        model.train()
        tl, nb = 0.0, 0
        for X, y, m in dl:
            X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            loss = masked_bce(model(X), y, m, fire_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
            if max_batches and nb >= max_batches:
                break
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for X, y, m in val:
                X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                vl += masked_bce(model(X), y, m, fire_weight).item(); vn += 1
                if max_batches and vn >= max_batches:
                    break
        vl /= max(vn, 1)
        log.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "min": (time.time() - t0) / 60})
        print(f"  epoch {ep}: train {tl / max(nb, 1):.4f}, inner-val {vl:.4f}, {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if vl < best:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if max_batches:
            break
    model.load_state_dict(best_state)
    return model, stats, keep, inner_val, log


def run_fold(fold: int, **kw):
    """Train, pick the threshold on the inner-validation fold (whole fires),
    score the held-out fold (whole fires); save the results."""
    tag = kw.pop("tag", None) or f"unet_f{fold}"
    physics, drop = kw.get("physics", False), kw.get("drop", ())
    model, stats, keep, inner_val, log = train(fold, **kw)
    data = Data()
    pv, nv = score_fold(model, data, inner_val, stats, keep, physics)
    thr = best_threshold(pv, nv)
    pt, nt = score_fold(model, data, fold, stats, keep, physics)
    res = {"fold": fold, "tag": tag, "threshold": thr / PROB_BINS, "inner_val_fold": inner_val,
           **{k: v for k, v in kw.items() if k not in ("max_batches",)}, **metrics_from_counts(pt, nt, thr)}
    out = PROCESSED / "unet_runs"
    out.mkdir(exist_ok=True)
    np.savez(out / f"{tag}_counts.npz", pos=pt, neg=nt, pos_val=pv, neg_val=nv)
    pd.DataFrame(log).to_csv(out / f"{tag}_log.csv", index=False)
    (out / f"{tag}.json").write_text(json.dumps(res, default=str, indent=1))
    torch.save(model.state_dict(), out / f"{tag}.pt")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, default=str))
    return res


def smoke() -> None:
    """A few training batches, then one whole-fire inference, with timings."""
    cw = pd.read_parquet(CACHE / "windows.parquet")
    stats = json.loads(STATS.read_text())["fold_0"]
    have_phys = (CACHE / "physics.npy").exists()
    for physics in ((False, True) if have_phys else (False,)):
        keep = channel_index([])
        ds = Windows(cw.index[cw.fold != "0"][:64], stats, keep, physics, augment=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
        model = ResUNet(len(keep) + int(physics), base=16).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        t0 = time.time()
        for i, (X, y, m) in enumerate(dl):
            X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            loss = masked_bce(model(X), y, m, 10.0)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"  physics={physics} batch {i}: X {tuple(X.shape)}, loss {loss.item():.4f}, "
                  f"finite {torch.isfinite(loss).item()}", flush=True)
        print(f"  {(time.time() - t0) / len(dl):.2f} s per batch of 16 ({DEVICE})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters (base 16): {n_params:,}")
    data = Data()
    t0 = time.time()
    pos, neg = score_fold(model, data, 0, stats, keep, have_phys)
    print(f"  whole-fire scoring of fold 0 (untrained model): {(time.time() - t0) / 60:.1f} min, "
          f"positives {int(pos.sum())}, negatives {int(neg.sum()):,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    sub.add_parser("physics_cache")
    sub.add_parser("smoke")
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--physics", action="store_true")
    t.add_argument("--drop", nargs="*", default=[], choices=list(GROUPS))
    t.add_argument("--base", type=int, default=16)
    t.add_argument("--fire-weight", type=float, default=10.0)
    t.add_argument("--no-burn-rate", type=float, default=0.5)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--tag")
    a = parser.parse_args()
    if a.cmd == "cache":
        cache()
    elif a.cmd == "physics_cache":
        physics_cache()
    elif a.cmd == "smoke":
        smoke()
    else:
        run_fold(a.fold, physics=a.physics, drop=a.drop, base=a.base, fire_weight=a.fire_weight,
                 no_burn_rate=a.no_burn_rate, epochs=a.epochs, tag=a.tag)


if __name__ == "__main__":
    main()
