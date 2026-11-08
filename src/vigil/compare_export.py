"""Export model predictions for the website's Compare page (app/compare.html).

Only out-of-sample predictions:
  test fires (2019-20): every test fire, predicted by the final test models
                        (trained on all 12 other seasons; seed 0 shown).
  CV fires:             a hand-picked set from other seasons, each predicted by
                        the fold model that never trained on its season (seed 0).
Models: main (U-Net, no raw weather, + physics), plain U-Net, wind-shifted
persistence from active fire, physics alone; thresholds and settings exactly
as scored (test: test_run/real/settings.json; CV: that fold's own choices).

Per fire-day D (predicting D+1), one small RGB PNG in Web Mercator:
  R  main model probability x 255 (for the probability view)
  G  flags: 1 burnt on D+1, 2 burnt by D, 4 not scored (edge, uncertain or
     another fire), 8 main predicts, 16 U-Net, 32 baseline, 64 physics,
     128 outside the scored area
No alpha channel, so the browser can't premultiply the values away.
Per-day hits / misses / false alarms for every model are computed on the
whole scored area (the same cells as in evaluation) and go in the JSON.

  app/data/compare/fires.json
  app/data/compare/<event>/<k>.png    k = index of the fire-day

Usage: python -m vigil.compare_export
"""

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy import ndimage

from vigil.burn_history import grid
from vigil.config import CV_FOLDS, GRID_CRS, PROCESSED, ROOT
from vigil.dataset import CHANNELS, STATS, Data
from vigil.evaluate import BIN, N_BINS, smear, wind_at
from vigil.test_eval import region_box
from vigil.unet import DEVICE, PROB_BINS, ResUNet, channel_index, load_channel_store, physics_slice, predict_region
from vigil.web_export import TOWNS, rasterio_bounds

OUT = ROOT / "app" / "data" / "compare"
WORK = PROCESSED / "compare_work"
EPOCH = pd.Timestamp("2000-01-01")
MODELS = ["main", "unet", "baseline", "physics"]
BIT = {"pos": 1, "burnt": 2, "unscored": 4, "main": 8, "unet": 16, "baseline": 32, "physics": 64, "outside": 128}
# Other-season fires: (season, event). Big forest fires that go well and
# grass fires that go badly, east and west.
CV_FIRES = [(2014, 4990), (2013, 1181), (2013, 1191), (2019, 23798), (2025, 44819), (2025, 44714),
            (2014, 4997), (2014, 5084), (2018, 20169), (2024, 40999), (2013, 1607), (2025, 45064)]
MORE_TOWNS = {
    "Kilmore": (-37.297, 144.952), "Terang": (-38.240, 142.918), "Nhill": (-36.333, 141.650),
    "Edenhope": (-37.036, 141.296), "Stawell": (-37.057, 142.780), "Heyfield": (-37.980, 146.785),
    "Harrietville": (-36.890, 147.063), "Noojee": (-37.890, 146.000), "Griffith": (-34.288, 146.040),
    "Casterton": (-37.584, 141.403), "Apsley": (-36.968, 141.083), "Goongerah": (-37.330, 148.680),
    "Wangaratta": (-36.358, 146.312), "Leeton": (-34.553, 146.404), "Dunkeld": (-37.650, 142.343),
}
GAZ = {**TOWNS, **MORE_TOWNS}


def log(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


# ------------------------------------------------------------------ models and settings
def load_unet(path, drop, physics):
    m = ResUNet(len(channel_index(drop)) + int(physics)).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.eval()


def model_sets():
    """fold key ('test' or 0-4) -> settings and loaded U-Nets."""
    stats = json.loads(STATS.read_text())
    sets = {}
    t = json.loads((PROCESSED / "test_run/real/settings.json").read_text())
    main_cfg, unet_cfg = t["Main: U-Net no weather + physics"], t["U-Net"]
    base = t["Wind-shifted persistence (active fire)"]
    sets["test"] = {"stats": stats["all"],
                    "main": (PROCESSED / "test_run/real/Main_s0.pt", main_cfg["threshold_bin"]),
                    "unet": (PROCESSED / "test_run/real/U-Net_s0.pt", unet_cfg["threshold_bin"]),
                    "beta": base["beta"], "k_bin": base["k_bin"], "scale": t["Physics alone"]["scale"],
                    "kind": "test"}
    res = pd.read_csv(PROCESSED / "unet_results_v2.csv")
    chosen = json.loads((PROCESSED / "unet_runs/v2/main_chosen_settings.json").read_text())
    b = pd.read_csv(PROCESSED / "baseline_cv.csv").query("model == 'wind_active'").set_index("fold")
    p = pd.read_csv(PROCESSED / "physics_cv.csv").set_index("fold")
    for f in CV_FOLDS:
        fw, r = chosen[str(f)]
        row = lambda m: res[(res.model == m) & (res.fold == f) & (res.seed == 0)].iloc[0]
        sets[f] = {"stats": stats[f"fold_{f}"],
                   "main": (PROCESSED / f"unet_runs/v2/Main_grid_w{fw:g}_r{r:g}_f{f}_s0.pt",
                            int(round(row("Main (tuned)").threshold * PROB_BINS))),
                   "unet": (PROCESSED / f"unet_runs/v2/U-Net_f{f}_s0.pt",
                            int(round(row("U-Net").threshold * PROB_BINS))),
                   "beta": float(b.loc[f, "variant"].split("@")[1]),
                   "k_bin": int(round(b.loc[f, "k_km"] / (BIN * 0.375))),
                   "scale": float(p.loc[f, "scale"]), "kind": "cv"}
    return sets


def fire_list(data):
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    w = w[[data.has_weather(int(d)) for d in w.day]]
    out = []
    for (s, e), g in w[w.fold == "test"].groupby(["season", "event"]):
        out.append((int(s), int(e), "test", sorted(int(d) for d in g.day.unique())))
    for s, e in CV_FIRES:
        g = w[(w.season == s) & (w.event == e)]
        assert len(g), f"no windows for {s} {e}"
        out.append((s, e, int(g.fold.iloc[0]), sorted(int(d) for d in g.day.unique())))
    return out


# ------------------------------------------------------------------ pass 1: U-Nets + baseline (GPU)
def gpu_pass(data, fires, sets):
    ib, ia = CHANNELS.index("own_burnt"), CHANNELS.index("active_D")
    keep_main, keep_unet = channel_index(["weather"]), channel_index([])
    by_fold = {}
    for f in fires:
        by_fold.setdefault(f[2], []).append(f)
    for fold, group in by_fold.items():
        S = sets[fold]
        main = load_unet(S["main"][0], ["weather"], True)
        unet = load_unet(S["unet"][0], [], False)
        for season, event, _, days in group:
            out = WORK / f"{event}.npz"
            if out.exists():
                continue
            ev = data.labels.event.sel(season=season).values
            r0, r1, c0, c1 = region_box(ev, event, data.H, data.W)
            store = load_channel_store(season)
            flags = np.zeros((len(days), r1 - r0, c1 - c0), np.uint8)
            prob = np.zeros_like(flags)
            with torch.no_grad():
                for k, D in enumerate(days):
                    X, y, mask = data.region(season, event, D, r0, r1, c0, c1)
                    burnt = X[ib] > 0
                    if not burnt.any():          # not scored in evaluation either
                        flags[k] = (y > 0) * BIT["pos"] | BIT["unscored"]
                        continue
                    phys =physics_slice(store, event, D, r0, c0, r1 - r0, c1 - c0)
                    pm = predict_region(main, X, S["stats"], keep_main, phys)
                    pu = predict_region(unet, X, S["stats"], keep_unet, None)
                    to_bin = lambda p: np.minimum((p * PROB_BINS).astype(int), PROB_BINS - 1)
                    fl = (y > 0) * BIT["pos"] + burnt * BIT["burnt"] + (~mask & ~burnt) * BIT["unscored"]
                    fl += (to_bin(pm) >= S["main"][1]) * BIT["main"] + (to_bin(pu) >= S["unet"][1]) * BIT["unet"]
                    if True:
                        active = X[ia] > 0
                        br, bc = np.nonzero(burnt)
                        u, v = wind_at(data, D + 1, br.mean() + r0, bc.mean() + c0)
                        eff = (ndimage.distance_transform_edt(~smear(active, u, v, S["beta"])) if active.any()
                               else ndimage.distance_transform_edt(~burnt))
                        fl += (np.minimum((eff / BIN).astype(int), N_BINS - 1) <= S["k_bin"]) * BIT["baseline"]
                    flags[k], prob[k] = fl, np.round(pm * 255)
            np.savez_compressed(out, flags=flags, prob=prob, box=[r0, r1, c0, c1], days=days)
            log(f"  {season} event {event}: {len(days)} fire-days, region {r1 - r0}x{c1 - c0}")
        del main, unet


# ------------------------------------------------------------------ pass 2: physics (CPU)
_CTX = None


def _init():
    global _CTX
    from vigil.physics import Ctx
    _CTX = Ctx()


def _physics(task):
    from vigil.physics import prepare, run
    season, event, scale = task
    z = np.load(WORK / f"{event}.npz")
    r0, r1, c0, c1 = (int(v) for v in z["box"])
    reached = np.zeros(z["flags"].shape, bool)
    for k, D in enumerate(z["days"]):
        if not (z["flags"][k] & BIT["burnt"]).any():
            continue
        arr = run(prepare(_CTX, season, event, int(D), r0, r1, c0, c1, start="active"), scale)
        reached[k] = np.isfinite(arr)
    return event, np.packbits(reached), reached.shape


def physics_pass(fires, sets, workers=3):
    tasks = [(s, e, sets[f]["scale"]) for s, e, f, _ in fires if not (WORK / f"{e}_physics.npz").exists()]
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        for event, packed, shape in pool.map(_physics, tasks):
            np.savez_compressed(WORK / f"{event}_physics.npz", packed=packed, shape=shape)
            log(f"  physics event {event}")


# ------------------------------------------------------------------ pass 3: images and JSON
def counts(fl, bit):
    scored = (fl & (BIT["unscored"] | BIT["burnt"])) == 0
    pos, pred = (fl & BIT["pos"]) > 0, (fl & bit) > 0
    return [int((scored & pos & pred).sum()), int((scored & ~pos & pred).sum()), int((scored & pos & ~pred).sum())]


def f1(tp, fp, fn):
    return None if tp + fp + fn == 0 else round(2 * tp / (2 * tp + fp + fn), 3)


def assemble(fires, sets):
    tr, H, W = grid()
    to_ll = Transformer.from_crs(GRID_CRS, 4326, always_xy=True)
    merc_to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
    official = pd.read_csv(ROOT / "docs/results/test_by_fire.csv")
    name_of = {"Main: U-Net no weather + physics": "main", "U-Net": "unet",
               "Wind-shifted persistence (active fire)": "baseline", "Physics alone": "physics"}
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    out = []
    for season, event, fold, days in fires:
        z = np.load(WORK / f"{event}.npz")
        flags, prob = z["flags"].copy(), z["prob"]
        r0, r1, c0, c1 = (int(v) for v in z["box"])
        zp = np.load(WORK / f"{event}_physics.npz")
        reached = np.unpackbits(zp["packed"])[:np.prod(zp["shape"])].reshape(zp["shape"]).astype(bool)
        flags |= (reached * BIT["physics"]).astype(np.uint8)
        per_day = {m: [counts(fl, BIT[m]) for fl in flags] for m in MODELS}
        # crop to what's worth drawing: burnt, burning or predicted by anyone, plus a margin
        rel = ((flags & ~np.uint8(BIT["unscored"])) > 0).any(0) | (prob >= 13).any(0)
        rr, cc = np.nonzero(rel)
        pad = 10
        a0, a1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, flags.shape[1])
        b0, b1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, flags.shape[2])
        flags, prob = flags[:, a0:a1, b0:b1], prob[:, a0:a1, b0:b1]
        R0, C0 = r0 + a0, c0 + b0
        h, w = flags.shape[1:]
        # one nearest-neighbour index map, Web Mercator <- grid cell, used for every day
        src_tr = from_origin(tr.c + C0 * tr.a, tr.f + R0 * tr.e, tr.a, -tr.e)
        src_crs, dst_crs = CRS.from_epsg(GRID_CRS), CRS.from_epsg(3857)
        dst_tr, dw, dh = calculate_default_transform(src_crs, dst_crs, w, h, *rasterio_bounds(src_tr, (h, w)),
                                                     resolution=470)
        idx = np.zeros((dh, dw), np.int32)
        reproject(np.arange(1, h * w + 1, dtype=np.int32).reshape(h, w), idx, src_transform=src_tr,
                  src_crs=src_crs, dst_transform=dst_tr, dst_crs=dst_crs, resampling=Resampling.nearest,
                  src_nodata=0, dst_nodata=0)
        inside, flat = idx > 0, np.maximum(idx - 1, 0)
        d = OUT / str(event)
        d.mkdir()
        for k in range(len(days)):
            img = np.zeros((dh, dw, 3), np.uint8)
            img[..., 0] = np.where(inside, prob[k].ravel()[flat], 0)
            img[..., 1] = np.where(inside, flags[k].ravel()[flat], BIT["outside"])
            Image.fromarray(img, "RGB").save(d / f"{k}.png", optimize=True)
        x0, y1 = dst_tr.c, dst_tr.f
        x1, y0 = x0 + dw * dst_tr.a, y1 + dh * dst_tr.e
        corners = [list(merc_to_ll.transform(x, y)) for x, y in ((x0, y1), (x1, y1), (x1, y0), (x0, y0))]
        L = (z["flags"] & BIT["pos"]).any(0) | (z["flags"][-1] & BIT["burnt"] > 0)
        lr, lc = np.nonzero(L) if L.any() else np.nonzero(rel)
        cx, cy = to_ll.transform(tr.c + tr.a * (c0 + lc.mean() + 0.5), tr.f + tr.e * (r0 + lr.mean() + 0.5))
        fr = (z["flags"][:, a0:a1, b0:b1] & (BIT["pos"] | BIT["burnt"])).any(0)
        fr_r, fr_c = np.nonzero(fr) if fr.any() else np.nonzero(rel[a0:a1, b0:b1])
        focus = [list(to_ll.transform(tr.c + tr.a * (C0 + c), tr.f + tr.e * (R0 + r)))
                 for r, c in ((fr_r.max() + 1, fr_c.min()), (fr_r.min(), fr_c.max() + 1))]
        town = min(GAZ, key=lambda t: (GAZ[t][0] - cy) ** 2 + (GAZ[t][1] - cx) ** 2)
        total = {m: [sum(c[i] for c in per_day[m]) for i in range(3)] for m in MODELS}
        fire = {"id": event, "season": season, "kind": sets[fold]["kind"],
                "fold_seasons": None if fold == "test" else CV_FOLDS[fold], "town": town,
                "center": [cx, cy], "corners": corners, "focus": focus,
                "days": [(EPOCH + pd.Timedelta(days=D + 1)).strftime("%Y-%m-%d") for D in days],
                "new_burn": [int(((fl & BIT["pos"]) > 0).sum()) for fl in z["flags"]],
                "counts": per_day, "f1": {m: f1(*total[m]) for m in MODELS}, "f1_shown_run": None}
        if fold == "test":
            o = official[official.event == event]
            fire["f1_shown_run"] = fire["f1"]
            fire["f1"] = {name_of[r.model]: round(float(r.F1), 3) for r in o.itertuples() if r.model in name_of}
        out.append(fire)
        log(f"  {season} {event}: {len(days)} images {dw}x{dh}, main F1 {fire['f1'].get('main')}")
    eg = max((f for f in out if f["kind"] == "test" and f["center"][0] > 147.5), key=lambda f: sum(f["new_burn"]))
    for f in out:
        f["label"] = "East Gippsland" if f is eg else f"Near {f['town']}"
        f["default"] = f is eg
    out.sort(key=lambda f: (not f["default"], f["kind"] != "test", -sum(f["new_burn"])))
    meta = {"models": {"main": "Main model", "unet": "Plain U-Net", "baseline": "Wind-shifted persistence",
                       "physics": "Physics alone"},
            "main_threshold": {k: v["main"][1] / PROB_BINS for k, v in sets.items()}, "fires": out}
    (OUT / "fires.json").write_text(json.dumps(meta, separators=(",", ":")))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    log(f"{len(out)} fires, {sum(len(f['days']) for f in out)} fire-days -> {OUT} ({size:.1f} MB)")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sets = model_sets()
    data = Data()
    fires = fire_list(data)
    log(f"{len(fires)} fires, {sum(len(f[3]) for f in fires)} fire-days; GPU pass")
    gpu_pass(data, fires, sets)
    del data
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    log("physics pass (CPU, after the GPU pass)")
    physics_pass(fires, sets)
    assemble(fires, sets)


if __name__ == "__main__":
    main()
