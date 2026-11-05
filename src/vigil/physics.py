"""Physics baseline: a cellular automaton using CSIRO spread models.

Fire spreads over the 375 m grid as arrival times (a time-dependent shortest-
path over a 16-neighbour stencil), through the 24 h of fire day D+1 (06:00 to
06:00 local) with hourly BARRA-C2 weather. When spread from a cell is
impossible in the current hour (e.g. damp fuel at night), the fire waits for
the next hour when it can spread. Cells reached within 24 h are the
predicted new burn; earlier arrival = higher score (for PR-AUC).

Assumptions (approved 2026-11-05, see CLAUDE.md):
  Fuel model per cell, from the dominant CSIRO BFC fuel group:
    forest, woodland, pine/plantation -> Vesta (dry eucalypt forest)
    shrubland / heath                 -> mallee-heath (surface fire)
    grassland, cropland, WUI          -> CSIRO grassland
    wetland, non-burnable             -> no spread
  Vesta fuel hazard from time since last burnt, via accumulation curves
    FHS_s = 3.5 (1 - e^-0.2t), FHS_ns = 3 (1 - e^-0.2t), H_ns = 25 (1 - e^-0.15t) cm
  Grass curing from a seasonal curve (no daily curing data).
  Slope: ROS x exp(0.069 theta), theta = slope along the spread direction (deg).
  Fire shape: ellipse with length-to-breadth 1 + 0.0012 U^2.154 (U km/h, <= 8);
    rate toward direction phi from the head = R (1 - e) / (1 - e cos phi).
  No spotting, no suppression. One global ROS scale (tuned later in CV).

Equations:
  Grass (Cheney et al. 1998, natural pasture), km/h:
    U >= 5: (1.4 + 0.838 (U - 5)^0.844) phi_M phi_C;  U < 5: (0.054 + 0.269 U) phi_M phi_C
    M = 9.58 - 0.205 T + 0.138 RH;  phi_M = exp(-0.108 M) (M < 12), else linear decay
    phi_C = 1.036 / (1 + 103.99 exp(-0.0996 (C - 20)))  (Cruz et al. 2015)
  Vesta (Cheney et al. 2012), m/h:
    U > 5: [30 + 1.5308 (U - 5)^0.8576 FHS_s^0.9301 (FHS_ns H_ns)^0.6366 1.03] phi(Mf)
    U <= 5: 30 phi(Mf);  phi(Mf) = 18.35 Mf^-1.495
    Mf (Gould et al. 2007): 12-17 h: 2.76 + 0.124 RH - 0.0187 T; 9-12 and 17-20 h:
    3.60 + 0.169 RH - 0.045 T; night: 3.08 + 0.198 RH - 0.0483 T
  Mallee-heath surface fire (Cruz et al. 2013), m/min:
    3.337 U^0.708 exp(-0.0697 MC);  MC = 4.79 + 0.173 RH - 0.1 (T - 25)
  U = 10 m wind speed in km/h; T in deg C; RH in %.

Usage:
  python -m vigil.physics example     # one fire-day: predicted vs actual figure
  python -m vigil.physics cv          # season CV, ROS scale tuned on the other folds
  python -m vigil.physics channel     # arrival-time input channel (scale 1) for the hybrid U-Net
"""

import heapq

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates

from vigil.burn_history import CAP_YEARS, tslb
from vigil.config import LOCAL_TZ, PROCESSED, RAW, ROOT, FIRE_DAY_START_HOUR

EPOCH = pd.Timestamp("2000-01-01")
CELL_M = 375.0
CURING = {10: 60, 11: 75, 12: 90, 1: 95, 2: 95, 3: 90, 4: 80}   # % by month (assumption)
GRASS, VESTA, MALLEE, NONE = 0, 1, 2, 3
# 16-neighbour stencil (rook, diagonal, knight moves): fewer direction artefacts.
STENCIL = [(dr, dc) for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2)
           if (dr, dc) != (0, 0) and (abs(dr) < 2 and abs(dc) < 2 or abs(dr) + abs(dc) == 3)]


def fuel_model(static: dict, sl) -> np.ndarray:
    groups = {VESTA: ["fuel_forest", "fuel_woodland", "fuel_plantation"],
              MALLEE: ["fuel_shrubland"],
              GRASS: ["fuel_grassland", "fuel_crop", "fuel_wui"],
              NONE: ["fuel_wetland", "fuel_nonburnable"]}
    stack = np.stack([sum(static[k][sl] for k in groups[m]) for m in (GRASS, VESTA, MALLEE, NONE)])
    return np.argmax(stack, axis=0)


class Ctx:
    """Light context for physics runs (no daily weather arrays), so several
    worker processes fit in memory: grid geometry, fuel/terrain layers, and
    one season of labels at a time."""

    def __init__(self):
        from pyproj import Transformer
        from vigil.burn_history import grid
        self.transform, self.H, self.W = grid()
        st = xr.open_dataset(PROCESSED / "static.nc", engine="h5netcdf")
        keep = [k for k in st.data_vars if k.startswith("fuel_")] + ["grad_x", "grad_y", "elevation"]
        self.static = {k: st[k].values for k in keep}
        ref = xr.open_dataset(sorted((RAW / "barra2" / "uas").glob("*.nc"))[0], engine="h5netcdf")
        blat, blon = ref.lat.values, ref.lon.values
        tr = self.transform
        X, Y = np.meshgrid(tr.c + tr.a * (np.arange(self.W) + 0.5), tr.f + tr.e * (np.arange(self.H) + 0.5))
        lon, lat = Transformer.from_crs(3111, 4326, always_xy=True).transform(X, Y)
        self.wrow = ((lat - blat[0]) / (blat[1] - blat[0])).astype(np.float32)
        self.wcol = ((lon - blon[0]) / (blon[1] - blon[0])).astype(np.float32)
        x2, y2 = Transformer.from_crs(4326, 3111, always_xy=True).transform(lon, lat + 0.01)
        self.gamma = np.arctan2(x2 - X, y2 - Y).astype(np.float32)
        self._labels = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf")
        self._season = {}

    def season(self, s: int):
        """(event, burn_day, uncertain_day) arrays for a season; keeps two seasons cached."""
        if s not in self._season:
            if len(self._season) >= 2:
                self._season.pop(next(iter(self._season)))
            L = self._labels.sel(season=s)
            self._season[s] = (L.event.values, L.burn_day.values, L.uncertain_day.values)
        return self._season[s]


def hourly_weather(ctx, day: int, sl) -> list[dict]:
    """24 hourly fields (T, RH, U km/h, wind unit vector in grid axes) for fire day `day`."""
    start = (EPOCH + pd.Timedelta(days=day) + pd.Timedelta(hours=FIRE_DAY_START_HOUR)) \
        .tz_localize(LOCAL_TZ).tz_convert("UTC").tz_localize(None)
    hours = pd.date_range(start, periods=24, freq="h")
    cache = {}

    def month(v, ts):
        key = (v, ts.strftime("%Y%m"))
        if key not in cache:
            cache[key] = xr.open_dataset(RAW / "barra2" / v / f"{key[1]}.nc", engine="h5netcdf")[v]
        return cache[key]

    rr, cc, g = ctx.wrow[sl], ctx.wcol[sl], ctx.gamma[sl]
    out = []
    for ts in hours:
        f = {v: map_coordinates(month(v, ts).sel(time=ts).values.astype(np.float32), [rr, cc],
                                order=1, mode="nearest") for v in ("uas", "vas", "tas", "hurs")}
        u = f["uas"] * np.cos(g) + f["vas"] * np.sin(g)
        v = -f["uas"] * np.sin(g) + f["vas"] * np.cos(g)
        speed = np.hypot(u, v)
        out.append({"T": f["tas"] - 273.15, "RH": np.clip(f["hurs"], 0, 100), "U": speed * 3.6,
                    "ux": u / np.maximum(speed, 1e-6), "uy": v / np.maximum(speed, 1e-6),
                    "local_hour": (FIRE_DAY_START_HOUR + len(out)) % 24})
    return out


def head_ros(w: dict, model: np.ndarray, tsf: np.ndarray, month: int) -> np.ndarray:
    """Head-fire rate of spread on flat ground, m/h, per cell for one hour."""
    T, RH, U, h = w["T"], w["RH"], w["U"], w["local_hour"]
    ros = np.zeros_like(T)
    # Grass
    M = 9.58 - 0.205 * T + 0.138 * RH
    phi_m = np.where(M < 12, np.exp(-0.108 * M),
                     np.where(U <= 10, 0.684 - 0.0342 * M, 0.547 - 0.0228 * M)).clip(0)
    C = CURING.get(month, 80)
    phi_c = 1.036 / (1 + 103.99 * np.exp(-0.0996 * (C - 20)))
    grass = np.where(U >= 5, 1.4 + 0.838 * np.maximum(U - 5, 0) ** 0.844, 0.054 + 0.269 * U)
    ros = np.where(model == GRASS, grass * phi_m * phi_c * 1000, ros)
    # Vesta
    if 12 <= h < 17:
        Mf = 2.76 + 0.124 * RH - 0.0187 * T
    elif 9 <= h < 12 or 17 <= h < 20:
        Mf = 3.60 + 0.169 * RH - 0.045 * T
    else:
        Mf = 3.08 + 0.198 * RH - 0.0483 * T
    phi_mf = 18.35 * np.maximum(Mf, 2.0) ** -1.495
    fhs_s = np.maximum(3.5 * (1 - np.exp(-0.2 * tsf)), 0.5)
    fhs_ns = np.maximum(3.0 * (1 - np.exp(-0.2 * tsf)), 0.5)
    h_ns = np.maximum(25 * (1 - np.exp(-0.15 * tsf)), 2.0)
    vesta = np.where(U > 5, 30 + 1.5308 * np.maximum(U - 5, 0) ** 0.8576 * fhs_s ** 0.9301
                     * (fhs_ns * h_ns) ** 0.6366 * 1.03, 30.0) * phi_mf
    ros = np.where(model == VESTA, vesta, ros)
    # Mallee-heath surface fire
    MC = 4.79 + 0.173 * RH - 0.1 * (T - 25)
    mallee = 3.337 * np.maximum(U, 0) ** 0.708 * np.exp(-0.0697 * MC) * 60
    ros = np.where(model == MALLEE, mallee, ros)
    return np.where(model == NONE, 0.0, ros).astype(np.float32)


STEPS = [(dr, dc, float(np.hypot(dr, dc)) * CELL_M, dc / np.hypot(dr, dc), -dr / np.hypot(dr, dc))
         for dr, dc in STENCIL]


def prepare(ctx, season: int, event: int, D: int, r0, r1, c0, c1, start: str = "active") -> dict:
    """Everything for one fire-day that doesn't depend on the ROS scale: start
    cells, dead burnt cells, and the spread rate (m/h, scale 1) from each cell
    in each of the 16 stencil directions for each of the 24 hours."""
    sl = (slice(r0, r1), slice(c0, c1))
    ev, bd, _ = ctx.season(season)
    own = ev[sl] == event
    burnt = own & (bd[sl] > 0) & (bd[sl] <= D)
    src = own & (bd[sl] == D) if start == "active" else burnt
    if not src.any():
        src = burnt
    model = fuel_model(ctx.static, sl)
    tsf = tslb(season, EPOCH + pd.Timedelta(days=D), event)[sl]
    gx, gy = ctx.static["grad_x"][sl], ctx.static["grad_y"][sl]
    month = (EPOCH + pd.Timedelta(days=D + 1)).month
    wx = hourly_weather(ctx, D + 1, sl)
    slope_f = [np.exp(0.069 * np.clip(np.degrees(np.arctan(gx * ex + gy * ey)), -20, 30))
               for _, _, _, ex, ey in STEPS]
    rate = np.zeros((24, len(STEPS)) + burnt.shape, np.float32)
    for h, w in enumerate(wx):
        R = head_ros(w, model, tsf, month)
        e = np.sqrt(1 - 1 / np.minimum(1 + 0.0012 * w["U"] ** 2.154, 8.0) ** 2)
        for k, (_, _, _, ex, ey) in enumerate(STEPS):
            cos_phi = ex * w["ux"] + ey * w["uy"]
            rate[h, k] = R * (1 - e) / (1 - e * cos_phi) * slope_f[k]
    burnable = head_ros(wx[12], model, tsf, month) > 0
    return {"src": src, "done": burnt & ~src, "burnt": burnt, "rate": rate,
            "burnable": burnable, "model": model}


def run(prep: dict, scale: float = 1.0, hours: float = 24.0) -> np.ndarray:
    """Arrival time (h after 06:00 on D+1) per cell; inf = not reached in `hours`."""
    src, done, rate, burnable = prep["src"], prep["done"], prep["rate"], prep["burnable"]
    H, W = src.shape
    arrival = np.full((H, W), np.inf)
    heap = [(0.0, int(r), int(c)) for r, c in zip(*np.nonzero(src))]
    for _, r, c in heap:
        arrival[r, c] = 0.0
    heapq.heapify(heap)
    blocked = done | ~burnable
    steps = [(k, dr, dc, dist) for k, (dr, dc, dist, _, _) in enumerate(STEPS)]
    while heap:
        t, r, c = heapq.heappop(heap)
        if t > arrival[r, c] or t > hours:
            continue
        h0 = min(int(t), 23)
        for k, dr, dc, dist in steps:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= H or cc >= W or blocked[rr, cc]:
                continue
            # If spread is impossible this hour (e.g. damp fuel at night), the
            # fire waits for the first later hour when it can spread. The rate
            # at departure is used for the whole step.
            ta = np.inf
            for h in range(h0, 24):
                v = rate[h, k, r, c] * scale
                if v > 0:
                    ta = max(t, h) + dist / v
                    break
            if ta < arrival[rr, cc] and ta <= hours:  # noqa: E501
                arrival[rr, cc] = ta
                heapq.heappush(heap, (ta, rr, cc))
    arrival[prep["burnt"]] = np.inf        # only new burn is predicted
    return arrival.astype(np.float32)


def simulate(ctx, season, event, D, r0, r1, c0, c1, start="active", scale=1.0, hours=24.0):
    prep = prepare(ctx, season, event, D, r0, r1, c0, c1, start)
    return run(prep, scale, hours), prep["model"]


def example() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from scipy.ndimage import distance_transform_edt

    data = Ctx()
    D = (pd.Timestamp("2014-02-08") - EPOCH).days
    ev, bd, un_all = data.season(2014)
    # East Gippsland event active on 8 Feb 2014 with the most new burn on 9 Feb.
    cand = pd.Series(ev[(bd == D + 1) & (ev >= 0)]).value_counts()
    event = int(cand.index[0])
    r, c = np.nonzero(ev == event)
    pad = 40
    r0, r1 = max(r.min() - pad, 0), min(r.max() + pad, data.H)
    c0, c1 = max(c.min() - pad, 0), min(c.max() + pad, data.W)
    sl = (slice(r0, r1), slice(c0, c1))
    own = ev[sl] == event
    burnt = own & (bd[sl] > 0) & (bd[sl] <= D)
    actual = own & (bd[sl] == D + 1)
    un = un_all[sl]
    valid = ~burnt & ~((un > 0) & (un <= D + 1))

    import time
    res = {}
    for start in ("active", "burnt"):
        t0 = time.time()
        arr, model = simulate(data, 2014, event, D, r0, r1, c0, c1, start=start)
        pred = arr <= 24
        tp, fp, fn = (pred & actual & valid).sum(), (pred & ~actual & valid).sum(), (~pred & actual & valid).sum()
        res[start] = (arr, pred, 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1),
                      time.time() - t0)
        print(f"start={start}: F1 {res[start][2]:.3f}, precision {res[start][3]:.3f}, recall {res[start][4]:.3f}, "
              f"predicted {pred.sum()} cells vs actual {actual.sum()}, {res[start][5]:.0f}s")
    # Persistence (active, 1-cell ring) for comparison
    act = own & (bd[sl] == D)
    per = (distance_transform_edt(~act) <= 1.5) & ~burnt
    tp, fp, fn = (per & actual & valid).sum(), (per & ~actual & valid).sum(), (~per & actual & valid).sum()
    print(f"persistence (active): F1 {2*tp/max(2*tp+fp+fn,1):.3f}")

    elev = data.static["elevation"][sl]
    hs = 0.55 + 0.4 * LightSource(315, 40).hillshade(elev, vert_exag=6, dx=375, dy=375)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    arr = res["active"][0]
    ax = axes[0]
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1)
    a = np.where(np.isfinite(arr) & (arr <= 24), arr, np.nan)
    im = ax.imshow(a, cmap="YlOrBr_r", vmin=0, vmax=24)
    ax.contour(burnt, levels=[0.5], colors="#222", linewidths=0.8)
    fig.colorbar(im, ax=ax, fraction=0.04, label="predicted arrival (hours after 06:00, 9 Feb)")
    ax.set_title("Physics model: arrival time (start = active fire on 8 Feb)", fontsize=11)
    for ax, (title, pred) in zip(axes[1:], [("Physics (start = active fire)", res["active"][1]),
                                            ("Physics (start = all burnt area)", res["burnt"][1])]):
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1)
        img = np.zeros(pred.shape + (4,))
        img[pred & actual] = (0.10, 0.45, 0.75, 0.9)      # hit (blue)
        img[pred & ~actual] = (0.95, 0.6, 0.15, 0.75)     # false alarm (amber)
        img[~pred & actual] = (0.85, 0.2, 0.2, 0.85)      # missed (red)
        img[burnt] = (0.25, 0.25, 0.25, 0.6)              # burnt by 8 Feb (grey)
        ax.imshow(img)
        f1, p, rc = res["burnt" if "all" in title else "active"][2:5]
        ax.set_title(f"{title}\nF1 {f1:.2f}, precision {p:.2f}, recall {rc:.2f}", fontsize=11)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=10, mfc=col, mec="none", label=lab) for col, lab in
               [((0.10, 0.45, 0.75), "hit"), ((0.95, 0.6, 0.15), "false alarm"), ((0.85, 0.2, 0.2), "missed"),
                ((0.25, 0.25, 0.25), "burnt by 8 Feb")]]
    axes[2].legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"Physics baseline, one fire-day (untuned, ROS scale 1): event {event}, "
                 f"predicting 9 Feb 2014 from 8 Feb. North up, 375 m cells.", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "docs" / "figures" / "physics_example_2014-02-09.png"
    fig.savefig(out, dpi=90, facecolor="white")
    print(f"-> {out}")


# ---------------------------------------------------------------- season CV
SCALES = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]   # global ROS scale grid (first grid 0.75-3 hit its lower edge)
T_BIN = 0.25                                # arrival-time bins, hours
N_T = int(24 / T_BIN) + 1                   # last bin = not reached in 24 h
_CTX = None


def _init_worker():
    global _CTX
    _CTX = Ctx()


def _fire_event(task):
    """All fire-days of one event: counts of positives/negatives per (scale,
    band, arrival bin) over the scored cells, as in evaluate.py."""
    from scipy.ndimage import distance_transform_edt
    from vigil.evaluate import BANDS, PAD
    season, event, days = task
    ev, bd, un = _CTX.season(season)
    r, c = np.nonzero(ev == event)
    r0, r1 = max(r.min() - PAD, 0), min(r.max() + PAD + 1, _CTX.H)
    c0, c1 = max(c.min() - PAD, 0), min(c.max() + PAD + 1, _CTX.W)
    E, B, U = ev[r0:r1, c0:c1], bd[r0:r1, c0:c1], un[r0:r1, c0:c1]
    own = E == event
    out = []
    for D, fold in days:
        burnt = own & (B > 0) & (B <= D)
        if not burnt.any():
            continue
        pos = own & (B == D + 1)
        other = (E >= 0) & ~own & (B > 0) & (B <= D + 1)
        valid = ~burnt & ~((U > 0) & (U <= D + 1)) & ~other
        band = np.digitize(distance_transform_edt(~burnt), BANDS, right=True)
        prep = prepare(_CTX, season, event, D, r0, r1, c0, c1, start="active")
        P = np.zeros((len(SCALES), 3, N_T))
        N = np.zeros((len(SCALES), 3, N_T))
        for i, sc in enumerate(SCALES):
            arr = run(prep, sc)
            b = np.where(np.isfinite(arr), np.minimum((arr / T_BIN).astype(int), N_T - 2), N_T - 1)
            for k in range(3):
                sel = valid & (band == k)
                P[i, k] = np.bincount(b[sel & pos], minlength=N_T)
                N[i, k] = np.bincount(b[sel & ~pos], minlength=N_T)
        out.append((fold, P, N))
    return out


def cv(workers: int = 5) -> None:
    """Physics baseline in the same season CV as the other baselines; the ROS
    scale is chosen on the other 4 folds (pooled F1 at the 24 h cutoff)."""
    import time
    from concurrent.futures import ProcessPoolExecutor
    from vigil.evaluate import BAND_NAMES
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    w = w[w.fold != "test"]
    wx_days = set((pd.to_datetime(xr.open_dataset(PROCESSED / "weather_daily.nc",
                   engine="h5netcdf").day.values) - EPOCH).days)
    fd = w.drop_duplicates(["season", "event", "day"])
    fd = fd[fd.day.isin(wx_days) & (fd.day + 1).isin(wx_days)]
    tasks = [(int(s), int(e), [(int(d), int(f)) for d, f in zip(g.day, g.fold)])
             for (s, e), g in fd.groupby(["season", "event"])]
    tasks.sort(key=lambda t: -len(t[2]))          # long events first, for load balance
    P = np.zeros((5, len(SCALES), 3, N_T))
    N = np.zeros((5, len(SCALES), 3, N_T))
    t0, done = time.time(), 0
    with ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
        for res in pool.map(_fire_event, tasks):
            for fold, p_, n_ in res:
                P[fold] += p_
                N[fold] += n_
                done += 1
            if done and done % 100 < len(res):
                print(f"  {done}/{len(fd)} fire-days, {(time.time() - t0) / 60:.0f} min", flush=True)
    np.savez(PROCESSED / "physics_counts.npz", P=P, N=N, scales=SCALES)
    report_cv(P, N)


def report_cv(P, N) -> None:
    from vigil.evaluate import BAND_NAMES
    reached = N_T - 1                              # bins 0..N_T-2 = reached within 24 h

    def f1_prec_rec(p, n):
        tp, fp, P_ = p[:reached].sum(), n[:reached].sum(), p.sum()
        prec, rec = tp / max(tp + fp, 1), tp / max(P_, 1)
        return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec

    def ap(p, n):
        tp, fp = np.cumsum(p), np.cumsum(n)
        prec, rec = tp / np.maximum(tp + fp, 1), tp / max(p.sum(), 1)
        return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))

    rows = []
    for f in range(5):
        others = [g for g in range(5) if g != f]
        f1_train = [f1_prec_rec(P[others, i].sum((0, 1)), N[others, i].sum((0, 1)))[0]
                    for i in range(len(SCALES))]
        i = int(np.argmax(f1_train))
        p, n = P[f, i].sum(0), N[f, i].sum(0)
        f1, prec, rec = f1_prec_rec(p, n)
        row = {"model": "physics", "fold": f, "scale": SCALES[i], "F1": f1, "IoU": f1 / (2 - f1),
               "precision": prec, "recall": rec, "PR_AUC": ap(p, n)}
        for k, name in enumerate(BAND_NAMES):
            fb, _, rb = f1_prec_rec(P[f, i, k], N[f, i, k])
            row[f"F1 {name}"], row[f"recall {name}"] = fb, rb
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(PROCESSED / "physics_cv.csv", index=False)
    print(r.round(3).to_string(index=False))
    # Comparison with the other baselines (mean +- sd over the 5 folds).
    base = pd.read_csv(PROCESSED / "baseline_cv.csv")
    pix = pd.read_csv(PROCESSED / "pixel_baselines_cv.csv")
    allr = pd.concat([base, pix[pix.model == "logreg"], r], ignore_index=True)
    cols = ["F1", "IoU", "precision", "recall", "PR_AUC", "F1 <=6 km", "F1 6-24 km", "F1 >24 km",
            "recall 6-24 km", "recall >24 km"]
    summ = allr.groupby("model")[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(PROCESSED / "physics_vs_baselines.csv")
    print(summ.T.to_string())


# ------------------------------------------------- physics input channel (hybrid)
CHANNEL_SCALE = 1.0
CHANNEL_PAD = 80          # half a window (64) + jitter (16), so windows can cut from it
CHANNEL_DIR = PROCESSED / "physics_channel"
UNREACHED = 255           # arrival stored in 0.1 h steps (0-240); 255 = not reached in 24 h


def _channel_event(task):
    season, event, days = task
    ev, bd, _ = _CTX.season(season)
    r, c = np.nonzero(ev == event)
    r0, r1 = max(r.min() - CHANNEL_PAD, 0), min(r.max() + CHANNEL_PAD + 1, _CTX.H)
    c0, c1 = max(c.min() - CHANNEL_PAD, 0), min(c.max() + CHANNEL_PAD + 1, _CTX.W)
    out = []
    for D in days:
        arr = run(prepare(_CTX, season, event, D, r0, r1, c0, c1, start="active"), CHANNEL_SCALE)
        q = np.where(np.isfinite(arr), np.minimum(np.round(arr * 10), 240), UNREACHED).astype(np.uint8)
        out.append((season, event, D, r0, c0, q))
    return out


def precompute_channel(workers: int = 5) -> None:
    """Physics arrival time (scale 1.0) for every fire-day with a window, CV and
    test alike (no tuning happens on test, so computing it there is fine)."""
    import time
    from concurrent.futures import ProcessPoolExecutor
    w = pd.read_parquet(PROCESSED / "windows.parquet")
    wx_days = set((pd.to_datetime(xr.open_dataset(PROCESSED / "weather_daily.nc",
                   engine="h5netcdf").day.values) - EPOCH).days)
    fd = w.drop_duplicates(["season", "event", "day"])
    fd = fd[fd.day.isin(wx_days) & (fd.day + 1).isin(wx_days)]
    tasks = [(int(s), int(e), [int(d) for d in g.day]) for (s, e), g in fd.groupby(["season", "event"])]
    tasks.sort(key=lambda t: -len(t[2]))
    CHANNEL_DIR.mkdir(parents=True, exist_ok=True)
    store, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
        for res in pool.map(_channel_event, tasks):
            for season, event, D, r0, c0, q in res:
                store.setdefault(season, {})[f"{event}_{D}"] = q
                store[season][f"{event}_{D}_origin"] = np.array([r0, c0])
                done += 1
            if done and done % 100 < len(res):
                print(f"  {done}/{len(fd)} fire-days, {(time.time() - t0) / 60:.0f} min", flush=True)
    for season, d in store.items():
        out = CHANNEL_DIR / f"{season}.npz"
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **d)
        tmp.rename(out)
    print(f"{done} fire-days -> {CHANNEL_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["example", "cv", "channel"])
    parser.add_argument("--workers", type=int, default=5)
    a = parser.parse_args()
    {"example": example, "cv": lambda: cv(a.workers),
     "channel": lambda: precompute_channel(a.workers)}[a.cmd]()
