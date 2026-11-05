"""Physics baseline: a cellular automaton using CSIRO spread models.

Fire spreads over the 375 m grid as arrival times (a time-dependent shortest-
path over a 16-neighbour stencil), through the 24 h of fire day D+1 (06:00 to
06:00 local) with hourly BARRA-C2 weather. Cells reached within 24 h are the
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
"""

import heapq

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates

from vigil.burn_history import CAP_YEARS, tslb
from vigil.config import LOCAL_TZ, PROCESSED, RAW, ROOT, FIRE_DAY_START_HOUR
from vigil.dataset import Data

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


def hourly_weather(data: Data, day: int, sl) -> list[dict]:
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

    rr, cc, g = data.wrow[sl], data.wcol[sl], data.gamma[sl]
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


def simulate(data: Data, season: int, event: int, D: int, r0, r1, c0, c1,
             start: str = "active", scale: float = 1.0, hours: float = 24.0):  # active: see CLAUDE.md
    """Arrival time (h from 06:00 on D+1) per cell; inf = not reached."""
    sl = (slice(r0, r1), slice(c0, c1))
    L = data.labels.sel(season=season)
    ev, bd = L.event.values[sl], L.burn_day.values[sl]
    own = ev == event
    burnt = own & (bd > 0) & (bd <= D)
    src = own & (bd == D) if start == "active" else burnt
    if not src.any():
        src = burnt
    model = fuel_model(data.static, sl)
    tsf = tslb(season, EPOCH + pd.Timedelta(days=D), event)[sl]
    gx, gy = data.static["grad_x"][sl], data.static["grad_y"][sl]
    month = (EPOCH + pd.Timedelta(days=D + 1)).month
    wx = hourly_weather(data, D + 1, sl)
    ros = [head_ros(w, model, tsf, month) * scale for w in wx]
    lb = [np.minimum(1 + 0.0012 * w["U"] ** 2.154, 8.0) for w in wx]
    ecc = [np.sqrt(1 - 1 / b ** 2) for b in lb]

    H, W = burnt.shape
    arrival = np.full((H, W), np.inf, np.float32)
    heap = []
    for r, c in zip(*np.nonzero(src)):
        arrival[r, c] = 0.0
        heap.append((0.0, r, c))
    heapq.heapify(heap)
    done = burnt & ~src            # dead burnt area: can't burn again, doesn't spread
    steps = [(dr, dc, np.hypot(dr, dc) * CELL_M, dc / np.hypot(dr, dc), -dr / np.hypot(dr, dc))
             for dr, dc in STENCIL]
    while heap:
        t, r, c = heapq.heappop(heap)
        if t > arrival[r, c] or t > hours:
            continue
        hr = min(int(t), 23)
        R = ros[hr][r, c]
        if R <= 0:
            continue
        e = ecc[hr][r, c]
        ux, uy = wx[hr]["ux"][r, c], wx[hr]["uy"][r, c]
        for dr, dc, dist, ex, ey in steps:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W) or done[rr, cc]:
                continue
            cos_phi = ex * ux + ey * uy
            rate = R * (1 - e) / (1 - e * cos_phi)
            slope_deg = np.degrees(np.arctan(gx[r, c] * ex + gy[r, c] * ey))
            rate *= np.exp(0.069 * np.clip(slope_deg, -20, 30))
            if ros[hr][rr, cc] <= 0:
                continue
            ta = t + dist / max(rate, 1e-3)
            if ta < arrival[rr, cc]:
                arrival[rr, cc] = ta
                heapq.heappush(heap, (ta, rr, cc))
    arrival[burnt] = np.inf        # only new burn is predicted
    return arrival, model


def example() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from scipy.ndimage import distance_transform_edt

    data = Data()
    D = (pd.Timestamp("2014-02-08") - EPOCH).days
    L = data.labels.sel(season=2014)
    ev, bd = L.event.values, L.burn_day.values
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
    un = L.uncertain_day.values[sl]
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["example"])
    parser.parse_args()
    example()
