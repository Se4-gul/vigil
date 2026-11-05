"""Figures from the labels: an animation of one fire and an interactive map.

  python -m vigil.viz animate      # 2019-20 East Gippsland fire, MP4 + GIF
  python -m vigil.viz map          # all 2020 fires, self-contained HTML with a day slider

Both draw burn days (labels.nc) over a hillshade of the 375 m elevation grid.
New burn on the shown day is bright; earlier burn is a darker shade of the same
hue; masked/uncertain cells aren't drawn.
"""

import argparse
import base64
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LightSource, ListedColormap
from PIL import Image
from pyproj import Transformer

from vigil.burn_history import grid
from vigil.config import PROCESSED, ROOT

FIG = ROOT / "docs" / "figures"
EPOCH = pd.Timestamp("2000-01-01")
NEW = "#ff7a1a"        # today's new burn
OLD = "#7a2414"        # burnt earlier
TOWNS = {"Mallacoota": (-37.553, 149.754), "Buchan": (-37.497, 148.176),
         "Orbost": (-37.708, 148.456)}
plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"


def hillshade(elev: np.ndarray) -> np.ndarray:
    """Muted grey hillshade (0-1); sea flat and pale."""
    hs = LightSource(azdeg=315, altdeg=40).hillshade(elev, vert_exag=6, dx=375, dy=375)
    hs = 0.55 + 0.4 * hs                     # compress toward light grey
    hs[elev <= 0] = 0.93
    return hs


def town_pixels(r0: int, c0: int) -> dict:
    tr, _, _ = grid()
    t = Transformer.from_crs(4326, 3111, always_xy=True)
    out = {}
    for name, (lat, lon) in TOWNS.items():
        x, y = t.transform(lon, lat)
        c, r = ~tr * (x, y)
        out[name] = (r - r0, c - c0)
    return out


def animate(event_hint=(-37.39, 148.76), season: int = 2020) -> None:
    L = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf").sel(season=season)
    ev, bd = L.event.values, L.burn_day.values
    # The largest labelled event near the hint is the East Gippsland fire.
    tr, H, W = grid()
    t = Transformer.from_crs(4326, 3111, always_xy=True)
    x, y = t.transform(event_hint[1], event_hint[0])
    ch, rh = ~tr * (x, y)
    ids, counts = np.unique(ev[ev >= 0], return_counts=True)
    near = [(n, e) for e, n in zip(ids, counts)
            if np.hypot(*(np.argwhere(ev == e).mean(0) - (rh, ch))) < 80]
    event = max(near)[1]
    r, c = np.nonzero(ev == event)
    pad = 25
    r0, r1 = max(r.min() - pad, 0), min(r.max() + pad, H)
    c0, c1 = max(c.min() - pad, 0), min(c.max() + pad + 40, W)   # room east for Mallacoota
    own = ev[r0:r1, c0:c1] == event
    days = np.where(own, bd[r0:r1, c0:c1], 0)
    elev = xr.open_dataset(PROCESSED / "static.nc", engine="h5netcdf").elevation.values[r0:r1, c0:c1]
    hs = hillshade(elev)
    first, last = days[days > 0].min(), days[days > 0].max()
    frames = list(range(first, last + 1))
    towns = town_pixels(r0, c0)
    burnt_kha = lambda d: ((days > 0) & (days <= d)).sum() * 0.375**2 * 100 / 1e3

    fig, ax = plt.subplots(figsize=(8, 8 * (r1 - r0) / (c1 - c0) + 0.6), dpi=110)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    layer = ax.imshow(np.zeros(days.shape), cmap=ListedColormap(["none", OLD, NEW]),
                      vmin=0, vmax=2, interpolation="nearest")
    for name, (tr_, tc_) in towns.items():
        ax.plot(tc_, tr_, "o", ms=5, mfc="white", mec="#222", mew=1.2)
        ax.text(tc_ + 6, tr_ - 4, name, fontsize=10, color="#222", weight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    stamp = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=15, weight="bold",
                    color="#222", va="top",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", alpha=0.9))
    ax.text(0.02, 0.025, "Burn day from MODIS MCD64A1 outlines + VIIRS timing (375 m). "
            "Fire days run 06:00-06:00.", transform=ax.transAxes, fontsize=7.5, color="#444",
            bbox=dict(fc="white", ec="none", alpha=0.75))
    handles = [plt.Line2D([], [], marker="s", ls="", ms=10, mfc=NEW, mec="none", label="Burnt this day"),
               plt.Line2D([], [], marker="s", ls="", ms=10, mfc=OLD, mec="none", label="Burnt earlier")]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9, framealpha=0.9)
    ax.set_axis_off()

    def draw(d):
        img = np.where(days == d, 2, np.where((days > 0) & (days < d), 1, 0))
        layer.set_data(img)
        date = (EPOCH + pd.Timedelta(days=int(d))).strftime("%a %d %b %Y")
        stamp.set_text(f"{date}\n{burnt_kha(d):,.0f} kha burnt")
        return layer, stamp

    anim = FuncAnimation(fig, draw, frames=frames + [last] * 8, blit=False)
    FIG.mkdir(parents=True, exist_ok=True)
    mp4, gif = FIG / "east_gippsland_2019_20.mp4", FIG / "east_gippsland_2019_20.gif"
    anim.save(mp4, writer=FFMpegWriter(fps=6, bitrate=2400))
    fig.set_dpi(70)
    anim.save(gif, writer=PillowWriter(fps=6))
    print(f"event {event}: {len(frames)} days, {(EPOCH + pd.Timedelta(days=int(first))).date()} to "
          f"{(EPOCH + pd.Timedelta(days=int(last))).date()}, {burnt_kha(last):,.0f} kha\n-> {mp4}\n-> {gif}")


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


MAP_TOWNS = {"Mallacoota": (-37.553, 149.754), "Buchan": (-37.497, 148.176),
             "Orbost": (-37.708, 148.456), "Corryong": (-36.197, 147.902),
             "Bright": (-36.730, 146.960), "Bairnsdale": (-37.823, 147.610),
             "Melbourne": (-37.814, 144.963), "Canberra": (-35.281, 149.130)}


def build_map(season: int = 2020) -> None:
    """Self-contained HTML: all labelled fires of a season, with a day slider."""
    tr, H, W = grid()
    L = xr.open_dataset(PROCESSED / "labels.nc", engine="h5netcdf").sel(season=season)
    ev, bd = L.event.values, L.burn_day.values
    # Scope: fire season only (1 Oct - 30 Apr), as for the models.
    lo = (pd.Timestamp(f"{season - 1}-10-01") - EPOCH).days
    hi = (pd.Timestamp(f"{season}-04-30") - EPOCH).days
    burnt = (ev >= 0) & (bd >= lo) & (bd <= hi)
    r, c = np.nonzero(burnt)
    pad = 30
    r0, r1 = max(r.min() - pad, 0), min(r.max() + pad, H)
    c0, c1 = max(c.min() - pad, 0), min(c.max() + pad, W)
    days = np.where(burnt, bd, 0)[r0:r1, c0:c1]
    first, last = lo, hi                       # slider spans the whole fire season
    off = np.where(days > 0, days - first, 0)
    data = np.zeros(days.shape + (4,), np.uint8)       # R,G = day offset; A = burnt
    data[..., 0] = off % 256
    data[..., 1] = off // 256
    data[..., 3] = np.where(days > 0, 255, 0)
    elev = xr.open_dataset(PROCESSED / "static.nc", engine="h5netcdf").elevation.values[r0:r1, c0:c1]
    hs = (hillshade(elev) * 255).astype(np.uint8)
    t = Transformer.from_crs(4326, 3111, always_xy=True)
    towns = {}
    for name, (lat, lon) in MAP_TOWNS.items():
        x, y = t.transform(lon, lat)
        cc, rr = ~tr * (x, y)
        if 0 <= rr - r0 < days.shape[0] and 0 <= cc - c0 < days.shape[1]:
            towns[name] = [rr - r0, cc - c0]
    daily = np.bincount(off[days > 0].ravel(), minlength=last - first + 1)
    meta = {"first": (EPOCH + pd.Timedelta(days=first)).strftime("%Y-%m-%d"),
            "ndays": last - first + 1, "w": days.shape[1], "h": days.shape[0],
            "cellKha": 0.375**2 * 100 / 1e3, "towns": towns, "daily": daily.tolist(),
            "season": season}
    html = MAP_HTML.replace("__META__", json.dumps(meta)) \
                   .replace("__HILLSHADE__", png_b64(hs)) \
                   .replace("__DATA__", png_b64(data))
    out = FIG / f"fires_{season}_map.html"
    out.write_text(html)
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB), {meta['ndays']} days, "
          f"{days.shape[1]} x {days.shape[0]} cells")


MAP_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Vigil: 2019-20 fire season, day by day</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --ink:#222; --muted:#666; --line:#ddd; --bg:#fafaf8; --new:#ff7a1a; --old:#7a2414; }
  body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; color:var(--ink); background:var(--bg); }
  header { padding:14px 20px 6px; }
  h1 { font-size:18px; margin:0 0 2px; }
  .sub { color:var(--muted); font-size:12.5px; }
  .controls { display:flex; gap:14px; align-items:center; padding:8px 20px; flex-wrap:wrap; }
  button { font:inherit; padding:5px 12px; border:1px solid #bbb; border-radius:6px; background:white; cursor:pointer; }
  input[type=range] { flex:1; min-width:260px; accent-color:var(--new); }
  .date { font-weight:700; font-size:16px; min-width:150px; }
  .stat { color:var(--muted); min-width:170px; }
  .legend { display:flex; gap:14px; color:var(--ink); font-size:12.5px; }
  .sw { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:5px; }
  .wrap { position:relative; margin:4px 20px 8px; }
  canvas { width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:4px; image-rendering:pixelated; }
  #tip { position:absolute; pointer-events:none; background:white; border:1px solid #ccc; border-radius:4px;
         padding:3px 7px; font-size:12px; display:none; box-shadow:0 1px 4px rgba(0,0,0,.12); white-space:nowrap; }
  #spark { width:100%; height:46px; display:block; }
  .sparkwrap { margin:0 20px 16px; }
  footer { color:var(--muted); font-size:11.5px; padding:0 20px 18px; }
</style></head><body>
<header>
  <h1>2019–20 fire season: burnt area by day</h1>
  <div class="sub">Every labelled fire from October to April (Victoria and the NSW/SA parts of the study area). 375 m cells; burn day from MODIS MCD64A1 outlines with VIIRS timing.</div>
</header>
<div class="controls">
  <button id="play" aria-label="Play or pause">▶ Play</button>
  <input id="slider" type="range" min="0" value="0" step="1" aria-label="Fire day">
  <span class="date" id="date"></span>
  <span class="stat" id="stat"></span>
  <span class="legend"><span><span class="sw" style="background:var(--new)"></span>Burnt this day</span>
  <span><span class="sw" style="background:var(--old)"></span>Burnt earlier</span></span>
</div>
<div class="wrap"><canvas id="map"></canvas><div id="tip"></div></div>
<div class="sparkwrap"><canvas id="spark"></canvas></div>
<footer>Fire days run 06:00–06:00 local time. Use ← → to step a day, space to play. Hover a burnt cell to see its burn date.</footer>
<script>
const META = __META__;
const HS = "data:image/png;base64,__HILLSHADE__";
const DATA = "data:image/png;base64,__DATA__";
const NEW = [255,122,26], OLD = [122,36,20];
const canvas = document.getElementById("map"), ctx = canvas.getContext("2d");
const slider = document.getElementById("slider"), tip = document.getElementById("tip");
canvas.width = META.w; canvas.height = META.h;
slider.max = META.ndays - 1;
const first = new Date(META.first + "T00:00:00");
const fmt = d => { const x = new Date(first); x.setDate(x.getDate() + d);
  return x.toLocaleDateString("en-AU", {weekday:"short", day:"numeric", month:"short", year:"numeric"}); };
const cum = []; META.daily.reduce((a, v, i) => (cum[i] = a + v), 0);

let hsImg, dayOff, overlay;
function load(src) { return new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = src; }); }
Promise.all([load(HS), load(DATA)]).then(([h, d]) => {
  hsImg = h;
  const off = document.createElement("canvas"); off.width = META.w; off.height = META.h;
  const oc = off.getContext("2d"); oc.drawImage(d, 0, 0);
  const px = oc.getImageData(0, 0, META.w, META.h).data;
  dayOff = new Int16Array(META.w * META.h).fill(-1);
  for (let i = 0; i < dayOff.length; i++) if (px[4*i+3] > 0) dayOff[i] = px[4*i] + 256 * px[4*i+1];
  overlay = ctx.createImageData(META.w, META.h);
  render(0); drawSpark();
});

function render(day) {
  const o = overlay.data;
  for (let i = 0; i < dayOff.length; i++) {
    const v = dayOff[i], k = 4 * i;
    if (v < 0 || v > day) { o[k+3] = 0; continue; }
    const c = v === day ? NEW : OLD;
    o[k] = c[0]; o[k+1] = c[1]; o[k+2] = c[2]; o[k+3] = 255;
  }
  ctx.drawImage(hsImg, 0, 0);
  const tmp = document.createElement("canvas"); tmp.width = META.w; tmp.height = META.h;
  tmp.getContext("2d").putImageData(overlay, 0, 0);
  ctx.drawImage(tmp, 0, 0);
  ctx.font = "bold 22px -apple-system, system-ui, sans-serif"; ctx.lineWidth = 4;
  for (const [name, [r, c]] of Object.entries(META.towns)) {
    ctx.beginPath(); ctx.arc(c, r, 5, 0, 2 * Math.PI); ctx.fillStyle = "white"; ctx.fill();
    ctx.strokeStyle = "#222"; ctx.lineWidth = 2; ctx.stroke();
    ctx.lineWidth = 5; ctx.strokeStyle = "white"; ctx.strokeText(name, c + 9, r - 7);
    ctx.fillStyle = "#222"; ctx.fillText(name, c + 9, r - 7);
  }
  document.getElementById("date").textContent = fmt(day);
  document.getElementById("stat").textContent =
    `${Math.round(cum[day] * META.cellKha).toLocaleString()} kha burnt so far (+${Math.round(META.daily[day] * META.cellKha).toLocaleString()} today)`;
  drawSpark(day);
}

const spark = document.getElementById("spark"), sctx = spark.getContext("2d");
function drawSpark(day = +slider.value) {
  const w = spark.clientWidth, h = spark.clientHeight, dpr = devicePixelRatio || 1;
  spark.width = w * dpr; spark.height = h * dpr; sctx.scale(dpr, dpr);
  const max = Math.max(...META.daily), bw = w / META.ndays;
  META.daily.forEach((v, i) => {
    const bh = (v / max) * (h - 14);
    sctx.fillStyle = i === day ? "#ff7a1a" : (i < day ? "#b9654f" : "#d9d6d0");
    sctx.fillRect(i * bw + 0.5, h - bh, Math.max(bw - 1, 1), bh);
  });
  sctx.fillStyle = "#666"; sctx.font = "11px -apple-system, system-ui, sans-serif";
  sctx.fillText("New burn per day", 2, 10);
}

slider.addEventListener("input", () => render(+slider.value));
let timer = null;
const playBtn = document.getElementById("play");
function toggle() {
  if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "▶ Play"; return; }
  if (+slider.value >= META.ndays - 1) slider.value = 0;
  playBtn.textContent = "❚❚ Pause";
  timer = setInterval(() => {
    if (+slider.value >= META.ndays - 1) return toggle();
    slider.value = +slider.value + 1; render(+slider.value);
  }, 160);
}
playBtn.addEventListener("click", toggle);
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") { slider.value = Math.min(+slider.value + 1, META.ndays - 1); render(+slider.value); }
  if (e.key === "ArrowLeft") { slider.value = Math.max(+slider.value - 1, 0); render(+slider.value); }
  if (e.key === " ") { e.preventDefault(); toggle(); }
});
canvas.addEventListener("mousemove", e => {
  if (!dayOff) return;
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor((e.clientX - rect.left) / rect.width * META.w);
  const r = Math.floor((e.clientY - rect.top) / rect.height * META.h);
  const v = dayOff[r * META.w + c];
  if (v >= 0 && v !== undefined) {
    tip.style.display = "block";
    tip.style.left = (e.clientX - rect.left + 12) + "px"; tip.style.top = (e.clientY - rect.top + 12) + "px";
    tip.textContent = `Burnt ${fmt(v)}`;
  } else tip.style.display = "none";
});
canvas.addEventListener("mouseleave", () => tip.style.display = "none");
window.addEventListener("resize", () => drawSpark());
</script></body></html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["animate", "map"])
    args = parser.parse_args()
    animate() if args.cmd == "animate" else build_map()
