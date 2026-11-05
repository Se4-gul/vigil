// Vigil · Replay. Real fires from our burn-day labels.
// Each fire is one PNG (Web Mercator): R + 256*G = days since the fire's first
// burn day, A = burnt. The browser colours cells by age on a canvas source.

// Colour by age (days since a cell's burn day): warm only while active and
// recent, then a cool, semi-transparent ash scar. Stops: [age, r, g, b, a].
// Palette C is the default; ?palette=a|b|c switches for comparison.
const PALETTES = {
  a: [[0, 240, 190, 120, 235], [2, 214, 140, 120, 220], [7, 132, 136, 142, 150]],
  b: [[0, 246, 208, 146, 235], [2.5, 196, 124, 132, 210], [8, 116, 125, 136, 140]],
  c: [[0, 236, 178, 106, 235], [3, 186, 118, 86, 215], [9, 146, 148, 150, 150]],
};
const params = new URLSearchParams(location.search);
const STOPS = PALETTES[params.get("palette")] || PALETTES.c;
const LUT_STEP = 0.05, LUT_MAX = 12;
const LUT = (() => {
  const n = Math.ceil(LUT_MAX / LUT_STEP) + 1, out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    const age = i * LUT_STEP;
    let j = 0;
    while (j < STOPS.length - 2 && age > STOPS[j + 1][0]) j++;
    const [a0, ...c0] = STOPS[j], [a1, ...c1] = STOPS[Math.min(j + 1, STOPS.length - 1)];
    let f = a1 > a0 ? (age - a0) / (a1 - a0) : 1;
    f = Math.max(0, Math.min(1, f)); f = f * f * (3 - 2 * f);
    for (let k = 0; k < 4; k++) out[4 * i + k] = c0[k] + (c1[k] - c0[k]) * f;
  }
  return out;
})();
const lutAt = age => 4 * Math.min(Math.floor(Math.max(age, 0) / LUT_STEP), LUT.length / 4 - 1);
const GLOW = [255, 196, 120];
const DAYS_PER_SECOND = 2.2;
const REDUCED_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;

// Basemap: OpenFreeMap "dark" (vector tiles, free, no API key), lifted a step:
// water darkest, land a notch lighter, roads and labels dimmed.
const map = new maplibregl.Map({
  container: "map",
  attributionControl: { compact: true },
  style: "https://tiles.openfreemap.org/styles/dark",
  center: [148.6, -37.3], zoom: 7, maxZoom: 12,
});
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

let firstLabelLayer;
map.on("style.load", () => {
  const set = (id, prop, v) => map.getLayer(id) && map.setPaintProperty(id, prop, v);
  set("background", "background-color", "#26292d");
  set("water", "fill-color", "#0f1215");
  set("waterway", "line-color", "#0f1215");
  for (const id of ["landcover_wood", "landuse_park"]) set(id, "fill-color", "#2a2d31");
  set("landuse_residential", "fill-color", "#2c2f33");
  for (const layer of map.getStyle().layers) {
    if (layer.type === "symbol" && !firstLabelLayer) firstLabelLayer = layer.id;
    if (layer.type === "line" && layer["source-layer"] === "transportation") set(layer.id, "line-opacity", 0.4);
    if (layer.type === "line" && layer["source-layer"] === "boundary") set(layer.id, "line-opacity", 0.55);
    if (layer.type === "symbol" && layer.layout && layer.layout["text-field"]) set(layer.id, "text-opacity", 0.5);
  }
  // Very subtle hillshade: AWS Terrain Tiles (Terrarium; free, no key). The
  // virtual-host URL is the one that sends CORS headers.
  map.addSource("dem", {
    type: "raster-dem", encoding: "terrarium", tileSize: 256, maxzoom: 12,
    tiles: ["https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"],
    attribution: 'Terrain: <a href="https://registry.opendata.aws/terrain-tiles/">Terrain Tiles (AWS Open Data, Mapzen/Tilezen)</a>',
  });
  const firstLine = map.getStyle().layers.find(l => l.type === "line").id;
  map.addLayer({ id: "hillshade", type: "hillshade", source: "dem",
                 paint: { "hillshade-exaggeration": 0.32, "hillshade-shadow-color": "#0b0d0f",
                          "hillshade-highlight-color": "#4c4f54", "hillshade-accent-color": "#0b0d0f" } },
               firstLine);
  init();
});

const $ = id => document.getElementById(id);
const canvas = document.createElement("canvas"), ctx = canvas.getContext("2d");
const activeCanvas = document.createElement("canvas"), actx = activeCanvas.getContext("2d");
const glowCanvas = document.createElement("canvas"), gctx = glowCanvas.getContext("2d");
let fires = [], fire = null, dayOff = null, img = null, activeImg = null;
let t = 0, playing = false, last = 0;

const dateOf = d => { const x = new Date(fire.first + "T00:00:00"); x.setDate(x.getDate() + d); return x; };
const longDate = x => x.toLocaleDateString("en-AU", { day: "numeric", month: "long", year: "numeric" });
const shortDate = x => x.toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });
const smooth = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);

function renderFire() {
  const o = img.data, g = activeImg.data;
  for (let i = 0, n = dayOff.length; i < n; i++) {
    const d = dayOff[i], k = 4 * i;
    g[k + 3] = 0;
    const age = d < 0 ? -1 : t - d;
    if (age <= 0) { o[k + 3] = 0; continue; }
    const li = lutAt(age), ease = age < 1 ? smooth(age) : 1;      // burning today eases in
    o[k] = LUT[li]; o[k + 1] = LUT[li + 1]; o[k + 2] = LUT[li + 2]; o[k + 3] = LUT[li + 3] * ease;
    if (age < 1.5) {                                              // halo on the active front
      g[k] = GLOW[0]; g[k + 1] = GLOW[1]; g[k + 2] = GLOW[2];
      g[k + 3] = 255 * ease * (1 - Math.max(0, age - 0.5));
    }
  }
  ctx.putImageData(img, 0, 0);
  actx.putImageData(activeImg, 0, 0);
  gctx.clearRect(0, 0, glowCanvas.width, glowCanvas.height);
  gctx.filter = "blur(5px)";
  gctx.drawImage(activeCanvas, 0, 0);
  gctx.filter = "none";
}

// Timeline: the scrubber is a bar chart of each day's new burn, coloured with
// the map's own palette (future days faint, past days by age, today amber).
const bars = $("bars"), bctx = bars.getContext("2d");
function renderBars() {
  const dpr = devicePixelRatio || 1, w = bars.clientWidth, h = bars.clientHeight;
  if (bars.width !== w * dpr) { bars.width = w * dpr; bars.height = h * dpr; }
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, w, h);
  const n = fire.days, max = Math.max(...fire.daily_kha), bw = w / n, base = h - 6;
  bctx.fillStyle = "rgba(217, 218, 214, 0.18)";                 // baseline
  bctx.fillRect(0, base, w, 1);
  for (let i = 0; i < n; i++) {
    const v = fire.daily_kha[i], bh = Math.max(v > 0 ? 1.5 : 0, (v / max) * (base - 4));
    const age = t - i;
    if (age <= 0) bctx.fillStyle = "rgba(146, 148, 150, 0.42)";   // future: dim ash, readable before play
    else {
      const li = lutAt(age);
      bctx.fillStyle = `rgba(${LUT[li]},${LUT[li + 1]},${LUT[li + 2]},${Math.max(LUT[li + 3] / 255, 0.55)})`;
    }
    bctx.fillRect(i * bw + 0.5, base - bh, Math.max(bw - 1, 1), bh);
  }
  const x = Math.min(t / n, 1) * w;                               // playhead
  bctx.fillStyle = "#d9dad6";
  bctx.fillRect(Math.round(x) - 0.5, 2, 1, base);
}

function render() {
  if (!dayOff) return;
  renderFire();
  renderBars();
  const day = Math.min(Math.floor(t), fire.days - 1);
  $("date").textContent = longDate(dateOf(day));
  const burnt = fire.daily_kha.slice(0, day + 1).reduce((a, b) => a + b, 0);
  $("stat").textContent = `${Math.round(burnt).toLocaleString()} kha burnt in ${day + 1} ${day ? "days" : "day"}`;
  const s = $("slider");
  s.value = t;
  s.setAttribute("aria-valuetext", `${longDate(dateOf(day))}, ${Math.round(burnt)} thousand hectares burnt`);
}

function frame(ts) {
  if (playing) {
    const dt = last ? Math.min((ts - last) / 1000, 0.1) : 0;
    t = Math.min(t + dt * DAYS_PER_SECOND, fire.days);
    if (t >= fire.days) setPlaying(false);
    render();
  }
  last = ts;
  requestAnimationFrame(frame);
}

function setPlaying(p) {
  if (p && t >= fire.days) t = 0;
  playing = p;
  // A class toggle, not SVG .hidden (SVG elements don't support that property,
  // which is why the icon never changed before).
  $("play").classList.toggle("is-playing", p);
  $("play").setAttribute("aria-label", p ? "Pause" : "Play");
  if (REDUCED_MOTION && p) { t = Math.min(Math.floor(t) + 1, fire.days); render(); setPlaying(false); }
}

async function loadFire(i, { animateView = true } = {}) {
  fire = fires[i];
  const im = new Image();
  im.src = `data/fire_${fire.id}.png`;
  await im.decode();
  for (const c of [canvas, activeCanvas, glowCanvas]) { c.width = im.width; c.height = im.height; }
  ctx.clearRect(0, 0, im.width, im.height);
  ctx.drawImage(im, 0, 0);
  const px = ctx.getImageData(0, 0, im.width, im.height).data;
  dayOff = new Int16Array(im.width * im.height);
  for (let k = 0; k < dayOff.length; k++) dayOff[k] = px[4 * k + 3] ? px[4 * k] + 256 * px[4 * k + 1] : -1;
  img = ctx.createImageData(im.width, im.height);
  activeImg = actx.createImageData(im.width, im.height);
  for (const id of ["fire", "glow"]) if (map.getSource(id)) { map.removeLayer(id); map.removeSource(id); }
  map.addSource("glow", { type: "canvas", canvas: glowCanvas, coordinates: fire.corners, animate: true });
  map.addLayer({ id: "glow", type: "raster", source: "glow",
                 paint: { "raster-resampling": "linear", "raster-opacity": 0.5, "raster-fade-duration": 0 } },
               firstLabelLayer);
  // Linear resampling softens the 375 m cell edges slightly, without
  // inventing detail finer than the data.
  map.addSource("fire", { type: "canvas", canvas, coordinates: fire.corners, animate: true });
  map.addLayer({ id: "fire", type: "raster", source: "fire",
                 paint: { "raster-resampling": "linear", "raster-fade-duration": 0 } }, firstLabelLayer);
  const lons = fire.corners.map(c => c[0]), lats = fire.corners.map(c => c[1]);
  map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
                { padding: clearArea(),
                  duration: animateView && !params.has("day") && !REDUCED_MOTION ? 900 : 0 });
  $("fire-label").textContent = fire.label;
  $("fire-range").textContent = `${shortDate(dateOf(0))} to ${shortDate(dateOf(fire.days - 1))}, ` +
                                `${Math.round(fire.kha).toLocaleString()} kha in total`;
  for (const b of $("fire-list").querySelectorAll("button")) b.setAttribute("aria-selected", +b.dataset.i === i);
  $("slider").max = fire.days;
  t = 0;
  render();
}

// The clear part of the map: below the title block and above the date and
// timeline, measured from the layout so the fire never sits under the text.
function clearArea() {
  const vh = innerHeight, vw = innerWidth, gap = 24;
  const top = document.querySelector(".top").getBoundingClientRect().bottom + gap;
  const bottom = vh - document.querySelector(".readout").getBoundingClientRect().top + gap;
  const side = Math.min(80, vw * 0.06);
  // Keep at least a third of the height for the fire on short screens.
  const spare = vh - top - bottom;
  const shrink = spare < vh / 3 ? (vh / 3 - spare) / 2 : 0;
  return { top: Math.max(top - shrink, 10), bottom: Math.max(bottom - shrink, 10), left: side, right: side };
}

// Fire picker: the fire's name is the control; it opens a plain list.
function openList(open) {
  $("fire-list").hidden = !open;
  $("fire-name").setAttribute("aria-expanded", open);
  if (open) $("fire-list").querySelector('[aria-selected="true"]')?.focus();
}
function buildList() {
  const ul = $("fire-list");
  fires.forEach((f, i) => {
    const li = document.createElement("li"), b = document.createElement("button");
    b.dataset.i = i;
    b.setAttribute("role", "option");
    const when = new Date(f.first + "T00:00:00").toLocaleDateString("en-AU", { month: "short", year: "numeric" });
    b.innerHTML = `<span>${f.label}</span><span class="when">${when}, ${Math.round(f.kha).toLocaleString()} kha</span>`;
    b.onclick = () => { openList(false); setPlaying(false); loadFire(i); $("fire-name").focus(); };
    li.appendChild(b);
    ul.appendChild(li);
  });
}
$("fire-name").onclick = () => openList($("fire-list").hidden);
document.addEventListener("click", e => {
  if (!e.target.closest(".fire")) openList(false);
});

async function init() {
  const data = await (await fetch("data/fires.json")).json();
  fires = data.fires;
  buildList();
  const pick = params.has("fire") ? Math.max(0, fires.findIndex((f, i) => String(f.id) === params.get("fire") || String(i) === params.get("fire"))) : 0;
  await loadFire(pick, { animateView: false });
  if (params.has("day")) { t = params.get("day") === "end" ? fire.days : +params.get("day"); render(); }
  requestAnimationFrame(frame);
  if (params.get("play") === "1") setPlaying(true);
}

$("play").onclick = () => setPlaying(!playing);
$("slider").oninput = e => { t = +e.target.value; render(); };
window.addEventListener("resize", () => fire && renderBars());
document.addEventListener("keydown", e => {
  if (e.key === "Escape") return openList(false);
  // Leave keys to focused controls (buttons, the slider, the list), so space
  // doesn't toggle twice and arrows don't double-step.
  if (e.target.closest("button, input, select, .fire-list")) return;
  if (!fire) return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  if (e.key === "ArrowRight") { t = Math.min(Math.floor(t) + 1, fire.days); render(); }
  if (e.key === "ArrowLeft") { t = Math.max(Math.ceil(t) - 1, 0); render(); }
});
