// Vigil v0: replay of real fires from burn-day labels.
// Each fire is one PNG (Web Mercator): R + 256*G = days since the fire's first
// burn day, A = burnt. The browser colours cells by age on a canvas source:
// burning cells ease in (soft amber), then fade to muted embers over a few days.

// Colour by age (days since the cell's burn day). Warm only while active and
// recent; then a cool, desaturated ash a little lighter than land (a scar).
// Stops: [age, r, g, b, alpha 0-255]. Pick with ?palette=a|b|c.
const PALETTES = {
  a: { name: "Amber → rose → ash",
       stops: [[0, 240, 190, 120, 235], [2, 214, 140, 120, 220], [7, 132, 136, 142, 150]] },
  b: { name: "Pale gold → dusty rose → slate ash",
       stops: [[0, 246, 208, 146, 235], [2.5, 196, 124, 132, 210], [8, 116, 125, 136, 140]] },
  c: { name: "Amber → copper → light ash",
       stops: [[0, 236, 178, 106, 235], [3, 186, 118, 86, 215], [9, 146, 148, 150, 155]] },
};
const PALETTE = PALETTES[new URLSearchParams(location.search).get("palette")] || PALETTES.a;
const LUT_STEP = 0.05, LUT_MAX = 12;
const LUT = (() => {                      // age -> rgba, precomputed
  const n = Math.ceil(LUT_MAX / LUT_STEP) + 1, out = new Uint8ClampedArray(4 * n), st = PALETTE.stops;
  for (let i = 0; i < n; i++) {
    const age = i * LUT_STEP;
    let j = 0;
    while (j < st.length - 2 && age > st[j + 1][0]) j++;
    const [a0, ...c0] = st[j], [a1, ...c1] = st[Math.min(j + 1, st.length - 1)];
    let f = a1 > a0 ? (age - a0) / (a1 - a0) : 1;
    f = Math.max(0, Math.min(1, f)); f = f * f * (3 - 2 * f);
    for (let k = 0; k < 4; k++) out[4 * i + k] = c0[k] + (c1[k] - c0[k]) * f;
  }
  return out;
})();
const GLOW = [255, 196, 120];             // halo on the active front
const DAYS_PER_SECOND = 2.2;              // playback speed

// Basemap: OpenFreeMap "dark" (vector tiles, free, no API key; OpenStreetMap
// data, attribution comes with the style). Muted further below.
const map = new maplibregl.Map({
  container: "map",
  attributionControl: { compact: true },
  style: "https://tiles.openfreemap.org/styles/dark",
  center: [148.6, -37.3], zoom: 7, maxZoom: 12,
});
let firstLabelLayer;
map.on("style.load", () => {
  const set = (id, prop, v) => map.getLayer(id) && map.setPaintProperty(id, prop, v);
  set("background", "background-color", "rgb(25,26,28)");           // land: a step lighter
  set("water", "fill-color", "rgb(9,10,12)");                       // water: darkest
  set("waterway", "line-color", "rgb(9,10,12)");
  for (const id of ["landcover_wood", "landuse_park"]) set(id, "fill-color", "rgb(28,29,31)");
  set("landuse_residential", "fill-color", "rgb(30,30,32)");
  for (const layer of map.getStyle().layers) {
    if (layer.type === "symbol" && !firstLabelLayer) firstLabelLayer = layer.id;   // fire goes under labels
    if (layer.type === "line" && layer["source-layer"] === "transportation") set(layer.id, "line-opacity", 0.45);
    if (layer.type === "line" && layer["source-layer"] === "boundary") set(layer.id, "line-opacity", 0.6);
    if (layer.type === "symbol" && layer.layout && layer.layout["text-field"]) set(layer.id, "text-opacity", 0.55);
  }
  // Very subtle hillshade: AWS Terrain Tiles (Terrarium encoding; free, no key).
  map.addSource("dem", {
    type: "raster-dem", encoding: "terrarium", tileSize: 256, maxzoom: 12,
    tiles: ["https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"],
    attribution: 'Terrain: <a href="https://registry.opendata.aws/terrain-tiles/">Terrain Tiles (AWS Open Data, Mapzen/Tilezen)</a>',
  });
  const firstLine = map.getStyle().layers.find(l => l.type === "line").id;
  map.addLayer({ id: "hillshade", type: "hillshade", source: "dem",
                 paint: { "hillshade-exaggeration": 0.35, "hillshade-shadow-color": "#000000",
                          "hillshade-highlight-color": "#46484c", "hillshade-accent-color": "#000000" } },
               firstLine);
  init();   // don't wait for every tile (the 'load' event)
});
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

const $ = id => document.getElementById(id);
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
const activeCanvas = document.createElement("canvas"), actx = activeCanvas.getContext("2d");
const glowCanvas = document.createElement("canvas"), gctx = glowCanvas.getContext("2d");
let activeImg = null;
let fire = null, dayOff = null, img = null, t = 0, playing = false, last = 0;

const fmtDate = (first, d) => {
  const x = new Date(first + "T00:00:00");
  x.setDate(x.getDate() + Math.floor(d));
  return x.toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });
};
const smooth = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);

function render() {
  if (!dayOff) return;
  const o = img.data, g = activeImg.data;
  for (let i = 0, n = dayOff.length; i < n; i++) {
    const d = dayOff[i], k = 4 * i;
    g[k + 3] = 0;
    if (d < 0) { o[k + 3] = 0; continue; }
    const age = t - d;
    if (age <= 0) { o[k + 3] = 0; continue; }
    const li = 4 * Math.min(Math.floor(age / LUT_STEP), LUT.length / 4 - 1);
    const ease = age < 1 ? smooth(age) : 1;           // burning today: ease in
    o[k] = LUT[li]; o[k + 1] = LUT[li + 1]; o[k + 2] = LUT[li + 2]; o[k + 3] = LUT[li + 3] * ease;
    if (age < 1.5) {                                  // halo source: the active front
      g[k] = GLOW[0]; g[k + 1] = GLOW[1]; g[k + 2] = GLOW[2];
      g[k + 3] = 255 * ease * (1 - Math.max(0, age - 0.5));
    }
  }
  actx.putImageData(activeImg, 0, 0);
  gctx.clearRect(0, 0, glowCanvas.width, glowCanvas.height);
  gctx.filter = "blur(5px)";
  gctx.drawImage(activeCanvas, 0, 0);
  gctx.filter = "none";
  ctx.putImageData(img, 0, 0);
  const day = Math.min(Math.floor(t), fire.days - 1);
  $("date").textContent = fmtDate(fire.first, day);
  const burnt = fire.daily_kha.slice(0, day + 1).reduce((a, b) => a + b, 0);
  $("stat").textContent = `${Math.round(burnt).toLocaleString()} kha burnt · day ${day + 1} of ${fire.days}`;
  $("slider").value = t;
}

function frame(ts) {
  if (playing) {
    const dt = last ? (ts - last) / 1000 : 0;
    t = Math.min(t + dt * DAYS_PER_SECOND, fire.days);
    if (t >= fire.days) setPlaying(false);
    render();
  }
  last = ts;
  requestAnimationFrame(frame);
}

function setPlaying(p) {
  playing = p;
  if (p && t >= fire.days) t = 0;
  $("icon-play").hidden = p; $("icon-pause").hidden = !p;
  $("play").setAttribute("aria-label", p ? "Pause" : "Play");
}

async function loadFire(f) {
  fire = f;
  const im = new Image();
  im.src = `data/fire_${f.id}.png`;
  await im.decode();
  canvas.width = im.width; canvas.height = im.height;
  ctx.clearRect(0, 0, im.width, im.height);
  ctx.drawImage(im, 0, 0);
  const px = ctx.getImageData(0, 0, im.width, im.height).data;
  dayOff = new Int16Array(im.width * im.height);
  for (let i = 0; i < dayOff.length; i++) dayOff[i] = px[4 * i + 3] ? px[4 * i] + 256 * px[4 * i + 1] : -1;
  img = ctx.createImageData(im.width, im.height);
  for (const c of [activeCanvas, glowCanvas]) { c.width = im.width; c.height = im.height; }
  activeImg = actx.createImageData(im.width, im.height);
  $("slider").max = f.days;
  t = 0;
  for (const id of ["fire", "glow"]) if (map.getSource(id)) { map.removeLayer(id); map.removeSource(id); }
  map.addSource("glow", { type: "canvas", canvas: glowCanvas, coordinates: f.corners, animate: true });
  map.addLayer({ id: "glow", type: "raster", source: "glow",
                 paint: { "raster-resampling": "linear", "raster-opacity": 0.55, "raster-fade-duration": 0 } },
               firstLabelLayer);
  map.addSource("fire", { type: "canvas", canvas, coordinates: f.corners, animate: true });
  map.addLayer({ id: "fire", type: "raster", source: "fire",
                 paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 } }, firstLabelLayer);
  const lons = f.corners.map(c => c[0]), lats = f.corners.map(c => c[1]);
  map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
                { padding: { top: 60, bottom: 110, left: 340, right: 60 },
                  duration: new URLSearchParams(location.search).has("day") ? 0 : 900 });
  render();
}

async function init() {
  const data = await (await fetch("data/fires.json")).json();
  $("season").textContent = `${data.season} fire season · replay`;
  const sel = $("fire");
  data.fires.forEach((f, i) => {
    const o = document.createElement("option");
    const month = new Date(f.first + "T00:00:00").toLocaleDateString("en-AU", { month: "short", year: "numeric" });
    o.value = i; o.textContent = `${f.label} · ${month} · ${Math.round(f.kha).toLocaleString()} kha`;
    sel.appendChild(o);
  });
  sel.onchange = () => { setPlaying(false); loadFire(data.fires[+sel.value]); };
  await loadFire(data.fires[0]);
  requestAnimationFrame(frame);
  const params = new URLSearchParams(location.search);
  if (params.has("day")) { t = params.get("day") === "end" ? fire.days : +params.get("day"); render(); }
  if (params.get("play") === "1") setPlaying(true);
}

$("play").onclick = () => setPlaying(!playing);
$("slider").oninput = e => { t = +e.target.value; render(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  if (e.key === "ArrowRight") { t = Math.min(Math.floor(t) + 1, fire.days); render(); }
  if (e.key === "ArrowLeft") { t = Math.max(Math.ceil(t) - 1, 0); render(); }
});
