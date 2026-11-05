// Vigil v0: replay of real fires from burn-day labels.
// Each fire is one PNG (Web Mercator): R + 256*G = days since the fire's first
// burn day, A = burnt. The browser colours cells by age on a canvas source:
// burning cells ease in (soft amber), then fade to muted embers over a few days.

const AMBER = [239, 184, 115];
const EMBER = [122, 90, 76];
const FADE_DAYS = 6;          // amber -> ember
const DAYS_PER_SECOND = 2.2;  // playback speed

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
  for (const layer of map.getStyle().layers) {
    if (layer.type === "symbol" && !firstLabelLayer) firstLabelLayer = layer.id;   // fire goes under labels
    if (layer.type === "symbol" && layer.layout && layer.layout["text-field"]) {
      map.setPaintProperty(layer.id, "text-opacity", 0.7);
    }
  }
  init();   // don't wait for every tile (the 'load' event)
});
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

const $ = id => document.getElementById(id);
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
let fire = null, dayOff = null, img = null, t = 0, playing = false, last = 0;

const fmtDate = (first, d) => {
  const x = new Date(first + "T00:00:00");
  x.setDate(x.getDate() + Math.floor(d));
  return x.toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });
};
const smooth = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);

function render() {
  if (!dayOff) return;
  const o = img.data;
  for (let i = 0, n = dayOff.length; i < n; i++) {
    const d = dayOff[i], k = 4 * i;
    if (d < 0) { o[k + 3] = 0; continue; }
    const age = t - d;
    if (age <= 0) { o[k + 3] = 0; continue; }
    if (age < 1) {                       // burning today: ease in
      const a = smooth(age);
      o[k] = AMBER[0]; o[k + 1] = AMBER[1]; o[k + 2] = AMBER[2]; o[k + 3] = 235 * a;
    } else {                             // then fade to embers
      const f = smooth((age - 1) / FADE_DAYS);
      o[k] = AMBER[0] + (EMBER[0] - AMBER[0]) * f;
      o[k + 1] = AMBER[1] + (EMBER[1] - AMBER[1]) * f;
      o[k + 2] = AMBER[2] + (EMBER[2] - AMBER[2]) * f;
      o[k + 3] = 235 - 95 * f;
    }
  }
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
  $("slider").max = f.days;
  t = 0;
  const src = map.getSource("fire");
  if (src) map.removeLayer("fire"), map.removeSource("fire");
  map.addSource("fire", { type: "canvas", canvas, coordinates: f.corners, animate: true });
  map.addLayer({ id: "fire", type: "raster", source: "fire",
                 paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 } }, firstLabelLayer);
  const lons = f.corners.map(c => c[0]), lats = f.corners.map(c => c[1]);
  map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
                { padding: { top: 60, bottom: 110, left: 340, right: 60 }, duration: 900 });
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
  if (params.has("day")) { t = +params.get("day"); render(); }
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
