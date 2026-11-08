// Vigil · Landscape. Years since the land last burnt, as at 1 October of each
// year (before that season's fires), from MODIS MCD64A1 on both sides of the
// border. One greyscale PNG per October (value = years, 10 = 10+), loaded on
// demand; python -m vigil.landscape_export.
//
// Colour: recent burn is a light, cool ash; as fuel grows back it warms and
// darkens towards the land colour, and 10+ years is transparent, so old
// country is just basemap. No amber or red: amber means burning on this site.
const RAMP = [ // [years, r, g, b, alpha 0-1]
  [1, 164, 166, 168, 0.66], [3, 148, 140, 130, 0.54], [5, 132, 116, 100, 0.46],
  [7, 114, 96, 80, 0.32], [9, 96, 82, 70, 0.18],
];
// Fuel groups: four muted earth tones; non-burnable also hatched, because in
// colour alone it sits too close to forest for tritanopes (checked in OKLab).
// Grass and crop is most of Victoria, so it's the quietest; forest the clearest.
const FUEL = { 1: [100, 146, 102, 0.80], 2: [216, 186, 122, 0.80], 3: [168, 156, 116, 0.30], 4: [120, 132, 156, 0.60] };
const OPENING = { year: 2020, bounds: [[146.0, -38.15], [150.15, -35.95]] };   // East Gippsland and the Alps
const STEP_SECONDS = 1.4, FADE_MS = 500;
const REDUCED_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;
const CAVEAT = "From NASA's MODIS burned-area record, which starts in 2002. Cells are about 500 m across, so small fires may be missed. The same record covers both sides of the border, within the study area shown.";
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);

const LUT = (() => {                                   // years 0-10 -> RGBA
  const out = new Uint8ClampedArray(4 * 11);
  for (let v = 1; v <= 9; v++) {
    let j = 0;
    while (j < RAMP.length - 2 && v > RAMP[j + 1][0]) j++;
    const [a0, ...c0] = RAMP[j], [a1, ...c1] = RAMP[j + 1];
    const f = Math.max(0, Math.min(1, (v - a0) / (a1 - a0)));
    for (let k = 0; k < 3; k++) out[4 * v + k] = c0[k] + (c1[k] - c0[k]) * f;
    out[4 * v + 3] = 255 * (c0[3] + (c1[3] - c0[3]) * f);
  }
  return out;                                          // 0 (no data) and 10 (10+) stay transparent
})();

let meta, years = [], yi = 0, fuelOn = false, playing = false, lastStep = 0, front = 0, label;
const cache = new Map();
const reader = document.createElement("canvas"), rctx = reader.getContext("2d", { willReadFrequently: true });
const layers = [0, 1].map(() => { const c = document.createElement("canvas"); return { canvas: c, ctx: c.getContext("2d") }; });
const fuelLayer = (() => { const c = document.createElement("canvas"); return { canvas: c, ctx: c.getContext("2d") }; })();

const map = makeBasemap((m, l) => { label = l; init(); }, { center: [148, -37.1], zoom: 7 });

async function values(name) {
  if (cache.has(name)) return cache.get(name);
  const p = (async () => {
    const im = new Image();
    im.src = `data/landscape/${name}.png`;
    await im.decode();
    reader.width = im.width; reader.height = im.height;
    rctx.drawImage(im, 0, 0);
    return { w: im.width, h: im.height, px: rctx.getImageData(0, 0, im.width, im.height).data };
  })();
  cache.set(name, p);
  return p;
}

// The data stop at the study-area box; fade the last few km so the edge isn't
// a hard line across New South Wales.
const EDGE_PX = 14;
const edgeFade = (x, y, w, h) => Math.min(1, Math.min(x, y, w - 1 - x, h - 1 - y) / EDGE_PX);

function paintTsb(L, { w, h, px }) {
  if (L.canvas.width !== w) { L.canvas.width = w; L.canvas.height = h; }
  const img = L.ctx.createImageData(w, h), o = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = 4 * (y * w + x), v = Math.min(px[i], 10) * 4;
      if (!LUT[v + 3]) continue;
      o[i] = LUT[v]; o[i + 1] = LUT[v + 1]; o[i + 2] = LUT[v + 2];
      o[i + 3] = LUT[v + 3] * (x < EDGE_PX || y < EDGE_PX || x >= w - EDGE_PX || y >= h - EDGE_PX ? edgeFade(x, y, w, h) : 1);
    }
  }
  L.ctx.putImageData(img, 0, 0);
}

function paintFuel({ w, h, px }) {
  const L = fuelLayer;
  L.canvas.width = w; L.canvas.height = h;
  const img = L.ctx.createImageData(w, h), o = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = 4 * (y * w + x), c = FUEL[px[i]];
      if (!c) continue;
      o[i] = c[0]; o[i + 1] = c[1]; o[i + 2] = c[2];
      o[i + 3] = 255 * c[3] * (px[i] === 4 && (x + y) % 4 >= 2 ? 0.35 : 1)   // hatch "not burnable"
                 * (x < EDGE_PX || y < EDGE_PX || x >= w - EDGE_PX || y >= h - EDGE_PX ? edgeFade(x, y, w, h) : 1);
    }
  }
  L.ctx.putImageData(img, 0, 0);
}

function addCanvas(id, canvas, opacity) {
  map.addSource(id, { type: "canvas", canvas, coordinates: meta.corners, animate: false });
  map.addLayer({ id, type: "raster", source: id,
                 paint: { "raster-opacity": opacity, "raster-resampling": "linear", "raster-fade-duration": 0,
                          "raster-opacity-transition": { duration: REDUCED_MOTION ? 0 : FADE_MS } } }, label);
}

function refresh(id) {                                 // canvas sources upload only while "playing"
  const src = map.getSource(id);
  src.play();
  requestAnimationFrame(() => requestAnimationFrame(() => src.pause()));
}

// Cross-fade: draw the new October into the hidden layer, then swap opacities.
async function showYear(i) {
  yi = Math.max(0, Math.min(i, years.length - 1));
  const Y = years[yi], data = await values(`tsb_${Y}`);
  if (years[yi] !== Y) return;
  const back = 1 - front;
  paintTsb(layers[back], data);
  refresh(`tsb${back}`);
  if (!fuelOn) {
    map.setPaintProperty(`tsb${back}`, "raster-opacity", 1);
    map.setPaintProperty(`tsb${front}`, "raster-opacity", 0);
  }
  front = back;
  if (yi + 1 < years.length) values(`tsb_${years[yi + 1]}`);   // prefetch the next October
  readout();
  renderBars();
}

const pct = x => `${(100 * x).toFixed(x < 0.1 ? 1 : 0)}%`;
// The bars are a rolling window: the five fire seasons up to the one that ended
// before this October (e.g. October 2025: 2020–21 to 2024–25). A drop means a
// big season left the window, not that the land has regrown.
const season = s => `${s - 1}–${String(s).slice(2)}`;
const windowText = Y => `the five fire seasons ${season(Y - 4)} to ${season(Y)}`;

function readout() {
  const Y = years[yi], s = meta.share_vic_last5[Y];
  if (fuelOn) {
    const f = meta.fuel_share_vic;
    $("date").textContent = "Fuel types";
    $("stat").textContent = `Victoria is ${pct(f["forest/woodland"])} forest or woodland and ${pct(f["grass/crop"])} grass or crop.`;
  } else {
    $("date").textContent = `October ${Y}`;
    $("stat").textContent = `${pct(s)} of Victoria burnt in ${windowText(Y)}.`;
  }
  $("slider").value = yi;
  $("slider").setAttribute("aria-valuetext", `October ${Y}: ${pct(s)} of Victoria burnt in ${windowText(Y)}`);
}

// ------------------------------------------------------------------ timeline: one bar per October
const bars = $("bars"), bctx = bars.getContext("2d");
function renderBars() {
  const dpr = devicePixelRatio || 1, w = bars.clientWidth, h = bars.clientHeight;
  if (bars.width !== Math.round(w * dpr)) { bars.width = Math.round(w * dpr); bars.height = Math.round(h * dpr); }
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, w, h);
  const n = years.length, bw = w / n, base = h - 16;
  const max = Math.max(...years.map(Y => meta.share_vic_last5[Y]));
  const every = bw < 44 ? 3 : 1;
  bctx.fillStyle = "rgba(217, 218, 214, 0.18)";
  bctx.fillRect(0, base, w, 1);
  bctx.font = `400 11px "Barlow Semi Condensed", sans-serif`;
  bctx.textAlign = "center";
  years.forEach((Y, i) => {
    const v = meta.share_vic_last5[Y], bh = Math.max(1.5, (v / max) * (base - 4));
    const cur = i === yi && !fuelOn;
    bctx.fillStyle = cur ? "rgba(217, 218, 214, 0.95)" : "rgba(146, 148, 150, 0.55)";
    bctx.fillRect(i * bw + bw * 0.18, base - bh, bw * 0.64, bh);
    if (i % every === 0 || cur) {
      bctx.fillStyle = cur ? "#d9dad6" : "#6f7275";
      bctx.fillText(String(Y), i * bw + bw / 2, h - 2);
    }
  });
}

function hover(e) {
  const r = bars.getBoundingClientRect(), i = Math.min(years.length - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * years.length)));
  const Y = years[i], tip = $("hover");
  tip.textContent = `October ${Y}: ${pct(meta.share_vic_last5[Y])} burnt in ${windowText(Y)}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(Math.max((i + 0.5) / years.length * r.width, 190), r.width - 190)}px`;
}

// ------------------------------------------------------------------ controls
function setFuel(on) {
  fuelOn = on;
  $("fuel").checked = on;
  $("title").textContent = on ? "What grows there" : "How long since the land last burnt";
  $("key-tsb").hidden = on;
  $("key-fuel").hidden = !on;
  $("fuel-credit").hidden = !on;
  $("caveat").textContent = on
    ? "CSIRO's national bushfire fuel map (2025), grouped into four types; each cell shows its most common type."
    : CAVEAT;
  document.querySelector(".timeline").classList.toggle("is-off", on);
  $("slider").disabled = on;
  $("play").disabled = on;
  if (on) setPlaying(false);
  map.setPaintProperty("fuel", "raster-opacity", on ? 1 : 0);
  map.setPaintProperty(`tsb${front}`, "raster-opacity", on ? 0 : 1);
  readout();
  renderBars();
}

function setPlaying(p) {
  playing = p;
  $("play").classList.toggle("is-playing", p);
  $("play").setAttribute("aria-label", p ? "Pause" : "Play");
  if (p && yi >= years.length - 1) showYear(0);
  lastStep = performance.now();
  if (REDUCED_MOTION && p) { showYear(yi + 1); setPlaying(false); }
}

function frame(ts) {
  if (playing && ts - lastStep > STEP_SECONDS * 1000) {
    lastStep = ts;
    if (yi >= years.length - 1) setPlaying(false); else showYear(yi + 1);
  }
  requestAnimationFrame(frame);
}

// Legend in plain language: four bands, each swatch a small gradient of the
// colours inside that band (the map itself steps season by season).
const BANDS = [[1, 1, "burnt last season"], [2, 3, "2–3 seasons ago"], [4, 9, "4–9 seasons ago"], [10, 10, "10+ seasons, or not in the record"]];
function paintKey() {
  const col = v => `rgba(${LUT[4 * v]},${LUT[4 * v + 1]},${LUT[4 * v + 2]},${(LUT[4 * v + 3] / 255).toFixed(2)})`;
  for (const [lo, hi, text] of BANDS) {
    const span = document.createElement("span"), sw = document.createElement("i");
    sw.className = "sw band";
    if (lo === 10) sw.classList.add("none");
    else sw.style.background = lo === hi ? col(lo)
      : `linear-gradient(90deg, ${Array.from({ length: hi - lo + 1 }, (_, j) => col(lo + j)).join(", ")})`;
    span.append(sw, text);
    $("key-tsb").appendChild(span);
  }
}

// Wide screens: the text column owns the left, so the landscape goes to its right.
function framing() {
  const vw = innerWidth, vh = innerHeight;
  const textRight = Math.max(...[".top", ".readout"].map(q => document.querySelector(q).getBoundingClientRect().right));
  if (vw - textRight > 480) {
    const bottom = vh - document.querySelector(".timeline").getBoundingClientRect().top + 24;
    return { left: textRight + 30, right: 40, top: 40, bottom };
  }
  return clearArea();
}

async function init() {
  meta = await (await fetch("data/landscape/meta.json")).json();
  years = meta.years;
  $("slider").max = years.length - 1;
  addCanvas("tsb0", layers[0].canvas, 1);
  addCanvas("tsb1", layers[1].canvas, 0);
  paintFuel(await values("fuel"));
  addCanvas("fuel", fuelLayer.canvas, 0);
  refresh("fuel");
  paintKey();
  $("what").querySelector("p").textContent = $("explainer").textContent;   // same words, tap to open on short phones
  const want = years.indexOf(+params.get("year"));
  await document.fonts.ready;
  map.fitBounds(OPENING.bounds, { padding: framing(), duration: 0 });
  await showYear(want >= 0 ? want : years.indexOf(OPENING.year));
  if (params.get("fuel") === "1") setFuel(true);
  requestAnimationFrame(frame);
}

$("play").onclick = () => setPlaying(!playing);
$("slider").oninput = e => { setPlaying(false); showYear(+e.target.value); };
$("fuel").onchange = e => setFuel(e.target.checked);
const track = document.querySelector(".track");
track.addEventListener("pointermove", e => { if (!fuelOn) hover(e); });
track.addEventListener("pointerleave", () => { $("hover").hidden = true; });
window.addEventListener("resize", () => meta && renderBars());
document.addEventListener("keydown", e => {
  if (e.target.closest("button, input, select")) return;
  if (!meta || fuelOn) return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  if (e.key === "ArrowRight") showYear(yi + 1);
  if (e.key === "ArrowLeft") showYear(yi - 1);
});
