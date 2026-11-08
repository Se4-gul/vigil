// Vigil · Compare. For a fire and a day, what each model predicted for the
// next day against what actually burnt. Every prediction is out of sample:
// test fires by the final models, other seasons by the fold model that never
// trained on that season. Data: app/data/compare/ (python -m vigil.compare_export).
// Each fire-day is one RGB PNG: R = main model probability x 255, G = flags.
const BIT = { pos: 1, burnt: 2, unscored: 4, main: 8, unet: 16, baseline: 32, physics: 64, outside: 128 };
const MODEL_ORDER = ["main", "unet", "baseline", "physics"];
const MODEL_NOTE = {
  main: "U-Net with the physics model as an input, no raw weather",
  unet: "U-Net with all inputs, including raw weather",
  baseline: "Active fire pushed downwind; the best simple rule",
  physics: "Cellular automaton with CSIRO spread models",
};
// Palette C. Warm = it burnt (hit amber, missed copper); the one cool colour is
// a prediction that didn't happen. Burnt earlier is the Replay's ash scar; not
// scored is a dark hatched veil. Checked in OKLab: the three classes stay
// >= 13 apart under simulated colour blindness.
const C = {
  hit: [236, 178, 106, 242], missed: [186, 118, 86, 235], false: [159, 180, 196, 217],
  ash: [146, 148, 150, 77], veil: [15, 18, 21, 150], outline: [235, 236, 232, 235],
};
const STEP_SECONDS = 0.7;
const REDUCED_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);

let firstLabelLayer, meta, fires = [], fire = null, k = 0, model = "main", probMode = false;
let playing = false, lastStep = 0, loadToken = 0;
const cache = new Map();
const canvas = document.createElement("canvas"), ctx = canvas.getContext("2d");
const reader = document.createElement("canvas"), rctx = reader.getContext("2d", { willReadFrequently: true });

const map = makeBasemap((m, label) => { firstLabelLayer = label; init(); });

// ------------------------------------------------------------------ helpers
const parseDate = s => new Date(s + "T00:00:00");
const longDate = s => parseDate(s).toLocaleDateString("en-AU", { day: "numeric", month: "long", year: "numeric" });
const shortDate = s => parseDate(s).toLocaleDateString("en-AU", { day: "numeric", month: "short" });
const dayDiff = (a, b) => Math.round((parseDate(b) - parseDate(a)) / 86400000);
const fmt = x => x == null ? "–" : x.toFixed(2);
const n = x => x.toLocaleString("en-AU");
const f1Of = ([tp, fp, fn]) => tp + fp + fn === 0 ? null : 2 * tp / (2 * tp + fp + fn);
const seasonName = s => `${s - 1}–${String(s).slice(2)}`;

// ------------------------------------------------------------------ data
async function dayPixels(f, i) {
  const key = `${f.id}/${i}`;
  if (cache.has(key)) return cache.get(key);
  const p = (async () => {
    const im = new Image();
    im.src = `data/compare/${f.id}/${i}.png`;
    await im.decode();
    reader.width = im.width; reader.height = im.height;
    rctx.drawImage(im, 0, 0);
    return { w: im.width, h: im.height, px: rctx.getImageData(0, 0, im.width, im.height).data };
  })();
  cache.set(key, p);
  if (cache.size > 60) cache.delete(cache.keys().next().value);
  return p;
}

// ------------------------------------------------------------------ map drawing
function paint({ w, h, px }) {
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const out = ctx.createImageData(w, h), o = out.data, bit = BIT[model];
  const put = (i, c, a = c[3]) => { o[i] = c[0]; o[i + 1] = c[1]; o[i + 2] = c[2]; o[i + 3] = a; };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = 4 * (y * w + x), g = px[i + 1];
      if (g & BIT.outside) continue;
      if (g & BIT.burnt) { put(i, C.ash); continue; }
      if (g & BIT.unscored) { if ((x + y) % 4 < 2) put(i, C.veil); else put(i, C.veil, 60); continue; }
      const pos = g & BIT.pos;
      if (probMode) {
        const p = px[i] / 255;
        if (pos && isEdge(px, w, h, x, y)) { put(i, C.outline); continue; }
        if (p > 0.02) put(i, C.hit, Math.round(235 * Math.min(1, Math.pow(p, 0.75))));
        continue;
      }
      const pred = g & bit;
      if (pos && pred) put(i, C.hit);
      else if (pos) put(i, C.missed);
      else if (pred) put(i, C.false);
    }
  }
  ctx.putImageData(out, 0, 0);
}

function isEdge(px, w, h, x, y) {
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || yy < 0 || xx >= w || yy >= h) return true;
    if (!(px[4 * (yy * w + xx) + 1] & BIT.pos)) return true;
  }
  return false;
}

async function show() {
  const token = ++loadToken, f = fire, i = k;
  const day = await dayPixels(f, i);
  if (token !== loadToken) return;
  paint(day);
  let src = map.getSource("pred");
  if (!src) {
    // Not animated: the canvas only changes with the day, so repaint on demand.
    map.addSource("pred", { type: "canvas", canvas, coordinates: f.corners, animate: false });
    map.addLayer({ id: "pred", type: "raster", source: "pred",
                   paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 } }, firstLabelLayer);
    src = map.getSource("pred");
  }
  src.play();
  requestAnimationFrame(() => requestAnimationFrame(() => src.pause()));
  for (const j of [i + 1, i - 1]) if (j >= 0 && j < f.days.length) dayPixels(f, j);   // prefetch
  readout();
  renderBars();
}

// ------------------------------------------------------------------ text
function readout() {
  const date = fire.days[k], c = fire.counts[model][k], [tp, fp, fn] = c, f1 = f1Of(c);
  $("date").textContent = longDate(date);
  let s;
  if (tp + fn === 0 && fp === 0) s = "No new burn this day, and none predicted.";
  else if (tp + fn === 0) s = `No new burn this day; ${n(fp)} cells predicted that didn't burn.`;
  else s = `F1 ${fmt(f1)} this day: ${n(tp)} cells hit, ${n(fn)} missed, ${n(fp)} false alarms.`;
  const whole = fire.f1[model];
  $("stat").textContent = `${s} Whole fire ${fmt(whole)}.`;
  $("slider").setAttribute("aria-valuetext", `${longDate(date)}. ${s}`);
  $("slider").value = dayDiff(fire.days[0], date);
}

function kindLine(f) {
  if (f.kind === "test")
    return `${seasonName(f.season)} test fire, predicted by models that never saw this season. ` +
           "Whole-fire scores average three training runs; the map shows one.";
  return `${seasonName(f.season)} fire, predicted by a model trained without this season.`;
}

// ------------------------------------------------------------------ timeline: new burn above, F1 below
const bars = $("bars"), bctx = bars.getContext("2d");
function renderBars() {
  const dpr = devicePixelRatio || 1, w = bars.clientWidth, h = bars.clientHeight;
  if (bars.width !== Math.round(w * dpr)) { bars.width = Math.round(w * dpr); bars.height = Math.round(h * dpr); }
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, w, h);
  const span = dayDiff(fire.days[0], fire.days.at(-1)) + 1, bw = w / span;
  const topBase = 30, f1Top = 36, f1Base = h - 2, max = Math.max(...fire.new_burn, 1);
  bctx.fillStyle = "rgba(217, 218, 214, 0.18)";
  bctx.fillRect(0, topBase, w, 1);
  bctx.fillRect(0, f1Base, w, 1);
  fire.days.forEach((d, i) => {
    const x = dayDiff(fire.days[0], d) * bw, cur = i === k;
    const v = fire.new_burn[i], bh = Math.max(v > 0 ? 1.5 : 0, (v / max) * (topBase - 3));
    bctx.fillStyle = cur ? "rgba(236, 178, 106, 0.95)" : "rgba(146, 148, 150, 0.55)";
    bctx.fillRect(x + 0.5, topBase - bh, Math.max(bw - 1, 1), bh);
    const f1 = f1Of(fire.counts[model][i]);
    if (f1 == null) return;                                  // no new burn, nothing predicted
    const fh = Math.max(1, f1 * (f1Base - f1Top));
    bctx.fillStyle = cur ? "rgba(236, 178, 106, 1)" : "rgba(236, 178, 106, 0.55)";
    bctx.fillRect(x + 0.5, f1Base - fh, Math.max(bw - 1, 1), fh);
  });
  const x = (dayDiff(fire.days[0], fire.days[k]) + 0.5) * bw;   // playhead
  bctx.fillStyle = "#d9dad6";
  bctx.fillRect(Math.round(x) - 0.5, 0, 1, h);
}

function nearestIndex(offset) {
  let best = 0, bd = Infinity;
  fire.days.forEach((d, i) => { const dd = Math.abs(dayDiff(fire.days[0], d) - offset); if (dd < bd) { bd = dd; best = i; } });
  return best;
}

function hover(e) {
  const r = bars.getBoundingClientRect(), span = dayDiff(fire.days[0], fire.days.at(-1)) + 1;
  const i = nearestIndex(Math.floor((e.clientX - r.left) / r.width * span));
  const c = fire.counts[model][i], f1 = f1Of(c), tip = $("hover");
  tip.textContent = `${shortDate(fire.days[i])}: ${f1 == null ? "no new burn" : "F1 " + fmt(f1)}, ` +
                    `${n(fire.new_burn[i])} cells burnt`;
  tip.hidden = false;
  const x = (dayDiff(fire.days[0], fire.days[i]) + 0.5) / span * r.width;
  tip.style.left = `${Math.min(Math.max(x, 60), r.width - 60)}px`;
}

// ------------------------------------------------------------------ controls
function setDay(i) { k = Math.max(0, Math.min(i, fire.days.length - 1)); show(); }

function setModel(m) {
  model = m;
  if (m !== "main") setProb(false);
  $("prob").disabled = m !== "main";
  for (const b of $("model-list").children) b.setAttribute("aria-checked", b.dataset.m === m);
  buildList();
  show();
}

function setProb(p) {
  probMode = p;
  $("prob").checked = p;
  $("key-classes").hidden = p;
  $("key-prob").hidden = !p;
  $("explain").textContent = p
    ? "Brighter amber means the main model gave that cell a higher chance of burning the next day. The light outline is what actually burnt."
    : "A hit burnt where the model said it would. A miss burnt without warning. A false alarm was predicted but didn't burn.";
}

function setPlaying(p) {
  playing = p;
  $("play").classList.toggle("is-playing", p);
  $("play").setAttribute("aria-label", p ? "Pause" : "Play");
  if (p && k >= fire.days.length - 1) setDay(0);
  if (REDUCED_MOTION && p) { setDay(k + 1); setPlaying(false); }
}

function frame(ts) {
  if (playing && ts - lastStep > STEP_SECONDS * 1000) {
    lastStep = ts;
    if (k >= fire.days.length - 1) setPlaying(false); else setDay(k + 1);
  }
  requestAnimationFrame(frame);
}

async function loadFire(i, day) {
  fire = fires[i];
  cache.clear();
  if (map.getSource("pred")) { map.removeLayer("pred"); map.removeSource("pred"); }
  $("fire-label").textContent = fire.label;
  $("fire-kind").textContent = kindLine(fire);
  $("slider").max = dayDiff(fire.days[0], fire.days.at(-1));
  buildList();
  // Default day: the fire's biggest run, so the page opens on something to compare.
  k = day ?? fire.new_burn.indexOf(Math.max(...fire.new_burn));
  await show();
  await document.fonts.ready;      // frame after the text is set: its width decides where the fire goes
  fit();
}

const fit = () => map.fitBounds(fire.focus, { padding: framing(), duration: 0, maxZoom: 10.5 });  // what burnt

// On wide screens the text column owns the left side, so the fire goes in the
// space to its right; on narrow ones, between the title block and the readout.
function framing() {
  const vw = innerWidth, vh = innerHeight;
  const textRight = Math.max(...[".top", ".readout"].map(s => document.querySelector(s).getBoundingClientRect().right));
  if (vw - textRight > 420) {
    const bottom = vh - document.querySelector(".timeline").getBoundingClientRect().top + 40;
    return { left: textRight + 40, right: 60, top: 60, bottom };
  }
  return clearArea();
}

function buildList() {
  const box = $("fire-list");
  box.textContent = "";
  for (const [kind, title] of [["test", "2019–20 test fires"], ["cv", "Other seasons, each held out of training"]]) {
    const h = document.createElement("p");
    h.className = "group";
    h.textContent = title;
    box.appendChild(h);
    fires.forEach((f, i) => {
      if (f.kind !== kind) return;
      const b = document.createElement("button");
      b.setAttribute("role", "option");
      b.setAttribute("aria-selected", f === fire);
      const when = parseDate(f.days[0]).toLocaleDateString("en-AU", { month: "short", year: "numeric" });
      b.innerHTML = `<span>${f.label}</span><span class="when">${when}</span>` +
                    `<span class="score">F1 ${fmt(f.f1[model])}</span>`;
      b.onclick = () => { openList(false); setPlaying(false); loadFire(i); $("fire-name").focus(); };
      box.appendChild(b);
    });
  }
}

function openList(open) {
  $("fire-list").hidden = !open;
  $("fire-name").setAttribute("aria-expanded", open);
  if (open) $("fire-list").querySelector('[aria-selected="true"]')?.focus();
}

function buildModels() {
  const box = $("model-list");
  for (const m of MODEL_ORDER) {
    const b = document.createElement("button");
    b.dataset.m = m;
    b.setAttribute("role", "radio");
    b.title = MODEL_NOTE[m];
    b.textContent = meta.models[m];
    b.onclick = () => setModel(m);
    box.appendChild(b);
  }
}

async function init() {
  meta = await (await fetch("data/compare/fires.json")).json();
  fires = meta.fires;
  buildModels();
  const pick = Math.max(0, fires.findIndex(f => String(f.id) === params.get("fire")));
  if (MODEL_ORDER.includes(params.get("model"))) model = params.get("model");
  setModel(model);
  setProb(params.get("prob") === "1" && model === "main");
  const f = fires[pick];
  const day = params.has("day") ? f.days.indexOf(params.get("day")) : -1;
  await loadFire(pick, day >= 0 ? day : undefined);
  requestAnimationFrame(frame);
}

$("fire-name").onclick = () => openList($("fire-list").hidden);
document.addEventListener("click", e => { if (!e.target.closest(".fire")) openList(false); });
$("prob").onchange = e => { setProb(e.target.checked); show(); };
$("play").onclick = () => setPlaying(!playing);
$("slider").oninput = e => setDay(nearestIndex(+e.target.value));
const track = document.querySelector(".track");
track.addEventListener("pointermove", hover);
track.addEventListener("pointerleave", () => { $("hover").hidden = true; });
window.addEventListener("resize", () => { if (fire) { renderBars(); fit(); } });
document.addEventListener("keydown", e => {
  if (e.key === "Escape") return openList(false);
  if (e.target.closest("button, input, select, .fire-list")) return;
  if (!fire) return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  if (e.key === "ArrowRight") setDay(k + 1);
  if (e.key === "ArrowLeft") setDay(k - 1);
});
