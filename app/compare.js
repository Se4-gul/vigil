// Vigil · Compare. Each day the model sees where the fire is and guesses where
// it will burn tomorrow. Two synced maps: the main model against a comparison
// (the best simple rule by default). Every prediction is out of sample: test
// fires by the final models, other seasons by the fold model that never
// trained on that season. Data: app/data/compare/ (python -m vigil.compare_export).
// Each fire-day is one RGB PNG: R = main model probability x 255, G = flags.
const BIT = { pos: 1, burnt: 2, unscored: 4, main: 8, unet: 16, baseline: 32, physics: 64, outside: 128 };
const NAMES = { main: "Main model", baseline: "Simple rule", unet: "Plain U-Net", physics: "Physics model" };
// Palette C. Simple view: what burnt is a light outline, the guess an amber
// fill, so a right guess is fill inside the outline. Detail view: warm = it
// burnt (hit amber, missed copper), the one cool colour = guessed but didn't
// burn. Burnt before is the Replay's ash; not scored a hatched dark veil.
const C = {
  guess: [236, 178, 106, 205], outline: [240, 241, 236, 255],
  hit: [236, 178, 106, 242], missed: [186, 118, 86, 235], false: [159, 180, 196, 217],
  ash: [146, 148, 150, 70], veil: [15, 18, 21, 150],
};
const DEFAULT = { fire: 26900, day: "2020-01-04" };   // main model clearly ahead of the simple rule
const STEP_SECONDS = 0.8;
const REDUCED_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);

let meta, fires = [], fire = null, k = 0, modelB = "baseline", detail = false;
let playing = false, lastStep = 0, loadToken = 0, ready = 0;
const cache = new Map();
const reader = document.createElement("canvas"), rctx = reader.getContext("2d", { willReadFrequently: true });

// ------------------------------------------------------------------ two synced maps
function panel(container, model) {
  const canvas = document.createElement("canvas");
  const p = { canvas, ctx: canvas.getContext("2d"), model, label: null };
  p.map = makeBasemap((m, label) => { p.label = label; if (++ready === 2) init(); },
                      undefined, { container, scale: container === "map-b", attribution: false });
  return p;
}
const A = panel("map-a", "main"), B = panel("map-b", "baseline");
let syncing = false;
for (const [src, dst] of [[A, B], [B, A]]) {
  src.map.on("move", () => {
    if (syncing) return;
    syncing = true;
    dst.map.jumpTo({ center: src.map.getCenter(), zoom: src.map.getZoom(), bearing: src.map.getBearing() });
    syncing = false;
  });
}

// ------------------------------------------------------------------ helpers
const parseDate = s => new Date(s + "T00:00:00");
const longDate = s => parseDate(s).toLocaleDateString("en-AU", { day: "numeric", month: "long", year: "numeric" });
const shortDate = s => parseDate(s).toLocaleDateString("en-AU", { day: "numeric", month: "short" });
const dayDiff = (a, b) => Math.round((parseDate(b) - parseDate(a)) / 86400000);
const n = x => x.toLocaleString("en-AU");
const f1Of = ([tp, fp, fn]) => tp + fp + fn === 0 ? null : 2 * tp / (2 * tp + fp + fn);
const seasonName = s => `${s - 1}–${String(s).slice(2)}`;
const inTen = (a, b) => { const r = a / b; return r > 0 && r < 0.05 ? "fewer than 1 in 10" : `${Math.round(10 * r)} in 10`; };

// Plain language: of the cells that burnt, how many it caught; of its guesses, how many were right.
function plain([tp, fp, fn]) {
  if (tp + fn === 0) return fp ? `Nothing burnt this day, but it guessed ${n(fp)} cells.` : "Nothing burnt this day, and it guessed nothing.";
  if (tp + fp === 0) return `Guessed nothing; ${n(fn)} cells burnt.`;
  return `Caught ${inTen(tp, tp + fn)} cells that burnt; ${inTen(tp, tp + fp)} of its guesses were right.`;
}
const caught = ([tp, , fn]) => tp + fn ? `caught ${inTen(tp, tp + fn)}` : "nothing burnt";

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

// ------------------------------------------------------------------ drawing
function isEdge(px, w, h, x, y) {
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || yy < 0 || xx >= w || yy >= h || !(px[4 * (yy * w + xx) + 1] & BIT.pos)) return true;
  }
  return false;
}

function paint(P, { w, h, px }) {
  const { canvas, ctx } = P;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const out = ctx.createImageData(w, h), o = out.data, bit = BIT[P.model];
  const put = (i, c, a = c[3]) => { o[i] = c[0]; o[i + 1] = c[1]; o[i + 2] = c[2]; o[i + 3] = a; };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = 4 * (y * w + x), g = px[i + 1];
      if (g & BIT.outside) continue;
      if (g & BIT.burnt) { put(i, C.ash); continue; }
      if (g & BIT.unscored) { put(i, C.veil, (x + y) % 4 < 2 ? 150 : 60); continue; }
      const pos = g & BIT.pos, pred = g & bit;
      if (detail) {
        if (pos && pred) put(i, C.hit); else if (pos) put(i, C.missed); else if (pred) put(i, C.false);
      } else if (pos && isEdge(px, w, h, x, y)) put(i, C.outline);
      else if (pred) put(i, C.guess);
    }
  }
  ctx.putImageData(out, 0, 0);
}

function draw(P, day) {
  paint(P, day);
  let src = P.map.getSource("pred");
  if (!src) {
    // Not animated: the canvas only changes with the day, so repaint on demand.
    P.map.addSource("pred", { type: "canvas", canvas: P.canvas, coordinates: fire.corners, animate: false });
    P.map.addLayer({ id: "pred", type: "raster", source: "pred",
                     paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 } }, P.label);
    src = P.map.getSource("pred");
  }
  src.play();
  requestAnimationFrame(() => requestAnimationFrame(() => src.pause()));
}

async function show() {
  const token = ++loadToken, f = fire, i = k;
  const day = await dayPixels(f, i);
  if (token !== loadToken) return;
  draw(A, day);
  draw(B, day);
  for (const j of [i + 1, i - 1]) if (j >= 0 && j < f.days.length) dayPixels(f, j);   // prefetch
  readout();
  renderBars();
}

// ------------------------------------------------------------------ text
function scoreLine(model) {
  const c = fire.counts[model][k], f1 = f1Of(c);
  return plain(c) + (detail && f1 != null ? ` F1 ${f1.toFixed(2)}.` : "");
}

function readout() {
  const date = fire.days[k];
  $("date").textContent = longDate(date);
  $("score-a").textContent = scoreLine("main");
  $("score-b").textContent = scoreLine(modelB);
  $("slider").value = dayDiff(fire.days[0], date);
  $("slider").setAttribute("aria-valuetext",
    `${longDate(date)}. Main model: ${scoreLine("main")} ${NAMES[modelB]}: ${scoreLine(modelB)}`);
}

function kindLine(f) {
  return f.kind === "test"
    ? `A ${seasonName(f.season)} fire, kept out of training to test the models.`
    : `A ${seasonName(f.season)} fire, predicted by models trained without that season.`;
}

// ------------------------------------------------------------------ timeline: one bar per day of new burn
const bars = $("bars"), bctx = bars.getContext("2d");
function renderBars() {
  const dpr = devicePixelRatio || 1, w = bars.clientWidth, h = bars.clientHeight;
  if (bars.width !== Math.round(w * dpr)) { bars.width = Math.round(w * dpr); bars.height = Math.round(h * dpr); }
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, w, h);
  const span = dayDiff(fire.days[0], fire.days.at(-1)) + 1, bw = w / span, base = h - 6;
  const max = Math.max(...fire.new_burn, 1);
  bctx.fillStyle = "rgba(217, 218, 214, 0.18)";
  bctx.fillRect(0, base, w, 1);
  fire.days.forEach((d, i) => {
    const v = fire.new_burn[i], bh = Math.max(v > 0 ? 1.5 : 0, (v / max) * (base - 4));
    bctx.fillStyle = i === k ? "rgba(236, 178, 106, 1)" : "rgba(146, 148, 150, 0.6)";
    bctx.fillRect(dayDiff(fire.days[0], d) * bw + 0.5, base - bh, Math.max(bw - 1, 1), bh);
  });
  const x = (dayDiff(fire.days[0], fire.days[k]) + 0.5) * bw;
  bctx.fillStyle = "#d9dad6";
  bctx.fillRect(Math.round(x) - 0.5, 0, 1, base);
}

function nearestIndex(offset) {
  let best = 0, bd = Infinity;
  fire.days.forEach((d, i) => { const dd = Math.abs(dayDiff(fire.days[0], d) - offset); if (dd < bd) { bd = dd; best = i; } });
  return best;
}

function hover(e) {
  const r = bars.getBoundingClientRect(), span = dayDiff(fire.days[0], fire.days.at(-1)) + 1;
  const i = nearestIndex(Math.floor((e.clientX - r.left) / r.width * span));
  const a = fire.counts.main[i], b = fire.counts[modelB][i], fa = f1Of(a), fb = f1Of(b), tip = $("hover");
  tip.textContent = `${shortDate(fire.days[i])}: main model ${caught(a)}, ${NAMES[modelB].toLowerCase()} ${caught(b)}` +
                    (fa != null ? ` (F1 ${fa.toFixed(2)} vs ${(fb ?? 0).toFixed(2)})` : "");
  tip.hidden = false;
  const x = (dayDiff(fire.days[0], fire.days[i]) + 0.5) / span * r.width;
  tip.style.left = `${Math.min(Math.max(x, 170), r.width - 170)}px`;
}

// ------------------------------------------------------------------ controls
function setDay(i) { k = Math.max(0, Math.min(i, fire.days.length - 1)); show(); }

function setDetail(d) {
  detail = d;
  $("detail").checked = d;
  $("key-simple").hidden = d;
  $("key-detail").hidden = !d;
  $("hint").textContent = d ? "Scores add F1, which balances the two: 1 is perfect, 0 is no overlap."
                            : "Where the fill sits inside the outline, the guess was right.";
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

const fit = () => {
  const pad = { top: 56, bottom: 64, left: 24, right: 24 };      // clear of the name and the score
  A.map.fitBounds(fire.focus, { padding: pad, duration: 0, maxZoom: 10.5 });
};

async function loadFire(i, day) {
  fire = fires[i];
  cache.clear();
  for (const P of [A, B]) if (P.map.getSource("pred")) { P.map.removeLayer("pred"); P.map.removeSource("pred"); }
  $("fire-label").textContent = fire.label;
  $("fire-kind").textContent = kindLine(fire);
  $("slider").max = dayDiff(fire.days[0], fire.days.at(-1));
  buildList();
  k = day ?? fire.new_burn.indexOf(Math.max(...fire.new_burn));
  fit();                                    // fixed framing per fire: what burnt, both maps
  await show();
}

// Picker: fires with a single scored day are left out (they're still in the scores).
const listed = f => !(f.kind === "test" && f.days.length === 1);

function buildList() {
  const box = $("fire-list");
  box.textContent = "";
  for (const [kind, title] of [["test", "Test fires, 2019–20"], ["cv", "Other seasons"]]) {
    const h = document.createElement("p");
    h.className = "group";
    h.textContent = title;
    box.appendChild(h);
    fires.forEach((f, i) => {
      if (f.kind !== kind || !listed(f)) return;
      const b = document.createElement("button");
      b.setAttribute("role", "option");
      b.setAttribute("aria-selected", f === fire);
      const when = parseDate(f.days[0]).toLocaleDateString("en-AU", { month: "short", year: "numeric" });
      const tot = [0, 1, 2].map(j => f.counts.main.reduce((s, c) => s + c[j], 0));
      b.innerHTML = `<span>${f.label}</span><span class="when">${when}</span>` +
                    `<span class="score-small">${caught(tot)}</span>`;
      b.title = `Main model over the whole fire: ${plain(tot)}`;
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

async function init() {
  meta = await (await fetch("data/compare/fires.json")).json();
  fires = meta.fires;
  if (["baseline", "unet", "physics"].includes(params.get("vs"))) modelB = params.get("vs");
  B.model = modelB;
  $("model-b").value = modelB;
  setDetail(params.get("detail") === "1");
  const id = params.get("fire") ?? String(DEFAULT.fire);
  const pick = Math.max(0, fires.findIndex(f => String(f.id) === id));
  const wantDay = params.get("day") ?? (id === String(DEFAULT.fire) ? DEFAULT.day : null);
  const day = wantDay ? fires[pick].days.indexOf(wantDay) : -1;
  await loadFire(pick, day >= 0 ? day : undefined);
  requestAnimationFrame(frame);
}

$("fire-name").onclick = () => openList($("fire-list").hidden);
document.addEventListener("click", e => { if (!e.target.closest(".fire")) openList(false); });
$("detail").onchange = e => { setDetail(e.target.checked); show(); };
$("model-b").onchange = e => { modelB = B.model = e.target.value; show(); };
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
