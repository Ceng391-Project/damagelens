"use strict";
const COLORS = { damaged: "#ff9f1c", destroyed: "#e63946", intact: "#2ec4b6" };
const KEY_CLS = { "1": "damaged", "2": "destroyed", "3": "intact" };
const $ = (id) => document.getElementById(id);
const canvases = [$("c-pre"), $("c-post")];
const ctx = canvases.map((c) => c.getContext("2d"));

let manifest = null, order = [], idx = 0;
let ann = null, imgs = { pre: null, post: null };
let cls = "damaged", draft = [], selected = -1, hide = false, blink = false;
let view = { s: 1, x: 0, y: 0 };            // image px -> canvas px: c = s * p + (x, y)
let saveTimer = null, saving = false, dirty = false;

const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

function loadImage(src) {
  return new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = src; });
}

function shardFilter() {
  const s = $("shard").value;
  order = manifest.tiles.map((t, i) => ({ id: t.id, i })).filter((t) => s === "all" || t.i % 3 === Number(s)).map((t) => t.id);
}

function updateProgress() {
  const st = manifest.status;
  const n = order.length, done = order.filter((id) => st[id] === "done").length, skip = order.filter((id) => st[id] === "skip").length;
  $("progress").textContent = `${done} bitti · ${skip} atlandı · ${n - done - skip} kaldı (${n})`;
}

async function openTile(i) {
  await flush();
  idx = (i + order.length) % order.length;
  const id = order[idx];
  store.set(`dl-last-${manifest.name}`, id);
  $("tile-name").textContent = `${id} (${idx + 1}/${order.length})`;
  [ann, imgs.pre, imgs.post] = await Promise.all([api(`/api/ann/${id}`), loadImage(`/tiles/${id}_pre.png`), loadImage(`/tiles/${id}_post.png`)]);
  ann.polygons = ann.polygons || [];
  draft = []; selected = -1; dirty = false;
  resetView(); setStatusChip(); updateCounts(); draw();
}

function setStatusChip() {
  const c = $("status-chip"); c.textContent = ann.status; c.className = `chip ${ann.status}`;
}

function updateCounts() {
  const n = { damaged: 0, destroyed: 0, intact: 0 };
  ann.polygons.forEach((p) => n[p.cls]++);
  $("counts").textContent = `bu karoda: ${n.damaged} hasarlı, ${n.destroyed} yıkık, ${n.intact} hasarsız`;
}

function resetView() {
  const W = canvases[0].width; view = { s: W / manifest.tile, x: 0, y: 0 };
}

function fitCanvases() {
  canvases.forEach((c) => {
    const w = Math.round(c.getBoundingClientRect().width * devicePixelRatio);
    if (c.width !== w) { c.width = w; c.height = w; }
  });
  if (manifest) { resetView(); draw(); }
}

function toImage(c, e) {
  const r = c.getBoundingClientRect(), k = c.width / r.width;
  return [((e.clientX - r.left) * k - view.x) / view.s, ((e.clientY - r.top) * k - view.y) / view.s];
}

function path(g, pts) {
  g.beginPath();
  pts.forEach(([x, y], j) => (j ? g.lineTo : g.moveTo).call(g, x * view.s + view.x, y * view.s + view.y));
}

function draw() {
  if (!ann) return;
  canvases.forEach((c, k) => {
    const g = ctx[k];
    g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#000"; g.fillRect(0, 0, c.width, c.height);
    const im = k === 0 || blink ? imgs.pre : imgs.post;
    if (im) { g.imageSmoothingEnabled = view.s < 2; g.drawImage(im, view.x, view.y, manifest.tile * view.s, manifest.tile * view.s); }
    if (hide) return;
    g.lineWidth = Math.max(1.5, devicePixelRatio * 1.5);
    ann.polygons.forEach((p, j) => {
      path(g, p.points); g.closePath();
      g.strokeStyle = COLORS[p.cls]; g.fillStyle = COLORS[p.cls] + (j === selected ? "66" : "2a");
      g.fill(); g.stroke();
      if (j === selected) { g.setLineDash([6, 4]); g.strokeStyle = "#fff"; g.stroke(); g.setLineDash([]); }
    });
    if (draft.length) {
      path(g, draft); g.strokeStyle = COLORS[cls]; g.stroke();
      draft.forEach(([x, y]) => { g.fillStyle = COLORS[cls]; g.fillRect(x * view.s + view.x - 3, y * view.s + view.y - 3, 6, 6); });
    }
  });
}

function inside([x, y], pts) {
  let hit = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

function changed() { dirty = true; updateCounts(); draw(); scheduleSave(); }

function scheduleSave() {
  $("save-state").textContent = "kaydediliyor…"; $("save-state").className = "saving";
  clearTimeout(saveTimer); saveTimer = setTimeout(flush, 400);
}

async function flush() {
  clearTimeout(saveTimer);
  if (!dirty || !ann || saving) return;
  saving = true; dirty = false;
  const body = { status: ann.status, user: $("user").value.trim(), polygons: ann.polygons };
  try {
    await api(`/api/ann/${ann.tile}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    manifest.status[ann.tile] = ann.status; updateProgress();
    $("save-state").textContent = "kaydedildi"; $("save-state").className = "saved";
  } catch (err) {
    dirty = true; $("save-state").textContent = "kaydedilemedi: sunucu çalışıyor mu?"; $("save-state").className = "error";
  } finally { saving = false; }
  if (dirty) scheduleSave();
}

function closeDraft() {
  if (draft.length >= 3) { ann.polygons.push({ cls, points: draft }); selected = ann.polygons.length - 1; }
  draft = []; changed();
}

function setClass(c) {
  cls = c;
  document.querySelectorAll(".cls").forEach((b) => b.classList.toggle("on", b.dataset.cls === c));
  if (selected >= 0) { ann.polygons[selected].cls = c; changed(); }
}

function setStatus(s) {
  if (draft.length >= 3) closeDraft();
  ann.status = ann.status === s ? "todo" : s; setStatusChip(); changed();
}

let pan = null, spaceDown = false;
canvases.forEach((c) => {
  c.addEventListener("contextmenu", (e) => e.preventDefault());
  c.addEventListener("pointerdown", (e) => {
    if (e.button === 2 || e.button === 1 || spaceDown) { pan = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, c }; c.setPointerCapture(e.pointerId); return; }
    if (e.button !== 0) return;
    const p = toImage(c, e);
    if (!draft.length) {
      const hit = ann.polygons.map((q, j) => (inside(p, q.points) ? j : -1)).filter((j) => j >= 0).pop();
      if (hit !== undefined) { selected = hit; setClass(ann.polygons[hit].cls); draw(); return; }
    }
    selected = -1;
    draft.push([Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]); draw();
  });
  c.addEventListener("pointermove", (e) => {
    if (!pan) return;
    const k = c.width / c.getBoundingClientRect().width;
    view.x = pan.vx + (e.clientX - pan.x) * k; view.y = pan.vy + (e.clientY - pan.y) * k; draw();
  });
  c.addEventListener("pointerup", () => { pan = null; });
  c.addEventListener("dblclick", (e) => { e.preventDefault(); if (draft.length > 3) draft.pop(); closeDraft(); });
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = c.getBoundingClientRect(), k = c.width / r.width;
    const cx = (e.clientX - r.left) * k, cy = (e.clientY - r.top) * k;
    const f = Math.exp(-e.deltaY * 0.0015), s = Math.min(Math.max(view.s * f, c.width / manifest.tile), 12 * c.width / manifest.tile);
    view.x = cx - ((cx - view.x) * s) / view.s; view.y = cy - ((cy - view.y) * s) / view.s; view.s = s; draw();
  }, { passive: false });
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key;
  if (k === " ") { spaceDown = true; e.preventDefault(); }
  else if (KEY_CLS[k]) setClass(KEY_CLS[k]);
  else if (k === "Enter") closeDraft();
  else if (k === "Escape") { draft = []; selected = -1; draw(); }
  else if (k === "Backspace") { e.preventDefault(); if (draft.length) { draft.pop(); draw(); } }
  else if (k === "Delete" && selected >= 0) { ann.polygons.splice(selected, 1); selected = -1; changed(); }
  else if (k === "b" || k === "B") { if (!blink) { blink = true; $("blink-hint").style.visibility = "visible"; draw(); } }
  else if (k === "h" || k === "H") { hide = !hide; draw(); }
  else if (k === "0") { resetView(); draw(); }
  else if (k === "v" || k === "V") setStatus("done");
  else if (k === "s" || k === "S") setStatus("skip");
  else if (k === "n" || k === "N" || k === "ArrowRight") openTile(idx + 1);
  else if (k === "p" || k === "P" || k === "ArrowLeft") openTile(idx - 1);
  else if (k === "t" || k === "T") nextTodo();
});
document.addEventListener("keyup", (e) => {
  if (e.key === " ") spaceDown = false;
  if ((e.key === "b" || e.key === "B") && blink) { blink = false; $("blink-hint").style.visibility = "hidden"; draw(); }
});

function nextTodo() {
  for (let j = 1; j <= order.length; j++) {
    const i = (idx + j) % order.length;
    if ((manifest.status[order[i]] || "todo") === "todo") return openTile(i);
  }
}

$("prev").onclick = () => openTile(idx - 1);
$("next").onclick = () => openTile(idx + 1);
$("next-todo").onclick = nextTodo;
$("mark-done").onclick = () => setStatus("done");
$("mark-skip").onclick = () => setStatus("skip");
document.querySelectorAll(".cls").forEach((b) => (b.onclick = () => setClass(b.dataset.cls)));
$("user").value = store.get("dl-user", "");
$("user").onchange = () => store.set("dl-user", $("user").value.trim());
$("shard").onchange = () => { store.set("dl-shard", $("shard").value); shardFilter(); updateProgress(); openTile(0); };
window.addEventListener("resize", fitCanvases);
window.addEventListener("beforeunload", (e) => { if (dirty) { flush(); e.preventDefault(); } });

(async function init() {
  manifest = await api("/api/manifest");
  document.title = `DamageLens etiketleme · ${manifest.name}`;
  $("pre-date").textContent = manifest.pre_date; $("post-date").textContent = manifest.post_date;
  $("shard").value = store.get("dl-shard", "all");
  shardFilter(); updateProgress(); fitCanvases();
  const last = order.indexOf(store.get(`dl-last-${manifest.name}`, ""));
  await openTile(last >= 0 ? last : 0);
})();
