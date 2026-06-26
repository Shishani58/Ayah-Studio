"use strict";

// ---- tiny API helpers ----------------------------------------------------
const api = {
  async get(url) { const r = await fetch(url); if (!r.ok) throw new Error((await r.json()).detail || r.status); return r.json(); },
  async post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    if (!r.ok) { let d; try { d = (await r.json()).detail; } catch { d = r.statusText; } throw new Error(d || r.status); }
    return r.json();
  },
};
const $ = (id) => document.getElementById(id);
function fmt(ms) {
  if (ms == null) return "—";
  const s = Math.max(0, ms) / 1000, m = Math.floor(s / 60), r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
}
const setStatus = (msg) => ($("status").textContent = msg);

// ===========================================================================
//  Dashboard
// ===========================================================================
async function refreshSuras() {
  const suras = await api.get("/api/suras");
  const grid = $("grid");
  grid.innerHTML = "";
  for (const s of suras) {
    const tile = document.createElement("div");
    tile.className = "tile" + (s.has_audio ? "" : " no-audio");
    tile.innerHTML = `
      <div class="audio-dot ${s.has_audio ? "on" : ""}" title="${s.has_audio ? "есть аудио" : "нет аудио"}"></div>
      <div class="num">${s.surah}</div>
      <div class="meta">${s.ayah_count} аят.</div>
      <span class="badge ${s.status}">${badgeLabel(s.status)}</span>`;
    if (s.has_audio) tile.onclick = () => openReview(s.surah, true);
    grid.appendChild(tile);
  }
  updateGenControls(suras);
}
const badgeLabel = (st) => ({ "not-processed": "—", "processed": "черновик", "verified": "✓ проверена" }[st] || st);

// reflect progress on the generate buttons (resume-aware)
function updateGenControls(suras) {
  const withAudio = suras.filter((s) => s.has_audio);
  const remaining = withAudio.filter((s) => s.status === "not-processed").length;
  const done = withAudio.length - remaining;
  const ready = withAudio.length > 0 && !!$("dbPath").value.trim();
  const gen = $("genBtn"), regen = $("regenBtn");
  if (!ready) { gen.disabled = true; gen.textContent = "▶ Сгенерировать всё"; regen.classList.add("hidden"); return; }
  if (done === 0) { gen.disabled = false; gen.textContent = `▶ Сгенерировать всё (${withAudio.length})`; regen.classList.add("hidden"); }
  else if (remaining > 0) { gen.disabled = false; gen.textContent = `▶ Продолжить · осталось ${remaining}`; regen.classList.remove("hidden"); }
  else { gen.disabled = true; gen.textContent = `✓ Все ${withAudio.length} готовы`; regen.classList.remove("hidden"); }
}

async function openProject() {
  const audio_dir = $("audioDir").value.trim();
  const db_path = $("dbPath").value.trim();
  if (!audio_dir && !db_path) { setStatus("Укажите папку аудио и/или файл проекта."); return; }
  try {
    const p = await api.post("/api/project/open", { audio_dir, db_path });
    audioCount = p.audio_count || 0;
    const ready = p.db_path && p.audio_count;
    setStatus(ready
      ? `Готово: ${p.audio_count} сур с аудио. Нажмите «Сгенерировать всё» или откройте суру.`
      : `Аудио: ${p.audio_count} сур · Проект: ${p.db_path ? "✓" : "не выбран"}. ${p.db_path ? "" : "Выберите .db."}`);
    $("genBtn").disabled = !ready;
    if (ready) $("help").classList.add("collapsed");
    await refreshSuras();
  } catch (e) { setStatus("Ошибка: " + e.message); }
}

let audioCount = 0;
const GEN = { es: null, timer: null, startMs: 0, total: 0 };

function runGeneration(onlyMissing = true) {
  const mode = $("mode").value;
  // Explicit start: confirm the deliberate, long-running action.
  const what = onlyMissing
    ? "недостающих (ещё не сгенерированных) сур"
    : `ВСЕХ ${audioCount || ""} сур заново`;
  const msg = `Запустить генерацию ${what} в режиме «${mode}»?\n`
    + `Размеченные вручную аяты (edited/verified) не затрагиваются. Можно отменить (после текущей суры).`;
  if (!confirm(msg)) return;
  startGenUI();
  api.post("/api/generate", { mode, only_missing: onlyMissing })
    .then(() => {
      const es = new EventSource("/api/generate/stream");
      GEN.es = es;
      es.onmessage = (ev) => handleGenEvent(JSON.parse(ev.data));
      es.onerror = () => { if (GEN.es) endGenUI("error"); };
    })
    .catch((e) => { setStatus("Ошибка: " + e.message); endGenUI("error"); });
}
function startGenUI() {
  $("genBtn").disabled = true;
  const panel = $("genPanel");
  panel.classList.remove("hidden", "done", "cancelled");
  $("cancelBtn").disabled = false;
  $("cancelBtn").textContent = "✕ Отменить";
  $("genTitle").textContent = "Генерация…";
  $("genCur").textContent = "подготовка…";
  GEN.startMs = Date.now();
  $("genStart").textContent = new Date(GEN.startMs).toLocaleTimeString();
  $("genElapsed").textContent = "0:00";
  setProgress(0, 1, "запуск…");
  if (GEN.timer) clearInterval(GEN.timer);
  GEN.timer = setInterval(updateElapsed, 1000);
}
function updateElapsed() {
  const s = Math.floor((Date.now() - GEN.startMs) / 1000);
  $("genElapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function handleGenEvent(d) {
  if (d.type === "start") { GEN.total = d.total || 1; setProgress(0, GEN.total, `0 / ${GEN.total}`); }
  else if (d.type === "sura_start") { $("genCur").textContent = `обрабатываю суру ${d.surah}…`; }
  else if (d.type === "sura") {
    setProgress(d.index + 1, GEN.total, `${d.index + 1} / ${GEN.total}`);
    $("genCur").textContent = `сура ${d.surah} готова${d.written != null ? ` · ${d.written} аят.` : ""}`;
    refreshSuras();
  }
  else if (d.type === "done") { endGenUI("done", d.completed); }
  else if (d.type === "cancelled") { endGenUI("cancelled", d.completed); }
  else if (d.type === "error") { setStatus("Ошибка генерации: " + d.message); endGenUI("error"); }
}
function endGenUI(kind, completed) {
  if (GEN.es) { GEN.es.close(); GEN.es = null; }
  if (GEN.timer) { clearInterval(GEN.timer); GEN.timer = null; }
  updateElapsed();
  const elapsed = $("genElapsed").textContent;
  $("genBtn").disabled = false;
  $("cancelBtn").disabled = true;
  const panel = $("genPanel");
  if (kind === "done") {
    panel.classList.add("done"); $("genTitle").textContent = "✓ Готово";
    $("genCur").textContent = `обработано ${completed ?? ""} сур за ${elapsed}`;
    setProgress(1, 1, "✓ готово");
  } else if (kind === "cancelled") {
    panel.classList.add("cancelled"); $("genTitle").textContent = "⏹ Отменено";
    $("genCur").textContent = `обработано ${completed ?? 0} сур за ${elapsed}`;
  } else {
    panel.classList.add("cancelled"); $("genTitle").textContent = "✕ Ошибка";
  }
  refreshSuras();
}
async function cancelGeneration() {
  $("cancelBtn").disabled = true;
  $("cancelBtn").textContent = "Отмена…";
  $("genCur").textContent = "отмена после текущей суры…";
  try { await api.post("/api/generate/cancel", {}); } catch (e) { setStatus("Ошибка отмены: " + e.message); }
}
function setProgress(done, total, text) {
  $("progressFill").style.right = (100 - (total ? (done / total) * 100 : 0)) + "%";
  $("progressText").textContent = text;
}

// ===========================================================================
//  Review
// ===========================================================================
const R = { surah: null, ayahs: [], duration: 0, peaks: [], audio: null, current: -1, loop: false, drag: -1 };

async function openReview(surah) {
  R.surah = surah;
  setStatus("Загрузка суры " + surah + "…");
  let sura, wave;
  try {
    [sura, wave] = await Promise.all([api.get(`/api/sura/${surah}`), api.get(`/api/sura/${surah}/waveform`)]);
  } catch (e) { setStatus("Ошибка: " + e.message); return; }
  R.ayahs = sura.ayahs; R.peaks = wave.peaks; R.duration = wave.duration_ms; R.current = -1;
  $("reviewTitle").textContent = `Сура ${surah} · ${sura.ayah_count} аят.`;
  const audio = $("audio");
  audio.src = `/api/audio/${surah}`;
  audio.playbackRate = parseFloat($("rate").value);
  R.audio = audio;
  $("reviewMode").value = $("mode").value;  // default per-sura mode = global
  $("suraGenStatus").textContent = "";
  $("suraGenStatus").className = "sura-gen-status";
  renderAyahList();
  updateGenHint();
  showView("review");
  requestAnimationFrame(() => { resizeWave(); drawWave(); });
}
// hint under the waveform + per-sura button label reflect whether it's generated
function updateGenHint() {
  const generated = R.ayahs.some((a) => a.time != null);
  const hint = $("waveHint");
  if (generated) {
    hint.textContent = "Клик по волне — перемотка · тяните золотую линию — двигайте начало аята";
    hint.classList.remove("warn");
    $("regenSuraBtn").textContent = "↻ Пересчитать суру";
  } else {
    hint.textContent = "⚠ Сура не рассчитана — нажмите «Рассчитать суру» (выберите режим слева). Без тайм-кодов подсветки и маркеров нет.";
    hint.classList.add("warn");
    $("regenSuraBtn").textContent = "▶ Рассчитать суру";
  }
}

// regenerate just this sura with the chosen per-sura mode
async function regenerateSura() {
  if (R.surah == null) return;
  const surah = R.surah, mode = $("reviewMode").value;
  const btn = $("regenSuraBtn"), st = $("suraGenStatus");
  btn.disabled = true;
  const t0 = Date.now();
  const tick = () => (st.textContent = `расчёт… ${Math.floor((Date.now() - t0) / 1000)}с`);
  st.className = "sura-gen-status running"; tick();
  const timer = setInterval(tick, 1000);
  const stop = (cls, text) => { clearInterval(timer); btn.disabled = false; st.className = "sura-gen-status " + cls; st.textContent = text; };
  try {
    await api.post("/api/generate", { suras: [surah], mode });
  } catch (e) { stop("err", "ошибка: " + e.message); return; }
  const es = new EventSource("/api/generate/stream");
  es.onmessage = async (ev) => {
    const d = JSON.parse(ev.data);
    if (d.type === "done" || d.type === "cancelled") {
      es.close(); stop("ok", `✓ готово за ${Math.floor((Date.now() - t0) / 1000)}с`);
      await reloadSura(); refreshSuras();
    } else if (d.type === "error") { es.close(); stop("err", "ошибка: " + d.message); }
  };
  es.onerror = () => { es.close(); if (btn.disabled) stop("", ""); };
}
// re-fetch this sura's timings after a recompute, keep audio position
async function reloadSura() {
  try {
    const sura = await api.get(`/api/sura/${R.surah}`);
    R.ayahs = sura.ayahs;
  } catch { return; }
  renderAyahList();
  updateGenHint();
  drawWave();
  const cur = R.current; R.current = -1;
  setCurrent(currentAyahByTime((R.audio ? R.audio.currentTime : 0) * 1000));
  if (R.current < 0 && cur >= 0) setCurrent(cur);
}

function renderAyahList() {
  const ul = $("ayahList");
  ul.innerHTML = "";
  R.ayahs.forEach((a, i) => {
    const li = document.createElement("li");
    li.dataset.i = i;
    if (a.status === "verified") li.classList.add("verified");
    else if (a.status === "edited") li.classList.add("edited");
    li.innerHTML = `
      <span class="a-conf" style="background:${confColor(a.confidence, a.status)}"></span>
      <span class="a-num">${a.ayah}</span>
      <span class="a-time">${fmt(a.time)}</span>
      <button class="a-verify">${a.status === "verified" ? "✓ пров." : "пров."}</button>
      <span class="a-text" dir="rtl" lang="ar">${escapeHtml(a.text || "")}</span>`;
    li.onclick = (e) => {
      if (e.target.classList.contains("a-verify")) { toggleVerify(i); e.stopPropagation(); }
      else seekAyah(i, true);
    };
    ul.appendChild(li);
  });
}
function confColor(conf, status) {
  if (status === "verified") return "var(--green)";
  if (status === "edited") return "var(--amber)";
  if (conf == null) return "var(--line)";
  if (conf >= 0.7) return "var(--green)";
  if (conf >= 0.4) return "var(--amber)";
  return "var(--red)";
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- waveform canvas ----
function resizeWave() {
  const c = $("wave"), dpr = window.devicePixelRatio || 1;
  c.width = Math.round(c.clientWidth * dpr); c.height = Math.round(c.clientHeight * dpr);
  c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
}
function drawWave() {
  const c = $("wave"), ctx = c.getContext("2d");
  const w = c.clientWidth, h = c.clientHeight, mid = h / 2;
  ctx.clearRect(0, 0, w, h);
  const n = R.peaks.length;
  ctx.strokeStyle = "#3a4456"; ctx.lineWidth = 1; ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / n) * w, amp = (R.peaks[i] / 100) * (mid - 2);
    ctx.moveTo(x, mid - amp); ctx.lineTo(x, mid + amp);
  }
  ctx.stroke();
  R.ayahs.forEach((a, i) => {
    if (a.time == null || !R.duration) return;
    const x = (a.time / R.duration) * w;
    ctx.strokeStyle = i === R.current ? "#ffd56b" : "#d4a843";
    ctx.lineWidth = i === R.current ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = "10px sans-serif";
    ctx.fillText(a.ayah, x + 3, 11);
  });
  if (R.audio && R.duration) {
    const px = (R.audio.currentTime * 1000 / R.duration) * w;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
  }
}
function xToMs(clientX) {
  const rect = $("wave").getBoundingClientRect();
  return Math.max(0, Math.min(R.duration, ((clientX - rect.left) / rect.width) * R.duration));
}
function markerNearX(clientX) {
  const rect = $("wave").getBoundingClientRect(), tol = 7;
  for (let i = 0; i < R.ayahs.length; i++) {
    const t = R.ayahs[i].time; if (t == null) continue;
    const x = rect.left + (t / R.duration) * rect.width;
    if (Math.abs(clientX - x) <= tol) return i;
  }
  return -1;
}

// ---- playback ----
function togglePlay() { if (!R.audio) return; R.audio.paused ? R.audio.play() : R.audio.pause(); }
function seekAyah(i, play) {
  const a = R.ayahs[i]; if (!a || a.time == null || !R.audio) return;
  R.audio.currentTime = a.time / 1000;
  setCurrent(i);
  if (play) R.audio.play();
}
function setCurrent(i) {
  if (i === R.current) return;
  R.current = i;
  document.querySelectorAll(".ayah-list li").forEach((li) => li.classList.toggle("active", +li.dataset.i === i));
  const li = document.querySelector(`.ayah-list li[data-i="${i}"]`);
  if (li) li.scrollIntoView({ block: "nearest" });
}
function currentAyahByTime(ms) {
  let idx = -1, first = -1;
  for (let i = 0; i < R.ayahs.length; i++) {
    const t = R.ayahs[i].time;
    if (t == null) continue;
    if (first < 0) first = i;
    if (t <= ms + 1) idx = i;
  }
  // Before the first marker (e.g. the intro basmala), keep the first ayah active
  // so something is always highlighted while playing a generated sura.
  return idx >= 0 ? idx : first;
}
function ayahEndMs(i) {
  for (let j = i + 1; j < R.ayahs.length; j++) if (R.ayahs[j].time != null) return R.ayahs[j].time;
  return R.duration;
}

async function toggleVerify(i) {
  const a = R.ayahs[i];
  const next = a.status === "verified" ? "auto" : "verified";
  try {
    await api.post("/api/verify", { surah: R.surah, ayah: a.ayah, status: next });
    a.status = next === "auto" ? null : next;
    renderAyahList(); const cur = R.current; R.current = -1; setCurrent(cur); drawWave();
  } catch (e) { setStatus("Ошибка: " + e.message); }
}
async function commitMarker(i, ms) {
  const a = R.ayahs[i];
  try {
    await api.post("/api/timing", { surah: R.surah, ayah: a.ayah, time: Math.round(ms) });
    a.time = Math.round(ms); a.status = "edited";
    renderAyahList(); drawWave();
  } catch (e) { setStatus("Ошибка: " + e.message); }
}
async function exportProject() {
  let dst = "reciter_export.db";
  if (window.pywebview) { const p = await window.pywebview.api.save_db("reciter_export.db"); if (!p) return; dst = p; }
  else { const p = prompt("Путь для экспорта .db:", dst); if (!p) return; dst = p; }
  try { const r = await api.post("/api/export", { dst }); setStatus("Экспортировано: " + r.path); }
  catch (e) { alert("Экспорт невозможен: " + e.message); }
}

// ===========================================================================
//  wiring
// ===========================================================================
function showView(name) {
  $("dashboard").classList.toggle("hidden", name !== "dashboard");
  $("review").classList.toggle("hidden", name !== "review");
}
const baseName = (p) => p.replace(/[/\\]+$/, "").split(/[/\\]/).pop() || "reciter";

function enableNativePickers() {
  $("browseAudio").classList.remove("hidden");
  $("browseDb").classList.remove("hidden");
  $("browseAudio").onclick = async () => {
    const p = await window.pywebview.api.pick_folder();
    if (!p) return;
    $("audioDir").value = p;
    if (!$("dbPath").value.trim()) $("dbPath").value = `${p}/${baseName(p)}.db`;
  };
  $("browseDb").onclick = async () => {
    const suggested = $("audioDir").value.trim() ? `${baseName($("audioDir").value.trim())}.db` : "reciter.db";
    const p = (await window.pywebview.api.pick_db()) || (await window.pywebview.api.save_db(suggested));
    if (p) $("dbPath").value = p;
  };
}

function wire() {
  $("helpBtn").onclick = () => $("help").classList.toggle("collapsed");
  $("openBtn").onclick = openProject;
  $("genBtn").onclick = () => runGeneration(true);
  $("regenBtn").onclick = () => runGeneration(false);
  $("cancelBtn").onclick = cancelGeneration;
  $("regenSuraBtn").onclick = regenerateSura;
  $("mode").onchange = () => api.post("/api/settings", { mode: $("mode").value }).catch(() => {});
  $("backBtn").onclick = () => { if (R.audio) R.audio.pause(); showView("dashboard"); refreshSuras(); };
  $("playBtn").onclick = togglePlay;
  $("prevBtn").onclick = () => seekAyah(Math.max(R.current - 1, 0), false);
  $("nextBtn").onclick = () => seekAyah(Math.min(R.current + 1, R.ayahs.length - 1), false);
  $("loopBtn").onclick = () => { R.loop = !R.loop; $("loopBtn").classList.toggle("active", R.loop); };
  $("rate").onchange = () => { if (R.audio) R.audio.playbackRate = parseFloat($("rate").value); };
  $("exportBtn").onclick = exportProject;

  const audio = $("audio");
  audio.addEventListener("play", () => ($("playBtn").textContent = "⏸"));
  audio.addEventListener("pause", () => ($("playBtn").textContent = "▶"));
  audio.addEventListener("timeupdate", () => {
    const ms = audio.currentTime * 1000;
    $("clock").textContent = `${fmt(ms)} / ${fmt(R.duration)}`;
    const i = currentAyahByTime(ms);
    if (i >= 0 && i !== R.current) setCurrent(i);
    if (R.loop && R.current >= 0 && ms >= ayahEndMs(R.current)) seekAyah(R.current, true);
    drawWave();
  });

  const c = $("wave");
  c.addEventListener("mousedown", (e) => {
    const mi = markerNearX(e.clientX);
    if (mi >= 0) { R.drag = mi; c.style.cursor = "ew-resize"; }
    else { audio.currentTime = xToMs(e.clientX) / 1000; drawWave(); }
  });
  window.addEventListener("mousemove", (e) => { if (R.drag < 0) return; R.ayahs[R.drag].time = xToMs(e.clientX); drawWave(); });
  window.addEventListener("mouseup", (e) => {
    if (R.drag < 0) return;
    const i = R.drag; R.drag = -1; c.style.cursor = "pointer";
    commitMarker(i, xToMs(e.clientX));
  });
  window.addEventListener("resize", () => { if (!$("review").classList.contains("hidden")) { resizeWave(); drawWave(); } });

  window.addEventListener("keydown", (e) => {
    if ($("review").classList.contains("hidden") || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.code === "ArrowDown" || e.code === "ArrowRight") { e.preventDefault(); $("nextBtn").click(); }
    else if (e.code === "ArrowUp" || e.code === "ArrowLeft") { e.preventDefault(); $("prevBtn").click(); }
    else if (e.key === "v") { if (R.current >= 0) toggleVerify(R.current); }
  });

  // native file dialogs only inside the pywebview window
  if (window.pywebview) enableNativePickers();
  else window.addEventListener("pywebviewready", enableNativePickers);
}

// init
(async function () {
  wire();
  try {
    const p = await api.get("/api/project");
    audioCount = p.audio_count || 0;
    if (p.mode) $("mode").value = p.mode;
    if (p.audio_dir) $("audioDir").value = p.audio_dir;
    if (p.db_path) $("dbPath").value = p.db_path;
    if (p.db_path && p.audio_count) { $("genBtn").disabled = false; $("help").classList.add("collapsed"); }
  } catch {}
  await refreshSuras();
})();
