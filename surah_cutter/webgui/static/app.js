"use strict";

// ---- tiny API helpers ----------------------------------------------------
const api = {
  async get(url) { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) throw new Error((await r.json()).detail || r.status); return r.json(); },
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
const modeLabel = (mode) => ({ full: "точно", fast: "быстро", hybrid: "hybrid" }[mode] || mode);
function selectedMode() { return $("genMode") ? $("genMode").value : "full"; }
function selectedSuraMode() { return $("suraMode") ? $("suraMode").value : selectedMode(); }
function setMode(mode) {
  const m = mode === "fast" ? "fast" : "full";
  if ($("genMode")) $("genMode").value = m;
  if ($("suraMode")) $("suraMode").value = m;
  updateModeHint();
}
function updateModeHint() {
  const mode = selectedMode();
  $("modeHint").textContent = mode === "fast"
    ? "паузы → растущее окно; трудные суры проверять внимательнее"
    : "полное нейровыравнивание (~57 мс)";
}

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
  const mode = selectedMode();
  // Explicit start: confirm the deliberate, long-running action.
  const what = onlyMissing
    ? "недостающих (ещё не сгенерированных) сур"
    : `ВСЕХ ${audioCount || ""} сур заново`;
  const msg = `Запустить генерацию ${what} в режиме «${modeLabel(mode)}»?\n`
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
    const fast = d.fast_trace ? ` · пауз ${d.fast_trace.n_pause}, wasl ${d.fast_trace.n_wasl}, окно ${(d.fast_trace.coverage || 0).toFixed(2)}×` : "";
    $("genCur").textContent = `сура ${d.surah} готова${d.written != null ? ` · ${d.written} аят.` : ""}${fast}`;
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
const R = {
  surah: null, ayahs: [], duration: 0, peaks: [], audio: null, current: -1,
  loop: false, drag: -1, zoom: 1, phase1: null, showPhase1: false, phaseCurrent: -1,
  dsp: null, showDsp: false,
  combined: null, showCombined: false,
  showTen: false, tenRanges: new Map(), tenLoading: false, tenTimer: null,
  tenThreshold: 0.5, tenMergeGap: 180,
  showVad: false, vadRanges: new Map(), vadLoading: false, vadTimer: null,
};

async function openReview(surah) {
  R.surah = surah;
  setStatus("Загрузка суры " + surah + "…");
  let sura, wave;
  try {
    [sura, wave] = await Promise.all([api.get(`/api/sura/${surah}`), api.get(`/api/sura/${surah}/waveform`)]);
  } catch (e) { setStatus("Ошибка: " + e.message); return; }
  R.ayahs = sura.ayahs; R.peaks = wave.peaks; R.duration = wave.duration_ms; R.current = -1;
  R.phase1 = null; R.showPhase1 = false; R.phaseCurrent = -1;
  R.dsp = null; R.showDsp = false;
  R.combined = null; R.showCombined = false;
  R.showTen = false; R.tenRanges = new Map(); R.tenLoading = false;
  if (R.tenTimer) { clearTimeout(R.tenTimer); R.tenTimer = null; }
  R.showVad = false; R.vadRanges = new Map(); R.vadLoading = false;
  if (R.vadTimer) { clearTimeout(R.vadTimer); R.vadTimer = null; }
  R.zoom = 1;
  $("zoomRange").value = "1";
  $("zoomLabel").textContent = "1×";
  $("phase1Btn").classList.remove("active");
  $("phase1Btn").textContent = "фаза 1";
  $("dspBtn").classList.remove("active");
  $("dspBtn").textContent = "DSP";
  $("combinedBtn").classList.remove("active");
  $("combinedBtn").textContent = "фаза 1 итог";
  $("tenBtn").classList.remove("active");
  $("tenBtn").textContent = "TEN";
  $("vadBtn").classList.remove("active");
  $("vadBtn").textContent = "VAD";
  $("tenThreshold").value = String(R.tenThreshold);
  $("tenMergeGap").value = String(R.tenMergeGap);
  updateWaveControls();
  $("reviewTitle").textContent = `Сура ${surah} · ${sura.ayah_count} аят.`;
  const audio = $("audio");
  audio.src = `/api/audio/${surah}`;
  audio.playbackRate = parseFloat($("rate").value);
  R.audio = audio;
  $("suraGenStatus").textContent = "";
  $("suraGenStatus").className = "sura-gen-status";
  renderAyahList();
  updateGenHint();
  updateSuraProgress();
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
    hint.textContent = "⚠ Сура не рассчитана — нажмите «Рассчитать суру». Без тайм-кодов подсветки и маркеров нет.";
    hint.classList.add("warn");
    $("regenSuraBtn").textContent = "▶ Рассчитать суру";
  }
}
function updateSuraProgress(state = "") {
  const total = R.ayahs.length;
  const marked = R.ayahs.filter((a) => a.time != null).length;
  const edited = R.ayahs.filter((a) => a.status === "edited").length;
  const verified = R.ayahs.filter((a) => a.status === "verified").length;
  const pct = total ? Math.round((marked / total) * 100) : 0;
  $("suraProgressFill").style.right = (100 - pct) + "%";
  $("suraProgressPct").textContent = `${pct}%`;
  $("suraProgressText").textContent = state === "running"
    ? `расчёт суры… размечено ${marked} / ${total}`
    : `размечено ${marked} / ${total}`;
  $("suraProgressMeta").textContent = `проверено ${verified} · вручную ${edited} · осталось ${Math.max(0, total - marked)}`;
  $("suraProgress").className = "sura-progress"
    + (state === "running" ? " running" : "")
    + (total > 0 && marked === total ? " done" : "");
}

async function togglePhase1() {
  if (R.surah == null) return;
  const btn = $("phase1Btn");
  if (R.showPhase1) {
    R.showPhase1 = false;
    R.phaseCurrent = -1;
    btn.classList.remove("active");
    btn.textContent = "фаза 1";
    drawWave();
    return;
  }
  btn.disabled = true;
  btn.textContent = "фаза 1…";
  try {
    if (!R.phase1) R.phase1 = await api.get(`/api/sura/${R.surah}/fast-phase1`);
    R.showPhase1 = true;
    R.phaseCurrent = currentPhaseByTime(R.audio ? R.audio.currentTime * 1000 : 0);
    btn.classList.add("active");
    btn.textContent = `фаза 1 · ${R.phase1.pauses.length}`;
    drawWave();
  } catch (e) {
    setStatus("Ошибка фазы 1: " + e.message);
    btn.textContent = "фаза 1";
  } finally {
    btn.disabled = false;
  }
}

async function toggleCombined() {
  if (R.surah == null) return;
  const btn = $("combinedBtn");
  if (R.showCombined) {
    R.showCombined = false;
    btn.classList.remove("active");
    btn.textContent = "фаза 1 итог";
    drawWave();
    return;
  }
  btn.disabled = true;
  btn.textContent = "фаза 1 итог…";
  try {
    if (!R.combined) R.combined = await api.get(`/api/sura/${R.surah}/fast-combined-phase1`);
    R.showCombined = true;
    btn.classList.add("active");
    btn.textContent = `фаза 1 итог · ${R.combined.pauses.length}`;
    drawWave();
  } catch (e) {
    setStatus("Ошибка фазы 1 итог: " + e.message);
    btn.textContent = "фаза 1 итог";
  } finally {
    btn.disabled = false;
  }
}

async function toggleDsp() {
  if (R.surah == null) return;
  const btn = $("dspBtn");
  if (R.showDsp) {
    R.showDsp = false;
    btn.classList.remove("active");
    btn.textContent = "DSP";
    drawWave();
    return;
  }
  btn.disabled = true;
  btn.textContent = "DSP…";
  try {
    if (!R.dsp) R.dsp = await api.get(`/api/sura/${R.surah}/pause-pipeline`);
    R.showDsp = true;
    const accepted = (R.dsp.decisions || []).filter((d) => d.accepted).length;
    btn.classList.add("active");
    btn.textContent = `DSP · ${accepted}/${R.dsp.decisions.length}`;
    drawWave();
  } catch (e) {
    setStatus("Ошибка DSP: " + e.message);
    btn.textContent = "DSP";
  } finally {
    btn.disabled = false;
  }
}

async function toggleTen() {
  if (R.surah == null) return;
  const btn = $("tenBtn");
  if (R.showTen) {
    R.showTen = false;
    btn.classList.remove("active");
    btn.textContent = "TEN";
    drawWave();
    return;
  }
  R.showTen = true;
  btn.classList.add("active");
  btn.textContent = "TEN…";
  drawWave();
  await loadVisibleTen();
}

async function toggleVad() {
  if (R.surah == null) return;
  const btn = $("vadBtn");
  if (R.showVad) {
    R.showVad = false;
    btn.classList.remove("active");
    btn.textContent = "VAD";
    drawWave();
    return;
  }
  R.showVad = true;
  btn.classList.add("active");
  btn.textContent = "VAD…";
  drawWave();
  await loadVisibleVad();
}

function visibleMsRange(padRatio = 0.15) {
  const wrap = $("waveScroll"), content = $("waveContent");
  const left = wrap.scrollLeft, right = left + wrap.clientWidth;
  const ms0 = (left / Math.max(1, content.clientWidth)) * R.duration;
  const ms1 = (right / Math.max(1, content.clientWidth)) * R.duration;
  const pad = Math.max(1000, (ms1 - ms0) * padRatio);
  return [Math.max(0, ms0 - pad), Math.min(R.duration, ms1 + pad)];
}

function tenRangeKey(startMs, endMs) {
  const bucket = 30_000;
  return [
    Math.floor(startMs / bucket) * bucket,
    Math.ceil(endMs / bucket) * bucket,
    R.tenThreshold.toFixed(2),
    Math.round(R.tenMergeGap),
  ].join(":");
}

function tenPausesForView() {
  const out = [];
  for (const r of R.tenRanges.values()) out.push(...(r.pauses || []));
  const seen = new Set();
  return out
    .sort((a, b) => a.start_ms - b.start_ms)
    .filter((p) => {
      const key = `${Math.round(p.start_ms)}:${Math.round(p.end_ms)}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

async function loadVisibleTen() {
  if (!R.showTen || R.tenLoading || R.surah == null || !R.duration) return;
  const [startMs, endMs] = visibleMsRange();
  const key = tenRangeKey(startMs, endMs);
  if (R.tenRanges.has(key)) { updateTenButton(); drawWave(); return; }
  const [start, end] = key.split(":").map(Number);
  R.tenLoading = true;
  updateTenButton("…");
  try {
    const params = new URLSearchParams({
      start_ms: String(start),
      end_ms: String(end),
      threshold: String(R.tenThreshold),
      merge_gap_ms: String(R.tenMergeGap),
    });
    const data = await api.get(`/api/sura/${R.surah}/ten-phase?${params.toString()}`);
    R.tenRanges.set(key, data);
    updateTenButton();
    drawWave();
  } catch (e) {
    setStatus("Ошибка TEN: " + e.message);
    R.showTen = false;
    $("tenBtn").classList.remove("active");
    $("tenBtn").textContent = "TEN";
  } finally {
    R.tenLoading = false;
  }
}

function scheduleTenLoad() {
  if (!R.showTen) return;
  if (R.tenTimer) clearTimeout(R.tenTimer);
  R.tenTimer = setTimeout(loadVisibleTen, 220);
}

function updateTenButton(suffix = "") {
  const n = tenPausesForView().length;
  $("tenBtn").textContent = suffix ? `TEN${suffix}` : `TEN · ${n}`;
}

function vadRangeKey(startMs, endMs) {
  const bucket = 30_000;
  return `${Math.floor(startMs / bucket) * bucket}:${Math.ceil(endMs / bucket) * bucket}`;
}

function vadPausesForView() {
  const out = [];
  for (const r of R.vadRanges.values()) out.push(...(r.pauses || []));
  const seen = new Set();
  return out
    .sort((a, b) => a.start_ms - b.start_ms)
    .filter((p) => {
      const key = `${Math.round(p.start_ms)}:${Math.round(p.end_ms)}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

async function loadVisibleVad() {
  if (!R.showVad || R.vadLoading || R.surah == null || !R.duration) return;
  const [startMs, endMs] = visibleMsRange();
  const key = vadRangeKey(startMs, endMs);
  if (R.vadRanges.has(key)) { updateVadButton(); drawWave(); return; }
  const [start, end] = key.split(":").map(Number);
  R.vadLoading = true;
  updateVadButton("…");
  try {
    const params = new URLSearchParams({
      start_ms: String(start),
      end_ms: String(end),
    });
    const data = await api.get(`/api/sura/${R.surah}/vad-phase?${params.toString()}`);
    R.vadRanges.set(key, data);
    updateVadButton();
    drawWave();
  } catch (e) {
    setStatus("Ошибка VAD: " + e.message);
    R.showVad = false;
    $("vadBtn").classList.remove("active");
    $("vadBtn").textContent = "VAD";
  } finally {
    R.vadLoading = false;
  }
}

function scheduleVadLoad() {
  if (!R.showVad) return;
  if (R.vadTimer) clearTimeout(R.vadTimer);
  R.vadTimer = setTimeout(loadVisibleVad, 220);
}

function updateVadButton(suffix = "") {
  const n = vadPausesForView().length;
  $("vadBtn").textContent = suffix ? `VAD${suffix}` : `VAD · ${n}`;
}

function updateTenSettings() {
  const threshold = parseFloat($("tenThreshold").value);
  const mergeGap = parseFloat($("tenMergeGap").value);
  R.tenThreshold = Number.isFinite(threshold) ? Math.max(0.05, Math.min(0.95, threshold)) : 0.5;
  R.tenMergeGap = Number.isFinite(mergeGap) ? Math.max(0, Math.min(800, mergeGap)) : 180;
  $("tenThreshold").value = String(R.tenThreshold);
  $("tenMergeGap").value = String(Math.round(R.tenMergeGap));
  R.tenRanges = new Map();
  if (R.showTen) {
    updateTenButton("…");
    drawWave();
    scheduleTenLoad();
  }
}

// regenerate just this sura with the chosen per-sura mode
async function regenerateSura() {
  if (R.surah == null) return;
  const surah = R.surah, mode = selectedSuraMode();
  const manual = R.ayahs.filter((a) => a.status === "edited" || a.status === "verified").length;
  if (manual > 0 && !confirm(
    `В этой суре ${manual} аят(ов) с ручной правкой/проверкой.\n`
    + `Пересчитать только auto-аяты (${modeLabel(mode)})? Ручные правки не затрагиваются.`)) return;
  const btn = $("regenSuraBtn"), st = $("suraGenStatus");
  btn.disabled = true;
  const t0 = Date.now();
  const tick = () => {
    st.textContent = `расчёт… ${Math.floor((Date.now() - t0) / 1000)}с`;
    updateSuraProgress("running");
  };
  st.className = "sura-gen-status running"; tick();
  const timer = setInterval(tick, 1000);
  const stop = (cls, text) => { clearInterval(timer); btn.disabled = false; st.className = "sura-gen-status " + cls; st.textContent = text; updateSuraProgress(); };
  try {
    await api.post("/api/generate", { suras: [surah], mode });
  } catch (e) { stop("err", "ошибка: " + e.message); return; }
  const es = new EventSource("/api/generate/stream");
  es.onmessage = async (ev) => {
    const d = JSON.parse(ev.data);
    if (d.type === "sura" && d.surah === surah) {
      await reloadSura();
      st.textContent = `записано ${d.written ?? 0} аят.`;
    }
    if (d.type === "done" || d.type === "cancelled") {
      es.close(); stop("ok", `✓ готово за ${Math.floor((Date.now() - t0) / 1000)}с`);
      await reloadSura(); refreshSuras();
    } else if (d.type === "error") { es.close(); stop("err", "ошибка: " + d.message); }
  };
  es.onerror = () => { es.close(); if (btn.disabled) stop("", ""); };
}

async function showFastLog() {
  if (R.surah == null) return;
  const btn = $("fastLogBtn");
  const panel = $("fastLogPanel");
  const pre = $("fastLogText");
  panel.classList.remove("hidden");
  pre.textContent = "читаю fast-лог…";
  btn.disabled = true;
  try {
    const data = await api.get(`/api/sura/${R.surah}/fast-log`);
    pre.textContent = data.log || "";
  } catch (e) {
    pre.textContent = "Ошибка лога: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// re-fetch this sura's timings after a recompute, keep audio position
async function reloadSura() {
  try {
    const sura = await api.get(`/api/sura/${R.surah}`);
    R.ayahs = sura.ayahs;
  } catch { return; }
  renderAyahList();
  updateGenHint();
  updateSuraProgress();
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
    const marked = a.time != null;
    li.classList.add(marked ? "marked" : "missing");
    if (a.status === "verified") li.classList.add("verified");
    else if (a.status === "edited") li.classList.add("edited");
    li.innerHTML = `
      <span class="a-conf" style="background:${confColor(a.confidence, a.status)}"></span>
      <span class="a-num">${a.ayah}</span>
      <span class="a-time">${fmt(a.time)}</span>
      <span class="a-mark">${marked ? "разм." : "нет"}</span>
      <button class="a-verify" ${marked ? "" : "disabled"}>${a.status === "verified" ? "✓ пров." : "пров."}</button>
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
  const c = $("wave"), wrap = $("waveScroll"), content = $("waveContent"), dpr = window.devicePixelRatio || 1;
  content.style.width = Math.max(wrap.clientWidth, wrap.clientWidth * R.zoom) + "px";
  c.style.width = wrap.clientWidth + "px";
  c.width = Math.round(wrap.clientWidth * dpr);
  c.height = Math.round(c.clientHeight * dpr);
  c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
}
function setWaveZoom(zoom, centerMs = null) {
  const oldZoom = R.zoom;
  const wrap = $("waveScroll"), content = $("waveContent");
  const centerRatio = centerMs != null && R.duration
    ? centerMs / R.duration
    : (wrap.scrollLeft + wrap.clientWidth / 2) / Math.max(1, content.clientWidth);
  R.zoom = Math.max(1, Math.min(256, Math.round(zoom)));
  updateWaveControls();
  resizeWave();
  if (R.zoom !== oldZoom) {
    wrap.scrollLeft = Math.max(0, centerRatio * content.clientWidth - wrap.clientWidth / 2);
  }
  drawWave();
  scheduleTenLoad();
  scheduleVadLoad();
}
function updateWaveControls() {
  if (!$("zoomRange")) return;
  $("zoomRange").value = String(R.zoom);
  $("zoomLabel").textContent = `${R.zoom}×`;
}
function msToVirtualX(ms) {
  return (ms / Math.max(1, R.duration)) * $("waveContent").clientWidth;
}
function drawWave() {
  const c = $("wave"), wrap = $("waveScroll"), content = $("waveContent"), ctx = c.getContext("2d");
  const w = c.clientWidth, h = c.clientHeight, mid = h / 2;
  ctx.clearRect(0, 0, w, h);
  const n = R.peaks.length;
  const left = wrap.scrollLeft, right = left + w, virtualW = content.clientWidth;
  ctx.strokeStyle = "#3a4456"; ctx.lineWidth = 1; ctx.beginPath();
  const i0 = Math.max(0, Math.floor((left / Math.max(1, virtualW)) * Math.max(0, n - 1)) - 2);
  const i1 = Math.min(n - 1, Math.ceil((right / Math.max(1, virtualW)) * Math.max(0, n - 1)) + 2);
  for (let i = i0; i <= i1; i++) {
    const vx = (i / Math.max(1, n - 1)) * virtualW;
    const x = vx - left, amp = (R.peaks[i] / 100) * (mid - 2);
    ctx.moveTo(x, mid - amp); ctx.lineTo(x, mid + amp);
  }
  ctx.stroke();
  if (R.showPhase1 && R.phase1) drawPhase1(ctx, left, right, h);
  if (R.showDsp && R.dsp) drawDsp(ctx, left, right, h);
  if (R.showCombined && R.combined) drawCombined(ctx, left, right, h);
  if (R.showTen) drawTen(ctx, left, right, h);
  if (R.showVad) drawVad(ctx, left, right, h);
  R.ayahs.forEach((a, i) => {
    if (a.time == null || !R.duration) return;
    const vx = msToVirtualX(a.time);
    if (vx < left - 1 || vx > right + 1) return;
    const x = vx - left;
    ctx.strokeStyle = i === R.current ? "#ffd56b" : "#d4a843";
    ctx.lineWidth = i === R.current ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = "10px sans-serif";
    ctx.fillText(a.ayah, x + 3, 11);
  });
  if (R.audio && R.duration) {
    const playMs = R.audio.currentTime * 1000;
    const px = msToVirtualX(playMs) - left;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
  }
}

function drawDsp(ctx, left, right, h) {
  ctx.font = "10px sans-serif";
  for (const p of R.dsp.decisions || []) {
    const start = msToVirtualX(p.start_ms);
    const end = msToVirtualX(p.end_ms);
    if (end < left - 1 || start > right + 1) continue;
    const x0 = Math.max(0, start - left);
    const x1 = Math.min(right - left, end - left);
    ctx.fillStyle = p.accepted ? "rgba(255, 166, 61, 0.10)" : "rgba(255, 93, 93, 0.14)";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    const mark = msToVirtualX(p.center_ms) - left;
    ctx.strokeStyle = p.accepted ? "#ffa63d" : "#ff5d5d";
    ctx.lineWidth = p.accepted ? 1.5 : 2;
    ctx.beginPath(); ctx.moveTo(mark, 0); ctx.lineTo(mark, h); ctx.stroke();
    if (mark >= 0 && mark <= right - left) {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillText(p.accepted ? Math.round(p.depth_ms) : "x", mark + 3, 66);
    }
  }
}

function drawCombined(ctx, left, right, h) {
  ctx.font = "10px sans-serif";
  for (const p of R.combined.pauses || []) {
    const start = msToVirtualX(p.start_ms);
    const end = msToVirtualX(p.end_ms);
    if (end < left - 1 || start > right + 1) continue;
    const x0 = Math.max(0, start - left);
    const x1 = Math.min(right - left, end - left);
    ctx.fillStyle = "rgba(49, 211, 214, 0.11)";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    const mark = msToVirtualX(p.center_ms) - left;
    ctx.strokeStyle = "#31d3d6";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(mark, 0); ctx.lineTo(mark, h); ctx.stroke();
    if (mark >= 0 && mark <= right - left) {
      ctx.fillStyle = "#9af4f6";
      ctx.fillText(Math.round(p.depth_ms), mark + 3, 52);
    }
  }
}

function drawTen(ctx, left, right, h) {
  ctx.font = "10px sans-serif";
  for (const p of tenPausesForView()) {
    const start = msToVirtualX(p.start_ms);
    const end = msToVirtualX(p.end_ms);
    if (end < left - 1 || start > right + 1) continue;
    const x0 = Math.max(0, start - left);
    const x1 = Math.min(right - left, end - left);
    ctx.fillStyle = "rgba(178, 113, 255, 0.10)";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    const mark = msToVirtualX(p.center_ms) - left;
    ctx.strokeStyle = "#b271ff";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(mark, 0); ctx.lineTo(mark, h); ctx.stroke();
    if (mark >= 0 && mark <= right - left) {
      ctx.fillStyle = "#d6bdff";
      ctx.fillText(Math.round(p.depth_ms), mark + 3, 24);
    }
  }
}

function drawVad(ctx, left, right, h) {
  ctx.font = "10px sans-serif";
  for (const p of vadPausesForView()) {
    const start = msToVirtualX(p.start_ms);
    const end = msToVirtualX(p.end_ms);
    if (end < left - 1 || start > right + 1) continue;
    const x0 = Math.max(0, start - left);
    const x1 = Math.min(right - left, end - left);
    ctx.fillStyle = "rgba(73, 214, 130, 0.10)";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    const mark = msToVirtualX(p.center_ms) - left;
    ctx.strokeStyle = "#49d682";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(mark, 0); ctx.lineTo(mark, h); ctx.stroke();
    if (mark >= 0 && mark <= right - left) {
      ctx.fillStyle = "#93f0ba";
      ctx.fillText(Math.round(p.depth_ms), mark + 3, 38);
    }
  }
}

function drawPhase1(ctx, left, right, h) {
  ctx.font = "10px sans-serif";
  const marks = phaseMarks();
  const cur = R.audio ? currentPhaseByTime(R.audio.currentTime * 1000) : R.phaseCurrent;
  if (marks && cur >= 0) {
    const start = msToVirtualX(marks[cur]);
    const end = msToVirtualX(marks[cur + 1]);
    if (end >= left && start <= right) {
      const x0 = Math.max(0, start - left);
      const x1 = Math.min(right - left, end - left);
      ctx.fillStyle = "rgba(74, 144, 217, 0.09)";
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    }
  }
  for (const p of R.phase1.pauses || []) {
    const start = msToVirtualX(p.start_ms);
    const end = msToVirtualX(p.end_ms);
    if (end < left - 1 || start > right + 1) continue;
    const x0 = Math.max(0, start - left);
    const x1 = Math.min(right - left, end - left);
    ctx.fillStyle = "rgba(74, 144, 217, 0.13)";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    const markMs = p.center_ms ?? p.onset_ms;
    const onset = msToVirtualX(markMs) - left;
    ctx.strokeStyle = "#4a90d9";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(onset, 0); ctx.lineTo(onset, h); ctx.stroke();
    if (onset >= 0 && onset <= right - left) {
      ctx.fillStyle = "#8fc4ff";
      ctx.fillText(Math.round(p.depth_ms), onset + 3, h - 7);
    }
  }
}
function xToMs(clientX) {
  const rect = $("wave").getBoundingClientRect();
  const vx = $("waveScroll").scrollLeft + (clientX - rect.left);
  return Math.max(0, Math.min(R.duration, (vx / Math.max(1, $("waveContent").clientWidth)) * R.duration));
}
function markerNearX(clientX) {
  const rect = $("wave").getBoundingClientRect(), tol = 7, left = $("waveScroll").scrollLeft;
  for (let i = 0; i < R.ayahs.length; i++) {
    const t = R.ayahs[i].time; if (t == null) continue;
    const x = rect.left + msToVirtualX(t) - left;
    if (Math.abs(clientX - x) <= tol) return i;
  }
  return -1;
}

// ---- playback ----
function togglePlay() { if (!R.audio) return; R.audio.paused ? R.audio.play() : R.audio.pause(); }
function scrollWaveToMs(ms) {
  if (R.zoom <= 1) return;
  const wrap = $("waveScroll"), x = msToVirtualX(ms);
  wrap.scrollLeft = Math.max(0, x - wrap.clientWidth / 2);
}
function seekAyah(i, play) {
  const a = R.ayahs[i]; if (!a || a.time == null || !R.audio) return;
  R.audio.currentTime = a.time / 1000;
  setCurrent(i);
  scrollWaveToMs(a.time);
  drawWave();
  if (play) R.audio.play();
}
function phaseMarks() {
  if (!R.showPhase1 || !R.phase1 || !R.duration) return null;
  const raw = [0, ...(R.phase1.pauses || []).map((p) => p.center_ms ?? p.onset_ms), R.duration]
    .filter((t) => t != null && Number.isFinite(t))
    .map((t) => Math.max(0, Math.min(R.duration, t)))
    .sort((a, b) => a - b);
  const marks = [];
  for (const t of raw) {
    if (!marks.length || t - marks[marks.length - 1] > 1) marks.push(t);
  }
  return marks.length >= 2 ? marks : null;
}
function currentPhaseByTime(ms) {
  const marks = phaseMarks();
  if (!marks) return -1;
  let idx = 0;
  for (let i = 0; i < marks.length - 1; i++) {
    if (marks[i] <= ms + 1) idx = i;
    else break;
  }
  return Math.min(idx, marks.length - 2);
}
function phaseEndMs(i) {
  const marks = phaseMarks();
  if (!marks || i < 0) return R.duration;
  return marks[Math.min(i + 1, marks.length - 1)];
}
function seekPhase(i, play) {
  const marks = phaseMarks();
  if (!marks || !R.audio) return;
  const idx = Math.max(0, Math.min(i, marks.length - 2));
  R.phaseCurrent = idx;
  R.audio.currentTime = marks[idx] / 1000;
  scrollWaveToMs(marks[idx]);
  drawWave();
  if (play) R.audio.play();
}
function seekPrevUnit() {
  if (phaseMarks()) seekPhase(currentPhaseByTime(R.audio ? R.audio.currentTime * 1000 : 0) - 1, false);
  else seekAyah(Math.max(R.current - 1, 0), false);
}
function seekNextUnit() {
  const marks = phaseMarks();
  if (marks) seekPhase(currentPhaseByTime(R.audio ? R.audio.currentTime * 1000 : 0) + 1, false);
  else seekAyah(Math.min(R.current + 1, R.ayahs.length - 1), false);
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
    renderAyahList(); updateSuraProgress(); const cur = R.current; R.current = -1; setCurrent(cur); drawWave();
  } catch (e) { setStatus("Ошибка: " + e.message); }
}
async function commitMarker(i, ms) {
  const a = R.ayahs[i];
  try {
    await api.post("/api/timing", { surah: R.surah, ayah: a.ayah, time: Math.round(ms) });
    a.time = Math.round(ms); a.status = "edited";
    renderAyahList(); updateGenHint(); updateSuraProgress(); drawWave();
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
  $("fastLogBtn").onclick = showFastLog;
  $("closeFastLogBtn").onclick = () => $("fastLogPanel").classList.add("hidden");
  $("zoomOutBtn").onclick = () => setWaveZoom(R.zoom / 2);
  $("zoomInBtn").onclick = () => setWaveZoom(R.zoom * 2);
  $("zoomFitBtn").onclick = () => setWaveZoom(1, R.duration / 2);
  $("phase1Btn").onclick = togglePhase1;
  $("dspBtn").onclick = toggleDsp;
  $("combinedBtn").onclick = toggleCombined;
  $("tenBtn").onclick = toggleTen;
  $("vadBtn").onclick = toggleVad;
  $("tenThreshold").onchange = updateTenSettings;
  $("tenMergeGap").onchange = updateTenSettings;
  $("zoomRange").oninput = () => setWaveZoom(parseInt($("zoomRange").value, 10), R.audio ? R.audio.currentTime * 1000 : null);
  $("genMode").onchange = async () => {
    setMode(selectedMode());
    try { await api.post("/api/settings", { mode: selectedMode() }); } catch {}
  };
  $("suraMode").onchange = () => setMode(selectedSuraMode());
  $("backBtn").onclick = () => { if (R.audio) R.audio.pause(); showView("dashboard"); refreshSuras(); };
  $("playBtn").onclick = togglePlay;
  $("prevBtn").onclick = seekPrevUnit;
  $("nextBtn").onclick = seekNextUnit;
  $("loopBtn").onclick = () => { R.loop = !R.loop; $("loopBtn").classList.toggle("active", R.loop); };
  $("rate").onchange = () => { if (R.audio) R.audio.playbackRate = parseFloat($("rate").value); };
  $("exportBtn").onclick = exportProject;

  const audio = $("audio");
  audio.addEventListener("play", () => ($("playBtn").textContent = "⏸"));
  audio.addEventListener("pause", () => ($("playBtn").textContent = "▶"));
  audio.addEventListener("timeupdate", () => {
    const ms = audio.currentTime * 1000;
    $("clock").textContent = `${fmt(ms)} / ${fmt(R.duration)}`;
    const activePhase = phaseMarks();
    if (activePhase) R.phaseCurrent = currentPhaseByTime(ms);
    else {
      const i = currentAyahByTime(ms);
      if (i >= 0 && i !== R.current) setCurrent(i);
    }
    if (R.zoom > 1) {
      const wrap = $("waveScroll"), x = msToVirtualX(ms);
      if (x < wrap.scrollLeft || x > wrap.scrollLeft + wrap.clientWidth) {
        wrap.scrollLeft = Math.max(0, x - wrap.clientWidth / 2);
      }
    }
    if (R.loop && activePhase && R.phaseCurrent >= 0 && ms >= phaseEndMs(R.phaseCurrent)) seekPhase(R.phaseCurrent, true);
    else if (R.loop && !activePhase && R.current >= 0 && ms >= ayahEndMs(R.current)) seekAyah(R.current, true);
    drawWave();
  });

  const c = $("wave");
  $("waveScroll").addEventListener("scroll", () => { drawWave(); scheduleTenLoad(); scheduleVadLoad(); });
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
    if (p.audio_dir) $("audioDir").value = p.audio_dir;
    if (p.db_path) $("dbPath").value = p.db_path;
    setMode(p.mode);
    if (p.db_path && p.audio_count) { $("genBtn").disabled = false; $("help").classList.add("collapsed"); }
  } catch {}
  await refreshSuras();
})();
