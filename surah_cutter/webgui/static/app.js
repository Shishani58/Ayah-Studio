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
function resetLiveLog(id) {
  const el = $(id);
  if (!el) return;
  el.textContent = "";
  el.dataset.lines = "0";
  updateLogCopyButton(id);
}
function appendLiveLog(id, line) {
  const el = $(id);
  if (!el) return;
  const box = el.closest(".log-box");
  if (box && box.classList.contains("hidden") && id !== "suraLiveLog") {
    box.classList.remove("hidden");
  }
  if (el.classList.contains("hidden")) el.classList.remove("hidden");
  const lines = el.textContent ? el.textContent.split("\n").filter(Boolean) : [];
  lines.push(line);
  while (lines.length > 200) lines.shift();
  el.textContent = lines.join("\n");
  el.dataset.lines = String(lines.length);
  updateLogCopyButton(id);
  el.scrollTop = el.scrollHeight;
}
function logCopyButton(id) {
  return document.querySelector(`[data-copy-log="${id}"]`);
}
function updateLogCopyButton(id) {
  const btn = logCopyButton(id), el = $(id);
  if (!btn || !el) return;
  btn.disabled = !el.textContent.trim();
  if (!btn.disabled && !btn.dataset.busy) btn.textContent = "Копировать";
}
async function copyLiveLog(id) {
  const el = $(id), btn = logCopyButton(id);
  const text = el ? el.textContent.trim() : "";
  if (!text || !btn) return;
  btn.disabled = true;
  btn.dataset.busy = "1";
  try {
    await writeClipboard(text);
    btn.textContent = "Скопировано";
  } catch {
    btn.textContent = "Не скопировано";
  } finally {
    setTimeout(() => {
      delete btn.dataset.busy;
      updateLogCopyButton(id);
    }, 1200);
  }
}
async function writeClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  let ok = false;
  try {
    area.select();
    ok = document.execCommand("copy");
  } finally {
    document.body.removeChild(area);
  }
  if (!ok) throw new Error("copy failed");
}
const modeLabel = (mode) => ({ auto: "авто", full: "точно", tiled: "точно·большие", fast: "быстро", hybrid: "hybrid" }[mode] || mode);
function selectedMode() { return $("genMode") ? $("genMode").value : "full"; }
function selectedSuraMode() { return $("suraMode") ? $("suraMode").value : selectedMode(); }
function setMode(mode) {
  const m = (mode === "auto" || mode === "fast" || mode === "tiled" || mode === "full") ? mode : "auto";
  if ($("genMode")) $("genMode").value = m;
  if ($("suraMode")) $("suraMode").value = m;
  updateModeHint();
}
function updateModeHint() {
  const mode = selectedMode();
  $("modeHint").textContent = mode === "auto"
    ? "сам выбирает режим, калибрует лаг и ремонтирует подозрительные авто-метки"
    : mode === "fast"
    ? "паузы → растущее окно; трудные суры проверять внимательнее"
    : mode === "tiled"
    ? "точное выравнивание тайлами — для длинных сур (Бакъара и т.п.) без перегрузки памяти"
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
  resetLiveLog("genLog");
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
  else if (d.type === "sura_log") {
    appendLiveLog("genLog", `Сура ${d.surah}: ${d.line}`);
    if (R.surah === d.surah) appendLiveLog("suraLiveLog", d.line);
  }
  else if (d.type === "sura") {
    setProgress(d.index + 1, GEN.total, `${d.index + 1} / ${GEN.total}`);
    const fast = d.fast_trace
      ? ` · пауз ${d.fast_trace.n_pause}, wasl ${d.fast_trace.n_wasl}, окно ${(d.fast_trace.coverage || 0).toFixed(2)}×${d.fast_trace.drift_detected ? " · drift" : ""}`
      : "";
    $("genCur").textContent = `сура ${d.surah} готова${d.written != null ? ` · ${d.written} аят.` : ""}${fast}`;
    if (d.log_paths && d.log_paths.log) appendLiveLog("genLog", `Лог сохранён: ${d.log_paths.log}`);
    if (R.surah === d.surah) reloadSura();
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
  loop: false, drag: -1, zoom: 1, recheck: null,
  pausesVisible: false, pauseDecisions: null,
  pinch: null, logResize: null,
};

function setSuraLogVisible(visible) {
  $("suraLiveLogBox").classList.toggle("hidden", !visible);
  $("review").classList.toggle("log-open", visible);
}

function setSuraLogHeight(px) {
  const review = $("review");
  const min = 90;
  const max = Math.max(min, Math.round(window.innerHeight * 0.72));
  const height = Math.max(min, Math.min(max, Math.round(px)));
  review.style.setProperty("--sura-log-height", `${height}px`);
}

async function openReview(surah) {
  R.surah = surah;
  setStatus("Загрузка суры " + surah + "…");
  let sura, wave;
  try {
    [sura, wave] = await Promise.all([api.get(`/api/sura/${surah}`), api.get(`/api/sura/${surah}/waveform`)]);
  } catch (e) { setStatus("Ошибка: " + e.message); return; }
  R.ayahs = sura.ayahs; R.peaks = wave.peaks; R.duration = wave.duration_ms; R.current = -1;
  R.recheck = null;
  R.pausesVisible = false;
  R.pauseDecisions = null;
  R.zoom = adaptiveWaveZoom();
  R.pinch = null;
  resetLiveLog("suraLiveLog");
  setSuraLogVisible(false);
  updateWaveControls();
  $("reviewTitle").textContent = `Сура ${surah} · ${sura.ayah_count} аят.`;
  const audio = $("audio");
  audio.src = `/api/audio/${surah}`;
  audio.playbackRate = parseFloat($("rate").value);
  R.audio = audio;
  $("suraGenStatus").textContent = "";
  $("suraGenStatus").className = "sura-gen-status";
  updatePauseOverlayUi();
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
    hint.textContent = "Сура без тайм-кодов: выберите аят, слушайте волну и нажмите «Поставить старт» или «Разметить по паузам».";
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
  resetLiveLog("suraLiveLog");
  setSuraLogVisible(true);
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
    if (d.type === "sura_log" && d.surah === surah) {
      appendLiveLog("suraLiveLog", d.line);
      st.textContent = d.line;
    }
    if (d.type === "sura" && d.surah === surah) {
      await reloadSura();
      st.textContent = `записано ${d.written ?? 0} аят.`;
      if (d.log_paths && d.log_paths.log) appendLiveLog("suraLiveLog", `Лог сохранён: ${d.log_paths.log}`);
    }
    if (d.type === "done" || d.type === "cancelled") {
      es.close(); stop("ok", `✓ готово за ${Math.floor((Date.now() - t0) / 1000)}с`);
      await reloadSura(); refreshSuras();
    } else if (d.type === "error") { es.close(); stop("err", "ошибка: " + d.message); }
  };
  es.onerror = () => { es.close(); if (btn.disabled) stop("", ""); };
}

async function draftPausesSura() {
  if (R.surah == null) return;
  const btn = $("draftPausesBtn"), st = $("suraGenStatus");
  if (!confirm("Разметить суру по паузам? Ручные и проверенные метки не будут затронуты.")) return;
  btn.disabled = true;
  st.className = "sura-gen-status running";
  st.textContent = "разметка по паузам…";
  try {
    const data = await api.post(`/api/sura/${R.surah}/draft-pauses`, {});
    await reloadSura();
    refreshSuras();
    st.className = "sura-gen-status ok";
    st.textContent = `паузы: записано ${data.written}, найдено ${data.pauses}`;
  } catch (e) {
    st.className = "sura-gen-status err";
    st.textContent = "ошибка: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function correctDraftSura() {
  if (R.surah == null) return;
  const generated = R.ayahs.some((a) => a.time != null);
  if (!generated) { setStatus("Сначала поставьте метки или разметьте суру по паузам."); return; }
  const mode = selectedSuraMode();
  if (!confirm(`Запустить корректировку auto-тайм-кодов в режиме «${modeLabel(mode)}»? Ручные и проверенные метки не будут затронуты.`)) return;
  const btn = $("correctDraftBtn"), st = $("suraGenStatus");
  btn.disabled = true;
  const t0 = Date.now();
  const tick = () => { st.textContent = `корректировка… ${Math.floor((Date.now() - t0) / 1000)}с`; };
  st.className = "sura-gen-status running"; tick();
  const timer = setInterval(tick, 1000);
  try {
    const data = await api.post(`/api/sura/${R.surah}/correct-draft`, { mode });
    clearInterval(timer);
    await reloadSura();
    refreshSuras();
    st.className = "sura-gen-status ok";
    st.textContent = `корректировка: записано ${data.written} аят.`;
  } catch (e) {
    clearInterval(timer);
    st.className = "sura-gen-status err";
    st.textContent = "ошибка: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// Перепроверка: аудит готовых тайм-кодов суры (read-only, БД не меняет).
async function recheckSura() {
  if (R.surah == null) return;
  const generated = R.ayahs.some((a) => a.time != null);
  if (!generated) { setStatus("Сура не рассчитана — нечего перепроверять."); return; }
  const btn = $("recheckBtn"), st = $("suraGenStatus");
  btn.disabled = true;
  st.className = "sura-gen-status running";
  const t0 = Date.now();
  const tick = () => { st.textContent = `перепроверка… ${Math.floor((Date.now() - t0) / 1000)}с`; };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const data = await api.get(`/api/sura/${R.surah}/recheck`);
    R.recheck = {};
    for (const r of data.results) R.recheck[r.ayah] = r;
    const s = data.summary || {};
    clearInterval(timer);
    btn.disabled = false;
    const suspect = s.suspect || 0;
    st.className = "sura-gen-status " + (suspect ? "warn" : "ok");
    st.textContent = suspect
      ? `⚠ перепроверка: ${suspect} подозрит. из ${s.checked || 0}`
      : `✓ перепроверка: все ${s.checked || 0} в норме`;
    renderAyahList();
    drawWave();
  } catch (e) {
    clearInterval(timer);
    btn.disabled = false;
    st.className = "sura-gen-status err";
    st.textContent = "ошибка: " + e.message;
  }
}

// re-fetch this sura's timings after a recompute, keep audio position
async function reloadSura() {
  try {
    const sura = await api.get(`/api/sura/${R.surah}`);
    R.ayahs = sura.ayahs;
  } catch { return; }
  R.recheck = null;  // times changed → stale verification
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
    const rc = R.recheck ? R.recheck[a.ayah] : null;
    if (rc && rc.verdict === "shift") li.classList.add("rc-suspect");
    li.innerHTML = `
      <span class="a-conf" style="background:${confColor(a.confidence, a.status)}"></span>
      <span class="a-num">${a.ayah}</span>
      <span class="a-time">${fmt(a.time)}</span>
      <span class="a-mark">${marked ? "разм." : "нет"}</span>
      ${recheckBadge(rc)}
      <button class="a-verify" ${marked ? "" : "disabled"}>${a.status === "verified" ? "✓ пров." : "пров."}</button>
      <span class="a-text" dir="rtl" lang="ar">${escapeHtml(a.text || "")}</span>`;
    li.onclick = (e) => {
      if (e.target.classList.contains("a-verify")) { toggleVerify(i); e.stopPropagation(); }
      else if (marked) seekAyah(i, true);
      else setCurrent(i);
    };
    ul.appendChild(li);
  });
}
// recheck verdict pill: ✓ ok / →N "звучит как соседний аят". Empty until re-checked.
function recheckBadge(rc) {
  if (!rc || rc.verdict === "no_time") return `<span class="a-recheck"></span>`;
  const sc = (rc.score != null) ? rc.score.toFixed(2) : "—";
  if (rc.verdict === "shift") {
    return `<span class="a-recheck rc-shift" title="${escapeHtml(rc.hint)} (здесь слышно скорее начало аята ${rc.match_ayah}: ${rc.match_score.toFixed(2)} против ${sc})">→${rc.match_ayah}</span>`;
  }
  return `<span class="a-recheck rc-ok" title="слышно своё начало (совпадение ${sc})">✓</span>`;
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
function adaptiveWaveZoom() {
  if (!R.duration) return 1;
  const durationSec = R.duration / 1000;
  const ayahCount = Math.max(1, R.ayahs.length);
  const avgAyahSec = durationSec / ayahCount;
  const targetVisibleSec = Math.max(35, Math.min(80, avgAyahSec * 5));
  const zoom = durationSec / targetVisibleSec;
  return Math.max(1, Math.min(100, Math.round(zoom)));
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
  renderPauseInspector();
  drawWave();
}
function updateWaveControls() {
  if (!$("zoomLabel")) return;
  $("zoomLabel").textContent = `${R.zoom}×`;
}
function updatePauseOverlayUi() {
  const btn = $("pauseOverlayBtn"), legend = $("pauseLegend"), inspector = $("pauseInspector");
  if (!btn || !legend || !inspector) return;
  btn.classList.toggle("active", R.pausesVisible);
  legend.classList.toggle("hidden", !R.pausesVisible);
  inspector.classList.toggle("hidden", !R.pausesVisible);
  renderPauseInspector();
}
async function togglePauseOverlay() {
  if (R.surah == null) return;
  const btn = $("pauseOverlayBtn"), st = $("suraGenStatus");
  if (R.pausesVisible) {
    R.pausesVisible = false;
    updatePauseOverlayUi();
    drawWave();
    return;
  }
  R.pausesVisible = true;
  updatePauseOverlayUi();
  if (!R.pauseDecisions) {
    btn.disabled = true;
    st.className = "sura-gen-status running";
    st.textContent = "ищу паузы…";
    try {
      const data = await api.get(`/api/sura/${R.surah}/pause-pipeline`);
      R.pauseDecisions = data.decisions || [];
      const accepted = R.pauseDecisions.filter((d) => d.accepted).length;
      const weak = R.pauseDecisions.filter((d) => d.accepted && d.reason === "weak-dsp").length;
      const rejected = R.pauseDecisions.length - accepted;
      st.className = "sura-gen-status ok";
      st.textContent = `паузы: ${accepted} принято, weak ${weak}, откл. ${rejected}`;
    } catch (e) {
      R.pausesVisible = false;
      st.className = "sura-gen-status err";
      st.textContent = "ошибка пауз: " + e.message;
    } finally {
      btn.disabled = false;
      updatePauseOverlayUi();
    }
  }
  drawWave();
}
function visibleWaveMs() {
  const wrap = $("waveScroll"), content = $("waveContent");
  const totalW = Math.max(1, content.clientWidth);
  const start = (wrap.scrollLeft / totalW) * Math.max(1, R.duration);
  const end = ((wrap.scrollLeft + wrap.clientWidth) / totalW) * Math.max(1, R.duration);
  return [Math.max(0, start), Math.min(R.duration || end, end)];
}
function pauseDecisionLabel(d) {
  if (!d.accepted) return "отклонена";
  if (d.reason === "weak-dsp") return "weak";
  if (d.reason === "intro-vad-tight") return "intro vad";
  if (d.reason === "intro-vad") return "intro";
  return "strict";
}
function pauseDecisionClass(d) {
  if (!d.accepted) return "rejected";
  return d.reason === "weak-dsp" ? "weak" : "strong";
}
function renderPauseInspector() {
  const panel = $("pauseInspector"), rows = $("pauseRows");
  if (!panel || !rows) return;
  if (!R.pausesVisible) {
    rows.innerHTML = "";
    return;
  }
  if (!R.pauseDecisions) {
    $("pauseInspectorSummary").textContent = "загрузка…";
    rows.innerHTML = "";
    return;
  }
  const accepted = R.pauseDecisions.filter((d) => d.accepted).length;
  const weak = R.pauseDecisions.filter((d) => d.accepted && d.reason === "weak-dsp").length;
  const rejected = R.pauseDecisions.length - accepted;
  const [viewStart, viewEnd] = visibleWaveMs();
  const visible = R.pauseDecisions.filter((d) => d.end_ms >= viewStart && d.start_ms <= viewEnd);
  $("pauseInspectorSummary").textContent =
    `всего ${R.pauseDecisions.length} · принято ${accepted} · weak ${weak} · откл. ${rejected} · видно ${visible.length}`;
  rows.innerHTML = "";
  for (const d of visible.slice(0, 120)) {
    const tr = document.createElement("tr");
    tr.className = "pause-row " + pauseDecisionClass(d);
    tr.title = `центр ${fmt(d.center_ms)} · ${d.reason}`;
    tr.innerHTML = `
      <td>${fmt(d.start_ms)}–${fmt(d.end_ms)}</td>
      <td>${Math.round(d.depth_ms)}мс</td>
      <td><span class="pause-pill ${pauseDecisionClass(d)}">${pauseDecisionLabel(d)}</span></td>
      <td>${Number(d.rms_ratio).toFixed(3)}</td>
      <td>${Number(d.flatness).toFixed(3)}</td>
      <td>${Number(d.zcr).toFixed(3)}</td>
      <td>${d.vad_ok == null ? "—" : (d.vad_ok ? "ok" : "нет")}</td>`;
    tr.onclick = () => {
      if (R.audio) R.audio.currentTime = d.center_ms / 1000;
      scrollWaveToMs(d.center_ms);
      drawWave();
    };
    rows.appendChild(tr);
  }
  if (visible.length > 120) {
    const tr = document.createElement("tr");
    tr.className = "pause-row muted";
    tr.innerHTML = `<td colspan="7">ещё ${visible.length - 120} в этом диапазоне — увеличьте масштаб</td>`;
    rows.appendChild(tr);
  }
}
function msToVirtualX(ms) {
  return (ms / Math.max(1, R.duration)) * $("waveContent").clientWidth;
}
function drawPauseOverlay(ctx, left, right, h) {
  if (!R.pausesVisible || !R.pauseDecisions || !R.duration) return;
  for (const d of R.pauseDecisions) {
    const x0 = msToVirtualX(d.start_ms);
    const x1 = msToVirtualX(d.end_ms);
    if (x1 < left || x0 > right) continue;
    const x = Math.max(0, x0 - left);
    const w = Math.max(2, Math.min(right, x1) - Math.max(left, x0));
    const isWeak = d.accepted && d.reason === "weak-dsp";
    if (d.accepted) {
      ctx.fillStyle = isWeak ? "rgba(217,164,65,.24)" : "rgba(74,144,217,.22)";
      ctx.strokeStyle = isWeak ? "#d9a441" : "#4a90d9";
      ctx.setLineDash([]);
    } else {
      ctx.fillStyle = "rgba(217,106,90,.10)";
      ctx.strokeStyle = "rgba(217,106,90,.75)";
      ctx.setLineDash([4, 3]);
    }
    ctx.fillRect(x, 14, w, h - 28);
    ctx.lineWidth = d.accepted ? 1.5 : 1;
    ctx.strokeRect(x + 0.5, 14.5, Math.max(1, w - 1), h - 29);
    const cx = msToVirtualX(d.center_ms) - left;
    if (cx >= 0 && cx <= ctx.canvas.clientWidth) {
      ctx.beginPath(); ctx.moveTo(cx, 8); ctx.lineTo(cx, h - 8); ctx.stroke();
    }
  }
  ctx.setLineDash([]);
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
  drawPauseOverlay(ctx, left, right, h);
  R.ayahs.forEach((a, i) => {
    if (a.time == null || !R.duration) return;
    const vx = msToVirtualX(a.time);
    if (vx < left - 1 || vx > right + 1) return;
    const x = vx - left;
    const rc = R.recheck ? R.recheck[a.ayah] : null;
    const suspect = rc && rc.verdict === "shift";
    ctx.strokeStyle = suspect ? "#ff6b6b" : (i === R.current ? "#ffd56b" : "#d4a843");
    ctx.lineWidth = (suspect || i === R.current) ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = "10px sans-serif";
    ctx.fillText(suspect ? a.ayah + "→" + rc.match_ayah : a.ayah, x + 3, 11);
  });
  if (R.audio && R.duration) {
    const playMs = R.audio.currentTime * 1000;
    const px = msToVirtualX(playMs) - left;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
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
function scrollWaveToEdge(edge) {
  const wrap = $("waveScroll");
  wrap.scrollLeft = edge === "end" ? wrap.scrollWidth - wrap.clientWidth : 0;
  renderPauseInspector();
  drawWave();
}
function seekAyah(i, play) {
  const a = R.ayahs[i]; if (!a || a.time == null || !R.audio) return;
  R.audio.currentTime = a.time / 1000;
  setCurrent(i);
  scrollWaveToMs(a.time);
  drawWave();
  if (play) R.audio.play();
}
function seekPrevUnit() {
  seekAyah(Math.max(R.current - 1, 0), false);
}
function seekNextUnit() {
  seekAyah(Math.min(R.current + 1, R.ayahs.length - 1), false);
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

async function markCurrentStart() {
  if (!R.audio || R.surah == null) return;
  let idx = R.current;
  if (idx < 0) idx = currentAyahByTime(R.audio.currentTime * 1000);
  if (idx < 0) idx = R.ayahs.findIndex((a) => a.time == null);
  if (idx < 0) idx = 0;
  setCurrent(idx);
  await commitMarker(idx, R.audio.currentTime * 1000);
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
  $("draftPausesBtn").onclick = draftPausesSura;
  $("correctDraftBtn").onclick = correctDraftSura;
  $("recheckBtn").onclick = recheckSura;
  $("markCurrentBtn").onclick = markCurrentStart;
  $("zoomOutBtn").onclick = () => setWaveZoom(R.zoom / 2);
  $("zoomInBtn").onclick = () => setWaveZoom(R.zoom * 2);
  $("zoomFitBtn").onclick = () => setWaveZoom(1, R.duration / 2);
  $("waveStartBtn").onclick = () => scrollWaveToEdge("start");
  $("waveEndBtn").onclick = () => scrollWaveToEdge("end");
  $("hideSuraLogBtn").onclick = () => setSuraLogVisible(false);
  $("suraLogResizeHandle").addEventListener("pointerdown", (e) => {
    const box = $("suraLiveLogBox");
    box.setPointerCapture(e.pointerId);
    R.logResize = { pointerId: e.pointerId, startY: e.clientY, startHeight: box.getBoundingClientRect().height };
    e.preventDefault();
  });
  $("pauseOverlayBtn").onclick = togglePauseOverlay;
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
  for (const btn of document.querySelectorAll("[data-copy-log]")) {
    btn.addEventListener("click", () => copyLiveLog(btn.dataset.copyLog));
    updateLogCopyButton(btn.dataset.copyLog);
  }

  const audio = $("audio");
  audio.addEventListener("play", () => ($("playBtn").textContent = "⏸"));
  audio.addEventListener("pause", () => ($("playBtn").textContent = "▶"));
  audio.addEventListener("timeupdate", () => {
    const ms = audio.currentTime * 1000;
    $("clock").textContent = `${fmt(ms)} / ${fmt(R.duration)}`;
    const i = currentAyahByTime(ms);
    if (i >= 0 && i !== R.current) setCurrent(i);
    if (R.zoom > 1) {
      const wrap = $("waveScroll"), x = msToVirtualX(ms);
      if (x < wrap.scrollLeft || x > wrap.scrollLeft + wrap.clientWidth) {
        wrap.scrollLeft = Math.max(0, x - wrap.clientWidth / 2);
      }
    }
    if (R.loop && R.current >= 0 && ms >= ayahEndMs(R.current)) seekAyah(R.current, true);
    drawWave();
  });

  const c = $("wave");
  const waveScroll = $("waveScroll");
  waveScroll.addEventListener("scroll", () => { renderPauseInspector(); drawWave(); });
  waveScroll.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const ms = xToMs(e.clientX);
    const factor = Math.exp(-e.deltaY / 180);
    setWaveZoom(R.zoom * factor, ms);
  }, { passive: false });
  waveScroll.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse") return;
    waveScroll.setPointerCapture(e.pointerId);
    if (!R.pinch) R.pinch = { points: new Map(), startDistance: 0, startZoom: R.zoom, centerMs: null };
    R.pinch.points.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (R.pinch.points.size === 2) {
      const pts = [...R.pinch.points.values()];
      R.pinch.startDistance = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      R.pinch.startZoom = R.zoom;
      R.pinch.centerMs = xToMs((pts[0].x + pts[1].x) / 2);
    }
  });
  waveScroll.addEventListener("pointermove", (e) => {
    if (!R.pinch || !R.pinch.points.has(e.pointerId)) return;
    R.pinch.points.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (R.pinch.points.size !== 2 || R.pinch.startDistance <= 0) return;
    e.preventDefault();
    const pts = [...R.pinch.points.values()];
    const distance = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    setWaveZoom(R.pinch.startZoom * (distance / R.pinch.startDistance), R.pinch.centerMs);
  });
  const endPinchPoint = (e) => {
    if (!R.pinch) return;
    R.pinch.points.delete(e.pointerId);
    if (R.pinch.points.size < 2) R.pinch = null;
  };
  waveScroll.addEventListener("pointerup", endPinchPoint);
  waveScroll.addEventListener("pointercancel", endPinchPoint);
  window.addEventListener("pointermove", (e) => {
    if (!R.logResize || e.pointerId !== R.logResize.pointerId) return;
    setSuraLogHeight(R.logResize.startHeight + (R.logResize.startY - e.clientY));
    e.preventDefault();
  });
  window.addEventListener("pointerup", (e) => {
    if (R.logResize && e.pointerId === R.logResize.pointerId) R.logResize = null;
  });
  window.addEventListener("pointercancel", (e) => {
    if (R.logResize && e.pointerId === R.logResize.pointerId) R.logResize = null;
  });
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
  window.addEventListener("resize", () => {
    if (!$("review").classList.contains("hidden")) {
      resizeWave();
      renderPauseInspector();
      drawWave();
    }
  });

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
