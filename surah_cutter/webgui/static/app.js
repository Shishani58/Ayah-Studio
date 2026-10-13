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
function studioIcon(name) {
  const paths = {
    play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M8 5v14M16 5v14" stroke-width="4"/>',
    previous: '<path d="M6 5v14M18 5l-9 7 9 7z"/>',
    next: '<path d="M18 5v14M6 5l9 7-9 7z"/>',
    marker: '<path d="M6 21V3l12 1v9l-12-1"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 0 1 6 0c0 2-3 2-3 4M12 17h.01"/>',
    filter: '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
    addProject: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7zM3 9h18M12 12v6M9 15h6"/>',
    recording: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7zM3 9h18"/>',
    back: '<path d="M20 12H4m6-6-6 6 6 6"/>',
    settings: '<path d="m9 3-1 3-3 1-2 3 2 2v3l3 2 1 4h6l1-4 3-2v-3l2-2-2-3-3-1-1-3z"/><circle cx="12" cy="12" r="3"/>',
    export: '<path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5"/>',
    download: '<path d="M12 3v12m-4-4 4 4 4-4M4 17v4h16v-4"/>',
    trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
    refresh: '<path d="M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-2l2 7M4 12l2 7a7 7 0 0 0 12-2"/>',
    logs: '<path d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7"/>',
    tools: '<path d="m14 6 4 4 3-3a6 6 0 0 1-8 7l-7 7-3-3 7-7a6 6 0 0 1 7-8z"/>',
    mode: '<path d="M4 5h16M4 12h16M4 19h16M8 3v4M16 10v4M10 17v4"/>',
    up: '<path d="m6 15 6-6 6 6"/>',
    down: '<path d="m6 9 6 6 6-6"/>',
  };
  return `<svg class="studio-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || ""}</svg>`;
}
function fmt(ms) {
  if (ms == null) return "—";
  const s = Math.max(0, ms) / 1000, m = Math.floor(s / 60), r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
}
const setStatus = (msg) => {
  $("status").textContent = msg;
  $("workspaceStatus").textContent = msg;
  $("workspaceStatus").classList.toggle("error", /ошибка|не удалось|не сохран/i.test(msg));
};
let currentProject = {}, projectLibrary = [], timeStep = 10, editBusy = false, reviewLoading = false;
function exactTime(ms) {
  if (ms == null) return "";
  const value = Math.round(ms), minutes = Math.floor(value / 60000);
  return `${String(minutes).padStart(2, "0")}:${String(Math.floor(value / 1000) % 60).padStart(2, "0")}.${String(value % 1000).padStart(3, "0")}`;
}
function parseTime(value) {
  const match = /^(\d+):([0-5]\d)\.(\d{3})$/.exec(value.trim());
  return match ? Number(match[1]) * 60000 + Number(match[2]) * 1000 + Number(match[3]) : NaN;
}
const ayahLabel = (count) => `${count} ${{ one: "аят", few: "аята", many: "аятов", other: "аята" }[new Intl.PluralRules("ru").select(count)]}`;
function saveState(message, state = "") {
  $("saveState").textContent = message;
  $("saveState").className = "save-state " + state;
}
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
const modeLabel = (mode) => ({ auto: "авто", full: "полный", words: "по словам", tiled: "точно·большие", fast: "быстрый", hybrid: "hybrid" }[mode] || mode);
function selectedMode() { return $("genMode") ? $("genMode").value : "full"; }
function selectedSuraMode() { return $("suraMode") ? $("suraMode").value : selectedMode(); }
function setMode(mode) {
  const m = (mode === "auto" || mode === "fast" || mode === "tiled" || mode === "full" || mode === "words") ? mode : "auto";
  if ($("genMode")) $("genMode").value = m;
  if ($("suraMode")) $("suraMode").value = m;
  document.querySelectorAll("[data-review-mode], [data-dashboard-mode]").forEach(button => {
    button.setAttribute("aria-pressed", String((button.dataset.reviewMode || button.dataset.dashboardMode) === m));
    button.disabled = currentProject.alignment_model === "arabic-ctc" && !["full", "tiled", "words"].includes(button.dataset.reviewMode || button.dataset.dashboardMode);
  });
  for (const id of ["genMode", "suraMode", "settingsMode"]) {
    if ($(id)) [...$(id).options].forEach(option => option.disabled = currentProject.alignment_model === "arabic-ctc" && !["full", "tiled", "words"].includes(option.value));
  }
  $("dashboardModeToggle").title = `Режим разметки: ${modeLabel(m)}`;
  updateModeHint();
  applyReviewMode(m);
  updateGenControls(dashboardSuras);
  renderSuras();
}
function updateModeHint() {
  const mode = selectedMode();
  $("modeHint").textContent = mode === "words"
    ? `Пословная разметка всей суры. Границы аятов сохраняются; недостающие рассчитываются ${currentProject.alignment_model === "arabic-ctc" ? "полным" : "быстрым"} режимом.`
    : mode === "auto"
    ? "Автоматический выбор режима. После генерации проверьте результат на слух."
    : mode === "fast"
    ? "Быстрый черновик по паузам. Сложные участки требуют внимательной проверки."
    : mode === "tiled"
    ? "Для длинных сур: обработка частями с ограничением памяти."
    : "Полное выравнивание аудио с текстом аятов.";
}

// ===========================================================================
//  Dashboard
// ===========================================================================
let dashboardSuras = [];
async function refreshSuras() {
  dashboardSuras = await api.get("/api/suras");
  renderSuras();
  renderSuraSelector();
  updateGenControls(dashboardSuras);
  const available = dashboardSuras.filter(s => s.has_audio).length;
  const verified = dashboardSuras.filter(s => s.status === "verified").length;
  $("librarySummary").textContent = `${available} с аудио · ${verified} проверено · ${dashboardSuras.length} всего`;
  $("dashboardExportBtn").disabled = !$("dbPath").value.trim();
}
function updateSuraFilterPicker() {
  const select = $("suraFilter");
  const label = select.selectedOptions[0].textContent;
  $("filterMenuToggle").title = `Показать: ${label}`;
  $("filterMenuToggle").setAttribute("aria-label", `Фильтр сур: ${label}`);
  $("filterMenuToggle").classList.toggle("filtered", select.value !== "audio");
  document.querySelectorAll("[data-sura-filter]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.suraFilter === select.value)));
}
function renderSuras() {
  const query = $("suraSearch").value.trim();
  const filter = $("suraFilter").value;
  const suras = dashboardSuras.filter(s => (!query || String(s.surah).includes(query))
    && (filter === "all" || (filter === "audio" ? s.has_audio : s.has_audio && s.status === filter)));
  const grid = $("grid");
  grid.innerHTML = "";
  for (const s of suras) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.disabled = !s.has_audio;
    tile.className = "tile" + (s.has_audio ? "" : " no-audio");
    tile.setAttribute("aria-label", `Сура ${s.surah}, ${ayahLabel(s.ayah_count)}, ${s.has_audio ? badgeLabel(s.status) : "нет аудио"}`);
    tile.innerHTML = `
      <div class="audio-dot ${s.has_audio ? "on" : ""}" title="${s.has_audio ? "есть аудио" : "нет аудио"}"></div>
      <span class="num">Сура ${s.surah}</span>
      <span class="meta">${s.ayah_count} аят.</span>
      <span class="badge ${s.status}">${selectedMode() === "words" ? (s.words_complete ? "слова размечены" : "слова не завершены") : badgeLabel(s.status)}</span>`;
    if (s.has_audio) tile.onclick = () => openReview(s.surah, true);
    grid.appendChild(tile);
  }
  $("gridEmpty").classList.toggle("hidden", suras.length > 0);
  const hasAudio = dashboardSuras.some(s => s.has_audio);
  $("gridEmpty").innerHTML = hasAudio
    ? "<h3>Суры не найдены</h3><p>Измените номер суры или выберите другой фильтр.</p>"
    : "<h3>Здесь появятся ваши суры</h3><p>Выберите папку с аудио выше и откройте проект.<br>Файлы должны называться 001.mp3 … 114.mp3.</p>";
}
const badgeLabel = (st) => ({ "not-processed": "без разметки", "partial": "неполная разметка", "processed": "черновик", "verified": "✓ проверена" }[st] || st);

// reflect progress on the generate buttons (resume-aware)
function updateGenControls(suras) {
  const withAudio = suras.filter((s) => s.has_audio);
  const remaining = withAudio.filter((s) => selectedMode() === "words" ? !s.words_complete : ["not-processed", "partial"].includes(s.status)).length;
  const done = withAudio.length - remaining;
  const ready = withAudio.length > 0 && !!$("dbPath").value.trim();
  const gen = $("genBtn"), regen = $("regenBtn");
  if (!ready) { gen.disabled = true; gen.innerHTML = studioIcon("play") + " Начать разметку"; regen.classList.add("hidden"); return; }
  if (done === 0) { gen.disabled = false; gen.innerHTML = studioIcon("play") + ` Разметить все · ${withAudio.length}`; regen.classList.add("hidden"); }
  else if (remaining > 0) { gen.disabled = false; gen.innerHTML = studioIcon("play") + ` Продолжить · ${remaining}`; regen.classList.remove("hidden"); }
  else { gen.disabled = true; gen.innerHTML = studioIcon("check") + ` Все ${withAudio.length} готовы`; regen.classList.remove("hidden"); }
}

async function openProject() {
  const button = $("openBtn");
  button.disabled = true;
  $("settingsMessage").textContent = "Копирование аудио и разметки…";
  try {
    const p = await api.post("/api/projects/create", {
      reciter: $("importReciter").value.trim(), recording: $("importRecording").value.trim(),
      audio_dir: $("audioDir").value.trim(), db_path: $("dbPath").value.trim(),
    });
    await applyProject(p);
    closeDialog($("settingsDialog"));
    setStatus("");
  } catch (e) { $("settingsMessage").textContent = "Не удалось создать проект: " + e.message; }
  finally { button.disabled = false; }
}

let audioCount = 0;
const GEN = { es: null, timer: null, startMs: 0, total: 0, wordFailures: 0 };

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
  GEN.wordFailures = 0;
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
    GEN.wordFailures += d.word_failures || 0;
    setProgress(d.index + 1, GEN.total, `${d.index + 1} / ${GEN.total}`);
    const fast = d.fast_trace
      ? ` · пауз ${d.fast_trace.n_pause}, wasl ${d.fast_trace.n_wasl}, unresolved ${d.fast_trace.unresolved || 0}, atomic ${d.fast_trace.atomic_wasl_reads || 0}, окно ${(d.fast_trace.coverage || 0).toFixed(2)}×${d.fast_trace.drift_detected ? " · drift" : ""}`
      : "";
    const wordResult = d.words_written != null ? ` · слов записано ${d.words_written}${d.word_failures ? ` · требуют проверки ${d.word_failures} аят.` : ""}` : "";
    $("genCur").textContent = `сура ${d.surah} обработана${wordResult}${d.written != null ? ` · ${d.written} аят.` : ""}${fast}`;
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
    panel.classList.add("done"); $("genTitle").textContent = GEN.wordFailures ? "Расчёт завершён · нужна проверка" : "✓ Готово";
    $("genCur").textContent = `обработано ${completed ?? ""} сур за ${elapsed}${GEN.wordFailures ? ` · требуют проверки ${GEN.wordFailures} аят.` : ""}`;
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
  request: 0, dataReady: false, audioReady: false, waveReady: false, dataLoading: false, waveLoading: false,
  surah: null, ayahs: [], wordProgress: null, duration: 0, peaks: [], audio: null, current: -1,
  loop: false, drag: -1, zoom: savedWaveZoom(), recheck: null,
  pausesVisible: false, pauseDecisions: null,
  waveBorn: [], waveFrame: 0, waveCancel: null, waveBucketMs: 100, waveMaximum: 1,
  pinch: null,
};

const W = { loading: false, mode: false, data: null, selected: 0, request: 0, busy: false, drag: -1, preview: null, stop: null, timer: null, unit: "s" };

function applyReviewMode(mode) {
  const changed = W.mode !== (mode === "words");
  W.mode = mode === "words";
  for (const id of ["draftPausesBtn", "correctDraftBtn", "recheckBtn"])
    $(id).classList.toggle("hidden", W.mode);
  $("regenSuraBtn").innerHTML = studioIcon("play") + (W.mode ? " Разметить слова" : " Начать разметку");
  $("regenSuraBtn").setAttribute("aria-label", W.mode ? "Разметить слова всей суры" : "Начать разметку суры");
  $("regenSuraBtn").title = W.mode ? "Сохранить границы аятов и рассчитать все слова суры" : "Рассчитать тайм-коды этой суры выбранным режимом";
  $("modeMenuToggle").title = `Режим разметки: ${modeLabel(mode)}`;
  $("verifySelectedBtn").textContent = W.mode ? "Отметить слово проверенным" : "Отметить выбранный проверенным";
  updateSuraProgress();
  if (!changed) return;
  W.stop = null;
  if (R.audio) R.audio.pause();
  $("alignWordsBtn").classList.toggle("hidden", !W.mode);
  $("wordUnitControl").classList.toggle("hidden", !W.mode);
  document.querySelector(".ayah-table-head").classList.toggle("word-mode", W.mode);
  $("ayahList").classList.toggle("word-mode", W.mode);
  const headings = document.querySelectorAll(".ayah-table-head span");
  headings[0].textContent = W.mode ? "Аят:слово" : "Аят";
  headings[2].textContent = W.mode ? "Слово" : "Текст аята";
  for (const [id, wordLabel, ayahLabel] of [["prevBtn", "Предыдущее слово", "Предыдущий аят"], ["nextBtn", "Следующее слово", "Следующий аят"]]) {
    const label = W.mode ? wordLabel : ayahLabel;
    $(id).setAttribute("aria-label", label);
    $(id).title = label;
  }
  $("waveHint").textContent = W.mode ? "Голубые линии — начала слов · тяните метку для правки" : "Клик по волне — перемотка · тяните золотую линию — двигайте начало аята";
  if (W.mode && R.dataReady && R.surah != null && !$("review").classList.contains("hidden")) { if (R.current < 0) setCurrent(0); else loadWords(); }
  else { resetWords(); if (R.dataReady) reviewLoadMessage("ayahLoadStatus", ""); }
  renderAyahList();
  drawWave();
  if (!W.mode && R.surah != null) {
    setWaveZoom(adaptiveWaveZoom());
    if (R.current >= 0 && R.ayahs[R.current].time != null) scrollWaveToMs(R.ayahs[R.current].time);
  }
}

function resetWords() {
  clearTimeout(W.timer); W.timer = null;
  W.loading = false; W.request++; W.data = null; W.selected = 0; W.drag = -1; W.preview = null; W.stop = null;
  if (W.mode) renderWords();
}
async function loadWords() {
  resetWords();
  if (!W.mode || !R.dataReady || R.surah == null) return;
  const request = W.request, surah = R.surah;
  W.loading = true;
  reviewLoadMessage("ayahLoadStatus", "Загрузка слов и разметки…");
  $("ayahList").setAttribute("aria-busy", "true");
  try {
    const data = await api.get(`/api/sura/${surah}/words`);
    if (request !== W.request) return;
    W.data = data; W.loading = false;
    reviewLoadMessage("ayahLoadStatus", "");
    $("ayahList").setAttribute("aria-busy", "false");
    const ayah = R.ayahs[R.current]?.ayah;
    W.selected = Math.max(0, data.words.findIndex(word => word.ayah === ayah));
    renderWords(); selectWord(W.selected, false);
  } catch (e) {
    if (request === W.request) {
      W.loading = false;
      reviewLoadMessage("ayahLoadStatus", "Не удалось загрузить слова. " + e.message + " ", true);
      $("ayahList").setAttribute("aria-busy", "false");
    }
  }
}
function wordPayload(data) {
  return { db_path: data.db_path, start_ms: data.start_ms, end_ms: data.end_ms };
}
function wordContext(word = W.data?.words[W.selected]) {
  return word && W.data ? { ...word, surah: W.data.surah, db_path: W.data.db_path } : null;
}
function mergeWordResult(result) {
  const updates = new Map(result.words.map(word => [word.word_index, word]));
  W.data.words = W.data.words.map(word => updates.has(word.word_index)
    ? { ...word, ...updates.get(word.word_index) } : word);
  W.data.word_progress = result.word_progress;
}
function wordDisplayTime(ms, unit = W.unit) {
  if (ms == null) return "";
  return unit === "ms" ? String(ms) : (ms / 1000).toFixed(3);
}
function parseWordTime(value, unit = W.unit) {
  const text = value.trim().replace(",", ".");
  const format = unit === "ms" ? /^\d+$/ : /^\d+(?:\.\d{1,3})?$/;
  if (!format.test(text)) return NaN;
  const ms = Math.round(Number(text) * (unit === "ms" ? 1 : 1000));
  return Number.isSafeInteger(ms) ? ms : NaN;
}
function renderWords() {
  if (!W.mode) return;
  if (W.data?.word_progress) R.wordProgress = W.data.word_progress;
  updateSuraProgress();
  const list = $("ayahList"); list.replaceChildren();
  list.classList.add("word-mode"); list.setAttribute("aria-label", "Слова суры");
  const words = W.data ? W.data.words : [];
  const groups = new Map();
  words.forEach((word, i) => {
    if (!groups.has(word.ayah)) {
      const group = document.createElement("li");
      group.className = "word-ayah"; group.dataset.ayah = word.ayah;
      group.innerHTML = `<div class="word-group-head"><strong>Аят ${word.ayah}</strong></div><ul class="word-cards" aria-label="Слова аята ${word.ayah}"></ul>`;
      list.appendChild(group); groups.set(word.ayah, group.querySelector("ul"));
    }
    const li = document.createElement("li");
    li.className = "word-card"; li.dataset.word = i;
    li.classList.toggle("active", i === W.selected);
    li.classList.toggle("missing", word.time == null);
    li.classList.toggle("stale", !!word.stale);
    const label = `${word.ayah}:${word.number}`;
    const status = word.stale ? "границы изменились" : ({ auto: "авто", edited: "правка", verified: "проверено" }[word.status] || "нет метки");
    const editable = Number.isInteger(word.start_ms) && Number.isInteger(word.end_ms);
    const editUnit = W.unit;
    const unit = editUnit === "ms" ? "мс" : "с";
    li.innerHTML = `
      <div class="word-card-head"><button class="a-num" aria-label="Прослушать слово ${word.number} аята ${word.ayah}" aria-pressed="${i === W.selected}">${word.number}</button><span class="word-meta">${status}</span></div>
      <button class="a-text word-play" dir="rtl" lang="ar" aria-label="Прослушать ${escapeHtml(word.text)}">${escapeHtml(word.text)}</button>
      <div class="word-time"><span class="word-unit">Начало, ${unit}</span><div class="time-editor">
        <input type="number" min="0" step="${W.unit === "ms" ? "1" : "0.001"}" class="a-time-input" inputmode="decimal" aria-label="Начало слова ${label}, ${W.unit === "ms" ? "миллисекунды" : "секунды"}" placeholder="Нет метки" value="${wordDisplayTime(word.time)}" ${editable ? "" : "disabled"} />
        <div class="time-arrows"><button type="button" aria-label="Увеличить время слова ${label}" ${editable ? "" : "disabled"}>${studioIcon("up")}</button><button type="button" aria-label="Уменьшить время слова ${label}" ${editable ? "" : "disabled"}>${studioIcon("down")}</button></div>
      </div></div>`;
    applyTimingStatus(li, word);
    const input = li.querySelector("input");
    input.onfocus = () => selectWord(i, false, false);
    const commitInput = async () => {
      if (input.value === wordDisplayTime(word.time, editUnit)) return;
      const ms = parseWordTime(input.value, editUnit);
      if (!Number.isSafeInteger(ms)) { input.setAttribute("aria-invalid", "true"); setStatus(editUnit === "ms" ? "Введите целое число миллисекунд." : "Введите секунды, например 6.120."); return; }
      const currentTime = W.data?.words.find(w => w.word_index === word.word_index)?.time;
      if (ms === currentTime) { input.value = wordDisplayTime(currentTime, editUnit); return; }
      W.selected = i; await editWord(ms);
    };
    input.onchange = commitInput;
    input.onblur = commitInput;
    input.onkeydown = async e => {
      if (e.key === "Enter") { e.preventDefault(); await commitInput(); }
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault(); const base = parseWordTime(input.value, editUnit);
        if (!Number.isSafeInteger(base)) return;
        W.selected = i; await editWord(base + (e.key === "ArrowUp" ? 1 : -1) * timeStep * (e.shiftKey ? 10 : 1));
      }
    };
    li.querySelectorAll(".time-arrows button").forEach((btn, j) => {
      btn.onclick = e => { W.selected = i; editWord((word.time ?? Math.round(R.audio.currentTime * 1000)) + (j === 0 ? 1 : -1) * timeStep * (e.shiftKey ? 10 : 1)); };
    });
    li.querySelector(".word-play").onclick = () => selectWord(i, true);
    li.querySelector(".a-num").onclick = () => selectWord(i, true);
    groups.get(word.ayah).appendChild(li);
  });
  const word = words[W.selected];
  $("alignWordsBtn").disabled = !word || !Number.isInteger(word.start_ms) || W.busy;
  $("verifySelectedBtn").disabled = !word || word.time == null || word.stale || W.busy;
}
function selectWord(i, play = true, focusWave = true) {
  const word = W.data?.words[i];
  if (!word) return;
  W.selected = i;
  R.current = R.ayahs.findIndex(a => a.ayah === word.ayah);
  document.querySelectorAll("#ayahList li[data-word]").forEach(li => {
    li.classList.toggle("active", Number(li.dataset.word) === i);
    li.querySelector(".a-num").setAttribute("aria-pressed", String(Number(li.dataset.word) === i));
  });
  $("verifySelectedBtn").disabled = word.time == null || word.stale;
  if (focusWave && word.time != null && !word.stale) {
    const next = W.data.words[i + 1];
    const end = next && next.ayah === word.ayah && next.time != null ? next.time : word.end_ms;
    const width = Math.max(700, (end || word.time + 700) - word.time);
    scrollWaveToMs(word.time + width / 2);
  }
  if (play) playWord();
  drawWave();
}
async function alignWords() {
  const data = wordContext(), request = W.request;
  if (!data || W.busy) return;
  W.busy = true;
  try {
    const result = await api.post(`/api/sura/${data.surah}/ayah/${data.ayah}/words/align`, wordPayload(data));
    if (request === W.request) mergeWordResult(result);
  } catch (e) { if (request === W.request) setStatus("Ошибка: " + e.message); }
  finally { W.busy = false; }
  renderWords(); drawWave();
}
async function editWord(ms, action) {
  const data = wordContext(), request = W.request;
  if (!data || W.busy) return;
  W.busy = true; saveState("Сохранение…");
  try {
    const result = await api.post(`/api/sura/${data.surah}/ayah/${data.ayah}/words/edit`,
      { ...wordPayload(data), word_index: data.word_index, time: Math.round(ms), action });
    if (request !== W.request) return;
    mergeWordResult(result); saveState("Сохранено", "saved"); setStatus("");
  } catch (e) { if (request === W.request) { saveState("Не сохранено", "error"); setStatus("Ошибка: " + e.message); } }
  finally { W.busy = false; }
  renderWords(); drawWave();
}
function playWord() {
  if (!R.audioReady || !R.dataReady) return;
  clearTimeout(W.timer); W.timer = null; W.stop = null;
  if (R.audio) R.audio.pause();
  const word = W.data?.words[W.selected];
  if (!word || word.time == null || word.stale) { setStatus("Сначала поставьте актуальную метку слова."); return; }
  const next = W.data.words[W.selected + 1];
  const sameAyah = next && next.ayah === word.ayah;
  if (sameAyah && (next.time == null || next.stale)) { setStatus("Сначала поставьте метку следующего слова."); return; }
  const end = sameAyah ? next.time : word.end_ms;
  if (!Number.isInteger(end) || end <= word.time) return;
  W.stop = { start: word.time, end };
  R.audio.currentTime = word.time / 1000;
  R.audio.play(); scheduleWordStop();
}
function finishWord() {
  const clip = W.stop;
  if (!clip) return;
  if (R.loop) { R.audio.currentTime = clip.start / 1000; R.audio.play(); scheduleWordStop(); }
  else { W.stop = null; R.audio.pause(); R.audio.currentTime = Math.max(clip.start, clip.end - 1) / 1000; }
}
function scheduleWordStop() {
  clearTimeout(W.timer); W.timer = null;
  if (!W.stop || !R.audio || R.audio.paused) return;
  const delay = Math.max(0, (W.stop.end - R.audio.currentTime * 1000) / R.audio.playbackRate);
  W.timer = setTimeout(finishWord, delay);
}
function wordNearX(clientX) {
  if (!W.mode || !W.data) return -1;
  const rect = $("wave").getBoundingClientRect(), left = $("waveScroll").scrollLeft;
  let nearest = -1, distance = 8;
  W.data.words.forEach((word, i) => {
    if (word.time == null || word.stale) return;
    const d = Math.abs(clientX - (rect.left + msToVirtualX(word.time) - left));
    if (d < distance) { distance = d; nearest = i; }
  });
  return nearest;
}

function setSuraLogVisible(visible) {
  const dialog = $("suraLiveLogBox");
  if (visible) {
    $("suraLogTitle").textContent = `Логи · Сура ${R.surah}`;
    openDialog(dialog);
    void loadSuraLogs();
  } else if (dialog.open) closeDialog(dialog, true);
}

let suraLogRequest = 0;
function showSuraLogPane(id) {
  for (const pane of ["suraLogList", "suraLiveLog", "suraSavedLog"]) $(pane).classList.toggle("hidden", pane !== id);
  $("suraLogBackBtn").classList.toggle("hidden", id === "suraLogList");
  for (const log of ["suraLiveLog", "suraSavedLog"]) logCopyButton(log).classList.toggle("hidden", log !== id);
  $("suraLogTitle").textContent = `Логи · Сура ${R.surah}`;
  $("suraLogEmpty").textContent = "";
  $("suraLogEmpty").classList.add("hidden");
}
async function loadSuraLogs() {
  const request = ++suraLogRequest, surah = R.surah, project = currentProject.db_path;
  const active = () => request === suraLogRequest && R.surah === surah && currentProject.db_path === project && $("suraLiveLogBox").open;
  showSuraLogPane("suraLogList");
  $("suraLogList").replaceChildren();
  $("suraLogEmpty").textContent = "Загрузка логов…";
  $("suraLogEmpty").classList.remove("hidden");
  const addEntry = (label, detail, select) => {
    const button = document.createElement("button");
    button.className = "ghost sura-log-entry";
    const title = document.createElement("strong"), meta = document.createElement("small");
    title.textContent = label; meta.textContent = detail;
    button.append(title, meta); button.onclick = select;
    $("suraLogList").append(button);
  };
  const addLive = () => {
    if (!$("suraLiveLog").textContent) return;
    addEntry("Текущий расчёт", "Записи текущего экрана", () => {
      suraLogRequest++; showSuraLogPane("suraLiveLog");
      $("suraLiveLog").scrollTop = $("suraLiveLog").scrollHeight;
      $("suraLogBackBtn").focus();
    });
  };
  try {
    const data = await api.get(`/api/sura/${surah}/logs`);
    if (!active()) return;
    addLive();
    for (const entry of data.logs) {
      addEntry(entry.id, `${entry.mode} · ${new Date(entry.modified_ms).toLocaleString("ru-RU")}`, async () => {
        const detailRequest = ++suraLogRequest;
        showSuraLogPane("suraSavedLog");
        resetLiveLog("suraSavedLog");
        $("suraLogTitle").textContent = entry.id;
        $("suraSavedLog").textContent = "Загрузка…";
        $("suraLogBackBtn").focus();
        try {
          const result = await api.get(`/api/sura/${surah}/logs/${encodeURIComponent(entry.id)}`);
          if (detailRequest !== suraLogRequest || R.surah !== surah || currentProject.db_path !== project || !$("suraLiveLogBox").open) return;
          $("suraSavedLog").textContent = [result.log, result.json === null ? null : `Данные JSON\n${result.json}`].filter(value => value !== null).join("\n\n");
          updateLogCopyButton("suraSavedLog");
        } catch (error) {
          if (detailRequest === suraLogRequest && $("suraLiveLogBox").open) $("suraSavedLog").textContent = `Не удалось открыть лог: ${error.message}. Вернитесь к списку и попробуйте снова.`;
        }
      });
    }
    $("suraLogEmpty").textContent = "Сохранённых логов этой суры пока нет.";
    $("suraLogEmpty").classList.toggle("hidden", !!$("suraLogList").children.length);
  } catch (error) {
    if (!active()) return;
    addLive();
    $("suraLogEmpty").textContent = `Не удалось загрузить логи: ${error.message}`;
    const retry = document.createElement("button"); retry.className = "ghost"; retry.textContent = "Повторить";
    retry.onclick = () => { void loadSuraLogs(); }; $("suraLogList").append(retry);
  }
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const dialogMotions = new WeakMap();
const menuMotions = new WeakMap();
let keyboardMotion = false;
document.addEventListener("keydown", () => { keyboardMotion = true; }, true);
document.addEventListener("pointerdown", () => { keyboardMotion = false; }, true);
function openDialog(dialog) {
  dialogMotions.get(dialog)?.cancel();
  dialog.classList.remove("motion-closing");
  dialog.classList.toggle("motion-instant", keyboardMotion);
  if (dialog.open) return;
  dialog.showModal();
  if (keyboardMotion) return;
  const animation = dialog.animate([
    { opacity: 0, transform: reducedMotion.matches ? "none" : "scale(.96)" },
    { opacity: 1, transform: "none" },
  ], { duration: reducedMotion.matches ? 100 : 220, easing: "cubic-bezier(0.23, 1, 0.32, 1)" });
  dialogMotions.set(dialog, animation);
}
function closeDialog(dialog, immediate = false) {
  if (!dialog.open) return;
  const style = getComputedStyle(dialog);
  const start = { opacity: style.opacity, transform: style.transform };
  dialogMotions.get(dialog)?.cancel();
  if (immediate || keyboardMotion) { dialog.close(); return; }
  dialog.classList.add("motion-closing");
  const animation = dialog.animate([
    start,
    { opacity: 0, transform: reducedMotion.matches ? "none" : "scale(.96)" },
  ], { duration: reducedMotion.matches ? 100 : 200, easing: "cubic-bezier(0.23, 1, 0.32, 1)", fill: "forwards" });
  dialogMotions.set(dialog, animation);
  animation.finished.then(() => {
    dialog.close(); animation.cancel(); dialog.classList.remove("motion-closing");
  }).catch(() => {});
}

function animateMenu(panel, anchor) {
  menuMotions.get(panel)?.cancel();
  if (keyboardMotion) return;
  const bounds = panel.getBoundingClientRect();
  const originX = Math.max(0, Math.min(bounds.width, anchor.left + anchor.width / 2 - bounds.left));
  panel.style.transformOrigin = `${originX}px ${anchor.top >= bounds.bottom ? "bottom" : "top"}`;
  menuMotions.set(panel, panel.animate([
    { opacity: 0, transform: reducedMotion.matches ? "none" : "scale(.97)" },
    { opacity: 1, transform: "none" },
  ], { duration: reducedMotion.matches ? 100 : 180, easing: "cubic-bezier(0.23, 1, 0.32, 1)" }));
}

function queueWaveDraw() {
  if (R.waveFrame) return;
  R.waveFrame = requestAnimationFrame(() => {
    R.waveFrame = 0;
    drawWave();
    if (!reducedMotion.matches && performance.now() - R.waveBorn.at(-1) < 180) queueWaveDraw();
  });
}
function loadWaveStream(surah, active) {
  return new Promise((resolve, reject) => {
    const stream = new EventSource(`/api/sura/${surah}/waveform/stream`);
    const stop = () => { stream.close(); if (R.waveCancel === cancel) R.waveCancel = null; };
    const cancel = () => { stop(); resolve(); };
    R.waveCancel = cancel;
    stream.onmessage = event => {
      if (!active()) { cancel(); return; }
      try {
        const wave = JSON.parse(event.data);
        if (wave.type === "start") {
          R.waveBucketMs = wave.bucket_ms;
          if (!R.duration && wave.duration_hint_ms > 0) {
            R.duration = wave.duration_hint_ms; setWaveZoom(adaptiveWaveZoom());
          }
        }
        else if (wave.type === "chunk") {
          if (wave.offset !== R.peaks.length) throw new Error("Нарушен порядок участков волны");
          R.peaks.push(...wave.peaks);
          const now = performance.now(); R.waveBorn.push(...wave.peaks.map(() => now));
          const progress = R.duration ? ` · ${Math.min(99, Math.floor(wave.decoded_ms / R.duration * 100))}%` : "";
          reviewLoadMessage("waveLoadStatus", "Построение волны" + progress);
          $("waveLoadStatus").classList.add("wave-partial");
          queueWaveDraw();
        } else if (wave.type === "done") {
          R.duration = wave.duration_ms; R.waveMaximum = wave.maximum || 1;
          R.waveReady = true; R.waveLoading = false;
          reviewLoadMessage("waveLoadStatus", "");
          $("clock").textContent = `${exactTime(R.audio.currentTime * 1000)} / ${exactTime(R.duration)}`;
          setWaveZoom(adaptiveWaveZoom()); updateReviewReadiness(); resizeWave(); queueWaveDraw();
          stop(); resolve();
        } else if (wave.type === "error") throw new Error(wave.message);
      } catch (error) { stop(); reject(error); }
    };
    stream.onerror = () => { stop(); reject(new Error("Соединение прервано")); };
  });
}

function reviewLoadMessage(id, message, failed = false) {
  const el = $(id);
  el.replaceChildren();
  el.classList.toggle("hidden", !message);
  el.classList.toggle("error", failed);
  if (!message) return;
  el.append(document.createTextNode(message));
  if (failed) {
    const retry = document.createElement("button");
    retry.className = "ghost"; retry.textContent = "Повторить";
    retry.onclick = () => openReview(R.surah);
    el.append(retry);
  }
}
function updateReviewReadiness() {
  for (const id of ["regenSuraBtn", "draftPausesBtn", "correctDraftBtn", "recheckBtn"])
    $(id).disabled = !R.dataReady;
  for (const id of ["playBtn", "prevBtn", "nextBtn", "rate", "rateBtn"])
    $(id).disabled = !R.dataReady || !R.audioReady;
  $("ayahList").setAttribute("aria-busy", String(R.dataLoading || (W.mode && W.loading)));
  $("waveScroll").setAttribute("aria-busy", String(R.waveLoading));
}
async function openReview(surah) {
  if (editBusy || W.busy) return;
  R.waveCancel?.();
  if (R.waveFrame) cancelAnimationFrame(R.waveFrame);
  R.waveFrame = 0;
  const request = ++R.request;
  const active = () => request === R.request;
  reviewLoading = true;
  if (R.audio) R.audio.pause();
  Object.assign(R, { surah, ayahs: [], wordProgress: null, peaks: [], duration: 0,
    waveBorn: [], waveBucketMs: 100, waveMaximum: 1,
    current: -1, recheck: null, pausesVisible: false, pauseDecisions: null,
    zoom: R.zoom, pinch: null, drag: -1, dataReady: false, audioReady: false, waveReady: false, dataLoading: true, waveLoading: true });
  resetWords();
  renderSuraSelector();
  resetLiveLog("suraLiveLog");
  setSuraLogVisible(false);
  updateWaveControls();
  const summary = dashboardSuras.find(s => s.surah === surah);
  $("reviewTitle").textContent = `Сура ${surah}${summary ? " · " + ayahLabel(summary.ayah_count) : ""}`;
  $("suraGenStatus").textContent = "";
  $("suraGenStatus").className = "sura-gen-status";
  $("verifySelectedBtn").disabled = true;
  $("alignWordsBtn").disabled = true;
  updatePauseOverlayUi();
  renderAyahList();
  $("ayahList").scrollTop = 0;
  updateSuraProgress();
  $("suraProgressText").textContent = "Загрузка разметки…";
  $("suraProgressMeta").textContent = "—";
  $("suraProgressPct").textContent = "—";
  $("waveHint").textContent = "";
  $("clock").textContent = `${exactTime(0)} / —`;
  reviewLoadMessage("ayahLoadStatus", "Загрузка аятов и разметки…");
  $("waveLoadStatus").classList.remove("wave-partial");
  reviewLoadMessage("waveLoadStatus", "Загрузка аудиоволны…");
  reviewLoadMessage("audioLoadStatus", "Загрузка аудио…");
  updateReviewReadiness();
  setStatus("");
  showView("review");
  requestAnimationFrame(() => { if (active()) { resizeWave(); drawWave(); } });

  const audio = $("audio");
  R.audio = audio;
  audio.onloadedmetadata = () => {
    if (!active()) return;
    if (!R.duration && Number.isFinite(audio.duration)) {
      R.duration = Math.round(audio.duration * 1000);
      setWaveZoom(adaptiveWaveZoom());
    }
    $("clock").textContent = `${exactTime(audio.currentTime * 1000)} / ${exactTime(R.duration)}`;
  };
  audio.oncanplay = () => {
    if (!active()) return;
    R.audioReady = true;
    reviewLoadMessage("audioLoadStatus", "");
    updateReviewReadiness();
  };
  audio.onerror = () => {
    if (!active()) return;
    R.audioReady = false;
    reviewLoadMessage("audioLoadStatus", "Не удалось загрузить аудио. ", true);
    updateReviewReadiness();
  };
  audio.src = `/api/audio/${surah}`;
  audio.playbackRate = parseFloat($("rate").value);
  audio.load();

  await Promise.allSettled([
    (async () => {
      try {
        const sura = await api.get(`/api/sura/${surah}`);
        if (!active()) return;
        R.ayahs = sura.ayahs; R.wordProgress = sura.word_progress; R.dataReady = true; R.dataLoading = false;
        $("reviewTitle").textContent = `Сура ${surah} · ${ayahLabel(sura.ayah_count)}`;
        document.title = `${$("reviewTitle").textContent} — Ayah Studio`;
        reviewLoadMessage("ayahLoadStatus", "");
        renderAyahList(); updateGenHint(); updateSuraProgress(); updateReviewReadiness();
        if (W.mode) setCurrent(0);
        drawWave();
      } catch (e) {
        if (!active()) return;
        R.dataLoading = false; updateReviewReadiness();
        reviewLoadMessage("ayahLoadStatus", "Не удалось загрузить аяты. " + e.message + " ", true);
        $("suraProgressText").textContent = "Разметка не загружена";
      }
    })(),
    (async () => {
      try {
        await loadWaveStream(surah, active);
      } catch (e) {
        if (active()) {
          R.waveLoading = false; updateReviewReadiness();
          reviewLoadMessage("waveLoadStatus", "Не удалось загрузить аудиоволну. " + e.message + " ", true);
        }
      }
    })(),
  ]);
  if (active()) reviewLoading = false;
}
// hint under the waveform + per-sura button label reflect whether it's generated
function updateGenHint() {
  const generated = R.ayahs.some((a) => a.time != null);
  const hint = $("waveHint");
  if (W.mode) {
    hint.textContent = generated ? "Голубые линии — начала слов · тяните метку для правки"
      : "Нажмите «Рассчитать слова всей суры», чтобы сначала получить границы аятов.";
    hint.classList.toggle("warn", !generated);
    $("regenSuraBtn").innerHTML = studioIcon("play") + (W.mode ? " Разметить слова" : " Начать разметку");
    $("regenSuraBtn").setAttribute("aria-label", W.mode ? "Разметить слова всей суры" : "Начать разметку суры");
    return;
  }
  if (generated) {
    hint.textContent = "Клик по волне — перемотка · тяните золотую линию — двигайте начало аята";
    hint.classList.remove("warn");
    $("regenSuraBtn").innerHTML = studioIcon("play") + (W.mode ? " Разметить слова" : " Начать разметку");
    $("regenSuraBtn").setAttribute("aria-label", W.mode ? "Разметить слова всей суры" : "Начать разметку суры");
  } else {
    hint.textContent = "Сура без тайм-кодов: задайте начало аята в поле времени или запустите разметку.";
    hint.classList.add("warn");
    $("regenSuraBtn").innerHTML = studioIcon("play") + (W.mode ? " Разметить слова" : " Начать разметку");
    $("regenSuraBtn").setAttribute("aria-label", W.mode ? "Разметить слова всей суры" : "Начать разметку суры");
  }
}
function updateSuraProgress(state = "") {
  const wordProgress = R.wordProgress || {};
  const total = W.mode ? (wordProgress.total || 0) : R.ayahs.length;
  const marked = W.mode ? (wordProgress.marked || 0) : R.ayahs.filter((a) => a.time != null).length;
  const edited = W.mode ? (wordProgress.edited || 0) : R.ayahs.filter((a) => a.status === "edited").length;
  const verified = W.mode ? (wordProgress.verified || 0) : R.ayahs.filter((a) => a.status === "verified").length;
  const pct = total ? Math.round((marked / total) * 100) : 0;
  $("suraProgressFill").style.right = (100 - pct) + "%";
  $("suraProgressPct").textContent = `${pct}%`;
  $("suraProgressText").textContent = state === "running"
    ? `расчёт суры… ${W.mode ? "слов" : "размечено"} ${marked} / ${total}`
    : `${W.mode ? "слов размечено" : "размечено"} ${marked} / ${total}`;
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
      st.textContent = d.words_written != null ? `слов записано ${d.words_written} · требуют проверки ${d.word_failures || 0} аят.` : `записано ${d.written ?? 0} аят.`;
      if (d.log_paths && d.log_paths.log) appendLiveLog("suraLiveLog", `Лог сохранён: ${d.log_paths.log}`);
    }
    if (d.type === "done" || d.type === "cancelled") {
      es.close();
      await reloadSura(); refreshSuras();
      const incomplete = mode === "words" && R.wordProgress && !R.wordProgress.complete;
      stop(incomplete ? "warn" : "ok", incomplete ? "Расчёт завершён: некоторые слова требуют проверки."
        : `✓ готово за ${Math.floor((Date.now() - t0) / 1000)}с`);
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
    R.ayahs = sura.ayahs; R.wordProgress = sura.word_progress;
  } catch { return; }
  if (W.mode) await loadWords();
  R.recheck = null;  // times changed → stale verification
  renderAyahList();
  updateGenHint();
  updateSuraProgress();
  drawWave();
  const cur = R.current; R.current = -1;
  setCurrent(currentAyahByTime((R.audio ? R.audio.currentTime : 0) * 1000));
  if (R.current < 0 && cur >= 0) setCurrent(cur);
}

function applyTimingStatus(row, item, suspect = false) {
  const status = item.time == null ? "missing"
    : item.status === "verified" && !item.stale && !suspect ? "verified" : "uncertain";
  const label = { missing: "Без разметки", verified: "Проверено", uncertain: "Требует проверки" }[status];
  row.dataset.timingStatus = status;
  const number = row.querySelector(".a-num");
  if (number) {
    number.dataset.baseLabel ||= number.getAttribute("aria-label");
    number.setAttribute("aria-label", `${number.dataset.baseLabel} · ${label}`);
    number.title = label;
  }
}

function renderAyahList() {
  if (W.mode) { renderWords(); return; }
  const ul = $("ayahList");
  ul.classList.remove("word-mode"); ul.setAttribute("aria-label", "Аяты суры");
  ul.innerHTML = "";
  R.ayahs.forEach((a, i) => {
    const li = document.createElement("li");
    li.dataset.i = i;
    li.classList.toggle("missing", a.time == null);
    li.classList.toggle("active", i === R.current);
    li.innerHTML = `
      <button class="a-num" aria-label="Выбрать и прослушать аят ${a.ayah}" aria-pressed="${i === R.current}">${a.ayah}</button>
      <div class="time-editor">
        <input type="text" class="a-time-input" inputmode="decimal" aria-label="Начало аята ${a.ayah}, минуты:секунды.миллисекунды" placeholder="Нет метки" value="${exactTime(a.time)}" />
        <div class="time-arrows"><button type="button" aria-label="Увеличить время аята ${a.ayah}" title="Позже (Shift — шаг ×10)">${studioIcon("up")}</button><button type="button" aria-label="Уменьшить время аята ${a.ayah}" title="Раньше (Shift — шаг ×10)">${studioIcon("down")}</button></div>
      </div>
      <span class="a-text" dir="rtl" lang="ar">${escapeHtml(a.text || "")}</span>`;
    applyTimingStatus(li, a, R.recheck?.[a.ayah]?.verdict === "shift");
    const input = li.querySelector("input");
    input.onfocus = () => setCurrent(i);
    const commitInput = async () => {
      if (input.value === exactTime(a.time)) return;
      const ms = parseTime(input.value);
      if (!Number.isSafeInteger(ms)) {
        input.setAttribute("aria-invalid", "true"); input.setAttribute("aria-describedby", "workspaceStatus");
        setStatus("Введите время в формате 00:10.320 (минуты:секунды.миллисекунды).");
        return;
      }
      await commitMarker(i, ms);
    };
    input.onchange = commitInput;
    input.onblur = commitInput;
    input.onkeydown = async (e) => {
      if (e.key === "Enter") { e.preventDefault(); await commitInput(); }
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const base = parseTime(input.value);
        if (!Number.isSafeInteger(base)) return;
        await commitMarker(i, base + (e.key === "ArrowUp" ? 1 : -1) * timeStep * (e.shiftKey ? 10 : 1));
      }
    };
    li.querySelectorAll(".time-arrows button").forEach((btn, j) => {
      btn.onclick = async (e) => {
        e.stopPropagation();
        setCurrent(i);
        const base = a.time == null ? (R.audio ? Math.round(R.audio.currentTime * 1000) : 0) : a.time;
        await commitMarker(i, base + (j === 0 ? 1 : -1) * timeStep * (e.shiftKey ? 10 : 1));
      };
    });
    li.onclick = (e) => {
      if (e.target.closest(".time-editor")) return;
      if (a.time != null) seekAyah(i, true); else setCurrent(i);
    };
    ul.appendChild(li);
  });
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
  return savedWaveZoom(Math.max(1, Math.min(100, Math.round(zoom))));
}
function savedWaveZoom(fallback = 1) {
  try {
    const stored = localStorage.getItem("ayah-studio-wave-zoom");
    const zoom = Number(stored);
    return stored !== null && Number.isFinite(zoom) ? Math.max(1, Math.min(256, Math.round(zoom))) : fallback;
  } catch { return fallback; }
}
function setWaveZoom(zoom, centerMs = null, persist = false) {
  const oldZoom = R.zoom;
  const wrap = $("waveScroll"), content = $("waveContent");
  const centerRatio = centerMs != null && R.duration
    ? centerMs / R.duration
    : (wrap.scrollLeft + wrap.clientWidth / 2) / Math.max(1, content.clientWidth);
  R.zoom = Math.max(1, Math.min(256, Math.round(zoom)));
  if (persist) { try { localStorage.setItem("ayah-studio-wave-zoom", String(R.zoom)); } catch {} }
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
  $("zoomOutBtn").disabled = R.zoom <= 1;
  $("zoomInBtn").disabled = R.zoom >= 256;
}
function updatePauseOverlayUi() {
  const btn = $("pauseOverlayBtn"), inspector = $("pauseInspector");
  if (!btn || !inspector) return;
  btn.classList.toggle("active", R.pausesVisible);
  btn.setAttribute("aria-pressed", String(R.pausesVisible));
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
  const word = W.mode ? W.data?.words[W.selected] : null;
  if (word && word.time != null && !word.stale) {
    const next = W.data.words[W.selected + 1];
    const endMs = next && next.ayah === word.ayah ? next.time : word.end_ms;
    if (endMs != null) {
      const start = msToVirtualX(word.time) - left, end = msToVirtualX(endMs) - left;
      ctx.fillStyle = "rgba(212,168,67,.13)"; ctx.fillRect(start, 0, end - start, h);
    }
  } else if (!W.mode && R.current >= 0 && R.ayahs[R.current].time != null) {
    const start = msToVirtualX(R.ayahs[R.current].time) - left;
    const end = msToVirtualX(ayahEndMs(R.current)) - left;
    ctx.fillStyle = "rgba(212,168,67,.13)";
    ctx.fillRect(start, 0, end - start, h);
  }
  const colors = getComputedStyle(document.documentElement);
  const color = token => colors.getPropertyValue(token).trim();
  ctx.strokeStyle = color("--blue"); ctx.lineWidth = 1; ctx.beginPath();
  const step = R.duration ? R.waveBucketMs / R.duration * virtualW : 0;
  const i0 = step ? Math.max(0, Math.floor(left / step) - 2) : 0;
  const i1 = step ? Math.min(n - 1, Math.ceil(right / step) + 2) : -1;
  const now = performance.now();
  for (let i = i0; i <= i1; i++) {
    const x = i * step - left, amp = Math.min(1, R.peaks[i] / R.waveMaximum) * (mid - 2);
    ctx.globalAlpha = reducedMotion.matches ? 1 : Math.min(1, (now - R.waveBorn[i]) / 180);
    ctx.beginPath(); ctx.moveTo(x, mid - amp); ctx.lineTo(x, mid + amp); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  drawPauseOverlay(ctx, left, right, h);
  R.ayahs.forEach((a, i) => {
    if (W.mode || a.time == null || !R.duration) return;
    const vx = msToVirtualX(a.time);
    if (vx < left - 1 || vx > right + 1) return;
    const x = vx - left;
    const rc = R.recheck ? R.recheck[a.ayah] : null;
    const suspect = rc && rc.verdict === "shift";
    ctx.strokeStyle = a.status === "verified" && !suspect ? color("--green") : color("--amber");
    ctx.lineWidth = (suspect || i === R.current) ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = "10px sans-serif";
    ctx.fillText(suspect ? a.ayah + "→" + rc.match_ayah : a.ayah, x + 3, 11);
  });
  if (W.mode && W.data) W.data.words.forEach((word, i) => {
    if (word.time == null) return;
    const x = msToVirtualX(W.drag === i && W.preview != null ? W.preview : word.time) - left;
    if (x < 0 || x > w) return;
    ctx.strokeStyle = word.stale ? color("--muted") : color("--blue");
    ctx.lineWidth = i === W.selected ? 2 : 1;
    ctx.setLineDash(word.stale ? [3, 3] : []);
    ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, h); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(`${word.ayah}:${word.number}`, x + 3, 28);
  });
  if (R.audio && R.duration) {
    const playMs = R.audio.currentTime * 1000;
    const px = msToVirtualX(playMs) - left;
    ctx.strokeStyle = color("--text"); ctx.lineWidth = 1;
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
function togglePlay() { if (!R.audio || !R.audioReady || !R.dataReady) return; if (W.mode) { R.audio.paused ? playWord() : R.audio.pause(); return; } R.audio.paused ? R.audio.play() : R.audio.pause(); }
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
  const a = R.ayahs[i]; if (!a || a.time == null || !R.audio || !R.audioReady) return;
  R.audio.currentTime = a.time / 1000;
  setCurrent(i);
  scrollWaveToMs(a.time);
  drawWave();
  if (play) R.audio.play();
}
function seekPrevUnit() {
  if (W.mode && !W.data) return;
  if (W.mode && W.data) { selectWord(Math.max(0, W.selected - 1)); return; }
  seekAyah(Math.max(R.current - 1, 0), false);
}
function seekNextUnit() {
  if (W.mode && !W.data) return;
  if (W.mode && W.data) { selectWord(Math.min(W.data.words.length - 1, W.selected + 1)); return; }
  seekAyah(Math.min(R.current + 1, R.ayahs.length - 1), false);
}
function setCurrent(i) {
  if (i === R.current) return;
  R.current = i;
  if (W.mode) { if (!W.data) loadWords(); return; }
  document.querySelectorAll(".ayah-list li").forEach((li) => {
    li.classList.toggle("active", +li.dataset.i === i);
    li.querySelector(".a-num").setAttribute("aria-pressed", String(+li.dataset.i === i));
  });
  $("verifySelectedBtn").disabled = !R.ayahs[i] || R.ayahs[i].time == null;
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
  if (W.mode) { await editWord(0, "verify"); return; }
  const a = R.ayahs[i];
  const next = a.status === "verified" ? "auto" : "verified";
  try {
    await api.post("/api/verify", { surah: R.surah, ayah: a.ayah, status: next });
    a.status = next === "auto" ? null : next;
    renderAyahList(); updateSuraProgress(); const cur = R.current; R.current = -1; setCurrent(cur); drawWave();
  } catch (e) { setStatus("Ошибка: " + e.message); }
}

async function markCurrentStart() {
  if (!R.audio || !R.audioReady || !R.dataReady || R.surah == null) return;
  if (W.mode) { await editWord(R.audio.currentTime * 1000); return; }
  let idx = R.current;
  if (idx < 0) idx = currentAyahByTime(R.audio.currentTime * 1000);
  if (idx < 0) idx = R.ayahs.findIndex((a) => a.time == null);
  if (idx < 0) idx = 0;
  setCurrent(idx);
  await commitMarker(idx, R.audio.currentTime * 1000);
}

async function commitMarker(i, ms) {
  if (editBusy || !R.ayahs[i]) return false;
  const a = R.ayahs[i], surah = R.surah;
  const oldTime = a.time;
  const row = document.querySelector(`.ayah-list li[data-i="${i}"]`);
  const input = row ? row.querySelector("input") : null;
  const focused = row && row.contains(document.activeElement) ? document.activeElement : null;
  const value = Math.round(ms);
  if (!Number.isSafeInteger(value) || value < 0 || value > R.duration) {
    if (input) { input.value = exactTime(oldTime); input.setAttribute("aria-invalid", "true"); input.setAttribute("aria-describedby", "workspaceStatus"); }
    setStatus("Время не сохранено. Укажите начало в пределах аудиозаписи.");
    saveState("Не сохранено", "error"); return false;
  }
  editBusy = true;
  if (row) row.querySelectorAll("input, .time-arrows button").forEach(el => el.disabled = true);
  saveState("Сохранение…");
  try {
    await api.post("/api/timing", { surah, ayah: a.ayah, time: value, db_path: currentProject.db_path });
    a.time = value; a.status = "edited";
    if (input) { input.value = exactTime(value); input.removeAttribute("aria-invalid"); input.removeAttribute("aria-describedby"); }
    if (R.recheck) delete R.recheck[a.ayah];
    if (row) { row.classList.remove("missing"); applyTimingStatus(row, a); }
    updateGenHint(); updateSuraProgress(); drawWave();
    if (W.mode) loadWords(); else resetWords();
    saveState("Сохранено", "saved");
    setStatus("");
    return true;
  } catch (e) {
    a.time = oldTime;
    if (input) { input.value = exactTime(oldTime); input.setAttribute("aria-invalid", "true"); input.setAttribute("aria-describedby", "workspaceStatus"); }
    saveState("Не сохранено", "error");
    setStatus("Время не сохранено. " + e.message);
    drawWave(); return false;
  } finally {
    editBusy = false;
    if (row) row.querySelectorAll("input, .time-arrows button").forEach(el => el.disabled = false);
    if (focused && document.activeElement === document.body) focused.focus({ preventScroll: true });
  }
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
  setRateMenuOpen(false);
  $("dashboardModeMenu").open = false;
  $("filterMenu").open = false;
  if (name !== "review") {
    setSuraLogVisible(false);
    R.request++; reviewLoading = false;
    R.waveCancel?.();
    if (R.waveFrame) cancelAnimationFrame(R.waveFrame);
    R.waveFrame = 0;
    R.audioReady = false;
    resetWords();
  }
  $("dashboard").classList.toggle("hidden", name !== "dashboard");
  $("review").classList.toggle("hidden", name !== "review");
  document.querySelector(".workspace-dashboard-actions").classList.toggle("hidden", name !== "dashboard");
  document.querySelector(".workspace-review-actions").classList.toggle("hidden", name !== "review");
  $("recordingMenu").open = false;
  $("suraMenu").open = false;
  $("modeMenu").open = false;
  document.querySelector(".review-tools").open = false;
  document.title = name === "review" ? `${$("reviewTitle").textContent} — Ayah Studio` : "Ayah Studio";
  (name === "review" ? $("reviewTitle") : $("suraSearch")).focus();
}
const baseName = (p) => p.replace(/[/\\]+$/, "").split(/[/\\]/).pop() || "reciter";

function enableNativePickers() {
  $("browseAudio").classList.remove("hidden");
  $("browseDb").classList.remove("hidden");
  $("browseAudio").onclick = async () => {
    const p = await window.pywebview.api.pick_folder();
    if (!p) return;
    $("audioDir").value = p;
    if (!$("importReciter").value.trim()) $("importReciter").value = baseName(p);
  };
  $("browseDb").onclick = async () => {
    const p = await window.pywebview.api.pick_db();
    if (p) $("dbPath").value = p;
  };

}

function renderProjectPicker() {
  const list = $("projectPickerList");
  const select = $("projectSelect");
  list.replaceChildren();
  if (!currentProject.project_id && currentProject.db_path) {
    const current = document.createElement("div");
    current.className = "project-picker-current";
    const name = document.createElement("strong"); name.textContent = currentProject.reciter || "Текущий проект";
    const detail = document.createElement("small"); detail.textContent = "Внешний проект · выбран";
    current.append(name, detail); list.append(current);
  }
  for (const project of projectLibrary) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.projectId = project.id;
    button.setAttribute("aria-pressed", String(project.id === currentProject.project_id));
    button.disabled = select.disabled || editBusy || reviewLoading;
    const name = document.createElement("span"); name.textContent = project.reciter;
    const detail = document.createElement("small"); detail.textContent = project.recording;
    button.append(name, detail);
    button.onclick = () => {
      if (select.disabled || editBusy || reviewLoading) return;
      $("recordingMenu").open = false;
      $("recordingMenuToggle").focus();
      if (project.id === currentProject.project_id) return;
      select.value = project.id;
      select.dispatchEvent(new Event("change"));
    };
    list.append(button);
  }
  if (!list.children.length) {
    const empty = document.createElement("p"); empty.className = "project-picker-empty";
    empty.textContent = "Пока нет проектов"; list.append(empty);
  }
}

async function refreshProjectLibrary() {
  projectLibrary = await api.get("/api/projects");
  const select = $("projectSelect");
  select.replaceChildren();
  if (!currentProject.project_id) {
    select.add(new Option(currentProject.db_path ? `${currentProject.reciter} · Внешний проект` : "Создайте проект в настройках", ""));
  }
  for (const p of projectLibrary) select.add(new Option(`${p.reciter} · ${p.recording}`, p.id));
  select.value = currentProject.project_id || "";
  const recordingLabel = currentProject.db_path ? `Проект: ${currentProject.reciter || ""} · ${currentProject.recording || "Внешний проект"}` : "Выбрать проект";
  $("recordingMenuToggle").title = recordingLabel;
  $("recordingMenuToggle").setAttribute("aria-label", recordingLabel);
  $("workspaceExportBtn").disabled = !currentProject.db_path;
  renderProjectPicker();
}

function renderSuraPicker() {
  const grid = $("suraPickerGrid");
  const query = $("suraPickerSearch").value.trim();
  grid.replaceChildren();
  for (const sura of dashboardSuras.filter(s => s.has_audio && String(s.surah).startsWith(query))) {
    const button = document.createElement("button");
    button.type = "button"; button.textContent = sura.surah; button.dataset.surah = sura.surah;
    button.setAttribute("aria-pressed", String(sura.surah === R.surah));
    const label = `Сура ${sura.surah} · ${ayahLabel(sura.ayah_count)}`;
    button.title = label; button.setAttribute("aria-label", label);
    button.disabled = editBusy || W.busy;
    button.onclick = () => {
      if (editBusy || W.busy) return;
      $("suraMenu").open = false;
      $("reviewSuraSelect").value = sura.surah;
      $("reviewSuraSelect").dispatchEvent(new Event("change"));
    };
    grid.append(button);
  }
  $("suraPickerEmpty").classList.toggle("hidden", grid.children.length > 0);
}

function renderSuraSelector() {
  const select = $("reviewSuraSelect");
  select.replaceChildren(new Option("Сура", ""));
  for (const s of dashboardSuras.filter(s => s.has_audio)) {
    select.add(new Option(`Сура ${s.surah}`, s.surah));
  }
  select.disabled = !dashboardSuras.some(s => s.has_audio);
  select.value = R.surah || "";
  $("suraPickerLabel").textContent = R.surah ? `Сура ${R.surah}` : "Сура";
  $("suraMenuToggle").setAttribute("aria-disabled", String(select.disabled));
  $("suraMenuToggle").setAttribute("aria-label", R.surah ? `Сура ${R.surah} · выбрать другую суру` : "Выбрать суру");
  renderSuraPicker();
}

async function applyProject(p) {
  if (R.audio) R.audio.pause();
  resetWords();
  currentProject = p;
  R.surah = null; R.ayahs = []; R.wordProgress = null; R.peaks = []; R.current = -1;
  $("audio").removeAttribute("src"); $("audio").load();
  $("audioDir").value = p.audio_dir || "";
  $("dbPath").value = p.db_path || "";
  setMode(p.mode);
  saveState("Сохранено", "saved");
  showView("dashboard");
  await refreshProjectLibrary();
  await refreshSuras();
}

function applyAppearance(theme, arabicSize) {
  document.documentElement.dataset.theme = theme === "light" ? "light" : "dark";
  document.documentElement.style.setProperty("--arabic-size", `${arabicSize || 30}px`);
  drawWave();
}

function settingsPanel(id) {
  document.querySelectorAll(".settings-panel").forEach(panel => panel.classList.toggle("hidden", panel.id !== id));
  document.querySelectorAll("[data-settings-panel]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.settingsPanel === id)));
  $("saveSettingsBtn").hidden = ["importSettings", "modelSettings"].includes(id);
  clearTimeout(modelPoll);
  if (id === "modelSettings") { modelSignature = ""; refreshModels(); }
}

function openSettings(panel = "interfaceSettings") {
  if (editBusy || reviewLoading) { setStatus("Дождитесь завершения сохранения или загрузки, затем откройте настройки."); return; }
  if (R.audio) R.audio.pause();
  const p = currentProject;
  $("settingsReciter").value = p.reciter || "";
  $("settingsRecording").value = p.recording || "";
  $("settingsReciter").disabled = !p.project_id;
  $("settingsRecording").disabled = !p.project_id;
  $("projectStorageSummary").textContent = p.db_path
    ? `${p.audio_count || 0} файлов аудио · ${p.project_id ? "Внутренний проект: аудио, разметка и история хранятся вместе." : "Внешний проект. Чтобы хранить копию внутри приложения, используйте вкладку «Импорт»."}`
    : "Нет открытого проекта. Создайте проект на вкладке «Импорт».";
  $("audioLocation").textContent = p.audio_dir || "Не выбрано";
  $("dbLocation").textContent = p.db_path || "Не выбрано";
  $("settingsExportBtn").disabled = !p.db_path;
  $("settingsStep").value = timeStep;
  $("settingsArabicSize").value = p.arabic_size || 30;
  document.querySelectorAll('[name="settingsTheme"]').forEach(input => input.checked = input.value === (p.theme || "dark"));
  $("settingsMode").replaceChildren(...[...$("genMode").children].map(option => option.cloneNode(true)));
  $("settingsMode").value = selectedMode();
  $("settingsMessage").textContent = "";
  settingsPanel(panel);
  openDialog($("settingsDialog"));
  if (panel === "modelSettings") refreshModels();
}

async function saveSettings() {
  const button = $("saveSettingsBtn");
  button.disabled = true;
  const data = {
    mode: $("settingsMode").value, time_step: Number($("settingsStep").value),
    arabic_size: Number($("settingsArabicSize").value),
    theme: document.querySelector('[name="settingsTheme"]:checked').value,
    project_id: currentProject.project_id,
    reciter: $("settingsReciter").value, recording: $("settingsRecording").value,
  };
  try {
    await api.post("/api/settings", data);
    Object.assign(currentProject, data);
    applyAppearance(data.theme, data.arabic_size);
    timeStep = data.time_step;
    currentProject = await api.get("/api/project");
    setMode(data.mode);
    document.documentElement.style.setProperty("--arabic-size", `${data.arabic_size}px`);
    await refreshProjectLibrary();
    closeDialog($("settingsDialog"));
    setStatus("Настройки сохранены.");
  } catch (e) { $("settingsMessage").textContent = "Не удалось сохранить настройки: " + e.message; }
  finally { button.disabled = false; }
}

function positionWorkspaceMenu(menu) {
  const panel = menu.querySelector(".toolbar-popover, .review-actions");
  if (!menu.open || !panel) return;
  const anchor = menu.querySelector("summary").getBoundingClientRect();
  const margin = 8;
  panel.style.maxHeight = `${Math.max(0, window.innerHeight - margin * 2)}px`;
  const bounds = panel.getBoundingClientRect();
  const left = Math.max(margin, Math.min(anchor.left, window.innerWidth - bounds.width - margin));
  const below = anchor.bottom + 6;
  const top = bounds.height <= window.innerHeight - below - margin ? below
    : Math.max(margin, Math.min(below, window.innerHeight - bounds.height - margin));
  panel.style.left = `${left}px`;
  panel.style.top = `${top}px`;
}

function setRateMenuOpen(open) {
  const wasOpen = !$("rateMenu").classList.contains("hidden");
  $("rateMenu").classList.toggle("hidden", !open);
  $("rateBtn").setAttribute("aria-expanded", String(open));
  if (open && !wasOpen) animateMenu($("rateMenu"), $("rateBtn").getBoundingClientRect());
  if (!open) menuMotions.get($("rateMenu"))?.cancel();
}
function updateRatePicker() {
  const value = $("rate").value;
  $("rateBtn").textContent = `${value}×`;
  $("rateBtn").setAttribute("aria-label", `Скорость воспроизведения: ${value}×`);
  $("rateMenu").querySelectorAll("button").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.rate === value));
  });
}

function wireWorkspace() {
  for (const [id, icon] of [["playBtn", "play"], ["prevBtn", "previous"], ["nextBtn", "next"], ["waveStartBtn", "previous"], ["waveEndBtn", "next"], ["backBtn", "back"], ["settingsBtn", "settings"], ["workspaceExportBtn", "export"], ["recordingMenuToggle", "recording"], ["modeMenuToggle", "mode"], ["reviewToolsToggle", "tools"], ["showSuraLogBtn", "logs"], ["dashboardModeToggle", "mode"], ["filterMenuToggle", "filter"], ["helpBtn", "help"], ["createProjectBtn", "addProject"], ["regenBtn", "refresh"]]) $(id).innerHTML = studioIcon(icon);
  document.querySelectorAll("[data-review-mode]").forEach(source => {
    const button = source.cloneNode(true);
    button.dataset.dashboardMode = button.dataset.reviewMode;
    delete button.dataset.reviewMode;
    button.onclick = () => {
      $("genMode").value = button.dataset.dashboardMode;
      $("genMode").dispatchEvent(new Event("change"));
      $("dashboardModeMenu").open = false; $("dashboardModeToggle").focus();
    };
    $("dashboardModePicker").append(button);
  });
  for (const option of $("suraFilter").options) {
    const button = document.createElement("button"); button.type = "button";
    button.dataset.suraFilter = option.value; button.textContent = option.textContent;
    button.onclick = () => {
      $("suraFilter").value = option.value;
      $("suraFilter").dispatchEvent(new Event("change"));
      $("filterMenu").open = false; $("filterMenuToggle").focus();
    };
    $("filterPicker").append(button);
  }
  updateSuraFilterPicker();
  $("suraMode").addEventListener("change", () => { $("modeMenu").open = false; });
  document.querySelectorAll("[data-review-mode]").forEach(button => {
    button.onclick = () => {
      $("suraMode").value = button.dataset.reviewMode;
      $("suraMode").dispatchEvent(new Event("change"));
      $("modeMenuToggle").focus();
    };
  });
  for (const menu of [$("recordingMenu"), $("suraMenu"), $("modeMenu"), $("dashboardModeMenu"), $("filterMenu"), document.querySelector(".review-tools")]) {
    menu.querySelector("summary").addEventListener("click", e => {
      e.preventDefault();
      if (menu.querySelector("summary").getAttribute("aria-disabled") === "true") return;
      menu.open = !menu.open;
      if (menu === $("recordingMenu") && menu.open) renderProjectPicker();
      if (menu === $("suraMenu") && menu.open) { $("suraPickerSearch").value = ""; renderSuraPicker(); }
      positionWorkspaceMenu(menu);
      if (menu === $("suraMenu") && menu.open) {
        $("suraPickerSearch").focus();
        $("suraPickerGrid").querySelector('[aria-pressed="true"]')?.scrollIntoView({ block: "nearest" });
      }
    });
    menu.addEventListener("toggle", () => {
      if (menu.open) {
        for (const other of [$("recordingMenu"), $("suraMenu"), $("modeMenu"), $("dashboardModeMenu"), $("filterMenu"), document.querySelector(".review-tools")]) {
          if (other !== menu) other.open = false;
        }
        positionWorkspaceMenu(menu);
        animateMenu(menu.querySelector(".toolbar-popover, .review-actions"), menu.querySelector("summary").getBoundingClientRect());
      } else {
        menuMotions.get(menu.querySelector(".toolbar-popover, .review-actions"))?.cancel();
      }
    });
    window.addEventListener("resize", () => positionWorkspaceMenu(menu));
    document.addEventListener("scroll", () => positionWorkspaceMenu(menu), true);
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && menu.open) { menu.open = false; menu.querySelector("summary").focus(); }
    });
    document.addEventListener("click", e => { if (!menu.contains(e.target)) menu.open = false; });
  }
  $("settingsBtn").onclick = () => openSettings();
  $("createProjectBtn").onclick = () => {
    openSettings("importSettings");
    $("audioDir").value = ""; $("dbPath").value = "";
    $("importReciter").focus();
  };
  for (const id of ["closeSettingsBtn", "cancelSettingsBtn"]) $(id).onclick = () => closeDialog($("settingsDialog"));
  $("settingsDialog").addEventListener("close", () => {
    clearTimeout(modelPoll);
    $("modelList").querySelectorAll('input[type="password"]').forEach(input => input.value = "");
    applyAppearance(currentProject.theme, currentProject.arabic_size);
  });
  const previewAppearance = () => applyAppearance(document.querySelector('[name="settingsTheme"]:checked').value, Number($("settingsArabicSize").value));
  document.querySelectorAll('[name="settingsTheme"]').forEach(input => input.onchange = previewAppearance);
  $("settingsArabicSize").onchange = previewAppearance;
  $("saveSettingsBtn").onclick = saveSettings;
  $("settingsExportBtn").onclick = exportProject;
  $("workspaceExportBtn").onclick = exportProject;
  document.querySelectorAll("[data-settings-panel]").forEach(btn => btn.onclick = () => settingsPanel(btn.dataset.settingsPanel));
  $("pickerCreateProjectBtn").onclick = () => {
    $("recordingMenu").open = false;
    $("createProjectBtn").click();
  };
  $("projectSelect").onchange = async (e) => {
    const id = e.target.value;
    if (!id || editBusy || reviewLoading) { e.target.value = currentProject.project_id || ""; return; }
    e.target.disabled = true;
    renderProjectPicker();
    $("recordingMenu").open = false;
    try { await applyProject(await api.post("/api/projects/activate", { id })); setStatus(""); }
    catch (error) { e.target.value = currentProject.project_id || ""; setStatus("Не удалось открыть проект. " + error.message); }
    finally { e.target.disabled = false; renderProjectPicker(); }
  };
  $("suraPickerSearch").oninput = () => { renderSuraPicker(); positionWorkspaceMenu($("suraMenu")); };
  $("suraPickerSearch").onkeydown = e => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    const number = Number(e.target.value.trim());
    const buttons = [...$("suraPickerGrid").querySelectorAll("button")];
    const exact = buttons.find(button => Number(button.dataset.surah) === number);
    if (exact) exact.click(); else if (buttons.length === 1) buttons[0].click();
  };
  $("reviewSuraSelect").onchange = (e) => {
    if (e.target.value) openReview(Number(e.target.value));
  };
  $("verifySelectedBtn").onclick = () => { if (R.current >= 0) toggleVerify(R.current); };
}

function wire() {
  wireWorkspace();
  for (const option of $("rate").options) {
    const button = document.createElement("button");
    button.type = "button"; button.dataset.rate = option.value; button.textContent = option.textContent;
    button.onclick = () => {
      $("rate").value = option.value;
      $("rate").dispatchEvent(new Event("change"));
      setRateMenuOpen(false); $("rateBtn").focus();
    };
    $("rateMenu").append(button);
  }
  updateRatePicker();
  $("rateBtn").onclick = () => setRateMenuOpen($("rateMenu").classList.contains("hidden"));
  document.addEventListener("click", e => {
    if (!document.querySelector(".player-rate").contains(e.target)) setRateMenuOpen(false);
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !$("rateMenu").classList.contains("hidden")) {
      setRateMenuOpen(false); $("rateBtn").focus();
    }
  });
  $("helpBtn").onclick = () => {
    const collapsed = $("help").classList.toggle("collapsed");
    $("helpBtn").setAttribute("aria-expanded", String(!collapsed));
  };
  $("projectForm").onsubmit = e => { e.preventDefault(); openProject(); };
  $("suraSearch").oninput = renderSuras;
  $("suraFilter").onchange = () => { updateSuraFilterPicker(); renderSuras(); };
  $("dashboardExportBtn").onclick = exportProject;
  $("genBtn").onclick = () => runGeneration(true);
  $("regenBtn").onclick = () => runGeneration(false);
  $("cancelBtn").onclick = cancelGeneration;
  $("regenSuraBtn").onclick = regenerateSura;
  $("draftPausesBtn").onclick = draftPausesSura;
  $("correctDraftBtn").onclick = correctDraftSura;
  $("recheckBtn").onclick = recheckSura;
  $("zoomOutBtn").onclick = () => setWaveZoom(R.zoom / 2, null, true);
  $("zoomInBtn").onclick = () => setWaveZoom(R.zoom * 2, null, true);
  $("waveStartBtn").onclick = () => scrollWaveToEdge("start");
  $("waveEndBtn").onclick = () => scrollWaveToEdge("end");
  $("hideSuraLogBtn").onclick = () => closeDialog($("suraLiveLogBox"));
  $("showSuraLogBtn").onclick = () => setSuraLogVisible(true);
  $("suraLogBackBtn").onclick = () => { void loadSuraLogs(); };
  $("suraLiveLogBox").addEventListener("close", () => { suraLogRequest++; });
  $("pauseOverlayBtn").onclick = togglePauseOverlay;
  $("genMode").onchange = async () => {
    setMode(selectedMode());
    try { await api.post("/api/settings", { mode: selectedMode() }); currentProject.mode = selectedMode(); }
    catch (e) { setMode(currentProject.mode); setStatus("Не удалось сохранить режим. Попробуйте снова: " + e.message); }
  };
  $("suraMode").onchange = async () => {
    const previousMode = selectedMode();
    const mode = selectedSuraMode();
    setMode(mode);
    try { await api.post("/api/settings", { mode }); currentProject.mode = mode; }
    catch (e) { setMode(previousMode); setStatus("Не удалось сохранить режим: " + e.message); }
  };
  $("backBtn").onclick = () => { if (R.audio) R.audio.pause(); showView("dashboard"); refreshSuras(); };
  $("playBtn").onclick = togglePlay;
  $("prevBtn").onclick = seekPrevUnit;
  $("nextBtn").onclick = seekNextUnit;
  $("rate").onchange = () => { if (R.audio) R.audio.playbackRate = parseFloat($("rate").value); updateRatePicker(); };
  $("exportBtn").onclick = exportProject;
  for (const btn of document.querySelectorAll("[data-copy-log]")) {
    btn.addEventListener("click", () => copyLiveLog(btn.dataset.copyLog));
    updateLogCopyButton(btn.dataset.copyLog);
  }

  $("alignWordsBtn").onclick = alignWords;
  $("wordTimeUnit").onchange = () => { W.unit = $("wordTimeUnit").value; renderWords(); };

  const audio = $("audio");
  audio.addEventListener("play", () => { $("playBtn").innerHTML = studioIcon("pause"); $("playBtn").setAttribute("aria-label", "Пауза"); });
  audio.addEventListener("pause", () => { $("playBtn").innerHTML = studioIcon("play"); $("playBtn").setAttribute("aria-label", "Воспроизвести"); });
  for (const event of ["play", "pause", "ratechange", "seeked"]) audio.addEventListener(event, scheduleWordStop);
  audio.addEventListener("ended", () => { if (W.mode && W.stop) finishWord(); });
  audio.addEventListener("timeupdate", () => {
    const ms = audio.currentTime * 1000;
    $("clock").textContent = `${exactTime(ms)} / ${exactTime(R.duration)}`;
    if (W.stop && ms >= W.stop.end) { finishWord(); drawWave(); return; }
    if (W.mode) { drawWave(); return; }
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
    setWaveZoom(R.zoom * factor, ms, true);
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
    setWaveZoom(R.pinch.startZoom * (distance / R.pinch.startDistance), R.pinch.centerMs, true);
  });
  const endPinchPoint = (e) => {
    if (!R.pinch) return;
    R.pinch.points.delete(e.pointerId);
    if (R.pinch.points.size < 2) R.pinch = null;
  };
  waveScroll.addEventListener("pointerup", endPinchPoint);
  waveScroll.addEventListener("pointercancel", endPinchPoint);
  c.addEventListener("mousedown", (e) => {
    if (!R.duration || !R.dataReady || !R.audioReady) return;
    W.stop = null;
    if (W.mode) {
      const wi = wordNearX(e.clientX);
      if (wi >= 0 && !W.busy) { W.drag = wi; W.selected = wi; renderWords(); c.style.cursor = "ew-resize"; }
      else { audio.pause(); audio.currentTime = xToMs(e.clientX) / 1000; drawWave(); }
      return;
    }
    const mi = markerNearX(e.clientX);
    if (mi >= 0 && !editBusy) { R.drag = mi; R.dragOriginal = R.ayahs[mi].time; c.style.cursor = "ew-resize"; }
    else { audio.currentTime = xToMs(e.clientX) / 1000; drawWave(); }
  });
  window.addEventListener("mousemove", (e) => {
    if (W.drag >= 0 && W.data) { W.preview = xToMs(e.clientX); drawWave(); return; }
    if (R.drag < 0) return; R.ayahs[R.drag].time = xToMs(e.clientX); drawWave(); });
  window.addEventListener("mouseup", (e) => {
    if (W.drag >= 0) { W.drag = -1; W.preview = null; c.style.cursor = "pointer"; editWord(xToMs(e.clientX)); return; }
    if (R.drag < 0) return;
    const i = R.drag; R.drag = -1; c.style.cursor = "pointer";
    R.ayahs[i].time = R.dragOriginal;
    commitMarker(i, xToMs(e.clientX));
  });
  window.addEventListener("resize", () => {
    if (!$("review").classList.contains("hidden")) {
      resizeWave();
      if (W.mode && W.data?.words[W.selected]?.time != null) scrollWaveToMs(W.data.words[W.selected].time);
      renderPauseInspector();
      drawWave();
    }
  });

  window.addEventListener("keydown", (e) => {
    if ($("suraLiveLogBox").open || $("settingsDialog").open || $("review").classList.contains("hidden") || e.target.closest("input, select, textarea, button, summary, [contenteditable]")) return;
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
    currentProject = p;
    timeStep = p.time_step || 10;
    applyAppearance(p.theme, p.arabic_size);
    await refreshProjectLibrary();
    saveState(p.db_path ? "Сохранено" : "Нет открытого проекта", p.db_path ? "saved" : "");
    audioCount = p.audio_count || 0;
    if (p.audio_dir) $("audioDir").value = p.audio_dir;
    if (p.db_path) $("dbPath").value = p.db_path;
    setMode(p.mode);
    if (p.db_path && p.audio_count) {
      $("genBtn").disabled = false;
      $("help").classList.add("collapsed");
      setStatus("");
    }
  } catch {}
  try { await refreshSuras(); } catch (e) { setStatus("Не удалось загрузить суры: " + e.message); }
})();
