"use strict";

// ---- tiny API helpers ----------------------------------------------------
const api = {
  async get(url) { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) throw new Error(translateMessage((await r.json()).detail) || r.status); return r.json(); },
  async post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    if (!r.ok) { let d; try { d = (await r.json()).detail; } catch { d = r.statusText; } throw new Error(translateMessage(d) || r.status); }
    return r.json();
  },
};
const $ = (id) => document.getElementById(id);
function studioIcon(name) {
  const paths = {
    play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="1" fill="currentColor" stroke="none"/>',
    pause: '<path d="M8 5v14M16 5v14" stroke-width="4"/>',
    previous: '<path d="M6 5v14M18 5l-9 7 9 7z"/>',
    next: '<path d="M18 5v14M6 5l9 7-9 7z"/>',
    skipPrevious: '<path d="M5 5h3v14H5zM19 5 9 12l10 7z" fill="currentColor" stroke="none"/>',
    skipNext: '<path d="M16 5h3v14h-3zM5 5l10 7-10 7z" fill="currentColor" stroke="none"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    marker: '<path d="M6 21V3l12 1v9l-12-1"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    edit: '<path d="m16 3 5 5-12 12-6 1 1-6zM14 5l5 5"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 0 1 6 0c0 2-3 2-3 4M12 17h.01"/>',
    filter: '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
    addProject: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7zM3 9h18M12 12v6M9 15h6"/>',
    recording: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7zM3 9h18"/>',
    sidebar: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16M6 8v1M6 12v1"/>',
    settings: '<path d="M10 3h4l.5 2.5 2 .8 2.1-1.4 2.1 3.5-2.1 1.4.3 2.2 2.1 1.4-2 3.5-2.3-1-1.8 1.2L14 20h-4l-.6-2.4-1.8-1.2-2.3 1-2-3.5 2.1-1.4.3-2.2-2.1-1.4 2.1-3.5 2.1 1.4 2-.8z"/><circle cx="12" cy="11.5" r="3"/>',
    export: '<path d="M12 15V3m-4 4 4-4 4 4M8 10H5v11h14V10h-3"/>',
    download: '<path d="M12 3v12m-4-4 4 4 4-4M4 17v4h16v-4"/>',
    trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
    refresh: '<path d="M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-2l2 7M4 12l2 7a7 7 0 0 0 12-2"/>',
    logs: '<path d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7"/>',
    tools: '<path d="m14 6 4 4 3-3a6 6 0 0 1-8 7l-7 7-3-3 7-7a6 6 0 0 1 7-8z"/>',
    fast: '<path d="m13 2-9 12h7l-1 8 10-12h-7z"/>',
    full: '<path d="M3 10v4M7 6v12M11 3v18M15 7v10M19 5v14M23 10v4"/>',
    words: '<path d="M4 5h16M4 10h7m4 0h5M4 15h10m4 0h2M4 20h6m4 0h6"/>',
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
  $("workspaceStatus").classList.toggle("error", /ошибка|не удалось|не сохран|error|failed|could not|not saved/i.test(msg));
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
const ayahLabel = (count) => `${count} ${uiLanguage === "en" ? (count === 1 ? "ayah" : "ayahs") : { one: "аят", few: "аята", many: "аятов", other: "аята" }[new Intl.PluralRules("ru").select(count)]}`;
function saveState(message, state = "") {
  if (state === "error") setStatus(message);
}
const suraName = number => (uiLanguage === "en" ? SURA_NAMES_EN : SURA_NAMES)[number] || t`Сура ${number}`;
const recordingName = project => project.recording_is_default ? t("Внешний проект") : project.recording || t("Внешний проект");
function updateReviewTitle(count) {
  $("reviewTitle").textContent = suraName(R.surah);
  $("reviewSubtitle").textContent = t`Сура ${R.surah} · ${ayahLabel(count || 0)}`;
  if (!$("review").classList.contains("hidden")) document.title = `${$("reviewTitle").textContent} — Ayah Studio`;
}
function syncSidebarStatus() {
  if (!R.dataReady) return;
  const sura = dashboardSuras.find(s => s.surah === R.surah);
  if (!sura) return;
  const marked = R.ayahs.filter(a => a.time != null);
  sura.status = !marked.length ? "not-processed" : marked.length < sura.ayah_count
    ? "partial" : marked.every(a => a.status === "verified") ? "verified" : "processed";
  renderSuras();
  const available = dashboardSuras.filter(s => s.has_audio).length;
  const verified = dashboardSuras.filter(s => s.status === "verified").length;
  $("librarySummary").textContent = t`${available} с аудио · ${verified} проверено · ${dashboardSuras.length} всего`;
  updateGenControls(dashboardSuras);
}
let sidebarAnimation = null, sidebarCompanionAnimations = [];
function setSidebarVisible(visible) {
  const sidebar = $("suraSidebar"), content = sidebar.firstElementChild;
  const navigation = document.querySelector(".workspace-navigation");
  const player = document.querySelector(".floating-player");
  const button = $("sidebarToggle"), reduce = reducedMotion.matches;
  const overlay = getComputedStyle(sidebar).position === "absolute";
  const fullWidth = getComputedStyle(content).width;
  // Snapshot every moving element before cancelling an interrupted transition.
  const width = sidebar.hidden ? "0px" : getComputedStyle(sidebar).width;
  const from = sidebar.hidden
    ? { transform: reduce ? "none" : "translateX(-100%)", opacity: 0 }
    : { transform: getComputedStyle(content).transform, opacity: getComputedStyle(content).opacity };
  const navigationBasis = getComputedStyle(navigation).flexBasis;
  const playerLeft = getComputedStyle(player).left;
  if (sidebarAnimation) sidebarAnimation.cancel();
  sidebarCompanionAnimations.forEach(animation => animation.cancel());
  sidebarCompanionAnimations = [];
  if (!visible && sidebar.contains(document.activeElement)) button.focus();
  sidebar.inert = !visible;
  sidebar.hidden = false;
  document.body.classList.toggle("sidebar-hidden", !visible);
  button.setAttribute("aria-expanded", String(visible));
  button.title = visible ? t("Скрыть боковую панель") : t("Показать боковую панель");
  button.setAttribute("aria-label", button.title);
  const options = {
    duration: reduce ? 100 : 220,
    easing: getComputedStyle(document.documentElement).getPropertyValue("--ease-out").trim(),
    fill: "forwards",
  };
  // The shell shrinks while its fixed-width contents slide: the main area
  // gains exactly the space being released, instead of jumping after exit.
  const targetWidth = visible || overlay ? fullWidth : "0px";
  sidebarAnimation = sidebar.animate([
    { width: overlay || reduce ? targetWidth : width }, { width: targetWidth },
  ], options);
  sidebarCompanionAnimations.push(content.animate([
    from, { transform: visible || reduce ? "none" : "translateX(-100%)", opacity: visible ? 1 : 0 },
  ], options));
  if (!reduce && window.innerWidth > 820) {
    sidebarCompanionAnimations.push(navigation.animate([
      { flexBasis: navigationBasis }, { flexBasis: getComputedStyle(navigation).flexBasis },
    ], options));
  }
  if (!reduce && !overlay) {
    sidebarCompanionAnimations.push(player.animate([
      { left: playerLeft }, { left: getComputedStyle(player).left },
    ], options));
  }
  const animation = sidebarAnimation;
  animation.onfinish = () => {
    sidebar.hidden = !visible;
    animation.cancel();
    sidebarCompanionAnimations.forEach(companion => companion.cancel());
    sidebarCompanionAnimations = [];
    sidebarAnimation = null;
    resizeWave(); drawWave();
  };
  const redraw = () => {
    if (sidebarAnimation !== animation) return;
    resizeWave(); drawWave();
    requestAnimationFrame(redraw);
  };
  requestAnimationFrame(redraw);
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
  if (!btn.disabled && !btn.dataset.busy) btn.textContent = t("Копировать");
}
async function copyLiveLog(id) {
  const el = $(id), btn = logCopyButton(id);
  const text = el ? el.textContent.trim() : "";
  if (!text || !btn) return;
  btn.disabled = true;
  btn.dataset.busy = "1";
  try {
    await writeClipboard(text);
    btn.textContent = t("Скопировано");
  } catch {
    btn.textContent = t("Не скопировано");
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
const modeLabel = (mode) => ({ auto: t("авто"), full: t("полный"), words: t("по словам"), tiled: t("точно·большие"), fast: t("быстрый"), hybrid: "hybrid" }[mode] || mode);
function selectedMode() { return $("genMode") ? $("genMode").value : "full"; }
function selectedSuraMode() { return $("suraMode") ? $("suraMode").value : selectedMode(); }
function modelSupportsFast() { return !currentProject.alignment_model || ["quran-phonetics", "quran-phonetics-fp32", "whisper-tarteel-tiny", "whisper-tarteel-base"].includes(currentProject.alignment_model); }
function setMode(mode) {
  const m = ["fast", "full", "words"].includes(mode) ? mode : "full";
  if ($("genMode")) $("genMode").value = m;
  if ($("suraMode")) $("suraMode").value = m;
  document.querySelectorAll("[data-review-mode], [data-dashboard-mode]").forEach(button => {
    button.setAttribute("aria-pressed", String((button.dataset.reviewMode || button.dataset.dashboardMode) === m));
    button.disabled = !modelSupportsFast() && !["full", "tiled", "words"].includes(button.dataset.reviewMode || button.dataset.dashboardMode);
  });
  for (const id of ["genMode", "suraMode"]) {
    if ($(id)) [...$(id).options].forEach(option => option.disabled = !modelSupportsFast() && !["full", "tiled", "words"].includes(option.value));
  }
  $("dashboardModeToggle").title = t`Режим разметки: ${modeLabel(m)}`;
  updateModeHint();
  applyReviewMode(m);
  updateGenControls(dashboardSuras);
  renderSuras();
}
function updateModeHint() {
  const mode = selectedMode();
  $("modeHint").textContent = mode === "words"
    ? t`Пословная разметка всей суры. Границы аятов сохраняются; недостающие рассчитываются ${modelSupportsFast() ? t("быстрым") : t("полным")} режимом.`
    : mode === "auto"
    ? t("Автоматический выбор режима. После генерации проверьте результат на слух.")
    : mode === "fast"
    ? t("Быстрый черновик по паузам. Сложные участки требуют внимательной проверки.")
    : mode === "tiled"
    ? t("Для длинных сур: обработка частями с ограничением памяти.")
    : t("Полное выравнивание аудио с текстом аятов.");
}

// ===========================================================================
//  Dashboard
// ===========================================================================
let dashboardSuras = [];
async function refreshSuras() {
  dashboardSuras = await api.get("/api/suras");
  renderSuras();
  updateGenControls(dashboardSuras);
  const available = dashboardSuras.filter(s => s.has_audio).length;
  const verified = dashboardSuras.filter(s => s.status === "verified").length;
  $("librarySummary").textContent = t`${available} с аудио · ${verified} проверено · ${dashboardSuras.length} всего`;

}
function updateSuraFilterPicker() {
  const select = $("suraFilter");
  const label = select.selectedOptions[0].textContent;
  $("filterMenuToggle").title = t`Показать: ${label}`;
  $("filterMenuToggle").setAttribute("aria-label", t`Фильтр сур: ${label}`);
  $("filterMenuToggle").classList.toggle("filtered", select.value !== "all");
  document.querySelectorAll("[data-sura-filter]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.suraFilter === select.value)));
}
function renderSuras() {
  const query = $("suraSearch").value.trim().toLocaleLowerCase(uiLanguage);
  const filter = $("suraFilter").value;
  const suras = dashboardSuras.filter(s => (!query || `${s.surah} ${SURA_NAMES[s.surah]} ${SURA_NAMES_EN[s.surah]}`.toLocaleLowerCase(uiLanguage).includes(query))
    && (filter === "all" || (filter === "audio" ? s.has_audio : s.has_audio && s.status === filter)));
  const grid = $("grid");
  const focusedSura = grid.contains(document.activeElement) ? document.activeElement.dataset.surah : null;
  const scrollTop = grid.scrollTop;
  grid.replaceChildren();
  for (const sura of suras) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.dataset.surah = sura.surah;
    tile.dataset.status = sura.status;
    // Keep absent audio discoverable through keyboard and its accessible name.
    tile.setAttribute("aria-disabled", String(!sura.has_audio));
    tile.className = "tile" + (sura.has_audio ? "" : " no-audio");
    tile.classList.toggle("active", sura.surah === R.surah);
    tile.setAttribute("aria-current", sura.surah === R.surah ? "true" : "false");
    const status = badgeLabel(sura.status);
    const detail = `${status}${sura.has_audio ? "" : t(" · Нет аудио")}`;
    tile.setAttribute("aria-label", `${sura.surah}. ${suraName(sura.surah)}, ${ayahLabel(sura.ayah_count)}, ${detail}`);
    tile.innerHTML = `<span class="num" aria-hidden="true">${sura.surah}</span><span class="sura-label"><span class="sura-name">${escapeHtml(suraName(sura.surah))}</span><span class="badge ${sura.status}">${detail}</span></span>`;
    tile.onclick = () => {
      if (!sura.has_audio) { setStatus(t`Сура ${sura.surah}: нет аудио. Импортируйте запись через меню проекта.`); return; }
      void openReview(sura.surah);
    };
    grid.appendChild(tile);
  }
  grid.scrollTop = scrollTop;
  if (focusedSura) grid.querySelector(`[data-surah="${focusedSura}"]`)?.focus({ preventScroll: true });
  $("gridEmpty").classList.toggle("hidden", suras.length > 0);
  $("gridEmpty").innerHTML = dashboardSuras.length
    ? t("<h3>Суры не найдены</h3><p>Измените поиск или фильтр.</p>")
    : t("<h3>Не удалось загрузить суры</h3><p>Откройте проект через кнопку с папкой.</p>");
}
const badgeLabel = status => ({ "not-processed": t("Не размечено"), "partial": t("Размечено частично"), "processed": t("Размечено"), "verified": t("Проверено") }[status] || t("Нет данных"));

// reflect progress on the generate buttons (resume-aware)
function suraNeedsGeneration(sura) {
  if (selectedMode() === "words") return !sura.words_complete;
  return ["not-processed", "partial"].includes(sura.status)
    || (Array.isArray(sura.alignment_modes) && sura.alignment_modes.length > 0
      && !sura.alignment_modes.every(mode => mode === selectedMode()));
}

function updateGenControls(suras) {
  if (GEN.running || generatingSurah != null) { syncAlignmentControls(); return; }
  const withAudio = suras.filter((s) => s.has_audio);
  const remaining = withAudio.filter(suraNeedsGeneration).length;
  const done = withAudio.length - remaining;
  const ready = withAudio.length > 0 && !!$("dbPath").value.trim();
  const gen = $("genBtn"), regen = $("regenBtn");
  if (!ready) { gen.disabled = true; gen.innerHTML = studioIcon("play") + t(" Начать разметку"); regen.classList.add("hidden"); return; }
  if (done === 0) { gen.disabled = false; gen.innerHTML = studioIcon("play") + t` Разметить все · ${withAudio.length}`; regen.classList.add("hidden"); }
  else if (remaining > 0) { gen.disabled = false; gen.innerHTML = studioIcon("play") + t` Продолжить · ${remaining}`; regen.classList.remove("hidden"); }
  else { gen.disabled = true; gen.innerHTML = studioIcon("check") + t` Все ${withAudio.length} готовы`; regen.classList.remove("hidden"); }
}

let editingProject = null, projectFormBusy = false;

function openProjectForm(project = null) {
  if (projectFormBusy || editBusy || reviewLoading || $("projectSelect").disabled) return;
  if (R.audio) R.audio.pause();
  editingProject = project;
  $("projectForm").reset();
  $("projectDialogTitle").textContent = project ? t("Изменить проект") : t("Добавить проект");
  $("importReciter").value = project?.reciter || "";
  $("importRecording").value = project ? recordingName(project) : t("Запись 1");
  $("projectImportFields").classList.toggle("hidden", !!project);
  $("audioDir").required = !project;
  $("audioDir").disabled = !!project;
  $("dbPath").disabled = !!project;
  $("projectFormMessage").textContent = "";
  openDialog($("projectDialog"));
  $("importReciter").focus();
  if (project) $("importReciter").select();
}

async function openProject() {
  if (projectFormBusy) return;
  const target = editingProject;
  const names = { reciter: $("importReciter").value.trim(), recording: $("importRecording").value.trim() };
  if (!names.reciter || !names.recording) {
    $("projectFormMessage").textContent = t("Укажите чтеца и название записи.");
    $(!names.reciter ? "importReciter" : "importRecording").focus();
    return;
  }
  projectFormBusy = true;
  $("projectForm").setAttribute("aria-busy", "true");
  const controls = [...$("projectDialog").querySelectorAll("input, button")];
  const disabled = controls.map(control => control.disabled);
  controls.forEach(control => control.disabled = true);
  $("projectFormMessage").textContent = target ? t("Сохранение…") : t("Копирование аудио и разметки…");
  let saved = false;
  try {
    if (!target) {
      const p = await api.post("/api/projects/create", { ...names,
        audio_dir: $("audioDir").value.trim(), db_path: $("dbPath").value.trim() });
      saved = true;
      await applyProject(p);
    } else {
      const p = target.id
        ? await api.post("/api/projects/update", { id: target.id, ...names })
        : await api.post("/api/settings", { ...names, db_path: target.db_path });
      saved = true;
      if (target.id === currentProject.project_id || (!target.id && target.db_path === currentProject.db_path)) {
        Object.assign(currentProject, target.id ? { reciter: p.reciter, recording: p.recording } : names, { recording_is_default: false });
      }
      await refreshProjectLibrary();
    }
    setStatus("");
  } catch (e) {
    if (saved) setStatus(t("Проект сохранён, но не удалось обновить экран: ") + translateMessage(e.message));
    else $("projectFormMessage").textContent = t("Не удалось сохранить проект: ") + translateMessage(e.message);
  } finally {
    projectFormBusy = false;
    $("projectForm").removeAttribute("aria-busy");
    controls.forEach((control, index) => control.disabled = disabled[index]);
    if (saved) closeDialog($("projectDialog"), true);
    if (!saved) $("openBtn").focus();
    else if ($("settingsDialog").open) {
      const edit = [...$("settingsProjectList").querySelectorAll("[data-edit-project]")]
        .find(button => button.dataset.editProject === (target?.id || target?.db_path));
      (edit || $("addProjectBtn")).focus();
    } else $("recordingMenuToggle").focus();
  }
}

let audioCount = 0;
const GEN = { epoch: 0, running: false, ready: false, es: null, jobId: null, timer: null, startMs: 0, total: 0, wordFailures: 0 };
const ALIGN = { startMs: null, timer: null, done: 0, total: 0, surah: null, ayah: 0, count: 0 };

function alignmentElapsed() {
  const seconds = Math.max(0, Math.floor((Date.now() - ALIGN.startMs) / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
function startAlignmentProgress() {
  ++GEN.epoch;
  if (ALIGN.timer) clearInterval(ALIGN.timer);
  Object.assign(ALIGN, { startMs: Date.now(), done: 0, total: 0, surah: null, ayah: 0, count: 0 });
  $("suraGenStatus").textContent = "";
  $("navigationProgress").classList.add("started");
  $("alignmentElapsed").textContent = "0:00";
  ALIGN.timer = setInterval(() => { $("alignmentElapsed").textContent = alignmentElapsed(); }, 1000);
  renderAlignmentProgress();
}
function stopAlignmentProgress() {
  if (ALIGN.timer) clearInterval(ALIGN.timer);
  ALIGN.timer = null;
  $("alignmentElapsed").textContent = alignmentElapsed();
  renderAlignmentProgress();
}
function renderAlignmentProgress() {
  const running = GEN.running || generatingSurah != null;
  const pct = ALIGN.total ? Math.min(running ? 99 : 100, Math.floor(ALIGN.done / ALIGN.total * 100)) : 0;
  $("alignmentPosition").textContent = ALIGN.ayah
    ? t`Сура ${ALIGN.surah} · аят ${ALIGN.ayah}/${ALIGN.count}`
    : ALIGN.surah ? t`Сура ${ALIGN.surah} · подготовка…` : t("Подготовка…");
  $("alignmentPercent").textContent = `${pct}%`;
  const bar = $("reviewGeneration");
  bar.setAttribute("aria-valuemax", String(ALIGN.total || 1));
  bar.setAttribute("aria-valuenow", String(ALIGN.done));
  bar.setAttribute("aria-valuetext", t`${$("alignmentPosition").textContent}; обработано ${ALIGN.done} из ${ALIGN.total} аятов`);
  bar.title = t`Обработано ${ALIGN.done} из ${ALIGN.total} аятов`;
  bar.firstElementChild.style.width = `${pct}%`;
}
function handleAlignmentEvent(d) {
  if (d.total_ayahs != null) ALIGN.total = d.total_ayahs;
  if (d.ayahs_done != null) ALIGN.done = Math.min(ALIGN.total, d.ayahs_done);
  if (d.type === "sura_start" || d.type === "ayah_progress") {
    ALIGN.surah = d.surah; ALIGN.ayah = d.ayah; ALIGN.count = d.ayah_count;
  } else if (d.type === "sura" && !d.skipped) ALIGN.ayah = ALIGN.count;
  renderAlignmentProgress();
}

function resetAlignmentProgress() {
  if (GEN.running || generatingSurah != null) return;
  ALIGN.startMs = null;
  $("navigationProgress").classList.remove("started");
  $("suraGenStatus").textContent = "";
}

async function runGeneration(onlyMissing = true) {
  if (GEN.running || generatingSurah != null) return;
  const mode = selectedMode();
  const project = currentProject;
  let prepared;
  try {
    prepared = await prepareGeneration({ mode, only_missing: onlyMissing, db_path: project.db_path },
      () => currentProject === project && selectedMode() === mode);
  } catch (e) { setStatus(t("Ошибка: ") + translateMessage(e.message)); return; }
  if (!prepared) return;
  // Explicit start: confirm the deliberate, long-running action.
  const what = onlyMissing
    ? mode === "words" ? t("недостающих (ещё не сгенерированных) сур")
      : t("недостающих сур и сур, размеченных другим режимом")
    : t`ВСЕХ ${audioCount || ""} сур заново`;
  const msg = t`Запустить генерацию ${what} в режиме «${modeLabel(mode)}»?\n`
    + t`Размеченные вручную аяты (edited/verified) не затрагиваются. Можно остановить текущий расчёт.`;
  if (!prepared.hasManual && !confirm(msg)) return;
  startGenUI();
  api.post("/api/generate", prepared.payload)
    .then((job) => {
      GEN.jobId = job.job_id;
      GEN.ready = true; syncAlignmentControls();
      GEN.es = watchGeneration({ jobId: GEN.jobId, onEvent: handleGenEvent,
        getStatus: () => api.get("/api/generate/status"),
        onSnapshot: applyGenerationSnapshot,
        onReconnect: () => setStatus(t("Соединение прервано. Подключаюсь к расчёту…")),
        onError: error => { setStatus(translateMessage(error.message)); endGenUI("error"); } });
    })
    .catch((e) => { setStatus(t("Ошибка: ") + translateMessage(e.message)); endGenUI("error"); });
}
function startGenUI() {
  GEN.running = true; GEN.ready = false; alignmentCancelling = false;
  startAlignmentProgress();
  syncAlignmentControls();
  const panel = $("genPanel");
  panel.classList.remove("hidden", "done", "cancelled");
  $("cancelBtn").disabled = false;
  $("cancelBtn").textContent = t("✕ Отменить");
  $("genTitle").textContent = t("Генерация…");
  $("genCur").textContent = t("подготовка…");
  GEN.wordFailures = 0;
  GEN.diagnosticsErrors = 0;
  GEN.startMs = Date.now();
  $("genStart").textContent = new Date(GEN.startMs).toLocaleTimeString(uiLanguage);
  $("genElapsed").textContent = "0:00";
  resetLiveLog("genLog");
  setProgress(0, 1, t("запуск…"));
  if (GEN.timer) clearInterval(GEN.timer);
  GEN.timer = setInterval(updateElapsed, 1000);
}
function updateElapsed() {
  const s = Math.floor((Date.now() - GEN.startMs) / 1000);
  $("genElapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function handleGenEvent(d) {
  if (d.total_ayahs != null) GEN.total = d.total_ayahs;
  handleAlignmentEvent(d);
  if (d.type === "start") { GEN.total = d.total_ayahs || 0; setProgress(0, GEN.total, t`0 / ${GEN.total} аятов`); }
  else if (d.type === "sura_start") { $("genCur").textContent = t`обрабатываю суру ${d.surah}…`; }
  else if (d.type === "ayah_progress") { setProgress(d.ayahs_done, GEN.total, t`${d.ayahs_done} / ${GEN.total} аятов`); }
  else if (d.type === "sura_log") {
    appendLiveLog("genLog", t`Сура ${d.surah}: ${d.line}`);
    if (R.surah === d.surah) appendLiveLog("suraLiveLog", d.line);
  }
  else if (d.type === "sura") {
    GEN.wordFailures += d.word_failures || 0;
    setProgress(ALIGN.done, GEN.total, t`${ALIGN.done} / ${GEN.total} аятов`);
    const fast = d.fast_trace
      ? t` · пауз ${d.fast_trace.n_pause}, wasl ${d.fast_trace.n_wasl}, unresolved ${d.fast_trace.unresolved || 0}, atomic ${d.fast_trace.atomic_wasl_reads || 0}, окно ${(d.fast_trace.coverage || 0).toFixed(2)}×${d.fast_trace.drift_detected ? " · drift" : ""}`
      : "";
    const wordResult = d.words_written != null ? t` · слов записано ${d.words_written}${d.word_failures ? t` · требуют проверки ${d.word_failures} аят.` : ""}` : "";
    $("genCur").textContent = t`сура ${d.surah} обработана${wordResult}${d.written != null ? t` · ${d.written} аят.` : ""}${fast}`;
    if (d.diagnostics_error) {
      GEN.diagnosticsErrors += 1;
      appendLiveLog("genLog", t`Сура ${d.surah}: разметка сохранена; ошибка диагностики: ${d.diagnostics_error}`);
    } else if (d.log_paths && d.log_paths.log) appendLiveLog("genLog", t`Лог сохранён: ${d.log_paths.log}`);
    if (R.surah === d.surah) reloadSura();
    refreshSuras();
  }
  else if (d.type === "done" || d.type === "cancelled") {
    GEN.diagnosticsErrors = d.diagnostics_errors?.length || GEN.diagnosticsErrors;
    endGenUI(d.type, d.completed);
  }
  else if (d.type === "error") { setStatus(t("Ошибка генерации: ") + translateMessage(d.message)); endGenUI("error"); }
}

function applyGenerationSnapshot(job) {
  GEN.total = job.total_ayahs || job.progress?.total_ayahs || GEN.total;
  GEN.wordFailures = job.word_failures ?? GEN.wordFailures;
  handleAlignmentEvent({ ...job.progress, total_ayahs: GEN.total, ayahs_done: job.ayahs_done || 0 });
  setProgress(ALIGN.done, GEN.total, t`${ALIGN.done} / ${GEN.total} аятов`);
}

async function restoreGeneration() {
  const project = currentProject, epoch = GEN.epoch;
  const job = await api.get("/api/generate/status");
  if (project !== currentProject || epoch !== GEN.epoch || GEN.running || generatingSurah != null) return;
  if (!job.job_id || job.same_project === false || (!job.running && !job.terminal)) return;
  startGenUI();
  GEN.jobId = job.job_id;
  GEN.ready = job.running;
  GEN.startMs = ALIGN.startMs = job.started_at_ms || Date.now();
  $("genStart").textContent = new Date(GEN.startMs).toLocaleTimeString(uiLanguage);
  updateElapsed();
  applyGenerationSnapshot(job);
  alignmentCancelling = job.cancel_requested;
  syncAlignmentControls();
  if (job.terminal) { handleGenEvent(job.terminal); return; }
  GEN.es = watchGeneration({ jobId: GEN.jobId, after: job.latest_event_id,
    onEvent: handleGenEvent, getStatus: () => api.get("/api/generate/status"),
    onSnapshot: applyGenerationSnapshot,
    onReconnect: () => setStatus(t("Соединение прервано. Подключаюсь к расчёту…")),
    onError: error => { setStatus(translateMessage(error.message)); endGenUI("error"); } });
}
function endGenUI(kind, completed) {
  GEN.running = false; GEN.ready = false; alignmentCancelling = false;
  stopAlignmentProgress();
  syncAlignmentControls();
  if (GEN.es) { GEN.es.close(); GEN.es = null; }
  if (GEN.timer) { clearInterval(GEN.timer); GEN.timer = null; }
  updateElapsed();
  const elapsed = $("genElapsed").textContent;
  $("genBtn").disabled = false;
  $("cancelBtn").disabled = true;
  const panel = $("genPanel");
  if (kind === "done") {
    panel.classList.add("done"); $("genTitle").textContent = GEN.wordFailures ? t("Расчёт завершён · нужна проверка") : t("✓ Готово");
    $("genCur").textContent = t`обработано ${completed ?? ""} сур за ${elapsed}${GEN.wordFailures ? t` · требуют проверки ${GEN.wordFailures} аят.` : ""}`;
    setProgress(ALIGN.done, ALIGN.total, t`${ALIGN.done} / ${ALIGN.total} аятов`);
  } else if (kind === "cancelled") {
    panel.classList.add("cancelled"); $("genTitle").textContent = t("⏹ Отменено");
    $("genCur").textContent = t`обработано ${completed ?? 0} сур за ${elapsed}`;
  } else {
    panel.classList.add("cancelled"); $("genTitle").textContent = t("✕ Ошибка");
  }
  $("suraGenStatus").className = "sura-gen-status " + (kind === "done" ? "ok" : kind === "error" ? "err" : "warn");
  $("suraGenStatus").textContent = kind === "done" ? t("Разметка завершена") : kind === "cancelled" ? t("Разметка остановлена") : t("Ошибка разметки. Проверьте логи.");
  if (GEN.diagnosticsErrors && kind !== "error") {
    const warning = kind === "cancelled"
      ? t("Разметка остановлена. Обработанные суры сохранены; диагностика — не полностью. Проверьте доступ к папке проекта.")
      : t("Разметка сохранена. Не удалось сохранить диагностику. Проверьте доступ к папке проекта.");
    $("genTitle").textContent = warning;
    $("suraGenStatus").textContent = warning;
    $("suraGenStatus").className = "sura-gen-status warn";
  }
  if (kind === "cancelled" && R.surah != null) reloadSura();
  refreshSuras();
}
async function cancelGeneration() {
  if (alignmentCancelling) return;
  alignmentCancelling = true; syncAlignmentControls();
  $("cancelBtn").disabled = true;
  $("cancelBtn").textContent = t("Отмена…");
  $("genCur").textContent = t("остановка расчёта…");
  $("suraGenStatus").textContent = t("Остановка расчёта…");
  try { await api.post("/api/generate/cancel", { job_id: GEN.jobId }); }
  catch (e) { alignmentCancelling = false; syncAlignmentControls(); $("cancelBtn").disabled = false; setStatus(t("Ошибка отмены: ") + translateMessage(e.message)); }
}
function setProgress(done, total, text) {
  $("progressFill").style.right = (100 - (total ? (done / total) * 100 : 0)) + "%";
  $("progressText").textContent = text;
  if (GEN.running) {
    $("suraGenStatus").textContent = alignmentCancelling ? t("Остановка расчёта…") : t`Обработано ${text}`;
  }
}

// ===========================================================================
//  Review
// ===========================================================================
const R = {
  request: 0, reloadRequest: 0, reloadPending: false, dataReady: false, audioReady: false, waveReady: false, dataLoading: false, waveLoading: false,
  surah: null, ayahs: [], wordProgress: null, duration: 0, peaks: [], audio: null, current: -1,
  loop: false, playbackMode: "continuous", playbackAyah: -1, playbackTimer: null,
  drag: -1, zoom: savedWaveZoom(), recheck: null,
  pausesVisible: false, pauseDecisions: null,
  waveBorn: [], waveFrame: 0, waveCancel: null, waveBucketMs: 100, waveMaximum: 1,
  pinch: null,
};

const W = { loading: false, mode: false, data: null, selected: 0, request: 0, busy: false, drag: -1, preview: null, stop: null, timer: null, unit: "s" };

function applyReviewMode(mode) {
  const changed = W.mode !== (mode === "words");
  W.mode = mode === "words";
  syncPlayerSettings();
  clearTimeout(R.playbackTimer);
  R.playbackAyah = -1;
  for (const id of ["draftPausesBtn", "correctDraftBtn", "recheckBtn"])
    $(id).classList.toggle("hidden", W.mode);
  syncAlignmentControls();
  $("regenSuraBtn").setAttribute("aria-label", W.mode ? t("Разметить слова всей суры") : t("Начать разметку суры"));
  $("regenSuraBtn").title = W.mode ? t("Сохранить границы аятов и рассчитать все слова суры") : t("Рассчитать тайм-коды этой суры выбранным режимом");
  for (const id of ["modeMenuToggle", "dashboardModeToggle"]) {
    $(id).innerHTML = studioIcon(mode) + t`<span><span class="mode-caption">Режим:</span> ${escapeHtml(modeLabel(mode))}</span>`;
    $(id).setAttribute("aria-label", t`Режим разметки: ${modeLabel(mode)}`);
  }
  $("modeMenuToggle").title = t`Режим разметки: ${modeLabel(mode)}`;
  $("verifySelectedBtn").textContent = W.mode ? t("Отметить слово проверенным") : t("Отметить выбранный проверенным");
  updateSuraProgress();
  if (!changed) return;
  W.stop = null;
  if (R.audio) R.audio.pause();
  $("alignWordsBtn").classList.toggle("hidden", !W.mode);
  $("wordUnitControl").classList.toggle("hidden", !W.mode);
  $("ayahList").classList.toggle("word-mode", W.mode);
  for (const [id, wordLabel, ayahLabel] of [["prevBtn", t("Предыдущее слово"), t("Предыдущий аят")], ["nextBtn", t("Следующее слово"), t("Следующий аят")]]) {
    const label = W.mode ? wordLabel : ayahLabel;
    $(id).setAttribute("aria-label", label);
    $(id).title = label;
  }
  $("waveHint").textContent = W.mode ? t("Синие линии — начала слов · тяните метку для правки") : t("Клик по волне — перемотка · тяните метку — двигайте начало аята");
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
  reviewLoadMessage("ayahLoadStatus", t("Загрузка слов и разметки…"));
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
      reviewLoadMessage("ayahLoadStatus", t("Не удалось загрузить слова. ") + translateMessage(e.message) + " ", true);
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
  list.classList.add("word-mode"); list.setAttribute("aria-label", t("Слова суры"));
  const words = W.data ? W.data.words : [];
  const groups = new Map();
  words.forEach((word, i) => {
    if (!groups.has(word.ayah)) {
      const group = document.createElement("li");
      group.className = "word-ayah"; group.dataset.ayah = word.ayah;
      group.innerHTML = t`<div class="word-group-head"><strong>Аят ${word.ayah}</strong></div><ul class="word-cards" aria-label="Слова аята ${word.ayah}"></ul>`;
      list.appendChild(group); groups.set(word.ayah, group.querySelector("ul"));
    }
    const li = document.createElement("li");
    li.className = "word-card"; li.dataset.word = i;
    li.classList.toggle("active", i === W.selected);
    li.classList.toggle("missing", word.time == null);
    li.classList.toggle("stale", !!word.stale);
    const label = `${word.ayah}:${word.number}`;
    const status = word.stale ? t("границы изменились") : ({ auto: t("Не помечен"), edited: t("Нужно ревью"), verified: t("Всё хорошо") }[word.status] || t("нет метки"));
    const editable = Number.isInteger(word.start_ms) && Number.isInteger(word.end_ms);
    const editUnit = W.unit;
    const unit = editUnit === "ms" ? t("мс") : t("с");
    li.innerHTML = t`
      <div class="word-card-head"><button class="a-num" aria-label="Изменить слово ${word.number} аята ${word.ayah}" aria-haspopup="dialog" aria-controls="wordEditor${word.word_index}" aria-pressed="${i === W.selected}">${word.number}</button><span class="word-meta">${status}</span><span class="timing-confidence"></span></div>
      <button class="a-text word-play" dir="rtl" lang="ar" aria-label="Прослушать ${escapeHtml(word.text)}">${escapeHtml(word.text)}</button>
      <dialog id="wordEditor${word.word_index}" class="ayah-time-popover word-editor" aria-labelledby="wordEditorTitle${word.word_index}">
      <div class="ayah-time-heading"><strong id="wordEditorTitle${word.word_index}">Аят ${word.ayah} · слово ${word.number}</strong><button type="button" class="ghost close-time" aria-label="Закрыть редактор слова">${studioIcon("close")}</button></div>
      <div class="word-time"><span class="word-unit">Начало, ${unit}</span><div class="time-editor">
        <input type="number" min="0" step="${W.unit === "ms" ? "1" : "0.001"}" class="a-time-input" inputmode="decimal" aria-label="Начало слова ${label}, ${W.unit === "ms" ? t("миллисекунды") : t("секунды")}" placeholder="Нет метки" value="${wordDisplayTime(word.time)}" ${editable ? "" : "disabled"} />
        <div class="time-arrows"><button type="button" aria-label="Увеличить время слова ${label}" ${editable ? "" : "disabled"}>${studioIcon("up")}</button><button type="button" aria-label="Уменьшить время слова ${label}" ${editable ? "" : "disabled"}>${studioIcon("down")}</button></div>
      </div></div>
      <div class="ayah-status-segments" role="group" aria-label="Статус слова ${label}">
        <button type="button" data-word-status="auto" title="Сбросить проверку, сохранив время"><span class="status-swatch" aria-hidden="true"></span><span>Не помечен</span></button>
        <button type="button" data-word-status="edited"><span class="status-swatch" aria-hidden="true"></span><span>Нужно ревью</span></button>
        <button type="button" data-word-status="verified"><span class="status-swatch" aria-hidden="true"></span><span>Всё хорошо</span></button>
      </div>
      <p class="ayah-time-help">${word.stale ? t("Границы изменились — обновите время.<br>") : ""}Шаг ${timeStep} мс · сохраняется автоматически</p>
      <p class="ayah-time-error hidden" role="status"></p>
      </dialog>`;
    applyTimingStatus(li, word);
    li.querySelectorAll("[data-word-status]").forEach(button => {
      button.setAttribute("aria-pressed", String(!word.stale && button.dataset.wordStatus === (word.status || "auto")));
      button.disabled = W.busy || word.time == null || word.stale || !editable;
      button.onclick = () => { W.selected = i; editWord(0, "status", button.dataset.wordStatus); };
    });
    const dialog = li.querySelector("dialog"), number = li.querySelector(".a-num");
    li.querySelector(".close-time").onclick = () => dialog.close();
    dialog.onclose = () => number.focus({ preventScroll: true });
    dialog.onclick = e => {
      e.stopPropagation();
      const bounds = dialog.getBoundingClientRect();
      if (e.target === dialog && (e.clientX < bounds.left || e.clientX > bounds.right || e.clientY < bounds.top || e.clientY > bounds.bottom)) dialog.close();
    };
    const input = li.querySelector("input");
    input.onfocus = () => selectWord(i, false, false);
    const commitInput = async () => {
      if (input.value === wordDisplayTime(word.time, editUnit)) return;
      const ms = parseWordTime(input.value, editUnit);
      if (!Number.isSafeInteger(ms)) { input.setAttribute("aria-invalid", "true"); setStatus(editUnit === "ms" ? t("Введите целое число миллисекунд.") : t("Введите секунды, например 6.120.")); return; }
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
    number.onclick = () => openWordEditor(i);
    groups.get(word.ayah).appendChild(li);
  });
  const word = words[W.selected];
  $("alignWordsBtn").disabled = !word || !Number.isInteger(word.start_ms) || W.busy;
  $("verifySelectedBtn").disabled = !word || word.time == null || word.stale || W.busy;
}
function openWordEditor(i, focusSelector = "input") {
  if (W.busy || reviewLoading) return;
  const row = document.querySelector(`#ayahList li[data-word="${i}"]`);
  const dialog = row?.querySelector("dialog"), number = row?.querySelector(".a-num");
  if (!dialog) return;
  R.audio?.pause(); selectWord(i, false, false);
  dialog.showModal();
  const anchor = number.getBoundingClientRect(), bounds = dialog.getBoundingClientRect();
  dialog.style.left = Math.max(8, Math.min(anchor.left, innerWidth - bounds.width - 8)) + "px";
  dialog.style.top = Math.max(8, Math.min(anchor.bottom + 8, innerHeight - bounds.height - 8)) + "px";
  const control = dialog.querySelector(focusSelector);
  if (control && !control.disabled) { control.focus({ preventScroll: true }); if (control.tagName === "INPUT") control.select(); }
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
  } catch (e) { if (request === W.request) setStatus(t("Ошибка: ") + translateMessage(e.message)); }
  finally { W.busy = false; }
  renderWords(); drawWave();
}
async function editWord(ms, action, status) {
  const data = wordContext(), request = W.request;
  if (!data || W.busy) return;
  const popup = document.querySelector(".word-editor[open]");
  const focusedStatus = document.activeElement?.dataset.wordStatus;
  const popupIndex = popup ? Number(popup.closest("[data-word]").dataset.word) : null;
  let error = "";
  W.busy = true; saveState(t("Сохранение…"));
  popup?.querySelectorAll("input, .time-arrows button, [data-word-status]").forEach(el => el.disabled = true);
  try {
    const result = await api.post(`/api/sura/${data.surah}/ayah/${data.ayah}/words/edit`,
      { ...wordPayload(data), word_index: data.word_index, time: Math.round(ms), action, status });
    if (request !== W.request) return;
    mergeWordResult(result); saveState(t("Сохранено"), "saved"); setStatus("");
  } catch (e) { if (request === W.request) { error = t("Не сохранено. ") + translateMessage(e.message); saveState(t("Не сохранено"), "error"); setStatus(error); } }
  finally { W.busy = false; }
  const reopen = popup?.open;
  renderWords(); drawWave();
  if (reopen && request === W.request) {
    openWordEditor(popupIndex, focusedStatus ? `[data-word-status="${focusedStatus}"]` : "input");
    const message = document.querySelector(".word-editor[open] .ayah-time-error");
    if (message) { message.textContent = error; message.classList.toggle("hidden", !error); }
  }
}
function playWord() {
  if (!R.audioReady || !R.dataReady) return;
  clearTimeout(W.timer); W.timer = null; W.stop = null;
  if (R.audio) R.audio.pause();
  const word = W.data?.words[W.selected];
  if (!word || word.time == null || word.stale) { setStatus(t("Сначала поставьте актуальную метку слова.")); return; }
  const next = W.data.words[W.selected + 1];
  const sameAyah = next && next.ayah === word.ayah;
  if (sameAyah && (next.time == null || next.stale)) { setStatus(t("Сначала поставьте метку следующего слова.")); return; }
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
    $("suraLogTitle").textContent = t`Логи · Сура ${R.surah}`;
    openDialog(dialog);
    void loadSuraLogs();
  } else if (dialog.open) closeDialog(dialog, true);
}

let suraLogRequest = 0;
const COMPARISON = { request: 0, pairRequest: 0, context: null, left: null, right: null, stopMs: null, hasAudio: false, audioSignature: null };

function sameComparisonAudio(a, b) {
  return a && b && a.name === b.name && a.size === b.size && a.mtime_ns === b.mtime_ns;
}

function comparisonActive(context, request) {
  return context.active() && COMPARISON.context === context && COMPARISON.request === request && $("comparisonDialog").open;
}

async function openComparison() {
  if (R.surah == null || editBusy || reviewLoading) return;
  const context = captureReviewContext(), request = ++COMPARISON.request;
  COMPARISON.context = context; COMPARISON.left = COMPARISON.right = null;
  R.audio?.pause();
  $("reviewToolsToggle").closest("details").open = false;
  $("comparisonTitle").textContent = t`Сравнение · Сура ${context.surah}`;
  $("comparisonRows").replaceChildren();
  $("comparisonEmpty").textContent = "";
  $("comparisonMessage").textContent = t("Загрузка результатов…");
  $("comparisonWarning").textContent = $("comparisonSources").textContent = "";
  $("comparisonDifferences").checked = false;
  for (const id of ["comparisonLeft", "comparisonRight"]) { $(id).replaceChildren(); $(id).disabled = true; }
  COMPARISON.hasAudio = false; COMPARISON.audioSignature = null;
  openDialog($("comparisonDialog"));
  try {
    const data = await api.get(`/api/sura/${context.surah}/runs`);
    if (!comparisonActive(context, request)) return;
    COMPARISON.hasAudio = data.has_audio; COMPARISON.audioSignature = data.audio_signature;
    if (data.has_audio) $("comparisonAudio").src = `/api/audio/${context.surah}`;
    for (const id of ["comparisonLeft", "comparisonRight"]) {
      const select = $(id), placeholder = document.createElement("option");
      placeholder.value = ""; placeholder.textContent = t("Выберите прогон"); select.append(placeholder);
      for (const run of data.runs) {
        const option = document.createElement("option"); option.value = run.id;
        const model = run.source.split(":")[1] || t("Нет данных о модели");
        const date = new Date(run.created_ms).toLocaleString(uiLanguage, {day:"2-digit", month:"2-digit", year:"numeric", hour:"2-digit", minute:"2-digit", second:"2-digit", fractionalSecondDigits:3});
        option.textContent = `${modeLabel(run.mode)} · ${date} · ${model}`;
        select.append(option);
      }
      select.disabled = !data.runs.length;
    }
    const left = data.runs.find(run => run.mode === "fast") || data.runs[0];
    const right = data.runs.find(run => run.mode === "full" && run.id !== left?.id)
      || data.runs.find(run => run.id !== left?.id);
    $("comparisonLeft").value = left?.id || ""; $("comparisonRight").value = right?.id || "";
    if (!right) {
      $("comparisonMessage").textContent = t("Для сравнения нужны два сохранённых прогона. Запустите быстрый и полный режимы.");
    } else await loadComparisonPair();
  } catch (error) {
    if (comparisonActive(context, request)) $("comparisonMessage").textContent = t("Не удалось загрузить результаты: ") + translateMessage(error.message);
  }
}

async function loadComparisonPair() {
  const context = COMPARISON.context, request = COMPARISON.request, pair = ++COMPARISON.pairRequest;
  $("comparisonAudio").pause(); COMPARISON.stopMs = null;
  $("comparisonPlaying").textContent = "";
  COMPARISON.left = COMPARISON.right = null; $("comparisonRows").replaceChildren();
  $("comparisonEmpty").textContent = "";
  $("comparisonWarning").textContent = $("comparisonSources").textContent = "";
  const leftId = $("comparisonLeft").value, rightId = $("comparisonRight").value;
  if (!context || !leftId || !rightId || leftId === rightId) {
    $("comparisonMessage").textContent = t("Выберите два разных прогона."); return;
  }
  $("comparisonMessage").textContent = t("Загрузка результатов…");
  try {
    const [left, right] = await Promise.all([leftId, rightId].map(id => api.get(`/api/sura/${context.surah}/runs/${encodeURIComponent(id)}`)));
    if (!comparisonActive(context, request) || pair !== COMPARISON.pairRequest) return;
    COMPARISON.left = left; COMPARISON.right = right;
    $("comparisonSources").textContent = t`А: ${modeLabel(left.mode)} · Б: ${modeLabel(right.mode)}`;
    $("comparisonSources").title = `${left.source}\n${right.source}`;
    $("comparisonWarning").textContent = !left.raw_model_result || !right.raw_model_result
      ? t("Старые результаты могут включать сохранённые ручные метки.") : "";
    if (!left.audio_signature || !right.audio_signature) {
      $("comparisonWarning").textContent += " " + t("Для старого прогона запись аудио не подтверждена.");
    } else if (!sameComparisonAudio(left.audio_signature, right.audio_signature)) {
      $("comparisonWarning").textContent += " " + t("Прогоны относятся к разным версиям аудио. Сравнивайте с осторожностью.");
    }
    const audioMismatch = [left, right].some(run => run.audio_signature && COMPARISON.audioSignature
      && !sameComparisonAudio(run.audio_signature, COMPARISON.audioSignature));
    if (audioMismatch) {
      $("comparisonWarning").textContent += " " + t("Текущая запись отличается от сохранённой. Прослушивание недоступно.");
      $("comparisonAudio").removeAttribute("src"); $("comparisonAudio").load();
    } else if (COMPARISON.hasAudio && !$("comparisonAudio").getAttribute("src")) {
      $("comparisonAudio").src = `/api/audio/${context.surah}`;
    }
    renderComparison();
  } catch (error) {
    if (comparisonActive(context, request) && pair === COMPARISON.pairRequest) $("comparisonMessage").textContent = t("Не удалось загрузить результаты: ") + translateMessage(error.message);
  }
}

function comparisonStats(left, right) {
  const byAyah = new Map(right.map(a => [a.ayah, a]));
  const deltas = left.filter(a => byAyah.has(a.ayah)).map(a => Math.abs(byAyah.get(a.ayah).time - a.time));
  return { count: deltas.length, mean: deltas.length ? Math.round(deltas.reduce((a, b) => a + b, 0) / deltas.length) : 0,
    max: deltas.length ? Math.max(...deltas) : 0, large: deltas.filter(d => d >= 500).length };
}

function renderComparison() {
  const { left, right } = COMPARISON;
  if (!left || !right) return;
  const stats = comparisonStats(left.ayahs, right.ayahs);
  $("comparisonMessage").textContent = t`Сравнено ${stats.count} аятов · средняя разница ${stats.mean} мс · максимум ${stats.max} мс · от 500 мс: ${stats.large}`;
  const maps = [left, right].map(run => new Map(run.ayahs.map(a => [a.ayah, a])));
  const tbody = $("comparisonRows"); tbody.replaceChildren();
  for (const ayah of R.ayahs) {
    const a = maps[0].get(ayah.ayah), b = maps[1].get(ayah.ayah);
    const delta = a && b ? b.time - a.time : null;
    if ($("comparisonDifferences").checked && delta != null && Math.abs(delta) < 500) continue;
    const tr = document.createElement("tr");
    if (delta != null && Math.abs(delta) >= 500) tr.className = "comparison-large";
    const number = document.createElement("th"); number.scope = "row"; number.textContent = ayah.ayah;
    number.title = ayah.text; tr.append(number);
    [a, b].forEach((item, side) => {
      const td = document.createElement("td");
      if (item) {
        const button = document.createElement("button"); button.className = "ghost comparison-listen";
        button.textContent = "▶ " + exactTime(item.time);
        button.setAttribute("aria-label", t`Прослушать аят ${ayah.ayah}, вариант ${side === 0 ? "A" : "B"}`);
        button.disabled = !COMPARISON.hasAudio || [left, right].some(run => run.audio_signature && COMPARISON.audioSignature
          && !sameComparisonAudio(run.audio_signature, COMPARISON.audioSignature));
        button.onclick = () => playComparison(side === 0 ? left : right, item.ayah, side === 0 ? "A" : "B");
        const score = document.createElement("span"), estimate = timingConfidence(item);
        score.className = "timing-confidence"; score.dataset.level = estimate.level; score.textContent = estimate.label;
        score.title = estimate.title;
        td.append(button, score);
      } else td.textContent = t("Нет метки");
      tr.append(td);
    });
    const difference = document.createElement("td"); difference.textContent = delta == null ? "—" : `${delta > 0 ? "+" : ""}${delta}`; tr.append(difference);
    tbody.append(tr);
  }
  $("comparisonEmpty").textContent = tbody.children.length ? "" : t("Различий от 500 мс нет.");
}

async function playComparison(run, ayah, variant) {
  const context = COMPARISON.context, request = COMPARISON.request, pair = COMPARISON.pairRequest;
  if (!context || !comparisonActive(context, request)) return;
  const index = run.ayahs.findIndex(a => a.ayah === ayah), item = run.ayahs[index];
  if (!item) return;
  const audio = $("comparisonAudio"); audio.pause();
  if (audio.readyState < 1) {
    $("comparisonPlaying").textContent = t("Аудио ещё загружается. Попробуйте снова."); return;
  }
  if (item.time >= audio.duration * 1000) {
    $("comparisonPlaying").textContent = t("Метка находится за пределами аудио."); return;
  }
  const next = run.ayahs[index + 1];
  COMPARISON.stopMs = next?.ayah === ayah + 1 ? next.time : ayah === R.ayahs.length
    ? audio.duration * 1000 : Math.min(item.time + 8000, audio.duration * 1000);
  audio.currentTime = item.time / 1000;
  $("comparisonPlaying").textContent = t`Аят ${ayah} · вариант ${variant}`;
  try { await audio.play(); }
  catch { if (comparisonActive(context, request) && pair === COMPARISON.pairRequest) $("comparisonPlaying").textContent = t("Не удалось воспроизвести аудио."); }
}

function stopComparison() {
  COMPARISON.request++; COMPARISON.pairRequest++; COMPARISON.context = null; COMPARISON.stopMs = null;
  $("comparisonAudio").pause(); $("comparisonAudio").removeAttribute("src"); $("comparisonAudio").load();
}

function showSuraLogPane(id) {
  for (const pane of ["suraLogList", "suraLiveLog", "suraSavedLog"]) $(pane).classList.toggle("hidden", pane !== id);
  $("suraLogBackBtn").classList.toggle("hidden", id === "suraLogList");
  for (const log of ["suraLiveLog", "suraSavedLog"]) logCopyButton(log).classList.toggle("hidden", log !== id);
  $("suraLogTitle").textContent = t`Логи · Сура ${R.surah}`;
  $("suraLogEmpty").textContent = "";
  $("suraLogEmpty").classList.add("hidden");
}
async function loadSuraLogs() {
  const request = ++suraLogRequest, surah = R.surah, project = currentProject.db_path;
  const active = () => request === suraLogRequest && R.surah === surah && currentProject.db_path === project && $("suraLiveLogBox").open;
  showSuraLogPane("suraLogList");
  $("suraLogList").replaceChildren();
  $("suraLogEmpty").textContent = t("Загрузка логов…");
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
    addEntry(t("Текущий расчёт"), t("Записи текущего экрана"), () => {
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
      addEntry(entry.id, `${entry.mode} · ${new Date(entry.modified_ms).toLocaleString(uiLanguage)}`, async () => {
        const detailRequest = ++suraLogRequest;
        showSuraLogPane("suraSavedLog");
        resetLiveLog("suraSavedLog");
        $("suraLogTitle").textContent = entry.id;
        $("suraSavedLog").textContent = t("Загрузка…");
        $("suraLogBackBtn").focus();
        try {
          const result = await api.get(`/api/sura/${surah}/logs/${encodeURIComponent(entry.id)}`);
          if (detailRequest !== suraLogRequest || R.surah !== surah || currentProject.db_path !== project || !$("suraLiveLogBox").open) return;
          $("suraSavedLog").textContent = [result.log, result.json === null ? null : t`Данные JSON\n${result.json}`].filter(value => value !== null).join("\n\n");
          updateLogCopyButton("suraSavedLog");
        } catch (error) {
          if (detailRequest === suraLogRequest && $("suraLiveLogBox").open) $("suraSavedLog").textContent = t`Не удалось открыть лог: ${error.message}. Вернитесь к списку и попробуйте снова.`;
        }
      });
    }
    $("suraLogEmpty").textContent = t("Сохранённых логов этой суры пока нет.");
    $("suraLogEmpty").classList.toggle("hidden", !!$("suraLogList").children.length);
  } catch (error) {
    if (!active()) return;
    addLive();
    $("suraLogEmpty").textContent = t`Не удалось загрузить логи: ${error.message}`;
    const retry = document.createElement("button"); retry.className = "ghost"; retry.textContent = t("Повторить");
    retry.onclick = () => { void loadSuraLogs(); }; $("suraLogList").append(retry);
  }
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const dialogMotions = new WeakMap();
const menuMotions = new WeakMap();
let keyboardMotion = false;
document.addEventListener("keydown", e => {
  keyboardMotion = true;
  if (["Tab", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
    document.body.dataset.focusMethod = "keyboard";
  }
}, true);
document.addEventListener("pointerdown", () => {
  keyboardMotion = false;
  document.body.dataset.focusMethod = "pointer";
}, true);
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
          if (wave.offset !== R.peaks.length) throw new Error(t("Нарушен порядок участков волны"));
          R.peaks.push(...wave.peaks);
          const now = performance.now(); R.waveBorn.push(...wave.peaks.map(() => now));
          const progress = R.duration ? ` · ${Math.min(99, Math.floor(wave.decoded_ms / R.duration * 100))}%` : "";
          reviewLoadMessage("waveLoadStatus", t("Построение волны") + progress);
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
    stream.onerror = () => { stop(); reject(new Error(t("Соединение прервано"))); };
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
    retry.className = "ghost"; retry.textContent = t("Повторить");
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
  syncAlignmentControls();
}
async function openReview(surah) {
  if (editBusy || W.busy) return;
  R.waveCancel?.();
  if (R.waveFrame) cancelAnimationFrame(R.waveFrame);
  R.waveFrame = 0;
  const request = ++R.request;
  R.reloadPending = false;
  $("pauseOverlayBtn").disabled = false;
  const active = () => request === R.request;
  reviewLoading = true;
  if (R.audio) R.audio.pause();
  Object.assign(R, { surah, ayahs: [], wordProgress: null, peaks: [], duration: 0,
    waveBorn: [], waveBucketMs: 100, waveMaximum: 1,
    current: -1, recheck: null, pausesVisible: false, pauseDecisions: null,
    zoom: R.zoom, pinch: null, drag: -1, dataReady: false, audioReady: false, waveReady: false, dataLoading: true, waveLoading: true });
  resetWords();
  renderSuras();
  resetLiveLog("suraLiveLog");
  setSuraLogVisible(false);
  updateWaveControls();
  const summary = dashboardSuras.find(s => s.surah === surah);
  updateReviewTitle(summary?.ayah_count);
  resetAlignmentProgress();
  $("suraGenStatus").textContent = "";
  $("suraGenStatus").className = "sura-gen-status";
  $("verifySelectedBtn").disabled = true;
  $("alignWordsBtn").disabled = true;
  updatePauseOverlayUi();
  renderAyahList();
  $("ayahList").scrollTop = 0;
  updateSuraProgress();
  $("suraProgressText").textContent = t("Загрузка разметки…");
  $("suraProgressMeta").textContent = "—";
  $("waveHint").textContent = "";
  $("clock").textContent = `${exactTime(0)} / —`;
  reviewLoadMessage("ayahLoadStatus", t("Загрузка аятов и разметки…"));
  $("waveLoadStatus").classList.remove("wave-partial");
  reviewLoadMessage("waveLoadStatus", t("Загрузка аудиоволны…"));
  reviewLoadMessage("audioLoadStatus", t("Загрузка аудио…"));
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
    reviewLoadMessage("audioLoadStatus", t("Не удалось загрузить аудио. "), true);
    updateReviewReadiness();
  };
  audio.src = `/api/audio/${surah}`;
  audio.defaultPlaybackRate = parseFloat($("rate").value);
  audio.playbackRate = parseFloat($("rate").value);
  audio.load();

  await Promise.allSettled([
    (async () => {
      try {
        const sura = await api.get(`/api/sura/${surah}`);
        if (!active()) return;
        R.ayahs = sura.ayahs; R.wordProgress = sura.word_progress; R.dataReady = true; R.dataLoading = false;
        updateReviewTitle(sura.ayah_count);
        document.title = `${$("reviewTitle").textContent} — Ayah Studio`;
        reviewLoadMessage("ayahLoadStatus", "");
        renderAyahList(); updateGenHint(); updateSuraProgress(); updateReviewReadiness();
        if (W.mode) setCurrent(0);
        drawWave();
      } catch (e) {
        if (!active()) return;
        R.dataLoading = false; updateReviewReadiness();
        reviewLoadMessage("ayahLoadStatus", t("Не удалось загрузить аяты. ") + translateMessage(e.message) + " ", true);
        $("suraProgressText").textContent = t("Разметка не загружена");
      }
    })(),
    (async () => {
      try {
        await loadWaveStream(surah, active);
      } catch (e) {
        if (active()) {
          R.waveLoading = false; updateReviewReadiness();
          reviewLoadMessage("waveLoadStatus", t("Не удалось загрузить аудиоволну. ") + translateMessage(e.message) + " ", true);
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
    hint.textContent = generated ? t("Синие линии — начала слов · тяните метку для правки")
      : t("Нажмите «Рассчитать слова всей суры», чтобы сначала получить границы аятов.");
    hint.classList.toggle("warn", !generated);
    syncAlignmentControls();
    return;
  }
  if (generated) {
    hint.textContent = t("Клик по волне — перемотка · тяните метку — двигайте начало аята");
    hint.classList.remove("warn");
    syncAlignmentControls();
  } else {
    hint.textContent = t("Сура без тайм-кодов: нажмите номер аята, чтобы задать начало, или запустите разметку.");
    hint.classList.add("warn");
    syncAlignmentControls();
  }
}
let generatingSurah = null, generatingProject = null, generationReady = false, alignmentCancelling = false;
function syncAlignmentControls() {
  const running = GEN.running || generatingSurah != null;
  const ready = GEN.running ? GEN.ready : generationReady;
  for (const id of ["regenSuraBtn", "genBtn"]) {
    const btn = $(id);
    btn.classList.toggle("alignment-stop", running);
    if (running) {
      const label = alignmentCancelling ? t("Остановка…") : t("Остановить разметку");
      btn.innerHTML = studioIcon("stop") + ` ${label}`;
      btn.setAttribute("aria-label", label);
      btn.title = t("Остановить текущий расчёт");
      btn.disabled = alignmentCancelling || !ready;
    } else if (id === "regenSuraBtn") {
      const label = W.mode ? t("Разметить слова") : t("Начать разметку");
      btn.innerHTML = studioIcon("play") + ` ${label}`;
      btn.setAttribute("aria-label", label);
      btn.title = t("Рассчитать разметку суры выбранным режимом");
      btn.disabled = !R.dataReady;
    }
  }
  $("navigationProgress").classList.toggle("running", running);
  $("workspaceSummary").classList.toggle("hidden", !running && !document.body.classList.contains("review-open"));
  $("reviewGeneration").classList.toggle("hidden", ALIGN.startMs == null);
}
function updateSuraProgress(state = R.surah != null && R.surah === generatingSurah && currentProject === generatingProject ? "running" : "") {
  syncAlignmentControls();
  const wordProgress = R.wordProgress || {};
  const total = W.mode ? (wordProgress.total || 0) : R.ayahs.length;
  const marked = W.mode ? (wordProgress.marked || 0) : R.ayahs.filter((a) => a.time != null).length;
  const edited = W.mode ? (wordProgress.edited || 0) : R.ayahs.filter((a) => a.status === "edited").length;
  const verified = W.mode ? (wordProgress.verified || 0) : R.ayahs.filter((a) => a.status === "verified").length;
  $("suraProgressText").textContent = `${W.mode ? t("Слов размечено") : t("Размечено")} ${marked}/${total}`;
  $("suraProgressMeta").textContent = t`Проверено ${verified}/${total} · Вручную ${edited} · Осталось ${Math.max(0, total - marked)}`;
  $("suraProgress").className = "sura-progress summary-counts" + (!document.body.classList.contains("review-open") ? " hidden" : "")
    + (state === "running" ? " running" : "")
    + (total > 0 && marked === total ? " done" : "");
}

let generationPreparing = false;
function chooseGenerationWrite(anchor) {
  const dialog = $("generationChoiceDialog");
  dialog.returnValue = "";
  return new Promise(resolve => {
    dialog.addEventListener("close", () => {
      anchor.focus({ preventScroll: true });
      resolve(dialog.returnValue === "overwrite" ? true : dialog.returnValue === "preserve" ? false : null);
    }, { once: true });
    $("preserveGenerationBtn").onclick = () => dialog.close("preserve");
    $("overwriteGenerationBtn").onclick = () => dialog.close("overwrite");
    dialog.onclick = e => {
      const bounds = dialog.getBoundingClientRect();
      if (e.target === dialog && (e.clientX < bounds.left || e.clientX > bounds.right
        || e.clientY < bounds.top || e.clientY > bounds.bottom)) dialog.close();
    };
    dialog.showModal();
    const origin = anchor.getBoundingClientRect(), bounds = dialog.getBoundingClientRect();
    dialog.style.left = Math.max(8, Math.min(origin.left, innerWidth - bounds.width - 8)) + "px";
    const top = origin.bottom + 8 + bounds.height <= innerHeight - 8
      ? origin.bottom + 8 : origin.top - bounds.height - 8;
    dialog.style.top = Math.max(8, Math.min(top, innerHeight - bounds.height - 8)) + "px";
    $("preserveGenerationBtn").focus({ preventScroll: true });
  });
}
async function prepareGeneration(payload, active) {
  if (generationPreparing || GEN.running || generatingSurah != null) return null;
  generationPreparing = true;
  const anchor = $(document.body.classList.contains("review-open") ? "regenSuraBtn" : "genBtn");
  try {
    const preview = await api.post("/api/generate/preview", payload);
    if (!active() || GEN.running || generatingSurah != null) return null;
    const hasManual = preview.manual_ayahs + preview.manual_words > 0;
    const overwrite = hasManual ? await chooseGenerationWrite(anchor) : false;
    if (overwrite === null || !active() || GEN.running || generatingSurah != null) return null;
    return { hasManual, payload: { ...payload, suras: preview.suras,
      only_missing: false, overwrite_manual: overwrite } };
  } finally { generationPreparing = false; }
}

// regenerate just this sura with the chosen per-sura mode
async function regenerateSura() {
  if (R.surah == null) return;
  if (GEN.running || generatingSurah != null) return cancelGeneration();
  const surah = R.surah, mode = selectedSuraMode();
  const project = currentProject;
  const active = () => R.surah === surah && currentProject === project;
  let prepared;
  try {
    prepared = await prepareGeneration({ suras: [surah], mode, db_path: project.db_path },
      () => active() && selectedSuraMode() === mode);
  } catch (e) { if (active()) setStatus(t("Ошибка: ") + translateMessage(e.message)); return; }
  if (!prepared) return;
  const btn = $("regenSuraBtn"), st = $("suraGenStatus");
  generatingSurah = surah;
  generatingProject = project;
  generationReady = false; alignmentCancelling = false;
  startAlignmentProgress();
  syncAlignmentControls();
  resetLiveLog("suraLiveLog");
  const t0 = Date.now();
  const tick = () => {
    if (!active()) return;
    if (alignmentCancelling) st.textContent = t("Остановка расчёта…");
    updateSuraProgress("running");
  };
  st.className = "sura-gen-status running"; tick();
  const timer = setInterval(tick, 1000);
  const stop = (cls, text) => {
    clearInterval(timer); generatingSurah = null; generatingProject = null; generationReady = false; alignmentCancelling = false; btn.disabled = false;
    GEN.es?.close(); GEN.es = null;
    stopAlignmentProgress();
    if (active()) { st.className = "sura-gen-status " + cls; st.textContent = text; }
    updateSuraProgress();
  };
  try {
    const job = await api.post("/api/generate", prepared.payload);
    GEN.jobId = job.job_id;
  } catch (e) { stop("err", t("ошибка: ") + translateMessage(e.message)); return; }
  generationReady = true; syncAlignmentControls();
  let diagnosticsError = null;
  const receive = async (d) => {
    handleAlignmentEvent(d);
    if (d.type === "sura_log" && d.surah === surah && active()) {
      appendLiveLog("suraLiveLog", d.line);
      if (!alignmentCancelling) st.textContent = d.line;
    }
    if (d.type === "sura" && d.surah === surah && active()) {
      await reloadSura();
      if (!active()) return;
      st.textContent = d.words_written != null ? t`слов записано ${d.words_written} · требуют проверки ${d.word_failures || 0} аят.` : t`записано ${d.written ?? 0} аят.`;
      if (d.diagnostics_error) {
        diagnosticsError = d.diagnostics_error;
        appendLiveLog("suraLiveLog", t`Сура ${d.surah}: разметка сохранена; ошибка диагностики: ${d.diagnostics_error}`);
      } else if (d.log_paths && d.log_paths.log) appendLiveLog("suraLiveLog", t`Лог сохранён: ${d.log_paths.log}`);
    }
    if (d.type === "done" || d.type === "cancelled") {
      if (active()) await reloadSura();
      refreshSuras();
      const incomplete = mode === "words" && R.wordProgress && !R.wordProgress.complete;
      diagnosticsError ||= d.diagnostics_errors?.find(item => item.surah === surah)?.error;
      const diagnosticWarning = d.type === "cancelled"
        ? t("Разметка остановлена. Обработанные суры сохранены; диагностика — не полностью. Проверьте доступ к папке проекта.")
        : t("Разметка сохранена. Не удалось сохранить диагностику. Проверьте доступ к папке проекта.");
      stop(diagnosticsError || d.type === "cancelled" || incomplete ? "warn" : "ok", diagnosticsError ? diagnosticWarning : d.type === "cancelled" ? t("Разметка остановлена") : incomplete ? t("Расчёт завершён: некоторые слова требуют проверки.")
        : t`✓ готово за ${Math.floor((Date.now() - t0) / 1000)}с`);
    } else if (d.type === "error") { stop("err", t("ошибка: ") + translateMessage(d.message)); }
  };
  GEN.es = watchGeneration({ jobId: GEN.jobId, onEvent: receive,
    getStatus: () => api.get("/api/generate/status"),
    onSnapshot: applyGenerationSnapshot,
    onReconnect: () => { if (active()) st.textContent = t("Соединение прервано. Подключаюсь к расчёту…"); },
    onError: error => stop("err", translateMessage(error.message)) });
}

function captureReviewContext() {
  const surah = R.surah, request = R.request, project = currentProject;
  const dbPath = project.db_path;
  return { surah, dbPath, active: () => R.request === request && R.surah === surah
    && currentProject === project && currentProject.db_path === dbPath };
}

async function draftPausesSura() {
  if (R.surah == null) return;
  const context = captureReviewContext();
  const btn = $("draftPausesBtn"), st = $("suraGenStatus");
  if (!confirm(t("Разметить суру по паузам? Ручные и проверенные метки не будут затронуты."))) return;
  btn.disabled = true;
  st.className = "sura-gen-status running";
  st.textContent = t("разметка по паузам…");
  try {
    const data = await api.post(`/api/sura/${context.surah}/draft-pauses`, { db_path: context.dbPath });
    if (!context.active()) return;
    await reloadSura();
    if (!context.active()) return;
    refreshSuras();
    st.className = "sura-gen-status ok";
    st.textContent = t`паузы: записано ${data.written}, найдено ${data.pauses}`;
  } catch (e) {
    if (!context.active()) return;
    st.className = "sura-gen-status err";
    st.textContent = t("ошибка: ") + translateMessage(e.message);
  } finally {
    if (context.active()) btn.disabled = false;
  }
}

async function correctDraftSura() {
  if (R.surah == null) return;
  const context = captureReviewContext();
  const generated = R.ayahs.some((a) => a.time != null);
  if (!generated) { setStatus(t("Сначала поставьте метки или разметьте суру по паузам.")); return; }
  const mode = selectedSuraMode();
  if (!confirm(t`Запустить корректировку auto-тайм-кодов в режиме «${modeLabel(mode)}»? Ручные и проверенные метки не будут затронуты.`)) return;
  const btn = $("correctDraftBtn"), st = $("suraGenStatus");
  btn.disabled = true;
  const t0 = Date.now();
  const tick = () => { if (context.active()) st.textContent = t`корректировка… ${Math.floor((Date.now() - t0) / 1000)}с`; };
  st.className = "sura-gen-status running"; tick();
  const timer = setInterval(tick, 1000);
  try {
    const data = await api.post(`/api/sura/${context.surah}/correct-draft`, { mode, db_path: context.dbPath });
    clearInterval(timer);
    if (!context.active()) return;
    await reloadSura();
    if (!context.active()) return;
    refreshSuras();
    st.className = "sura-gen-status ok";
    st.textContent = t`корректировка: записано ${data.written} аят.`;
  } catch (e) {
    clearInterval(timer);
    if (!context.active()) return;
    st.className = "sura-gen-status err";
    st.textContent = t("ошибка: ") + translateMessage(e.message);
  } finally {
    if (context.active()) btn.disabled = false;
  }
}

// Перепроверка: аудит готовых тайм-кодов суры (read-only, БД не меняет).
async function recheckSura() {
  if (R.surah == null) return;
  const context = captureReviewContext();
  const marks = R.ayahs.map(ayah => ayah.time).join(",");
  const generated = R.ayahs.some((a) => a.time != null);
  if (!generated) { setStatus(t("Сура не рассчитана — нечего перепроверять.")); return; }
  const btn = $("recheckBtn"), st = $("suraGenStatus");
  btn.disabled = true;
  st.className = "sura-gen-status running";
  const t0 = Date.now();
  const tick = () => { if (context.active()) st.textContent = t`перепроверка… ${Math.floor((Date.now() - t0) / 1000)}с`; };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const data = await api.get(`/api/sura/${context.surah}/recheck`);
    if (!context.active() || R.ayahs.map(ayah => ayah.time).join(",") !== marks) return;
    R.recheck = {};
    for (const r of data.results) R.recheck[r.ayah] = r;
    const s = data.summary || {};
    clearInterval(timer);
    btn.disabled = false;
    const suspect = s.suspect || 0;
    const unchecked = s.unchecked || 0;
    st.className = "sura-gen-status " + (suspect || unchecked ? "warn" : "ok");
    st.textContent = suspect
      ? t`⚠ перепроверка: ${suspect} подозрит. из ${s.checked || 0}`
      : unchecked ? t`Перепроверка: ${unchecked} аятов не удалось проверить`
      : t`✓ перепроверка: все ${s.checked || 0} в норме`;
    renderAyahList();
    drawWave();
  } catch (e) {
    if (!context.active()) return;
    st.className = "sura-gen-status err";
    st.textContent = t("ошибка: ") + translateMessage(e.message);
  } finally {
    clearInterval(timer);
    if (context.active()) btn.disabled = false;
  }
}

// re-fetch this sura's timings after a recompute, keep audio position
async function reloadSura() {
  if (editBusy) { R.reloadPending = true; return; }
  const context = captureReviewContext();
  const request = R.reloadRequest = (R.reloadRequest || 0) + 1;
  const active = () => context.active() && request === R.reloadRequest;
  try {
    const sura = await api.get(`/api/sura/${context.surah}`);
    if (!active()) { if (context.active() && editBusy) R.reloadPending = true; return; }
    R.ayahs = sura.ayahs; R.wordProgress = sura.word_progress;
  } catch { return; }
  if (W.mode) await loadWords();
  if (!active()) return;
  R.recheck = null;  // times changed → stale verification
  renderAyahList();
  updateGenHint();
  updateSuraProgress();
  drawWave();
  const cur = R.current; R.current = -1;
  setCurrent(currentAyahByTime((R.audio ? R.audio.currentTime : 0) * 1000));
  if (R.current < 0 && cur >= 0) setCurrent(cur);
}

function timingConfidence(item, suspect = false) {
  const unknown = { level: "unknown", label: t("Начало: нет оценки"),
    title: t("Положение начала отдельно не проверено. Процент распознавания слова не измеряет точность времени.") };
  if (item.time == null || item.stale || item.status === "edited") return unknown;
  if (suspect) {
    return { level: "low", label: t("Начало: сомнительное"),
      title: t("Повторная проверка отметила эту границу как сомнительную. Проверьте начало на слух.") };
  }
  if (item.status === "verified") {
    return { level: "high", label: t("Начало: проверено"),
      title: t("Метка отмечена как проверенная при ревью.") };
  }
  const evidence = item.onset_evidence;
  const boundary = evidence && evidence.boundary;
  const labels = { high: t("Начало: подтверждено"), medium: t("Начало: предварительное"),
    low: t("Начало: сомнительное"), unknown: unknown.label };
  const reasons = {
    "local-phrase-and-pause": t("Начальная фраза и положение границы проверены локально по речи и паузе."),
    "alignment-only": t("Выравнивание согласовано с текстом, но положение начала независимо не проверено."),
    "pause-edge-only": t("Метка близка к краю речевой паузы, но принадлежность начала аяту отдельно не подтверждена."),
    "pause-rechecked": t("Положение скорректировано по паузе; начальная фраза отдельно не подтверждена."),
    "weak-onset": t("В выбранной позиции нет достаточной акустической поддержки начала."),
    "late-onset-in-pause": t("Метка попала в неоднозначную часть длинной паузы."),
    "missing-local-proof": t("Для перенесённой метки нет сохранённого локального подтверждения."),
    "insufficient-timing-evidence": unknown.title,
  };
  const level = boundary && Object.hasOwn(labels, boundary.level) ? boundary.level : "unknown";
  let title = boundary ? reasons[boundary.reason] || unknown.title : unknown.title;
  title += " " + t("Это оценка оснований для границы, а не измеренная погрешность времени. Проверьте начало на слух.");
  const format = value => new Intl.NumberFormat(uiLanguage, { style: "percent", maximumFractionDigits: 1 }).format(value);
  const valid = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  if (evidence && valid(evidence.first_sound)) title += t` Сигнал первого звука: ${format(evidence.first_sound)}.`;
  if (valid(item.confidence)) title += t` Оценка распознавания: ${format(item.confidence)}.`;
  return { level, label: labels[level], title };
}

function applyTimingStatus(row, item, suspect = false) {
  const status = item.time == null ? "missing"
    : item.status === "verified" && !item.stale && !suspect ? "verified"
    : item.status == null || ["auto", "edited", "verified"].includes(item.status) ? "uncertain" : "neutral";
  const label = { missing: t("Без разметки"), verified: t("Проверено"), uncertain: t("Требует проверки"), neutral: t("Нет данных о состоянии") }[status];
  row.dataset.timingStatus = status;
  const confidence = row.querySelector(".timing-confidence");
  if (confidence) {
    const estimate = timingConfidence(item, suspect);
    confidence.dataset.level = estimate.level;
    confidence.textContent = estimate.label;
    confidence.title = estimate.title;
  }
  const number = row.querySelector(".a-num");
  if (number) {
    number.dataset.baseLabel ||= number.getAttribute("aria-label");
    number.setAttribute("aria-label", `${number.dataset.baseLabel} · ${label}`);
    number.title = label;
  }
}

function updateAyahStatusControls(row, item) {
  row?.querySelectorAll("[data-ayah-status]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.ayahStatus === (item.status || "auto")));
    button.disabled = editBusy || (item.time == null && button.dataset.ayahStatus !== "auto");
  });
}

function renderAyahList() {
  if (W.mode) { renderWords(); return; }
  const ul = $("ayahList");
  ul.classList.remove("word-mode"); ul.setAttribute("aria-label", t("Аяты суры"));
  ul.innerHTML = "";
  R.ayahs.forEach((a, i) => {
    const li = document.createElement("li");
    li.dataset.i = i;
    li.classList.toggle("missing", a.time == null);
    li.classList.toggle("active", i === R.current);
    li.innerHTML = t`
      <button class="a-num" aria-label="Изменить начало аята ${a.ayah}" aria-haspopup="dialog" aria-controls="ayahTime${a.ayah}" aria-pressed="${i === R.current}">${a.ayah}</button>
      <dialog id="ayahTime${a.ayah}" class="ayah-time-popover" aria-labelledby="ayahTimeTitle${a.ayah}">
      <div class="ayah-time-heading"><strong id="ayahTimeTitle${a.ayah}">Аят ${a.ayah} · начало</strong><button type="button" class="ghost close-time" aria-label="Закрыть редактор времени">${studioIcon("close")}</button></div>
      <div class="time-editor">
        <input type="text" class="a-time-input" inputmode="decimal" aria-label="Начало аята ${a.ayah}, минуты:секунды.миллисекунды" placeholder="Нет метки" value="${exactTime(a.time)}" />
        <div class="time-arrows"><button type="button" aria-label="Увеличить время аята ${a.ayah}" title="Позже (Shift — шаг ×10)">${studioIcon("up")}</button><button type="button" aria-label="Уменьшить время аята ${a.ayah}" title="Раньше (Shift — шаг ×10)">${studioIcon("down")}</button></div>
      </div>
      <div class="ayah-status-segments" role="group" aria-label="Статус аята ${a.ayah}">
        <button type="button" data-ayah-status="auto" title="Сбросить проверку, сохранив время"><span class="status-swatch" aria-hidden="true"></span><span>Не помечен</span></button>
        <button type="button" data-ayah-status="edited"><span class="status-swatch" aria-hidden="true"></span><span>Нужно ревью</span></button>
        <button type="button" data-ayah-status="verified"><span class="status-swatch" aria-hidden="true"></span><span>Всё хорошо</span></button>
      </div>
      <p class="ayah-time-help">Шаг ${timeStep} мс · сохраняется автоматически</p>
      <p id="ayahTimeError${a.ayah}" class="ayah-time-error hidden" role="status"></p>
      </dialog>
      <span class="a-text" dir="rtl" lang="ar">${escapeHtml(a.text || "")}</span>
      <span class="timing-confidence"></span>`;
    applyTimingStatus(li, a, ["shift", "insufficient_evidence"].includes(R.recheck?.[a.ayah]?.verdict));
    const input = li.querySelector("input");
    updateAyahStatusControls(li, a);
    li.querySelectorAll("[data-ayah-status]").forEach(button => {
      button.onclick = () => setAyahStatus(i, button.dataset.ayahStatus);
    });
    const dialog = li.querySelector("dialog"), number = li.querySelector(".a-num");
    number.onclick = e => {
      e.stopPropagation();
      if (editBusy || reviewLoading) return;
      if (R.audio) R.audio.pause();
      setCurrent(i);
      dialog.showModal();
      const anchor = number.getBoundingClientRect(), bounds = dialog.getBoundingClientRect();
      dialog.style.left = Math.max(8, Math.min(anchor.left, innerWidth - bounds.width - 8)) + "px";
      dialog.style.top = Math.max(8, Math.min(anchor.bottom + 8, innerHeight - bounds.height - 8)) + "px";
      input.focus(); input.select();
    };
    li.querySelector(".close-time").onclick = () => dialog.close();
    dialog.onclick = e => {
      e.stopPropagation();
      const bounds = dialog.getBoundingClientRect();
      if (e.target === dialog && (e.clientX < bounds.left || e.clientX > bounds.right || e.clientY < bounds.top || e.clientY > bounds.bottom)) dialog.close();
    };
    dialog.onclose = () => number.focus({ preventScroll: true });
    input.onfocus = () => setCurrent(i);
    const commitInput = async () => {
      if (input.value === exactTime(a.time)) return;
      const ms = parseTime(input.value);
      if (!Number.isSafeInteger(ms)) {
        input.setAttribute("aria-invalid", "true"); input.setAttribute("aria-describedby", `ayahTimeError${a.ayah}`);
        setStatus(t("Введите время в формате 00:10.320 (минуты:секунды.миллисекунды)."));
        li.querySelector(".ayah-time-error").textContent = t("Введите время в формате 00:10.320.");
        li.querySelector(".ayah-time-error").classList.remove("hidden");
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
      if (e.target.closest("dialog")) return;
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
  const context = captureReviewContext();
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
    st.textContent = t("ищу паузы…");
    try {
      const data = await api.get(`/api/sura/${context.surah}/pause-pipeline`);
      if (!context.active()) return;
      R.pauseDecisions = data.decisions || [];
      const accepted = R.pauseDecisions.filter((d) => d.accepted).length;
      const weak = R.pauseDecisions.filter((d) => d.accepted && d.reason === "weak-dsp").length;
      const rejected = R.pauseDecisions.length - accepted;
      st.className = "sura-gen-status ok";
      st.textContent = t`паузы: ${accepted} принято, weak ${weak}, откл. ${rejected}`;
    } catch (e) {
      if (!context.active()) return;
      R.pausesVisible = false;
      st.className = "sura-gen-status err";
      st.textContent = t("ошибка пауз: ") + translateMessage(e.message);
    } finally {
      if (context.active()) { btn.disabled = false; updatePauseOverlayUi(); }
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
  if (!d.accepted) return t("отклонена");
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
    $("pauseInspectorSummary").textContent = t("загрузка…");
    rows.innerHTML = "";
    return;
  }
  const accepted = R.pauseDecisions.filter((d) => d.accepted).length;
  const weak = R.pauseDecisions.filter((d) => d.accepted && d.reason === "weak-dsp").length;
  const rejected = R.pauseDecisions.length - accepted;
  const [viewStart, viewEnd] = visibleWaveMs();
  const visible = R.pauseDecisions.filter((d) => d.end_ms >= viewStart && d.start_ms <= viewEnd);
  $("pauseInspectorSummary").textContent =
    t`всего ${R.pauseDecisions.length} · принято ${accepted} · weak ${weak} · откл. ${rejected} · видно ${visible.length}`;
  rows.innerHTML = "";
  for (const d of visible.slice(0, 120)) {
    const tr = document.createElement("tr");
    tr.className = "pause-row " + pauseDecisionClass(d);
    tr.title = t`центр ${fmt(d.center_ms)} · ${d.reason}`;
    tr.innerHTML = t`
      <td>${fmt(d.start_ms)}–${fmt(d.end_ms)}</td>
      <td>${Math.round(d.depth_ms)}мс</td>
      <td><span class="pause-pill ${pauseDecisionClass(d)}">${pauseDecisionLabel(d)}</span></td>
      <td>${Number(d.rms_ratio).toFixed(3)}</td>
      <td>${Number(d.flatness).toFixed(3)}</td>
      <td>${Number(d.zcr).toFixed(3)}</td>
      <td>${d.vad_ok == null ? "—" : (d.vad_ok ? "ok" : t("нет"))}</td>`;
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
    tr.innerHTML = t`<td colspan="7">ещё ${visible.length - 120} в этом диапазоне — увеличьте масштаб</td>`;
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
  const w = c.clientWidth, h = c.clientHeight, waveTop = 26, waveBottom = h - 24, mid = (waveTop + waveBottom) / 2;
  ctx.clearRect(0, 0, w, h);
  const n = R.peaks.length;
  const left = wrap.scrollLeft, right = left + w, virtualW = content.clientWidth;
  const word = W.mode ? W.data?.words[W.selected] : null;
  if (word && word.time != null && !word.stale) {
    const next = W.data.words[W.selected + 1];
    const endMs = next && next.ayah === word.ayah ? next.time : word.end_ms;
    if (endMs != null) {
      const start = msToVirtualX(word.time) - left, end = msToVirtualX(endMs) - left;
      ctx.fillStyle = "rgba(0,122,255,.10)"; ctx.fillRect(start, 0, end - start, h);
    }
  } else if (!W.mode && R.current >= 0 && R.ayahs[R.current].time != null) {
    const start = msToVirtualX(R.ayahs[R.current].time) - left;
    const next = R.ayahs[R.current + 1];
    const endMs = next ? next.time : R.duration;
    // A missing next boundary is not a known ayah interval.
    if (endMs != null && endMs >= R.ayahs[R.current].time) {
      const end = msToVirtualX(endMs) - left;
      ctx.fillStyle = "rgba(0,122,255,.10)";
      ctx.fillRect(start, 0, end - start, h);
    }
  }
  const colors = getComputedStyle(document.documentElement);
  const color = token => colors.getPropertyValue(token).trim();
  ctx.strokeStyle = color("--wave-ink"); ctx.lineWidth = 1; ctx.beginPath();
  const step = R.duration ? R.waveBucketMs / R.duration * virtualW : 0;
  const i0 = step ? Math.max(0, Math.floor(left / step) - 2) : 0;
  const i1 = step ? Math.min(n - 1, Math.ceil(right / step) + 2) : -1;
  const now = performance.now();
  for (let i = i0; i <= i1; i++) {
    const x = i * step - left, amp = Math.min(1, R.peaks[i] / R.waveMaximum) * Math.max(0, (waveBottom - waveTop) / 2 - 2);
    ctx.globalAlpha = reducedMotion.matches ? 1 : Math.min(1, (now - R.waveBorn[i]) / 180);
    ctx.beginPath(); ctx.moveTo(x, mid - amp); ctx.lineTo(x, mid + amp); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  if (R.duration && w) {
    const visibleMs = R.duration / R.zoom;
    const rough = visibleMs / Math.max(2, w / 90);
    const magnitude = 10 ** Math.floor(Math.log10(Math.max(1, rough)));
    const tickMs = [1, 2, 5, 10].find(value => value * magnitude >= rough) * magnitude;
    const startMs = left / virtualW * R.duration;
    ctx.strokeStyle = color("--wave-marker"); ctx.fillStyle = color("--muted"); ctx.font = "11px -apple-system, sans-serif";
    for (let ms = Math.ceil(startMs / tickMs) * tickMs; ms <= Math.min(R.duration, startMs + visibleMs); ms += tickMs) {
      const x = msToVirtualX(ms) - left;
      ctx.beginPath(); ctx.moveTo(x, h - 20); ctx.lineTo(x, h - 14); ctx.stroke();
      ctx.fillText(exactTime(ms).slice(0, -4), Math.min(Math.max(0, x + 3), w - 40), h - 2);
    }
  }
  drawPauseOverlay(ctx, left, right, h);
  R.ayahs.forEach((a, i) => {
    if (W.mode || a.time == null || !R.duration) return;
    const vx = msToVirtualX(a.time);
    if (vx < left - 1 || vx > right + 1) return;
    const x = vx - left;
    const rc = R.recheck ? R.recheck[a.ayah] : null;
    const suspect = rc && ["shift", "insufficient_evidence"].includes(rc.verdict);
    ctx.strokeStyle = i === R.current ? color("--blue") : color("--wave-marker");
    ctx.lineWidth = (suspect || i === R.current) ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x, 5); ctx.lineTo(x, waveBottom); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath(); ctx.arc(x, 5, 2.5, 0, Math.PI * 2); ctx.fill();
    // Other boundaries keep a compact number; the selected interval has a label.
    if (i !== R.current) {
      ctx.font = "10px sans-serif";
      ctx.fillText(rc?.verdict === "shift" ? a.ayah + "→" + rc.match_ayah : a.ayah, x + 5, 15);
    }
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
  const selected = !W.mode ? R.ayahs[R.current] : null;
  if (selected?.time != null && R.duration) {
    const next = R.ayahs[R.current + 1];
    const endMs = next ? next.time : R.duration;
    const start = msToVirtualX(selected.time) - left;
    if (endMs != null && endMs > selected.time) {
      const end = msToVirtualX(endMs) - left;
      const from = Math.max(0, start), to = Math.min(w, end);
      if (to > from) {
        ctx.strokeStyle = color("--blue"); ctx.fillStyle = color("--blue"); ctx.lineWidth = 1;
        // Real endpoint lines remain clipped at the visible viewport edges.
        for (const x of [start, end]) if (x >= 0 && x <= w) {
          ctx.beginPath(); ctx.moveTo(x, 17); ctx.lineTo(x, waveBottom); ctx.stroke();
        }
        ctx.beginPath(); ctx.moveTo(from, 20); ctx.lineTo(to, 20); ctx.stroke();
        for (const [x, direction] of [[start, 1], [end, -1]]) if (x >= 0 && x <= w) {
          ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x + direction * 5, 17);
          ctx.lineTo(x + direction * 5, 23); ctx.closePath(); ctx.fill();
        }
        const label = t`Аят ${selected.ayah}`;
        ctx.font = "11px -apple-system, sans-serif";
        const labelWidth = ctx.measureText(label).width;
        // Avoid overflowing labels at very short intervals or high zoom.
        const labelX = Math.max(0, Math.min(w - labelWidth, (from + to - labelWidth) / 2));
        ctx.fillStyle = color("--text"); ctx.fillText(label, labelX, 13);
      }
    } else if (start >= 0 && start <= w) {
      ctx.fillStyle = color("--muted"); ctx.font = "10px sans-serif";
      ctx.fillText(String(selected.ayah), start + 5, 15);
    }
  }
  if (R.audio && R.duration) {
    const playMs = R.audio.currentTime * 1000;
    const px = msToVirtualX(playMs) - left;
    ctx.strokeStyle = color("--blue"); ctx.lineWidth = 2;
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
function togglePlay() {
  if (!R.audio || !R.audioReady || !R.dataReady) return;
  if (W.mode) { R.audio.paused ? playWord() : R.audio.pause(); return; }
  if (!R.audio.paused) { R.audio.pause(); return; }
  if (R.playbackMode !== "continuous" && R.current >= 0
      && R.audio.currentTime * 1000 >= ayahEndMs(R.current) - 2) {
    seekAyah(R.current, true);
  } else R.audio.play();
}
function scheduleAyahStop(reanchor = false) {
  clearTimeout(R.playbackTimer);
  if (W.mode || R.playbackMode === "continuous" || !R.audio || R.audio.paused || !R.dataReady) return;
  if (reanchor || R.playbackAyah < 0) R.playbackAyah = currentAyahByTime(R.audio.currentTime * 1000);
  const i = R.playbackAyah;
  if (i < 0 || R.ayahs[i]?.time == null || ayahEndMs(i) <= R.ayahs[i].time) return;
  const delay = Math.max(0, (ayahEndMs(i) - R.audio.currentTime * 1000) / R.audio.playbackRate);
  R.playbackTimer = setTimeout(() => {
    if (!finishAyahPlayback()) scheduleAyahStop();
  }, delay);
}
function finishAyahPlayback() {
  const i = R.playbackAyah;
  if (W.mode || R.playbackMode === "continuous" || !R.audio || R.audio.paused || i < 0) return false;
  const end = ayahEndMs(i);
  if (R.audio.currentTime * 1000 < end) return false;
  clearTimeout(R.playbackTimer);
  if (R.loop) seekAyah(i, true);
  else {
    R.audio.pause();
    R.audio.currentTime = Math.max(R.ayahs[i].time, end - 2) / 1000;
    setCurrent(i);
    drawWave();
  }
  return true;
}
function syncPlayerSettings() {
  const inputs = document.querySelectorAll('[name="playbackMode"]');
  for (const input of inputs) {
    input.checked = input.value === R.playbackMode;
    input.disabled = W.mode && input.value === "continuous";
    if (input.value === "single") input.nextSibling.textContent = W.mode ? t(" Остановиться после слова") : t(" Остановиться после аята");
    if (input.value === "repeat") input.nextSibling.textContent = W.mode ? t(" Повторять слово") : t(" Повторять аят");
  }
  $("playerSettingsHint").textContent = W.mode ? t("В режиме слов воспроизводится только выбранное слово.") : t("Границы берутся из текущей разметки.");
  // Word playback is always a clip, including the default continuous ayah mode.
  if (W.mode && R.playbackMode === "continuous") document.querySelector('[name="playbackMode"][value="single"]').checked = true;
  $("playerSettingsBtn").classList.toggle("active", R.playbackMode !== "continuous");
}
function setPlayerSettingsOpen(open) {
  $("playerSettingsMenu").classList.toggle("hidden", !open);
  $("playerSettingsBtn").setAttribute("aria-expanded", String(open));
  if (open) {
    setRateMenuOpen(false);
    $("playerSettingsMenu").querySelector("input:checked").focus();
  }
}
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
  R.playbackAyah = i;
  scheduleAyahStop();
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
  drawWave();
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
  await setAyahStatus(i, next);
}

async function setAyahStatus(i, status) {
  if (editBusy || reviewLoading || !R.ayahs[i]) return false;
  const a = R.ayahs[i];
  const row = document.querySelector(`.ayah-list li[data-i="${i}"]`);
  const error = row?.querySelector(".ayah-time-error");
  const focused = row?.contains(document.activeElement) ? document.activeElement : null;
  editBusy = true;
  ++R.reloadRequest;
  row?.querySelectorAll("input, [data-ayah-status], .time-arrows button").forEach(el => el.disabled = true);
  try {
    await api.post("/api/verify", { surah: R.surah, ayah: a.ayah, status, db_path: currentProject.db_path });
    a.status = status;
    if (R.recheck) delete R.recheck[a.ayah];
    if (row) applyTimingStatus(row, a);
    updateAyahStatusControls(row, a);
    updateSuraProgress(); syncSidebarStatus(); drawWave();
    if (error) { error.textContent = ""; error.classList.add("hidden"); }
    setStatus("");
    return true;
  } catch (e) {
    const message = t("Статус не сохранён. ") + translateMessage(e.message);
    if (error) { error.textContent = message; error.classList.remove("hidden"); }
    setStatus(message);
    return false;
  } finally {
    editBusy = false;
    row?.querySelectorAll("input, .time-arrows button").forEach(el => el.disabled = false);
    updateAyahStatusControls(row, a);
    if (focused && document.activeElement === document.body) focused.focus({ preventScroll: true });
    if (R.reloadPending) { R.reloadPending = false; void reloadSura(); }
  }
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
  const error = row?.querySelector(".ayah-time-error");
  const showError = message => { if (error) { error.textContent = message; error.classList.toggle("hidden", !message); } };
  const value = Math.round(ms);
  if (!Number.isSafeInteger(value) || value < 0 || value > R.duration) {
    if (input) { input.value = exactTime(oldTime); input.setAttribute("aria-invalid", "true"); input.setAttribute("aria-describedby", error?.id || "workspaceStatus"); }
    setStatus(t("Время не сохранено. Укажите начало в пределах аудиозаписи."));
    showError(t("Укажите начало в пределах аудиозаписи."));
    saveState(t("Не сохранено"), "error"); return false;
  }
  editBusy = true;
  ++R.reloadRequest;
  if (row) row.querySelectorAll("input, [data-ayah-status], .time-arrows button").forEach(el => el.disabled = true);
  saveState(t("Сохранение…"));
  try {
    await api.post("/api/timing", { surah, ayah: a.ayah, time: value, db_path: currentProject.db_path });
    a.time = value; a.status = "edited"; a.confidence = null;
    updateAyahStatusControls(row, a);
    if (input) { input.value = exactTime(value); input.removeAttribute("aria-invalid"); input.removeAttribute("aria-describedby"); }
    if (R.recheck) delete R.recheck[a.ayah];
    if (row) { row.classList.remove("missing"); applyTimingStatus(row, a); }
    updateGenHint(); updateSuraProgress(); syncSidebarStatus(); drawWave();
    if (W.mode) loadWords(); else resetWords();
    saveState(t("Сохранено"), "saved");
    setStatus("");
    showError("");
    return true;
  } catch (e) {
    a.time = oldTime;
    if (input) { input.value = exactTime(oldTime); input.setAttribute("aria-invalid", "true"); input.setAttribute("aria-describedby", error?.id || "workspaceStatus"); }
    saveState(t("Не сохранено"), "error");
    setStatus(t("Время не сохранено. ") + translateMessage(e.message));
    showError(t("Время не сохранено. ") + translateMessage(e.message));
    drawWave(); return false;
  } finally {
    editBusy = false;
    if (row) row.querySelectorAll("input, .time-arrows button").forEach(el => el.disabled = false);
    updateAyahStatusControls(row, a);
    if (focused && document.activeElement === document.body) focused.focus({ preventScroll: true });
    if (R.reloadPending) { R.reloadPending = false; void reloadSura(); }
  }
}
async function exportProject() {
  let dst = "reciter_export.db";
  if (window.pywebview) { const p = await window.pywebview.api.save_db("reciter_export.db"); if (!p) return; dst = p; }
  else { const p = prompt(t("Путь для экспорта .db:"), dst); if (!p) return; dst = p; }
  try { const r = await api.post("/api/export", { dst }); setStatus(t("Экспортировано: ") + r.path); }
  catch (e) { alert(t("Экспорт невозможен: ") + translateMessage(e.message)); }
}

// ===========================================================================
//  wiring
// ===========================================================================
function showView(name) {
  setRateMenuOpen(false);
  setPlayerSettingsOpen(false);
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
  document.body.classList.toggle("review-open", name === "review");
  $("genBtn").classList.toggle("hidden", name === "review");
  $("regenSuraBtn").classList.toggle("hidden", name !== "review");
  $("dashboard").classList.toggle("hidden", name !== "dashboard");
  $("review").classList.toggle("hidden", name !== "review");
  document.querySelector(".workspace-dashboard-actions").classList.toggle("hidden", name !== "dashboard");
  document.querySelector(".workspace-review-actions").classList.toggle("hidden", name !== "review");
  $("recordingMenu").open = false;
  $("modeMenu").open = false;
  document.querySelector(".review-tools").open = false;
  document.title = name === "review" ? `${$("reviewTitle").textContent} — Ayah Studio` : "Ayah Studio";
  $("modeMenu").classList.toggle("hidden", name !== "review");
  $("dashboardModeMenu").classList.toggle("hidden", name === "review");
  $("suraProgress").classList.toggle("hidden", name !== "review");
  (name === "review" ? $("regenSuraBtn") : $("suraSearch")).focus();
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
    const name = document.createElement("strong"); name.textContent = currentProject.reciter || t("Текущий проект");
    const detail = document.createElement("small"); detail.textContent = t("Внешний проект");
    const copy = document.createElement("div"); copy.className = "project-picker-copy";
    copy.append(name, detail);
    current.append(copy, projectPickerCheck(), projectPickerEdit()); list.append(current);
  }
  for (const project of projectLibrary) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.projectId = project.id;
    button.setAttribute("aria-pressed", String(project.id === currentProject.project_id));
    button.disabled = select.disabled || editBusy || reviewLoading;
    const name = document.createElement("span"); name.textContent = project.reciter;
    const detail = document.createElement("small"); detail.textContent = recordingName(project);
    button.append(name, detail);
    button.onclick = () => {
      if (select.disabled || editBusy || reviewLoading) return;
      $("recordingMenu").open = false;
      $("recordingMenuToggle").focus();
      if (project.id === currentProject.project_id) return;
      select.value = project.id;
      select.dispatchEvent(new Event("change"));
    };
    if (project.id === currentProject.project_id) {
      const row = document.createElement("div"); row.className = "project-picker-row";
      row.append(button, projectPickerEdit()); list.append(row);
    } else list.append(button);
  }
  if (!list.children.length) {
    const empty = document.createElement("p"); empty.className = "project-picker-empty";
    empty.textContent = t("Пока нет проектов"); list.append(empty);
  }
}

function projectPickerCheck() {
  const check = document.createElement("span");
  check.className = "project-picker-check";
  check.innerHTML = studioIcon("check");
  check.setAttribute("role", "img");
  check.setAttribute("aria-label", t("Текущий проект"));
  return check;
}

function renderSettingsProjects() {
  const list = $("settingsProjectList");
  list.replaceChildren();
  const entries = [...projectLibrary];
  if (!currentProject.project_id && currentProject.db_path) entries.unshift({ ...currentProject });
  for (const project of entries) {
    const active = project.id ? project.id === currentProject.project_id : project.db_path === currentProject.db_path;
    const row = document.createElement("div"); row.className = "settings-project-row";
    const copy = document.createElement("div"); copy.className = "project-picker-copy";
    const name = document.createElement("strong"); name.textContent = project.reciter;
    const detail = document.createElement("small"); detail.textContent = project.recording;
    copy.append(name, detail);
    if (active) {
      const status = document.createElement("small"); status.className = "project-active-label";
      status.textContent = t("Открыт"); copy.append(status);
    }
    const actions = document.createElement("div"); actions.className = "settings-project-actions";
    if (!active) {
      const open = document.createElement("button"); open.type = "button"; open.className = "ghost";
      open.textContent = t("Открыть"); open.setAttribute("aria-label", t`Открыть проект ${project.reciter} · ${recordingName(project)}`);
      open.disabled = $("projectSelect").disabled || editBusy || reviewLoading;
      open.onclick = () => {
        $("projectSelect").value = project.id;
        $("projectSelect").dispatchEvent(new Event("change"));
      };
      actions.append(open);
    }
    const edit = document.createElement("button"); edit.type = "button"; edit.className = "ghost";
    edit.textContent = t("Изменить"); edit.setAttribute("aria-label", t`Изменить проект ${project.reciter} · ${recordingName(project)}`);
    edit.dataset.editProject = project.id || project.db_path;
    edit.disabled = $("projectSelect").disabled || editBusy || reviewLoading;
    edit.onclick = () => openProjectForm(project);
    actions.append(edit); row.append(copy, actions); list.append(row);
  }
  if (!entries.length) {
    const empty = document.createElement("p"); empty.className = "projects-empty settings-help";
    empty.textContent = t("Пока нет проектов. Нажмите «Добавить проект» и выберите папку с аудио.");
    list.append(empty);
  }
  $("addProjectBtn").disabled = $("projectSelect").disabled || editBusy || reviewLoading;
}

function projectPickerEdit() {
  const edit = document.createElement("button");
  edit.type = "button"; edit.className = "project-picker-edit icon-btn ghost";
  edit.title = t("Изменить имя проекта"); edit.setAttribute("aria-label", edit.title);
  edit.innerHTML = studioIcon("edit");
  edit.disabled = $("projectSelect").disabled || editBusy || reviewLoading;
  edit.onclick = () => {
    $("recordingMenu").open = false;
    openProjectForm({ ...currentProject, id: currentProject.project_id });
  };
  return edit;
}

async function refreshProjectLibrary() {
  projectLibrary = await api.get("/api/projects");
  renderProjectLibrary();
}

function renderProjectLibrary() {
  const select = $("projectSelect");
  select.replaceChildren();
  if (!currentProject.project_id) {
    select.add(new Option(currentProject.db_path ? t`${currentProject.reciter} · Внешний проект` : t("Создайте проект в настройках"), ""));
  }
  for (const p of projectLibrary) select.add(new Option(`${p.reciter} · ${p.recording}`, p.id));
  select.value = currentProject.project_id || "";
  const recordingLabel = currentProject.db_path ? t`Проект: ${currentProject.reciter || ""} · ${recordingName(currentProject)}` : t("Выбрать проект");
  $("recordingMenuToggle").innerHTML = studioIcon("recording") + '<span id="projectName"></span>';
  $("projectName").textContent = currentProject.db_path ? currentProject.reciter || recordingName(currentProject) : t("Выбрать проект");
  $("recordingMenuToggle").title = recordingLabel;
  $("recordingMenuToggle").setAttribute("aria-label", recordingLabel);
  $("workspaceExportBtn").disabled = !currentProject.db_path;
  renderProjectPicker();
  renderSettingsProjects();
  updateSuraProgress();
}

async function applyProject(p) {
  ++R.request;
  R.waveCancel?.();
  resetAlignmentProgress();
  if (R.audio) R.audio.pause();
  resetWords();
  currentProject = p;
  R.surah = null; R.ayahs = []; R.wordProgress = null; R.peaks = []; R.current = -1;
  $("audio").removeAttribute("src"); $("audio").load();
  $("audioDir").value = p.audio_dir || "";
  $("dbPath").value = p.db_path || "";
  setMode(p.mode);
  saveState(t("Сохранено"), "saved");
  showView("dashboard");
  await refreshProjectLibrary();
  await refreshSuras();
}

function applyAppearance(theme, arabicSize) {
  document.documentElement.dataset.theme = theme === "dark" ? "dark" : "light";
  document.documentElement.style.setProperty("--arabic-size", `${arabicSize || 30}px`);
  drawWave();
}

function applyLanguage(language) {
  setUiLanguage(language);
  // Repaint copy from existing state; leave playback, jobs and saved data intact.
  document.querySelectorAll("[data-dashboard-mode]").forEach(button => {
    const source = document.querySelector(`[data-review-mode="${button.dataset.dashboardMode}"]`);
    button.innerHTML = source.innerHTML;
  });
  document.querySelectorAll("[data-sura-filter]").forEach(button => {
    button.textContent = [...$("suraFilter").options].find(option => option.value === button.dataset.suraFilter).textContent;
  });
  for (const id of ["status", "workspaceStatus", "settingsMessage", "modelMessage", "genTitle", "genCur", "suraGenStatus", "librarySummary", "ayahLoadStatus", "audioLoadStatus", "waveLoadStatus", "suraLogEmpty"]) {
    const walker = document.createTreeWalker($(id), NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) walker.currentNode.data = translateMessage(walker.currentNode.data);
  }
  renderProjectLibrary();
  const mode = selectedMode();
  for (const id of ["modeMenuToggle", "dashboardModeToggle"]) {
    $(id).innerHTML = studioIcon(mode) + t`<span><span class="mode-caption">Режим:</span> ${escapeHtml(modeLabel(mode))}</span>`;
    $(id).title = t`Режим разметки: ${modeLabel(mode)}`;
    $(id).setAttribute("aria-label", $(id).title);
  }
  renderSuras(); updateGenControls(dashboardSuras); updateModeHint();
  syncAlignmentControls(); updateSuraProgress();
  if (R.dataReady) { updateReviewTitle(R.ayahs.length); renderAyahList(); updateGenHint(); }
  if (ALIGN.startMs != null) renderAlignmentProgress();
  if (R.pauseDecisions) renderPauseInspector();
  syncPlayerSettings(); updateRatePicker(); updateSuraFilterPicker();
  $("sidebarToggle").title = $("sidebarToggle").getAttribute("aria-expanded") === "true" ? t("Скрыть боковую панель") : t("Показать боковую панель");
  $("sidebarToggle").setAttribute("aria-label", $("sidebarToggle").title);
  for (const [id, word, ayah] of [["prevBtn", t("Предыдущее слово"), t("Предыдущий аят")], ["nextBtn", t("Следующее слово"), t("Следующий аят")]]) {
    $(id).title = W.mode ? word : ayah;
    $(id).setAttribute("aria-label", $(id).title);
  }
  $("playBtn").setAttribute("aria-label", R.audio?.paused !== false ? t("Воспроизвести") : t("Пауза"));
  modelSignature = "";
  if ($("settingsDialog").open && !$("modelSettings").classList.contains("hidden")) void refreshModels();
}

function settingsPanel(id) {
  document.querySelectorAll(".settings-panel").forEach(panel => panel.classList.toggle("hidden", panel.id !== id));
  document.querySelectorAll("[data-settings-panel]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.settingsPanel === id)));
  clearTimeout(modelPoll);
  if (id === "modelSettings") { modelSignature = ""; refreshModels(); }
}

function openSettings(panel = "interfaceSettings") {
  if (editBusy || reviewLoading || W.busy) { setStatus(t("Дождитесь завершения сохранения или загрузки, затем откройте настройки.")); return; }
  if (R.audio) R.audio.pause();
  const p = currentProject;
  renderSettingsProjects();
  $("settingsArabicSize").value = p.arabic_size || 30;
  $("settingsLanguage").value = p.language || uiLanguage;
  document.querySelectorAll('[name="settingsTheme"]').forEach(input => input.checked = input.value === (p.theme || "light"));
  $("settingsMessage").textContent = "";
  settingsPanel(panel);
  openDialog($("settingsDialog"));
  if (panel === "modelSettings") refreshModels();
}

let settingsSaveQueue = Promise.resolve();
const confirmedSettings = new WeakMap();
const settingRevisions = new Map();

function applySettingsChange(data) {
  Object.assign(currentProject, data);
  if ("language" in data) applyLanguage(data.language);
  if ("theme" in data || "arabic_size" in data) {
    applyAppearance(currentProject.theme, currentProject.arabic_size);
    document.querySelectorAll('[name="settingsTheme"]').forEach(input => input.checked = input.value === currentProject.theme);
    $("settingsArabicSize").value = currentProject.arabic_size || 30;
  }
  if ("time_step" in data) timeStep = data.time_step;
  if ("mode" in data) setMode(data.mode);
}

function saveSettings(data) {
  const project = currentProject;
  if (!confirmedSettings.has(project)) confirmedSettings.set(project, { ...project, time_step: timeStep });
  const confirmed = confirmedSettings.get(project);
  const revisions = {};
  for (const key of Object.keys(data)) {
    revisions[key] = (settingRevisions.get(key) || 0) + 1;
    settingRevisions.set(key, revisions[key]);
  }
  applySettingsChange(data);
  $("settingsMessage").textContent = "";
  settingsSaveQueue = settingsSaveQueue.then(async () => {
    try {
      await api.post("/api/settings", data);
      Object.assign(confirmed, data);
    } catch (e) {
      if (project !== currentProject) return;
      const rollback = {};
      for (const key of Object.keys(data)) {
        if (settingRevisions.get(key) === revisions[key]) rollback[key] = confirmed[key];
      }
      applySettingsChange(rollback);
      $("settingsMessage").textContent = t("Не удалось сохранить настройки: ") + translateMessage(e.message);
      setStatus(t("Не удалось сохранить настройки: ") + translateMessage(e.message));
      return;
    }
  });
  return settingsSaveQueue;
}

function positionWorkspaceMenu(menu) {
  const panel = menu.querySelector(".toolbar-popover, .review-actions");
  if (!menu.open || !panel) return;
  const anchor = menu.querySelector("summary").getBoundingClientRect();
  const margin = 8;
  panel.style.maxHeight = `${Math.max(0, window.innerHeight - margin * 2)}px`;
  const bounds = panel.getBoundingClientRect();
  if (window.innerWidth <= 600) {
    panel.style.left = "12px";
    panel.style.top = `${Math.max(margin, window.innerHeight - bounds.height - 12)}px`;
    return;
  }
  const left = Math.max(margin, Math.min(anchor.left, window.innerWidth - bounds.width - margin));
  const below = anchor.bottom + 6;
  const top = bounds.height <= window.innerHeight - below - margin ? below
    : Math.max(margin, Math.min(below, window.innerHeight - bounds.height - margin));
  panel.style.left = `${left}px`;
  panel.style.top = `${top}px`;
}

function setRateMenuOpen(open) {
  if (open) setPlayerSettingsOpen(false);
  const wasOpen = !$("rateMenu").classList.contains("hidden");
  $("rateMenu").classList.toggle("hidden", !open);
  $("rateBtn").setAttribute("aria-expanded", String(open));
  if (open && !wasOpen) {
    if (window.innerWidth > 600) animateMenu($("rateMenu"), $("rateBtn").getBoundingClientRect());
    $("rate").focus();
  }
  if (!open) menuMotions.get($("rateMenu"))?.cancel();
}
function updateRatePicker() {
  const value = $("rate").value;
  $("rateBtn").textContent = `${value}×`;
  $("rateBtn").setAttribute("aria-label", t`Скорость воспроизведения: ${value}×`);
  $("rateValue").textContent = `${value}×`;
  $("rate").setAttribute("aria-valuetext", `${value}×`);
  $("rateMenu").querySelectorAll("button").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.rate === value));
  });
}

function wireWorkspace() {
  for (const [id, icon] of [["playBtn", "play"], ["prevBtn", "skipPrevious"], ["nextBtn", "skipNext"], ["waveStartBtn", "previous"], ["waveEndBtn", "next"], ["sidebarToggle", "sidebar"], ["settingsBtn", "settings"], ["workspaceExportBtn", "export"], ["recordingMenuToggle", "recording"], ["modeMenuToggle", "mode"], ["reviewToolsToggle", "tools"], ["dashboardModeToggle", "mode"], ["filterMenuToggle", "filter"], ["helpBtn", "help"], ["createProjectBtn", "addProject"], ["regenBtn", "refresh"]]) $(id).innerHTML = studioIcon(icon);
  document.querySelectorAll("[data-review-mode]").forEach(source => {
    source.querySelector(".mode-option-icon").innerHTML = studioIcon(source.dataset.reviewMode);
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
  for (const menu of [$("recordingMenu"), $("modeMenu"), $("dashboardModeMenu"), $("filterMenu"), document.querySelector(".review-tools")]) {
    menu.querySelector("summary").addEventListener("click", e => {
      e.preventDefault();
      if (menu.querySelector("summary").getAttribute("aria-disabled") === "true") return;
      menu.open = !menu.open;
      if (menu === $("recordingMenu") && menu.open) renderProjectPicker();
    });
    menu.addEventListener("toggle", () => {
      if (menu.open) {
        for (const other of [$("recordingMenu"), $("modeMenu"), $("dashboardModeMenu"), $("filterMenu"), document.querySelector(".review-tools")]) {
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
  $("createProjectBtn").onclick = () => openProjectForm();
  $("addProjectBtn").onclick = () => openProjectForm();
  for (const id of ["closeProjectBtn", "cancelProjectBtn"]) $(id).onclick = () => closeDialog($("projectDialog"));
  $("projectDialog").addEventListener("cancel", e => { if (projectFormBusy) e.preventDefault(); });
  $("closeSettingsBtn").onclick = () => closeDialog($("settingsDialog"));
  $("settingsDialog").addEventListener("close", () => {
    clearTimeout(modelPoll);
    $("modelList").querySelectorAll('input[type="password"]').forEach(input => input.value = "");
  });
  document.querySelectorAll('[name="settingsTheme"]').forEach(input => input.onchange = () => saveSettings({ theme: input.value }));
  $("settingsArabicSize").onchange = e => saveSettings({ arabic_size: Number(e.target.value) });
  $("settingsLanguage").onchange = e => saveSettings({ language: e.target.value });
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
    renderSettingsProjects();
    $("recordingMenu").open = false;
    try { await applyProject(await api.post("/api/projects/activate", { id })); setStatus(""); }
    catch (error) {
      e.target.value = currentProject.project_id || "";
      $("settingsMessage").textContent = t("Не удалось открыть проект: ") + translateMessage(error.message);
      setStatus(t("Не удалось открыть проект. ") + translateMessage(error.message));
    }
    finally {
      e.target.disabled = false; renderProjectPicker(); renderSettingsProjects();
      if ($("settingsDialog").open) $("addProjectBtn").focus();
    }
  };
  $("verifySelectedBtn").onclick = () => { if (R.current >= 0) toggleVerify(R.current); };
}

function wire() {
  wireWorkspace();
  $("playerSettingsBtn").innerHTML = studioIcon("settings");
  $("playerSettingsBtn").onclick = () => setPlayerSettingsOpen($("playerSettingsMenu").classList.contains("hidden"));
  for (const input of document.querySelectorAll('[name="playbackMode"]')) input.onchange = () => {
    R.playbackMode = input.value;
    R.loop = input.value === "repeat";
    syncPlayerSettings();
    scheduleAyahStop(true);
  };
  for (const value of [0.5, 0.8, 1, 1.5, 2]) {
    const button = document.createElement("button");
    button.type = "button"; button.dataset.rate = String(value); button.textContent = `${value}×`;
    button.onclick = () => {
      $("rate").value = value;
      $("rate").dispatchEvent(new Event("input"));
    };
    $("ratePresets").append(button);
  }
  updateRatePicker();
  $("rateBtn").onclick = () => setRateMenuOpen($("rateMenu").classList.contains("hidden"));
  document.addEventListener("click", e => {
    if (!document.querySelector(".player-rate").contains(e.target)) setRateMenuOpen(false);
    if (!document.querySelector(".player-settings").contains(e.target)) setPlayerSettingsOpen(false);
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !$("playerSettingsMenu").classList.contains("hidden")) {
      setPlayerSettingsOpen(false); $("playerSettingsBtn").focus();
    }
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
  $("grid").onkeydown = e => {
    if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) return;
    const buttons = [...$("grid").querySelectorAll("button")];
    const index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    e.preventDefault();
    const next = e.key === "Home" ? 0 : e.key === "End" ? buttons.length - 1
      : Math.max(0, Math.min(buttons.length - 1, index + (e.key === "ArrowDown" ? 1 : -1)));
    buttons[next].focus();
  };
  $("suraSearch").onkeydown = e => {
    if (e.key === "Enter" || e.key === "ArrowDown") {
      const first = $("grid").querySelector('button[aria-disabled="false"]');
      if (first) { e.preventDefault(); e.key === "Enter" ? first.click() : first.focus(); }
    }
  };
  $("suraFilter").onchange = () => { updateSuraFilterPicker(); renderSuras(); };
  $("genBtn").onclick = () => GEN.running || generatingSurah != null ? cancelGeneration() : runGeneration(true);
  $("batchGenerateBtn").onclick = () => runGeneration(true);
  $("batchRegenerateBtn").onclick = () => runGeneration(false);
  $("regenBtn").onclick = () => runGeneration(false);
  $("cancelBtn").onclick = cancelGeneration;
  $("regenSuraBtn").onclick = regenerateSura;
  $("draftPausesBtn").onclick = draftPausesSura;
  $("correctDraftBtn").onclick = correctDraftSura;
  $("recheckBtn").onclick = recheckSura;
  $("compareRunsBtn").onclick = openComparison;
  $("closeComparisonBtn").onclick = () => { stopComparison(); closeDialog($("comparisonDialog")); };
  for (const id of ["comparisonLeft", "comparisonRight"]) $(id).onchange = loadComparisonPair;
  $("comparisonDifferences").onchange = renderComparison;
  $("comparisonAudio").ontimeupdate = () => {
    const audio = $("comparisonAudio");
    if (COMPARISON.stopMs != null && audio.currentTime * 1000 >= COMPARISON.stopMs) audio.pause();
  };
  $("comparisonDialog").addEventListener("cancel", stopComparison);
  $("comparisonDialog").addEventListener("close", stopComparison);
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
    catch (e) { setMode(currentProject.mode); setStatus(t("Не удалось сохранить режим. Попробуйте снова: ") + translateMessage(e.message)); }
  };
  $("suraMode").onchange = async () => {
    const previousMode = selectedMode();
    const mode = selectedSuraMode();
    setMode(mode);
    try { await api.post("/api/settings", { mode }); currentProject.mode = mode; }
    catch (e) { setMode(previousMode); setStatus(t("Не удалось сохранить режим: ") + translateMessage(e.message)); }
  };
  $("sidebarToggle").onclick = () => setSidebarVisible($("sidebarToggle").getAttribute("aria-expanded") !== "true");
  $("playBtn").onclick = togglePlay;
  $("prevBtn").onclick = seekPrevUnit;
  $("nextBtn").onclick = seekNextUnit;
  $("rate").oninput = $("rate").onchange = () => {
    if (R.audio) {
      R.audio.defaultPlaybackRate = parseFloat($("rate").value);
      R.audio.playbackRate = parseFloat($("rate").value);
    }
    updateRatePicker();
  };
  for (const btn of document.querySelectorAll("[data-copy-log]")) {
    btn.addEventListener("click", () => copyLiveLog(btn.dataset.copyLog));
    updateLogCopyButton(btn.dataset.copyLog);
  }

  $("alignWordsBtn").onclick = alignWords;
  $("wordTimeUnit").onchange = () => { W.unit = $("wordTimeUnit").value; renderWords(); };

  const audio = $("audio");
  audio.addEventListener("play", () => { $("playBtn").innerHTML = studioIcon("pause"); $("playBtn").setAttribute("aria-label", t("Пауза")); });
  audio.addEventListener("pause", () => { $("playBtn").innerHTML = studioIcon("play"); $("playBtn").setAttribute("aria-label", t("Воспроизвести")); });
  for (const event of ["play", "pause", "ratechange", "seeked"]) audio.addEventListener(event, scheduleWordStop);
  for (const event of ["play", "seeking", "seeked"]) audio.addEventListener(event, () => scheduleAyahStop(true));
  for (const event of ["pause", "ratechange"]) audio.addEventListener(event, () => scheduleAyahStop());
  audio.addEventListener("ended", () => {
    if (!W.mode && R.loop && R.playbackAyah >= 0) seekAyah(R.playbackAyah, true);
  });
  audio.addEventListener("ended", () => { if (W.mode && W.stop) finishWord(); });
  audio.addEventListener("timeupdate", () => {
    const ms = audio.currentTime * 1000;
    $("clock").textContent = `${exactTime(ms)} / ${exactTime(R.duration)}`;
    if (W.stop && ms >= W.stop.end) { finishWord(); drawWave(); return; }
    if (W.mode) { drawWave(); return; }
    if (finishAyahPlayback()) { drawWave(); return; }
    const i = currentAyahByTime(ms);
    if (i >= 0 && i !== R.current) setCurrent(i);
    if (R.zoom > 1) {
      const wrap = $("waveScroll"), x = msToVirtualX(ms);
      if (x < wrap.scrollLeft || x > wrap.scrollLeft + wrap.clientWidth) {
        wrap.scrollLeft = Math.max(0, x - wrap.clientWidth / 2);
      }
    }
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
    if ($("generationChoiceDialog").open || $("comparisonDialog").open || $("suraLiveLogBox").open || $("settingsDialog").open || $("projectDialog").open || $("review").classList.contains("hidden") || e.target.closest("input, select, textarea, button, summary, [contenteditable]")) return;
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
    setUiLanguage(p.language);
    timeStep = p.time_step || 10;
    applyAppearance(p.theme, p.arabic_size);
    await refreshProjectLibrary();
    saveState(p.db_path ? t("Сохранено") : t("Нет открытого проекта"), p.db_path ? "saved" : "");
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
  try { await refreshSuras(); } catch (e) { setStatus(t("Не удалось загрузить суры: ") + translateMessage(e.message)); }
  try { await restoreGeneration(); } catch { /* project remains usable if status is temporarily unavailable */ }
})();
