/* Model downloads and selection inside the existing settings dialog. */
let modelPoll = null, modelSignature = "", modelBusy = false;
const modelSize = bytes => bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} ГБ` : `${Math.ceil(bytes / 1e6)} МБ`;

function modelElement(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text; node.className = className;
  return node;
}

function modelButton(icon, label) {
  const button = modelElement("button", "", "ghost icon-btn model-action");
  button.type = "button";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.innerHTML = studioIcon(icon);
  return button;
}

async function modelAction(id, action, body = {}) {
  modelBusy = true;
  $("modelMessage").textContent = action === "select" ? "Проверка и подключение модели…" : "";
  $("modelList").querySelectorAll("button").forEach(b => b.disabled = true);
  try {
    const result = await api.post(`/api/models/${id}/${action}`, body);
    if (action === "select") {
      currentProject.mode = result.mode;
      currentProject.alignment_model = id;
      setMode(result.mode); $("settingsMode").value = result.mode;
      $("modelMessage").textContent = "Модель выбрана. Следующий расчёт использует её.";
    }
  } catch (e) { $("modelMessage").textContent = e.message; }
  finally { modelBusy = false; modelSignature = ""; await refreshModels(); }
}

function renderModel(m, downloading) {
  const card = modelElement("article", "", "model-card");
  card.dataset.active = String(m.active); card.dataset.model = m.id;
  const info = modelElement("div", "", "model-info");
  const names = { "quran-phonetics": "Quran Phonetics · INT8", "quran-phonetics-fp32": "Quran Phonetics · FP32", "arabic-ctc": "Arabic Wav2Vec2 · INT8", muaalem: "Muaalem v3.2" };
  const status = m.active ? "Используется" : m.installed || m.ready ? "Скачано" : "Доступно для скачивания";
  info.append(modelElement("h4", names[m.id] || m.name));
  info.append(modelElement("span", `${modelSize(m.runtime_size || m.size)} · ${status}`, "model-meta"));
  card.append(info);
  const actions = modelElement("div", "", "model-actions");
  if (m.ready && !m.active) {
    const select = modelButton("check", `Использовать ${m.name}`);
    select.onclick = () => modelAction(m.id, "select"); actions.append(select);
  }
  if (!m.local_only && !m.installed && !m.ready && m.job.state !== "downloading") {
    const label = m.job.state === "error" ? `Повторить загрузку ${m.name}` : `Скачать ${m.name}`;
    const download = modelButton("download", label);
    download.disabled = downloading;
    download.onclick = () => modelAction(m.id, "download");
    actions.append(download);
  }
  if (m.job.state === "downloading") {
    const cancel = modelButton("close", `Отменить загрузку ${m.name}`);
    cancel.onclick = () => modelAction(m.id, "cancel"); actions.append(cancel);
    const progress = document.createElement("progress"); progress.max = m.job.total; progress.value = m.job.received;
    progress.setAttribute("aria-label", `Загрузка ${m.name}`);
    card.append(progress, modelElement("span", "", "model-meta model-progress"));
  }
  if (m.installed && !m.active && !m.local_only && m.job.state !== "downloading") {
    const remove = modelButton("trash", `Удалить скачанные файлы ${m.name}`);
    remove.onclick = () => modelAction(m.id, "remove"); actions.append(remove);
  }
  if (m.job.error) card.append(modelElement("p", m.job.error, "model-error"));
  card.append(actions); return card;
}

async function refreshModels() {
  clearTimeout(modelPoll);
  if (!$("settingsDialog").open || $("modelSettings").classList.contains("hidden")) return;
  if (modelBusy) { modelPoll = setTimeout(refreshModels, 1000); return; }
  let keepPolling = false;
  try {
    const data = await api.get("/api/models");
    keepPolling = data.models.some(m => m.job.state === "downloading");
    const signature = JSON.stringify(data.models.map(m => [m.id, m.active, m.ready, m.installed, m.job.state, m.job.error]));
    if (signature !== modelSignature) {
      modelSignature = signature;
      $("modelList").replaceChildren(...data.models.map(m => renderModel(m, data.models.some(x => x.job.state === "downloading"))));
    }
    for (const m of data.models) {
      if (m.job.state !== "downloading") continue;
      const card = [...$("modelList").children].find(c => c.dataset.model === m.id);
      if (!card) continue;
      card.querySelector("progress").value = m.job.received;
      card.querySelector(".model-progress").textContent = `${Math.floor(m.job.received * 100 / Math.max(1, m.job.total))}% · ${modelSize(m.job.received)} из ${modelSize(m.job.total)}`;
    }
  } catch (e) { $("modelMessage").textContent = "Не удалось загрузить каталог: " + e.message; }
  if (keepPolling) modelPoll = setTimeout(refreshModels, 1000);
}
