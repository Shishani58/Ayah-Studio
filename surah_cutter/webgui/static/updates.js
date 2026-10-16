"use strict";
let appUpdate = { phase: "unsupported" };
let appUpdatePoll;
let appUpdateRequest = false;

function renderAppUpdate() {
  const el = id => document.getElementById(id);
  const { phase, version, current, percent, error } = appUpdate;
  const busy = ["checking", "downloading", "preparing", "installing"].includes(phase);
  const messages = {
    unsupported: t("Автообновление доступно в установленном приложении для macOS и Windows."),
    idle: t("Проверьте, доступна ли новая версия."),
    checking: t("Проверяем обновления…"),
    current: t("Установлена последняя версия."),
    downloading: t("Скачиваем ${0}: ${1}%", version, percent),
    preparing: t("Проверяем и подготавливаем обновление…"),
    ready: t("Версия ${0} готова к установке.", version),
    installing: t("Закрываем приложение для установки…"),
    error: t("Не удалось подготовить обновление. Попробуйте снова.") + (error ? " " + error : ""),
  };
  el("appVersion").textContent = current ? t("Текущая версия: ${0}", current) : "";
  el("appUpdateMessage").textContent = messages[phase] || "";
  el("appUpdateMessage").dataset.kind = phase === "error" ? "error" : "status";
  el("checkAppUpdateBtn").disabled = busy || appUpdateRequest || phase === "ready" || phase === "unsupported";
  el("installAppUpdateBtn").hidden = !["ready", "installing"].includes(phase);
  el("installAppUpdateBtn").disabled = phase === "installing" || appUpdateRequest;
  el("appUpdateReadyHelp").hidden = phase !== "ready";
  el("appUpdateProgress").hidden = !["downloading", "preparing"].includes(phase);
  if (phase === "downloading") el("appUpdateProgress").value = percent;
  else el("appUpdateProgress").removeAttribute("value");
}

async function refreshAppUpdate() {
  clearTimeout(appUpdatePoll);
  const bridge = window.pywebview?.api;
  try {
    appUpdate = bridge?.update_status ? await bridge.update_status() : { phase: "unsupported" };
  } catch (error) {
    appUpdate = { ...appUpdate, phase: "error", error: error.message };
  }
  renderAppUpdate();
  if (["checking", "downloading", "preparing"].includes(appUpdate.phase))
    appUpdatePoll = setTimeout(refreshAppUpdate, 500);
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("checkAppUpdateBtn").onclick = async () => {
    if (appUpdateRequest) return;
    appUpdateRequest = true;
    renderAppUpdate();
    try {
      appUpdate = await window.pywebview.api.check_update();
      await refreshAppUpdate();
    } catch (error) {
      appUpdate = { ...appUpdate, phase: "error", error: error.message };
    } finally { appUpdateRequest = false; renderAppUpdate(); }
  };
  document.getElementById("installAppUpdateBtn").onclick = async () => {
    if (appUpdateRequest) return;
    appUpdateRequest = true;
    renderAppUpdate();
    try {
      await settingsSaveQueue;
      if (editBusy || reviewLoading || W.busy) throw new Error(t("Дождитесь завершения сохранения или загрузки, затем откройте настройки."));
      const result = await window.pywebview.api.install_update();
      if (result.error) throw new Error(translateMessage(result.error));
      appUpdate.phase = "installing";
      renderAppUpdate();
    } catch (error) {
      document.getElementById("appUpdateMessage").textContent = error.message;
      document.getElementById("appUpdateMessage").dataset.kind = "error";
    } finally {
      appUpdateRequest = false;
      // Preserve installation errors until the next user action.
      document.getElementById("installAppUpdateBtn").disabled = appUpdate.phase === "installing";
    }
  };
});
window.addEventListener("pywebviewready", refreshAppUpdate);
