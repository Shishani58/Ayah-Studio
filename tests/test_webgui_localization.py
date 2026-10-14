"""Language initialization and persistence must not change project data."""

import subprocess
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from surah_cutter.webgui import server, settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "settings.json")
    clear = settings.system_language.cache_clear
    clear()
    yield
    clear()


def endpoint(path):
    return next(route.endpoint for route in server.app.routes if route.path == path)


@pytest.mark.parametrize("locale,expected", [
    ("ru_RU.UTF-8", "ru"), ("ru-UA", "ru"), ("en_GB", "en"),
    ("ar_SA.UTF-8", "en"), ("C", "en"),
])
def test_linux_locale(locale, expected, monkeypatch):
    monkeypatch.setattr(settings.sys, "platform", "linux")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.setenv("LANG", locale)
    assert settings.system_language() == expected


def test_message_locale_precedes_lang(monkeypatch):
    monkeypatch.setattr(settings.sys, "platform", "linux")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.setenv("LC_MESSAGES", "ru_RU")
    monkeypatch.setenv("LANG", "en_US")
    assert settings.system_language() == "ru"


def test_macos_uses_first_preferred_language(monkeypatch):
    monkeypatch.setattr(settings.sys, "platform", "darwin")
    monkeypatch.setattr(settings.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout='(\n    "ru-RU",\n    "en-US"\n)'))
    monkeypatch.setattr(settings.locale, "getlocale", lambda: ("en_US", "UTF-8"))
    assert settings.system_language() == "ru"


def test_windows_uses_display_language(monkeypatch):
    import ctypes

    def get_locale(language_id, buffer, size, flags):
        assert language_id == 0x419
        buffer.value = "ru-RU"
        return len(buffer.value) + 1

    monkeypatch.setattr(settings.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        GetUserDefaultUILanguage=lambda: 0x419, LCIDToLocaleName=get_locale)), raising=False)
    assert settings.system_language() == "ru"


def test_detection_failure_falls_back(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("defaults", 2)

    monkeypatch.setattr(settings.sys, "platform", "darwin")
    monkeypatch.setattr(settings.subprocess, "run", fail)
    monkeypatch.setattr(settings.locale, "getlocale", lambda: ("en_US", "UTF-8"))
    assert settings.system_language() == "en"


def test_first_launch_is_persisted_and_manual_choice_wins(monkeypatch):
    monkeypatch.setattr(settings, "system_language", lambda: "ru")
    assert endpoint("/api/project")()["language"] == "ru"
    assert settings.load()["language"] == "ru"
    endpoint("/api/settings")({"language": "en"})
    assert endpoint("/api/project")()["language"] == "en"
    assert 'lang="en"' in endpoint("/")().body.decode()
    assert endpoint("/index.html")().body == endpoint("/")().body
    assert settings.load()["language"] == "en"


@pytest.mark.parametrize("language", ["ar", "EN", "", None, [], {}])
def test_invalid_language_does_not_overwrite_settings(language):
    settings.update(language="ru", theme="dark")
    before = settings.SETTINGS_PATH.read_bytes()
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/settings")({"language": language, "theme": "light"})
    assert exc.value.status_code == 400
    assert settings.SETTINGS_PATH.read_bytes() == before


def test_switch_preserves_all_other_settings():
    data = {"language": "ru", "theme": "dark", "db_path": "/project.db",
            "external_projects": {"/project.db": {"reciter": "Чтец", "recording": "Запись"}}}
    settings.save(data)
    endpoint("/api/settings")({"language": "en"})
    assert settings.load() == {**data, "language": "en"}


def test_save_failure_preserves_confirmed_language(monkeypatch):
    settings.update(language="ru")

    def fail(*args, **kwargs):
        raise OSError("read only")

    monkeypatch.setattr(settings, "save", fail)
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/settings")({"language": "en"})
    assert exc.value.status_code == 500
    assert settings.load()["language"] == "ru"


def test_frontend_messages_preserve_data_and_cover_static_ui():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to validate the shipped JavaScript catalog")
    static = Path(server.__file__).parent / "static"

    class Copy(HTMLParser):
        messages = []

        def handle_data(self, data):
            if data.strip() and data.strip() != "Русский":
                self.messages.append(data)

        def handle_starttag(self, tag, attrs):
            self.messages.extend(value for key, value in attrs
                                 if key in ("title", "aria-label", "placeholder") and value)

    parser = Copy()
    parser.feed((static / "index.html").read_text())
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const messages = JSON.parse(fs.readFileSync(0, 'utf8'));
const document = {
  documentElement: {lang:'en', removeAttribute() {}},
  createTreeWalker: () => ({nextNode: () => false}), querySelectorAll: () => [],
  getElementById: () => ({value:''})
};
const context = {document, NodeFilter:{SHOW_TEXT:4}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const run = code => vm.runInContext(code, context);
for (const source of messages) {
  if (!/[А-Яа-яЁё]/.test(source)) continue;
  assert(!/[А-Яа-яЁё]/.test(run(`t(${JSON.stringify(source)})`)), source);
}
assert.equal(run('t`Сура ${114} · аят ${2}/${6}`'), 'Surah 114 · ayah 2/6');
assert.equal(run('t`<p title="Статус слова ${3}">Аят ${5}</p>`'),
             '<p title="Word 3 status">Ayah 5</p>');
const arabic = 'قُلۡ أَعُوذُ بِرَبِّ ٱلنَّاسِ';
context.arabic = arabic;
assert.equal(run('t`<button lang="ar" aria-label="Прослушать ${arabic}">${arabic}</button>`'),
             `<button lang="ar" aria-label="Listen to ${arabic}">${arabic}</button>`);
assert.equal(run('t`Проект: ${"Сохранено"} · ${"Запись 1"}`'), 'Project: Сохранено · Запись 1');
assert.equal(run('t`Аят ${"${99}"}`'), 'Ayah ${99}');
assert.equal(run('translateMessage("Проект изменился. Откройте настройки заново.")'),
             'Project changed. Reopen Settings.');
run('setUiLanguage("ru")');
assert.equal(run('t`Сура ${114} · аят ${2}/${6}`'), 'Сура 114 · аят 2/6');
assert.equal(run('translateMessage("Surah 114 · ayah 2/6")'), 'Сура 114 · аят 2/6');
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
assert.equal(run('SURA_NAMES.length'), 115);
assert.equal(run('SURA_NAMES_EN.length'), 115);
"""
    subprocess.run([node, "-e", script, str(static / "i18n.js"), str(static / "sura-names.js")],
                   input=json.dumps(parser.messages), text=True, check=True, timeout=10)
