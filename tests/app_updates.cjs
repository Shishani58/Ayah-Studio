const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map();
const el = id => {
  if (!elements.has(id)) elements.set(id, {dataset: {}, removeAttribute() {}});
  return elements.get(id);
};
const events = {};
let state = {phase: 'idle', current: '1.0.0'};
let checks = 0, installs = 0, timer;
const c = {
  document: {getElementById: el, addEventListener: (name, fn) => { events[name] = fn; }},
  window: {addEventListener() {}, pywebview: {api: {
    update_status: async () => state,
    check_update: async () => { checks++; state = {...state, phase: 'downloading', percent: 25, version: '1.1.0'}; return state; },
    install_update: async () => { installs++; return {error: 'busy'}; },
  }}},
  t: (s, ...values) => s.replace(/\$\{(\d+)\}/g, (_, i) => values[i]),
  translateMessage: s => s, settingsSaveQueue: Promise.resolve(),
  editBusy: false, reviewLoading: false, W: {busy: false},
  clearTimeout() {timer = null;}, setTimeout(fn) {timer = fn; return 1;},
};
vm.createContext(c);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../surah_cutter/webgui/static/updates.js'), 'utf8'), c);
events.DOMContentLoaded();
(async () => {
  await c.refreshAppUpdate();
  assert.equal(el('checkAppUpdateBtn').disabled, false);
  await el('checkAppUpdateBtn').onclick();
  assert.equal(checks, 1);
  assert.equal(el('checkAppUpdateBtn').disabled, true);
  assert.equal(el('appUpdateProgress').value, 25);
  assert.equal(el('appUpdateProgress').hidden, false);
  assert(timer);
  state = {...state, phase: 'ready'};
  await timer();
  assert.equal(el('installAppUpdateBtn').hidden, false);
  assert.equal(el('appUpdateProgress').hidden, true);
  c.editBusy = true;
  await el('installAppUpdateBtn').onclick();
  assert.equal(installs, 0);
  c.editBusy = false;
  await el('installAppUpdateBtn').onclick();
  assert.equal(installs, 1);
  assert.equal(el('appUpdateMessage').textContent, 'busy');
  assert.equal(el('installAppUpdateBtn').disabled, false);
  c.window.pywebview.api.install_update = async () => ({ok: true});
  await el('installAppUpdateBtn').onclick();
  assert.equal(el('installAppUpdateBtn').disabled, true);
  state = {...state, phase: 'error', error: 'network'};
  await c.refreshAppUpdate();
  assert.equal(el('checkAppUpdateBtn').disabled, false);
  assert.equal(el('installAppUpdateBtn').hidden, true);
  delete c.window.pywebview;
  await c.refreshAppUpdate();
  assert.equal(el('checkAppUpdateBtn').disabled, true);
  console.log('Update UI: download, ready, busy, retry, install and browser states passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
