// Execute the shipped functions with controllable network/DOM boundaries.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '../surah_cutter/webgui/static');
const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
function extract(name) {
  const start = source.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert(start >= 0, name);
  return source.slice(start, source.indexOf('\n}', start) + 2);
}
const deferred = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
};
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness(names) {
  const elements = {};
  const context = {
    R: { request: 1, reloadRequest: 0, surah: 112, ayahs: [{ ayah: 1, time: 1000 }], current: 0 },
    currentProject: { db_path: 'one.db' }, editBusy: false, reviewLoading: false,
    W: { mode: false }, GEN: { epoch: 0, running: false }, generatingSurah: null,
    $: id => elements[id] ||= {}, api: {},
    document: { querySelector: () => null, activeElement: null,
      body: {classList: {contains: () => false}} },
    innerWidth: 1200, innerHeight: 800,
    setInterval: () => 1, clearInterval() {}, Date,
    uiLanguage: 'ru',
    t: (v, ...values) => typeof v === 'string' ? v : v.reduce((s, p, i) => s + p + (values[i] ?? ''), ''),
  };
  for (const name of ['setStatus', 'renderAyahList', 'drawWave', 'updateGenHint',
    'updateSuraProgress', 'setCurrent', 'saveState', 'updateAyahStatusControls',
    'syncSidebarStatus', 'resetWords', 'updatePauseOverlayUi']) context[name] = () => {};
  context.currentAyahByTime = () => 0;
  vm.createContext(context);
  vm.runInContext(names.map(extract).join('\n'), context);
  return context;
}
async function reviewRaces() {
  for (const mutate of [c => ++c.R.request, c => c.currentProject = { db_path: 'two.db' },
    c => c.R.ayahs[0].time = 2000]) {
    const c = harness(['captureReviewContext', 'recheckSura']);
    const reply = deferred(); c.api.get = () => reply.promise;
    const pending = c.recheckSura(); mutate(c);
    reply.resolve({ results: [{ ayah: 1, verdict: 'shift' }], summary: { suspect: 1 } });
    await pending; assert.equal(c.R.recheck, undefined);
  }
  for (const manual of ['commitMarker', 'setAyahStatus']) {
    const c = harness(['captureReviewContext', 'reloadSura', manual]);
    c.R.duration = 10000;
    const reply = deferred(); c.api.get = () => reply.promise;
    c.api.post = async () => {};
    const reload = c.reloadSura();
    await (manual === 'commitMarker' ? c.commitMarker(0, 2000) : c.setAyahStatus(0, 'verified'));
    reply.resolve({ ayahs: [{ ayah: 1, time: 1000, status: 'auto' }] }); await reload;
    assert.equal(c.R.ayahs[0][manual === 'commitMarker' ? 'time' : 'status'],
      manual === 'commitMarker' ? 2000 : 'verified');
  }
  const c = harness(['captureReviewContext', 'reloadSura']);
  c.editBusy = true; c.api.get = () => assert.fail('reload during write');
  await c.reloadSura(); assert.equal(c.R.reloadPending, true);
  for (const change of [c => ++c.GEN.epoch, c => c.currentProject = { db_path: 'two.db' }]) {
    const c = harness(['restoreGeneration']);
    c.startGenUI = () => assert.fail('stale startup restored');
    const reply = deferred(); c.api.get = () => reply.promise;
    const pending = c.restoreGeneration(); change(c);
    reply.resolve({ job_id: 'old', running: true }); await pending;
  }
  const p = harness(['captureReviewContext', 'togglePauseOverlay']);
  const reply = deferred(); p.api.get = () => reply.promise;
  const pending = p.togglePauseOverlay(); ++p.R.request;
  reply.resolve({ decisions: [{ accepted: true }] }); await pending;
  assert.equal(p.R.pauseDecisions, undefined);
}
async function generationRaces() {
  const context = { URLSearchParams, EventSource: class {
    close() { this.closed = true; }
  } };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root, 'generation.js'), 'utf8'), context);
  const log = [], gate = deferred();
  const stream = context.watchGeneration({ jobId: 'one', getStatus: async () => ({}),
    onError: error => { throw error; }, onEvent: async e => {
      if (e.type === 'sura') await gate.promise; log.push(e.type);
    } });
  const message = (type, id) => ({ data: JSON.stringify({ type }), lastEventId: String(id) });
  stream.onmessage(message('sura', 1)); stream.onmessage(message('sura', 1));
  stream.onmessage(message('done', 2)); assert.equal(stream.closed, true);
  gate.resolve(); await flush(); assert.deepEqual(log, ['sura', 'done']);
  const restored = [], status = deferred();
  const resumed = context.watchGeneration({ jobId: 'one', getStatus: () => status.promise,
    onSnapshot: s => restored.push(s.word_failures), onEvent: e => restored.push(e.type),
    onError: error => { throw error; } });
  const reconnect = resumed.onerror();
  status.resolve({ job_id: 'one', word_failures: 5, terminal: { type: 'done' } });
  await reconnect; await flush(); assert.deepEqual(restored, [5, 'done']);
  const offline = context.watchGeneration({ jobId: 'one', getStatus: async () => { throw Error('offline'); },
    onEvent() {}, onError: () => assert.fail('temporary disconnect ended job') });
  await offline.onerror(); assert.equal(offline.closed, undefined); offline.close();
}
function confidenceAndModes() {
  const c = harness(['timingConfidence', 'suraNeedsGeneration', 'applyTimingStatus']);
  for (const [value, level] of [[0, 'low'], [0.32, 'low'], [0.33, 'medium'], [0.65, 'medium'],
    [0.66, 'high'], [1, 'high'], [null, 'unknown'], [NaN, 'unknown'], [Infinity, 'unknown'],
    [-0.1, 'unknown'], [1.1, 'unknown'], ['0.9', 'unknown']]) {
    assert.equal(c.timingConfidence({time: 100, confidence: value, status: 'auto'}).level, level);
  }
  for (const extra of [{stale: true}, {status: 'edited'}, {time: null}]) {
    assert.equal(c.timingConfidence({time: 100, confidence: 0.9, ...extra}).level, 'unknown');
  }
  assert.match(c.timingConfidence({time: 100, confidence: 0}).label, /0/);
  const badge = {dataset: {}}, row = {dataset: {}, querySelector: selector => selector === '.timing-confidence' ? badge : null};
  c.applyTimingStatus(row, {time: 100, confidence: 0.9, status: 'auto'});
  assert.equal(row.dataset.timingStatus, 'uncertain'); // confidence never verifies a row
  assert.equal(badge.dataset.level, 'high'); assert.match(badge.textContent, /90/);
  c.applyTimingStatus(row, {time: 100, confidence: 0.9, status: 'edited'});
  assert.equal(badge.dataset.level, 'unknown');
  let mode = 'full'; c.selectedMode = () => mode;
  assert.equal(c.suraNeedsGeneration({status: 'processed', alignment_modes: ['fast']}), true);
  assert.equal(c.suraNeedsGeneration({status: 'processed', alignment_modes: ['full']}), false);
  assert.equal(c.suraNeedsGeneration({status: 'processed', alignment_modes: ['fast', 'full']}), true);
  assert.equal(c.suraNeedsGeneration({status: 'processed', alignment_modes: ['']}), true);
  assert.equal(c.suraNeedsGeneration({status: 'verified', alignment_modes: []}), false);
  assert.equal(c.suraNeedsGeneration({status: 'partial', alignment_modes: ['full']}), true);
  mode = 'words';
  assert.equal(c.suraNeedsGeneration({status: 'processed', words_complete: false}), true);
  assert.equal(c.suraNeedsGeneration({status: 'processed', words_complete: true}), false);
}
async function comparisonRaces() {
  const c = harness(['comparisonActive', 'loadComparisonPair', 'comparisonStats', 'sameComparisonAudio', 'playComparison', 'stopComparison']);
  const context = {surah:114, active:()=>true};
  c.COMPARISON = {context, request:1, pairRequest:0};
  c.$('comparisonDialog').open = true;
  c.$('comparisonRows').replaceChildren = () => {};
  c.$('comparisonLeft').value = 'fast'; c.$('comparisonRight').value = 'full';
  c.$('comparisonAudio').pause = () => {};
  c.modeLabel = mode => mode;
  c.renderComparison = () => assert.fail('late pair rendered');
  const reply = deferred(); c.api.get = () => reply.promise;
  const pending = c.loadComparisonPair(); ++c.COMPARISON.pairRequest;
  reply.resolve({ayahs:[],source:'',mode:'fast'}); await pending;
  assert.equal(c.COMPARISON.left, null);
  const scores = c.comparisonStats([{ayah:1,time:1000},{ayah:2,time:2000}],
    [{ayah:1,time:1090},{ayah:2,time:1400},{ayah:3,time:5000}]);
  assert.equal(scores.count, 2); assert.equal(scores.mean, 345); assert.equal(scores.max, 600); assert.equal(scores.large, 1);
  assert.equal(c.sameComparisonAudio({name:'a',size:1,mtime_ns:2},{mtime_ns:2,size:1,name:'a'}), true);
  const audio = c.$('comparisonAudio'); audio.readyState = 1; audio.duration = 20; audio.play = async () => {};
  c.R.ayahs = Array.from({length:6}, (_, i) => ({ayah:i+1}));
  await c.playComparison({ayahs:[{ayah:1,time:1000},{ayah:2,time:3000}]}, 1, 'A');
  assert.equal(audio.currentTime, 1); assert.equal(c.COMPARISON.stopMs, 3000);
  await c.playComparison({ayahs:[{ayah:1,time:1000},{ayah:3,time:19000}]}, 1, 'A');
  assert.equal(c.COMPARISON.stopMs, 9000); // don't play an incomplete span through other ayahs
  await c.playComparison({ayahs:[{ayah:6,time:19000}]}, 6, 'A');
  assert.equal(c.COMPARISON.stopMs, 20000);
  let paused = 0, unloaded = 0;
  audio.pause = () => paused++; audio.removeAttribute = key => assert.equal(key, 'src'); audio.load = () => unloaded++;
  c.stopComparison();
  assert.equal(c.COMPARISON.context, null); assert.equal(c.COMPARISON.stopMs, null);
  assert.equal(paused, 1); assert.equal(unloaded, 1);
}
async function generationWriteChoices() {
  for (const [button, value] of [['preserveGenerationBtn', false], ['overwriteGenerationBtn', true],
    [null, null]]) {
    const c = harness(['chooseGenerationWrite']);
    const dialog = c.$('generationChoiceDialog');
    let close, focused = false;
    const anchor = {focus: () => focused = true,
      getBoundingClientRect: () => ({left: 100, top: 10, bottom: 50})};
    dialog.style = {};
    dialog.showModal = () => {};
    dialog.getBoundingClientRect = () => ({width: 248, height: 96});
    c.$('preserveGenerationBtn').focus = () => {};
    dialog.addEventListener = (event, fn) => { assert.equal(event, 'close'); close = fn; };
    dialog.close = value => { if (value !== undefined) dialog.returnValue = value; close(); };
    c.openDialog = () => {};
    const pending = c.chooseGenerationWrite(anchor);
    assert.equal(dialog.style.left, '100px'); assert.equal(dialog.style.top, '58px');
    if (button) c.$(button).onclick(); else dialog.close(); // Escape
    assert.equal(await pending, value); assert.equal(focused, true);
  }
  for (const choice of [false, true, null]) {
    const c = harness(['prepareGeneration']); c.generationPreparing = false;
    c.api.post = async () => ({suras:[114], manual_ayahs:2, manual_words:0});
    c.chooseGenerationWrite = async () => choice;
    const result = await c.prepareGeneration({mode:'full', db_path:'one.db'}, () => true);
    if (choice === null) assert.equal(result, null);
    else { assert.equal(result.payload.overwrite_manual, choice); assert.equal(result.payload.only_missing, false); }
    assert.equal(c.generationPreparing, false);
  }
  const c = harness(['prepareGeneration']); c.generationPreparing = false;
  const reply = deferred(); let active = true;
  c.api.post = () => reply.promise;
  c.chooseGenerationWrite = () => assert.fail('stale dialog');
  const pending = c.prepareGeneration({}, () => active);
  assert.equal(await c.prepareGeneration({}, () => true), null); // Double Start
  active = false; reply.resolve({suras:[114], manual_ayahs:1, manual_words:0});
  assert.equal(await pending, null);
  c.api.post = async () => ({suras:[114], manual_ayahs:1, manual_words:0});
  active = true; c.chooseGenerationWrite = async () => { active = false; return true; };
  assert.equal(await c.prepareGeneration({}, () => active), null); // Changed while choosing
  c.api.post = async () => { throw Error('offline'); };
  await assert.rejects(c.prepareGeneration({}, () => true), /offline/);
  assert.equal(c.generationPreparing, false);
}
function diagnosticSaveOutcome() {
  const c = harness(['handleGenEvent', 'endGenUI']);
  const logs = [];
  c.GEN.wordFailures = 0; c.GEN.diagnosticsErrors = 0;
  c.ALIGN = {done: 6, total: 6};
  for (const name of ['handleAlignmentEvent', 'setProgress', 'refreshSuras', 'reloadSura',
    'stopAlignmentProgress', 'syncAlignmentControls', 'updateElapsed']) c[name] = () => {};
  c.appendLiveLog = (id, message) => logs.push(message);
  c.$('genPanel').classList = {add() {}};
  c.handleGenEvent({type: 'sura', surah: 114, written: 6,
    outcome: 'saved_with_diagnostics_error', diagnostics_error: 'disk full',
    log_paths: {log: '/missing.log'}});
  assert.equal(c.GEN.diagnosticsErrors, 1);
  assert.equal(logs.length, 1);
  assert(logs[0].includes('разметка сохранена'));
  assert(!logs[0].includes('Лог сохранён'));
  c.handleGenEvent({type: 'done', completed: 1, diagnostics_errors: [{surah: 114, error: 'disk full'}]});
  assert(c.$('genTitle').textContent.includes('Разметка сохранена'));
  assert.equal(c.$('suraGenStatus').className, 'sura-gen-status warn');
  c.handleGenEvent({type: 'cancelled', completed: 1, diagnostics_errors: [{surah: 114, error: 'disk full'}]});
  assert(c.$('genTitle').textContent.includes('Разметка остановлена'));
  assert(c.$('genTitle').textContent.includes('суры сохранены'));
  assert(c.$('genTitle').textContent.includes('диагностика'));
}
(async () => { diagnosticSaveOutcome(); confidenceAndModes(); await generationWriteChoices(); await comparisonRaces(); await reviewRaces(); await generationRaces(); console.log('async UI regressions passed'); })()
  .catch(error => { console.error(error); process.exitCode = 1; });
