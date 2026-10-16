const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../surah_cutter/webgui/static/app.js'), 'utf8');
function extract(name) {
  const start = source.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert(start >= 0, name);
  return source.slice(start, source.indexOf('\n}', start) + 2);
}
const slider = {value: '2'};
let timer;
const c = {
  R: {audio: {currentTime: 0, playbackRate: 2, paused: false}, dataReady: true,
    duration: 16000, ayahs: [{ayah: 1, time: 0}, {ayah: 2, time: 10000}]},
  W: {mode: false}, $: () => slider,
  clearTimeout() { timer = null; }, setTimeout(fn, delay) { timer = {fn, delay}; return 1; },
  syncPlayerSettings() {}, scheduleAyahStop() {}, updateRatePicker() {},
};
vm.createContext(c);
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname, '../surah_cutter/webgui/static/playback.js'), 'utf8'), c);
vm.runInContext(['currentAyahByTime', 'ayahEndMs', 'syncPlaybackRate', 'applyPlayerPreferences',
  'seekAyah', 'seekPrevUnit', 'seekNextUnit'].map(extract).join('\n'), c);
function at(seconds, expected = 2) {
  c.R.audio.currentTime = seconds;
  c.syncPlaybackRate();
  assert.equal(c.R.audio.playbackRate, expected, `at ${seconds}`);
  assert.equal(c.R.audio.defaultPlaybackRate, expected);
  assert.equal(timer, undefined, 'speed never schedules automatic changes');
  assert.equal(Number(slider.value), expected);
}
for (const mode of ['continuous', 'single', 'repeat']) {
  c.R.playbackMode = mode;
  for (const seconds of [0, 1, 5, 9, 9.5, 10, 11, 15, 16, 0]) at(seconds);
}
c.R.audio.paused = true; at(5);
c.R.audio.paused = false;
c.W.mode = true; c.W.stop = {start: 3000, end: 4500};
at(3); at(4); // short words also keep the selected speed
c.W.stop = {start: 3000, end: 8000};
at(3); at(4); at(7);
c.applyPlayerPreferences({playback_mode: 'repeat', playback_rate: .8});
assert.equal(c.R.loop, true); assert.equal(c.R.playbackMode, 'repeat');
assert.equal(Number(slider.value), .8);
for (const seconds of [3, 5, 7, 8]) at(seconds, .8);
c.applyPlayerPreferences({playback_mode: 'single', playback_rate: undefined});
assert.equal(c.R.loop, false); assert.equal(Number(slider.value), 1);

// Exercise real navigation and seeking from a paused player.
c.W.mode = false;
c.R.audioReady = true;
c.setCurrent = i => { c.R.current = i; };
c.scrollWaveToMs = () => {};
c.drawWave = () => {};
let plays = 0;
c.R.audio.play = () => { plays++; c.R.audio.paused = false; };
c.R.current = 0;
c.R.audio.paused = true;
c.seekNextUnit();
assert.equal(c.R.current, 1); assert.equal(c.R.playbackAyah, 1);
assert.equal(c.mediaTimeMs(c.R.audio), 10000);
assert.equal(plays, 1); assert.equal(c.R.audio.paused, false);
c.R.audio.paused = true;
c.seekPrevUnit();
assert.equal(c.R.current, 0); assert.equal(c.mediaTimeMs(c.R.audio), 0);
assert.equal(plays, 2); assert.equal(c.R.audio.paused, false);
for (const mode of ['continuous', 'repeat']) {
  c.R.playbackMode = mode; c.R.audio.paused = true;
  c.seekNextUnit(); c.seekPrevUnit();
  assert.equal(plays, 2); assert.equal(c.R.audio.paused, true);
}
c.R.playbackMode = 'single';
c.seekPrevUnit(); assert.equal(c.R.current, 0);
c.seekNextUnit(); c.seekNextUnit(); assert.equal(c.R.current, 1);
assert.equal(plays, 5);
console.log('Player preferences, constant speed and single-ayah navigation checks passed');
