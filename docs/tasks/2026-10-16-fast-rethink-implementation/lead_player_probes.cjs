const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync('surah_cutter/webgui/static/app.js', 'utf8');
const calls = { pause: 0, play: 0, seek: [], messages: [], ends: [] };
const state = {
  R: {
    ayahs: [{ ayah: 1, time: 1000 }, { ayah: 2, time: null },
            { ayah: 3, time: 5000 }, { ayah: 4, time: 7000 }],
    duration: 10000, playbackMode: 'single', audioReady: true, dataReady: true,
    audio: { paused: false, pause() { calls.pause++; this.paused = true; },
             play() { calls.play++; this.paused = false; } },
  },
  W: { mode: false },
  t: value => value,
  setStatus: value => calls.messages.push(value),
  seekAudioMs: (_, value) => calls.seek.push(value),
  syncPlaybackRate() {}, setCurrent() {},
  scrollWaveToMs() {}, drawWave() {},
  clearTimeout() {}, setTimeout() { return 1; },
  mediaTimeMs() { return 4000; },
  setAudioPlaybackEnd(_, end) { calls.ends.push(end); },
};
vm.createContext(state);
for (const name of ['ayahEndMs', 'currentAyahByTime', 'scheduleAyahStop', 'seekAyah']) {
  const match = source.match(new RegExp(`^function ${name}\\([^]*?\\n\\}`, 'm'));
  assert.ok(match, `${name} found in production source`);
  vm.runInContext(match[0], state);
}

assert.equal(state.ayahEndMs(0), null, 'missing immediate successor means unknown end');
assert.equal(state.ayahEndMs(1), null, 'missing start means unknown clip');
assert.equal(state.ayahEndMs(2), 7000);
assert.equal(state.ayahEndMs(3), 10000);
assert.equal(state.currentAyahByTime(4000), -1, 'unknown gap has no asserted current ayah');
assert.equal(state.currentAyahByTime(6000), 2);
state.seekAyah(0, true);
assert.equal(calls.pause, 1, 'rejecting bounded playback must stop already playing audio');
assert.equal(calls.play, 0);
assert.deepEqual(calls.seek, [], 'rejected bounded playback must not seek');
state.seekAyah(2, true);
assert.equal(calls.play, 1);
assert.deepEqual(calls.seek, [5000]);
state.R.audio.paused = false;
state.R.playbackAyah = 0;
state.scheduleAyahStop();
assert.equal(calls.pause, 2, 'switching to bounded playback with an unknown end stops audio');
assert.equal(calls.ends.at(-1), null);

const result = {
  passed: true, production_functions_executed: ['ayahEndMs', 'currentAyahByTime', 'scheduleAyahStop', 'seekAyah'],
  cases: ['unknown clip end', 'unknown start', 'immediate successor', 'last ayah',
          'unknown current ayah', 'known current ayah', 'reject while playing', 'valid clip',
          'switch from continuous to bounded playback'],
  scope: 'JavaScript function probes, not browser or native pywebview verification',
};
fs.writeFileSync(path.join(__dirname, 'lead-player-probes.json'), JSON.stringify(result, null, 2) + '\n');
process.stdout.write(JSON.stringify(result, null, 2) + '\n');
