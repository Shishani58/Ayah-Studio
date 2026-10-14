// Run the actual recheck function with a delayed response and a changed sura.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.resolve(__dirname, "../../../surah_cutter/webgui/static/app.js"), "utf8");
const start = source.indexOf("async function recheckSura()");
const end = source.indexOf("// re-fetch this sura", start);
assert(start >= 0 && end > start);
let reply;
const elements = {};
const context = {
  R: { surah: 112, ayahs: [{ time: 1000 }], recheck: null },
  $: id => elements[id] ||= {},
  api: { get: () => new Promise(resolve => { reply = resolve; }) },
  setStatus() {}, renderAyahList() {}, drawWave() {},
  setInterval: () => 1, clearInterval() {}, Date,
  t: (value, ...values) => typeof value === "string" ? value
    : value.reduce((text, part, i) => text + part + (values[i] ?? ""), ""),
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
(async () => {
  const pending = context.recheckSura();
  context.R.surah = 114;
  reply({ surah: 112, results: [{ ayah: 1, verdict: "shift", match_ayah: 2 }],
    summary: { suspect: 1, checked: 4 } });
  await pending;
  assert.equal(context.R.surah, 114);
  assert.equal(context.R.recheck[1].verdict, "shift");
  console.log(JSON.stringify({ current_sura: context.R.surah,
    recheck_from_sura: 112, applied: context.R.recheck }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });
