// Run against snapshot_server.py: only DOM state is changed, never stored timings.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 911 } });
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8879');
    await page.waitForFunction(() => dashboardSuras.length === 114);
    await page.evaluate(() => { void openReview(114); });
    await page.waitForFunction(() => R.dataReady && R.waveReady && R.audioReady);
    await page.evaluate(() => { window.originalAyahs = R.ayahs.map(ayah => ({...ayah})); applyAppearance('dark', 30); setWaveZoom(1); });
    const geometry = () => page.evaluate(() => {
      const canvas = $('wave').getBoundingClientRect(), row = document.querySelector('.wave-row').getBoundingClientRect();
      return { height: document.querySelector('.wave-wrap').getBoundingClientRect().height, top: canvas.top - row.top,
        bottom: row.bottom - $('waveScroll').getBoundingClientRect().bottom, hint: getComputedStyle($('waveHint')).display };
    });
    const marked = await geometry();
    assert.equal(marked.top, 8); assert.equal(marked.bottom, 8); assert.equal(marked.hint, 'none');
    await page.evaluate(() => { R.ayahs = R.ayahs.map(ayah => ({...ayah,time:null,status:null})); renderAyahList(); updateSuraProgress(); updateGenHint(); drawWave(); });
    const missing = await geometry(); assert.deepEqual(missing, marked);
    assert.match(await page.locator('#waveHint').textContent(), /без тайм-кодов/);
    assert.equal(await page.locator('#waveHint').isVisible(), false);
    await page.screenshot({ path: 'docs/tasks/2026-10-13-light-sidebar-studio/wave-unmarked-dark.png' });
    // The prior warn class remains during loading, but can't create an empty banner.
    await page.evaluate(() => $('waveHint').textContent = '');
    assert.deepEqual(await geometry(), marked);
    for (const zoom of [8, 32, 1]) {
      await page.evaluate(zoom => setWaveZoom(zoom), zoom);
      assert.deepEqual(await geometry(), marked);
    }
    await page.evaluate(() => { R.ayahs = originalAyahs; updateGenHint(); renderAyahList(); setCurrent(1); drawWave(); });
    assert.deepEqual(await geometry(), marked);
    assert.equal(await page.locator('#clock').isVisible(), true);
    assert.equal(await page.locator('#zoomInBtn').isVisible(), true);
    await page.locator('#playBtn').click(); await page.waitForFunction(() => !R.audio.paused); await page.locator('#playBtn').click();
    assert.deepEqual(errors, []);
    console.log('PASS: dark marked/missing/loading, constant 8px insets and section height at 1x/8x/32x, no hint overlay, clock/zoom/playback.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
