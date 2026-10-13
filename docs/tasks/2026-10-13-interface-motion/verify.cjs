// NODE_PATH=.../dependencies/node/node_modules node verify.cjs
// Serve webgui/static on 127.0.0.1:8877 first.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage({ viewport: { width: 1180, height: 900 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.addInitScript(() => {
      window.streams = [];
      window.EventSource = class {
        constructor(url) { this.url = url; this.closed = false; streams.push(this); }
        close() { this.closed = true; }
        emit(data) { this.onmessage?.({ data: JSON.stringify(data) }); }
      };
    });
    const wav = Buffer.alloc(320044);
    wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28);
    wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(320000, 40);
    await page.route('**/api/**', route => {
      const path = new URL(route.request().url()).pathname;
      if (path.startsWith('/api/audio/')) return route.fulfill({ contentType: 'audio/wav', body: wav });
      const body = path === '/api/projects' ? [] : path === '/api/suras'
        ? [113, 114].map(surah => ({ surah, ayah_count: 1, has_audio: true, status: 'processed' }))
        : /\/logs$/.test(path) ? { logs: [] }
        : /\/api\/sura\/\d+$/.test(path) ? { ayah_count: 1, ayahs: [{ ayah: 1, time: 0, status: 'auto', text: '' }] }
        : {};
      return route.fulfill({ json: body });
    });
    const open = n => page.evaluate(n => { void openReview(n); }, n);
    const emit = event => page.evaluate(event => streams.at(-1).emit(event), event);
    await page.goto('http://127.0.0.1:8877');
    await page.waitForFunction(() => dashboardSuras.length === 2);
    await open(114);
    await page.waitForFunction(() => R.dataReady && R.audioReady && R.duration === 10000);
    await emit({ type: 'start', bucket_ms: 100 });
    await emit({ type: 'chunk', offset: 0, peaks: Array.from({ length: 40 }, (_, i) => .1 + (i % 7) / 10), decoded_ms: 4000 });
    await page.waitForTimeout(200);
    assert.deepEqual(await page.evaluate(() => [R.peaks.length, R.waveReady, R.waveLoading]), [40, false, true]);
    assert.equal(await page.locator('#playBtn').isEnabled(), true);
    assert.equal(await page.locator('#waveLoadStatus').textContent(), 'Построение волны · 40%');
    assert.equal(await page.locator('#waveScroll').getAttribute('aria-busy'), 'true');
    await page.evaluate(() => setWaveZoom(1));
    await page.screenshot({ path: '/tmp/surah-motion-partial.png' });
    // Ready peaks occupy the first 40% of the timeline, never stretch to its end.
    const pixels = await page.evaluate(() => {
      const c = $('wave'), ctx = c.getContext('2d'), y = Math.round(c.height / 2 - 12);
      return [ctx.getImageData(Math.round(c.width * .2), y, 10, 1).data.some(v => v),
              ctx.getImageData(Math.round(c.width * .8), y, 10, 1).data.some(v => v)];
    });
    assert.deepEqual(pixels, [true, false]);
    await emit({ type: 'chunk', offset: 40, peaks: Array(60).fill(.5), decoded_ms: 10000 });
    await emit({ type: 'done', maximum: .8, duration_ms: 10000 });
    await page.waitForFunction(() => R.waveReady && !reviewLoading);
    assert.equal(await page.locator('#waveLoadStatus').isVisible(), false);
    assert.equal(await page.evaluate(() => streams.at(-1).closed), true);

    await page.locator('#showSuraLogBtn').click();
    assert.equal(await page.evaluate(() => $('suraLiveLogBox').open), true);
    assert.equal(await page.evaluate(() => $('suraLiveLogBox').getAnimations().length > 0), true);
    await page.waitForTimeout(240);
    await page.locator('#hideSuraLogBtn').click();
    assert.equal(await page.evaluate(() => $('suraLiveLogBox').open), true);
    await page.waitForFunction(() => !$('suraLiveLogBox').open);
    assert.equal(await page.evaluate(() => document.activeElement.id), 'showSuraLogBtn');
    await page.locator('#showSuraLogBtn').click();
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => $('suraLiveLogBox').open), false);

    await page.locator('#settingsBtn').click();
    await page.waitForFunction(() => $('settingsDialog').open);
    await page.screenshot({ path: '/tmp/surah-motion-settings.png' });
    // Reopening during an exit cancels that exit and keeps the native modal open.
    await page.evaluate(() => { closeDialog($('settingsDialog')); openDialog($('settingsDialog')); });
    await page.waitForTimeout(250);
    assert.equal(await page.evaluate(() => $('settingsDialog').open), true);
    await page.keyboard.press('Escape');
    await page.locator('#settingsBtn').focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => $('settingsDialog').open);
    assert.equal(await page.evaluate(() => $('settingsDialog').getAnimations().filter(a => a.playState === 'running').length), 0);
    await page.keyboard.press('Escape');

    await page.locator('#modeMenuToggle').click();
    await page.waitForFunction(() => $('modeMenu').open && menuMotions.get($('modeMenu').querySelector('.toolbar-popover')));
    assert.equal(await page.evaluate(() => menuMotions.get($('modeMenu').querySelector('.toolbar-popover')).effect.getTiming().duration), 180);
    await page.keyboard.press('Escape');
    await page.locator('#rateBtn').click();
    assert.equal(await page.locator('#rateMenu').isVisible(), true);
    assert.ok(await page.evaluate(() => Math.abs(parseFloat(getComputedStyle($('rateMenu')).transformOrigin.split(' ')[1]) - $('rateMenu').offsetHeight) <= 1));
    await page.keyboard.press('Escape');

    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.locator('#showSuraLogBtn').click();
    const frames = await page.evaluate(() => dialogMotions.get($('suraLiveLogBox')).effect.getKeyframes());
    assert.ok(frames.every(frame => frame.transform === 'none'));
    await page.keyboard.press('Escape');
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.setViewportSize({ width: 390, height: 844 });
    await open(113);
    await emit({ type: 'start', bucket_ms: 100 });
    await emit({ type: 'chunk', offset: 0, peaks: Array(20).fill(.5), decoded_ms: 2000 });
    await page.waitForTimeout(200);
    await page.screenshot({ path: '/tmp/surah-motion-mobile.png' });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await open(114);
    assert.equal(await page.evaluate(() => streams.at(-2).closed), true);
    await page.evaluate(() => streams.at(-2).emit({ type: 'done', maximum: 1, duration_ms: 99999 }));
    assert.equal(await page.evaluate(() => R.duration === 99999), false);
    await emit({ type: 'error', message: 'test decode failure' });
    await page.waitForFunction(() => !reviewLoading);
    assert.equal(await page.locator('#waveLoadStatus button').isVisible(), true);
    assert.equal(await page.locator('#waveScroll').getAttribute('aria-busy'), 'false');
    await page.locator('#waveLoadStatus button').click();
    await page.waitForFunction(() => R.waveLoading);
    await page.locator('#backBtn').click();
    assert.equal(await page.evaluate(() => streams.at(-1).closed), true);
    assert.deepEqual(errors, []);
    console.log('PASS: partial waveform position, completion, cancellation, stale events, errors/retry, modal entry/exit/focus/Escape/interruption, keyboard, reduced motion, desktop/mobile');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
