// Isolated fixture at :8895; generation and SSE mocked, no model or user DB.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 940}});
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    page.on('dialog', d => d.accept());
    let starts = 0, cancels = 0, cancelFails = false;
    await page.route('**/api/generate', r => { starts++; return r.fulfill({json: {started: true}}); });
    await page.route('**/api/generate/cancel', r => {
      cancels++;
      return r.fulfill(cancelFails ? {status: 500, json: {detail: 'cancel failed'}} : {json: {cancelling: true}});
    });
    await page.addInitScript(() => {
      const Native = window.EventSource;
      window.EventSource = class {
        constructor(url) {
          if (url !== '/api/generate/stream') return new Native(url);
          window.testGenerationStream = this;
        }
        close() { this.closed = true; }
      };
    });
    const emit = d => page.evaluate(d => window.testGenerationStream.onmessage({data: JSON.stringify(d)}), d);
    await page.goto('http://127.0.0.1:8895');
    await page.waitForFunction(() => dashboardSuras.length === 114);
    assert.equal(await page.locator('#modeMenu').isVisible(), false);
    for (const attr of ['reviewMode', 'dashboardMode']) {
      const modes = await page.locator(attr === 'reviewMode' ? '[data-review-mode]' : '[data-dashboard-mode]').evaluateAll((els, attr) => els.map(e => e.dataset[attr]), attr);
      assert.deepEqual(modes, ['fast', 'full', 'words']);
    }
    assert.equal(await page.locator('.mode-option-icon svg').count(), 6);
    for (const legacy of ['auto', 'tiled']) {
      await page.evaluate(mode => setMode(mode), legacy);
      assert.equal(await page.locator('#genMode').inputValue(), 'full');
      assert.equal(await page.locator('#suraMode').inputValue(), 'full');
    }
    await page.locator('#grid button[data-surah="114"]').click();
    await page.waitForFunction(() => R.dataReady && R.waveReady && R.audioReady);
    assert.equal(await page.locator('.review-summary').count(), 0);
    assert.equal(await page.locator('#dashboardModeMenu').isVisible(), false);
    assert.match(await page.locator('#suraProgressText').innerText(), /3\/6/);
    assert.match(await page.locator('#suraProgressMeta').innerText(), /Проверено 1\/6/);
    assert.equal(await page.locator('#navigationProgress').isVisible(), false);
    await page.locator('#recordingMenuToggle').click();
    assert.equal(await page.locator('#projectPickerList').isVisible(), true);
    await page.keyboard.press('Escape');
    await page.locator('#modeMenuToggle').click();
    await page.locator('[data-review-mode="fast"]').click();
    assert.match(await page.locator('#modeMenuToggle').innerText(), /быстрый/);
    await page.locator('#modeMenuToggle').click();
    await page.locator('[data-review-mode="full"]').click();
    await page.locator('#recordingMenuToggle').focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#projectPickerList').isVisible(), true);
    await page.keyboard.press('Escape');
    for (const theme of ['light', 'dark']) {
      await page.evaluate(t => applyAppearance(t, 30), theme);
      for (const width of [1440, 1180, 1060, 820, 600, 390, 320]) {
        await page.setViewportSize({width, height: 940});
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${theme} ${width}`);
        if (width > 820) {
          const delta = await page.evaluate(() => Math.abs($('regenSuraBtn').getBoundingClientRect().left - document.querySelector('.wave-wrap').getBoundingClientRect().left));
          assert.ok(delta < 1, `alignment ${width}: ${delta}`);
        }
        assert.equal(await page.locator('#modeMenuToggle .studio-icon').count(), 1);
        for (const id of ['regenSuraBtn', 'recordingMenuToggle', 'modeMenuToggle', 'suraProgress']) {
          const rect = await page.locator('#' + id).boundingBox();
          assert.ok(rect && rect.x >= 0 && rect.x + rect.width <= width + 1, `${id} ${width}`);
        }
        await page.locator('#modeMenuToggle').click();
        await page.waitForTimeout(250); // Existing menu enter animation and toggle event.
        const rect = await page.locator('#modeMenu .toolbar-popover').boundingBox();
        assert.ok(rect.x >= 0 && rect.x + rect.width <= width + 1);
        await page.keyboard.press('Escape');
      }
      await page.setViewportSize({width: 1440, height: 940});
      await page.screenshot({path: __dirname + `/navigation-${theme}.png`});
    }
    await page.locator('#regenSuraBtn').click();
    await page.waitForFunction(() => generationReady);
    assert.match(await page.locator('#regenSuraBtn').innerText(), /Остановить разметку/);
    assert.equal(await page.locator('#regenSuraBtn').evaluate(e => e.classList.contains('alignment-stop')), true);
    assert.equal(await page.locator('#reviewGeneration').isVisible(), true);
    assert.equal(await page.locator('#regenSuraBtn').evaluate(e => getComputedStyle(e).backgroundColor), 'rgb(199, 57, 54)');
    await page.screenshot({path: __dirname + '/navigation-running.png'});
    cancelFails = true;
    await page.locator('#regenSuraBtn').click();
    await page.waitForFunction(() => !alignmentCancelling);
    assert.equal(await page.locator('#regenSuraBtn').isEnabled(), true);
    cancelFails = false;
    await page.locator('#regenSuraBtn').click();
    await page.waitForFunction(() => alignmentCancelling);
    assert.match(await page.locator('#regenSuraBtn').innerText(), /Остановка/);
    assert.equal(await page.locator('#regenSuraBtn').isDisabled(), true);
    assert.equal(starts, 1); assert.equal(cancels, 2);
    await emit({type: 'cancelled', completed: 0});
    await page.waitForFunction(() => generatingSurah === null);
    assert.match(await page.locator('#regenSuraBtn').innerText(), /Начать разметку/);
    assert.equal(await page.locator('#reviewGeneration').isVisible(), false);
    assert.match(await page.locator('#suraGenStatus').innerText(), /остановлена/);
    await page.locator('#regenSuraBtn').click();
    await page.waitForFunction(() => generationReady);
    await emit({type: 'error', message: 'fixture error'});
    await page.waitForFunction(() => generatingSurah === null);
    assert.equal(await page.locator('#regenSuraBtn').isEnabled(), true);
    assert.match(await page.locator('#suraGenStatus').innerText(), /fixture error/);
    await page.locator('#regenSuraBtn').click();
    await page.waitForFunction(() => generationReady);
    await emit({type: 'done', completed: 1});
    await page.waitForFunction(() => generatingSurah === null);
    assert.match(await page.locator('#suraGenStatus').innerText(), /готово/);
    await page.evaluate(() => showView('dashboard'));
    await page.locator('#genBtn').click();
    await page.waitForFunction(() => GEN.ready);
    await emit({type: 'start', total: 3});
    await emit({type: 'sura', surah: 105, index: 0});
    assert.equal(await page.locator('#reviewGeneration').getAttribute('aria-valuenow'), '33');
    await page.locator('#genBtn').click();
    await emit({type: 'cancelled', completed: 1});
    await page.waitForFunction(() => !GEN.running);
    assert.equal(await page.locator('#genBtn').evaluate(e => e.classList.contains('alignment-stop')), false);
    assert.equal(await page.locator('#reviewGeneration').isVisible(), false);
    assert.deepEqual(errors, []);
    console.log('PASS: menus, keyboard, counts, two themes, seven widths, single/batch lifecycle, cancel failure/retry, no JS errors');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
