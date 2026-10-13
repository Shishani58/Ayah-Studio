// Run against the isolated /tmp fixture server, never a user's project.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 940 } });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:8878');
    await page.waitForFunction(() => dashboardSuras.length === 114);
    const tile = n => page.locator(`#grid button[data-surah="${n}"]`);
    assert.match(await tile(105).textContent(), /Проверено/);
    assert.match(await tile(107).textContent(), /Размечено/);
    assert.match(await tile(109).textContent(), /Не размечено/);
    assert.match(await tile(114).textContent(), /Размечено частично/);
    assert.match(await tile(1).textContent(), /Нет аудио/);
    assert.equal(await tile(1).getAttribute('aria-disabled'), 'true');
    await page.locator('#suraSearch').fill('Ан-Нас');
    assert.equal(await page.locator('#grid button').count(), 2); // Ан-Наср and Ан-Нас
    await page.locator('#suraSearch').fill('114');
    await page.locator('#suraSearch').press('Enter');
    await page.waitForFunction(() => R.waveReady && R.dataReady && R.audioReady);
    assert.equal(await page.locator('#reviewTitle').textContent(), 'Ан-Нас');
    assert.equal(await page.locator('#ayahList > li').count(), 6);
    const initial = await page.evaluate(() => ({ w: $('review').clientWidth, wave: $('wave').getBoundingClientRect().top, summary: $('reviewTitle').getBoundingClientRect().bottom }));
    assert.ok(initial.wave > initial.summary);
    await page.locator('#sidebarToggle').click();
    assert.equal(await page.locator('#suraSidebar').isVisible(), false);
    assert.ok(await page.evaluate(() => $('review').clientWidth) > initial.w);
    await page.locator('#sidebarToggle').click();
    assert.equal(await page.locator('#suraSidebar').isVisible(), true);
    const input = page.locator('#ayahList > li').nth(1).locator('input');
    await input.fill('00:02.137'); await input.press('Enter');
    await page.waitForFunction(() => R.ayahs[1].time === 2137 && !editBusy);
    assert.equal(await page.locator('#ayahList > li').nth(1).getAttribute('data-timing-status'), 'uncertain');
    await page.locator('#ayahList > li').nth(1).locator('.time-arrows button').first().click();
    await page.waitForFunction(() => R.ayahs[1].time === 2147 && !editBusy);
    await input.fill('invalid'); await input.press('Enter');
    assert.match(await page.locator('#workspaceStatus').textContent(), /формате/);
    assert.equal(await page.evaluate(() => R.ayahs[1].time), 2147);
    await input.fill('00:02.147'); await input.press('Enter');
    await page.route('**/api/timing', route => route.fulfill({ status: 422, json: { detail: 'test save failure' } }));
    await input.fill('00:02.200'); await input.press('Enter');
    await page.waitForFunction(() => !editBusy && $('workspaceStatus').textContent.includes('test save failure'));
    assert.equal(await input.inputValue(), '00:02.147');
    assert.equal(await page.evaluate(() => R.ayahs[1].time), 2147);
    await page.unroute('**/api/timing');
    // Existing verified metadata becomes uncertain on manual edit, never vice versa.
    await page.locator('#ayahList > li').first().locator('.time-arrows button').first().click();
    await page.waitForFunction(() => R.ayahs[0].status === 'edited' && !editBusy);
    assert.equal(await page.locator('#ayahList > li').first().getAttribute('data-timing-status'), 'uncertain');
    await page.locator('#reviewToolsToggle').click(); await page.locator('#verifySelectedBtn').click();
    await page.waitForFunction(() => R.ayahs[0].status === 'verified');
    await page.keyboard.press('Escape');
    assert.match(await page.locator('#suraProgressMeta').textContent(), /Проверено 1\/6/);
    await page.locator('#nextBtn').click();
    assert.equal(await page.evaluate(() => R.current), 1);
    await page.locator('#playBtn').click(); await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => R.audio.paused), false);
    await page.locator('#playBtn').click();
    await page.locator('#rateBtn').click(); await page.locator('#rateMenu button[data-rate="1.5"]').click();
    assert.equal(await page.evaluate(() => R.audio.playbackRate), 1.5);
    await page.locator('#modeMenuToggle').click(); await page.locator('[data-review-mode="words"]').click();
    await page.waitForFunction(() => W.data && !W.loading);
    const wordCount = await page.locator('#ayahList li[data-word]').count(); assert.ok(wordCount > 6);
    await page.evaluate(() => setWaveZoom(8, null, true));
    await page.locator('#ayahList li[data-word]').nth(1).locator('.a-num').click();
    assert.equal(await page.evaluate(() => R.zoom), 8);
    assert.equal(await page.locator('#wordUnitControl').isVisible(), true);
    assert.equal(await page.evaluate(() => {
      const row = document.createElement('li'); row.innerHTML = '<button class="a-num" aria-label="test"></button>';
      applyTimingStatus(row, {time:123,status:'unknown'}); return row.dataset.timingStatus;
    }), 'neutral');
    await page.locator('#wordTimeUnit').selectOption('ms');
    assert.equal(await page.evaluate(() => W.unit), 'ms');
    await page.locator('#modeMenuToggle').click(); await page.locator('[data-review-mode="full"]').click();
    await page.waitForFunction(() => !W.mode);
    await page.locator('#showSuraLogBtn').click(); await page.waitForFunction(() => $('suraLiveLogBox').open);
    await page.keyboard.press('Escape'); assert.equal(await page.evaluate(() => document.activeElement.id), 'showSuraLogBtn');
    await page.locator('#settingsBtn').click(); await page.waitForFunction(() => $('settingsDialog').open);
    await page.locator('[data-settings-panel="modelSettings"]').click();
    await page.waitForFunction(() => $('modelList').children.length > 0);
    await page.keyboard.press('Escape');
    // Bottom padding lets the final ayah clear the floating player's top edge.
    await page.locator('#ayahList > li').last().locator('input').focus();
    await page.evaluate(() => $('ayahList').scrollTop = $('ayahList').scrollHeight);
    const geometry = await page.evaluate(() => ({ last: $('ayahList').lastElementChild.getBoundingClientRect().bottom, player: document.querySelector('.floating-player').getBoundingClientRect().top }));
    assert.ok(geometry.last <= geometry.player, JSON.stringify(geometry));
    await page.locator('#suraSearch').fill(''); await tile(114).scrollIntoViewIfNeeded();
    await page.evaluate(() => { setWaveZoom(1); R.audio.pause(); R.audio.currentTime = 3; setCurrent(1); drawWave(); });
    const annotation = await page.evaluate(() => {
      const ctx = $('wave').getContext('2d'), originalText = ctx.fillText, originalMove = ctx.moveTo;
      let labels = [], arrows = [];
      ctx.fillText = function(text, ...args) { labels.push(text); return originalText.call(this, text, ...args); };
      ctx.moveTo = function(x, y) { if (y === 20) arrows.push(x); return originalMove.call(this, x, y); };
      try {
        setCurrent(1); drawWave();
        const known = { labels: [...labels], arrows: [...arrows], start: msToVirtualX(R.ayahs[1].time), end: msToVirtualX(R.ayahs[2].time) };
        labels = []; arrows = []; setCurrent(2); drawWave();
        const unknown = { labels: [...labels], arrows: [...arrows] };
        return { known, unknown };
      } finally { ctx.fillText = originalText; ctx.moveTo = originalMove; setCurrent(1); drawWave(); }
    });
    assert.ok(annotation.known.labels.includes('Аят 2'));
    assert.ok(annotation.known.arrows.some(x => Math.abs(x - annotation.known.start) < 1));
    assert.ok(annotation.known.arrows.some(x => Math.abs(x - annotation.known.end) < 1));
    assert.ok(!annotation.unknown.labels.includes('Аят 3'));
    assert.equal(annotation.unknown.arrows.length, 0);
    assert.equal(await tile(105).getAttribute('data-status'), 'verified');
    assert.equal(await tile(107).getAttribute('data-status'), 'processed');
    assert.equal(await tile(114).getAttribute('data-status'), 'partial');
    assert.equal(await tile(109).getAttribute('data-status'), 'not-processed');
    const colors = await tile(105).evaluate(el => ({bar:getComputedStyle(el,'::before').backgroundColor,label:getComputedStyle(el.querySelector('.badge')).color,neutral:getComputedStyle(document.documentElement).getPropertyValue('--muted').trim()}));
    assert.equal(colors.bar, 'rgb(22, 131, 62)');
    assert.equal(colors.label, 'rgb(104, 109, 119)');
    assert.equal(await page.locator('#showSuraLogBtn').textContent(), 'Логи');
    await page.waitForTimeout(250);
    await page.screenshot({ path: 'docs/tasks/2026-10-13-light-sidebar-studio/chrome-fixture.png' });
    for (const width of [1180, 820, 600, 390, 320]) {
      await page.setViewportSize({ width, height: 840 });
      if (width <= 600 && await page.locator('#suraSidebar').isVisible()) await page.locator('#sidebarToggle').click();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `overflow at ${width}`);
      await page.locator('#showSuraLogBtn').click(); await page.waitForFunction(() => $('suraLiveLogBox').open); await page.keyboard.press('Escape');
    }
    // Changing suras cancels the old progressive stream; late chunks can't win.
    const streaming = await browser.newPage({ viewport: { width: 1180, height: 840 } });
    await streaming.addInitScript(() => {
      window.streams = [];
      window.EventSource = class {
        constructor(url) { this.url = url; this.closed = false; streams.push(this); }
        close() { this.closed = true; }
        emit(data) { this.onmessage?.({ data: JSON.stringify(data) }); }
      };
    });
    await streaming.goto('http://127.0.0.1:8878');
    await streaming.waitForFunction(() => dashboardSuras.length === 114);
    await streaming.evaluate(() => { void openReview(114); });
    await streaming.waitForFunction(() => R.dataReady && streams.length === 1);
    await streaming.evaluate(() => {
      streams[0].emit({type:'start',bucket_ms:100,duration_hint_ms:14000});
      streams[0].emit({type:'chunk',offset:0,peaks:[.1,.2,.3],decoded_ms:300});
    });
    assert.equal(await streaming.evaluate(() => R.peaks.length), 3);
    await streaming.evaluate(() => { void openReview(113); });
    await streaming.waitForFunction(() => R.surah === 113 && R.dataReady && streams.length === 2);
    assert.equal(await streaming.evaluate(() => streams[0].closed), true);
    await streaming.evaluate(() => streams[0].emit({type:'chunk',offset:3,peaks:[.9],decoded_ms:400}));
    assert.equal(await streaming.evaluate(() => R.peaks.length), 0);
    // Generation payload/cancel UI, intercepted before the alignment pipeline.
    const requests = [];
    await streaming.route('**/api/generate', route => { requests.push(route.request().postDataJSON()); return route.fulfill({json:{started:true}}); });
    await streaming.route('**/api/generate/cancel', route => route.fulfill({json:{cancelled:true}}));
    streaming.on('dialog', dialog => dialog.accept());
    await streaming.locator('#regenSuraBtn').click();
    await streaming.waitForFunction(() => streams.some(s => s.url === '/api/generate/stream'));
    assert.deepEqual(requests.at(-1).suras, [113]);
    assert.equal(requests.at(-1).mode, 'full');
    await streaming.evaluate(() => streams.find(s => s.url === '/api/generate/stream').emit({type:'cancelled',completed:0}));
    await streaming.locator('#reviewToolsToggle').click();
    await streaming.locator('#batchGenerateBtn').click();
    await streaming.waitForFunction(() => GEN.es !== null);
    assert.equal(requests.at(-1).only_missing, true);
    await streaming.keyboard.press('Escape');
    assert.equal(await streaming.locator('#genPanel').isVisible(), true);
    await streaming.locator('#cancelBtn').click();
    assert.equal(await streaming.locator('#cancelBtn').textContent(), 'Отмена…');
    await streaming.evaluate(() => GEN.es.onmessage({data:JSON.stringify({type:'cancelled',completed:0})}));
    assert.match(await streaming.locator('#genTitle').textContent(), /Отменено/);
    await streaming.close();
    assert.deepEqual(errors, []);
    console.log('PASS: canonical rows; real API status/edit/verify; sidebar/search/focus; player/speed; words/zoom; settings/models/logs; final-row clearance; 1440/1180/820/600/390/320 geometry.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
