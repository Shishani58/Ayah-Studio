// Real API, canonical text, disposable fixture server on 8878; no model.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 940}});
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:8878');
    const data = await (await page.request.get('http://127.0.0.1:8878/api/sura/114/ayah/1/words')).json();
    const editUrl = 'http://127.0.0.1:8878/api/sura/114/ayah/1/words/edit';
    for (let i = 0; i < data.words.length; i++) {
      const response = await page.request.post(editUrl, {data: {db_path: data.db_path,
        start_ms: data.start_ms, end_ms: data.end_ms, word_index: data.words[i].word_index, time: 100 + i * 300}});
      assert.equal(response.ok(), true);
    }
    await page.waitForFunction(() => dashboardSuras.length === 114);
    await page.locator('#grid button[data-surah="114"]').click();
    await page.waitForFunction(() => R.dataReady && R.waveReady && R.audioReady);
    await page.evaluate(() => applyReviewMode('words'));
    await page.waitForFunction(() => W.data && !W.loading);
    let card = page.locator('.word-card').first();
    assert.equal(await page.locator('.word-card input:visible').count(), 0);
    const cardBounds = await card.boundingBox(); assert.ok(cardBounds.height < 160);
    await card.locator('.a-num').click();
    const editor = () => page.locator('.word-editor[open]');
    const selected = () => editor().locator('[data-word-status][aria-pressed="true"]').getAttribute('data-word-status');
    for (const status of ['verified', 'edited', 'auto']) {
      await editor().locator(`[data-word-status="${status}"]`).click();
      await page.waitForFunction(s => !W.busy && W.data.words[0].status === s, status);
      assert.equal(await editor().count(), 1); assert.equal(await selected(), status);
      const saved = await (await page.request.get('http://127.0.0.1:8878/api/sura/114/ayah/1/words')).json();
      assert.equal(saved.words[0].time, 100); assert.equal(saved.words[0].status, status);
    }
    await page.route('**/words/edit', r => r.fulfill({status: 409, json: {detail: 'test status failure'}}));
    await editor().locator('[data-word-status="verified"]').click();
    await page.waitForFunction(() => !W.busy && $('workspaceStatus').textContent.includes('test status failure'));
    assert.equal(await selected(), 'auto');
    assert.match(await editor().locator('.ayah-time-error').textContent(), /test status failure/);
    await page.unroute('**/words/edit');
    await editor().locator('[data-word-status="verified"]').focus();
    await editor().locator('[data-word-status="verified"]').press('Space');
    await page.waitForFunction(() => !W.busy && W.data.words[0].status === 'verified');
    assert.equal(await page.evaluate(() => document.activeElement.dataset.wordStatus), 'verified');
    await editor().locator('input').fill('0.150'); await editor().locator('input').press('Enter');
    await page.waitForFunction(() => !W.busy && W.data.words[0].time === 150);
    assert.equal(await selected(), 'edited');
    await page.evaluate(() => applyAppearance('dark', 30));
    await page.screenshot({path: __dirname + '/compact-dark.png'});
    await editor().screenshot({path: __dirname + '/editor-dark.png'});
    await page.evaluate(() => applyAppearance('light', 30));
    await editor().screenshot({path: __dirname + '/editor-light.png'});
    await editor().locator('input').press('Escape');
    await page.waitForFunction(() => !document.querySelector('.word-editor[open]') && document.activeElement === document.querySelector('.word-card .a-num'));
    assert.equal(await card.locator('.a-num').evaluate(e => e === document.activeElement), true);
    await page.evaluate(() => loadWords()); await page.waitForFunction(() => !W.loading && W.data);
    card = page.locator('.word-card').first(); await card.locator('.a-num').click();
    assert.equal(await selected(), 'edited'); assert.equal(await editor().locator('input').inputValue(), '0.150');
    await editor().locator('input').press('Escape');
    await page.locator('#sidebarToggle').click(); await page.waitForFunction(() => $('suraSidebar').hidden);
    for (const width of [820, 390, 320]) {
      await page.setViewportSize({width, height: 940}); await card.locator('.a-num').click();
      const rect = await editor().boundingBox(); assert.ok(rect.x >= 0 && rect.x + rect.width <= width);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      await editor().locator('input').press('Escape');
    }
    const missing = page.locator('.word-card').nth(data.words.length);
    await missing.locator('.a-num').click();
    assert.equal(await editor().locator('[data-word-status]:enabled').count(), 0);
    await editor().locator('.close-time').click();
    const timing = await page.request.post('http://127.0.0.1:8878/api/timing', {data: {surah:114, ayah:2, time:2100, db_path:data.db_path}});
    assert.equal(timing.ok(), true);
    await page.evaluate(() => loadWords()); await page.waitForFunction(() => !W.loading && W.data);
    await card.locator('.a-num').click();
    assert.equal(await editor().locator('[data-word-status]:enabled').count(), 0);
    assert.equal(await editor().locator('[data-word-status][aria-pressed="true"]').count(), 0);
    assert.match(await editor().locator('.ayah-time-help').textContent(), /Границы изменились/);
    await editor().locator('input').fill('0.160'); await editor().locator('input').press('Enter');
    await page.waitForFunction(() => !W.busy && !W.data.words[0].stale && W.data.words[0].time === 160);
    assert.equal(await editor().locator('[data-word-status]:enabled').count(), 3);
    assert.deepEqual(errors, []);
    console.log('PASS: compact word cards, real status/time API, reload, failures, stale/missing protection, keyboard/focus, themes, 1440/820/390/320.');
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exit(1)});
