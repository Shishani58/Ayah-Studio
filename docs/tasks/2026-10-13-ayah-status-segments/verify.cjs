// Disposable fixture server on 8878; no user data or alignment model.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 940}});
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:8878');
    await page.waitForFunction(() => dashboardSuras.length === 114);
    await page.locator('#grid button[data-surah="114"]').click();
    await page.waitForFunction(() => R.dataReady && R.audioReady && R.waveReady);
    let row = page.locator('#ayahList > li').nth(1);
    await row.locator('.a-num').click();
    const segments = row.locator('.ayah-status-segments');
    const selected = () => segments.locator('[aria-pressed="true"]').getAttribute('data-ayah-status');
    for (const status of ['verified', 'edited', 'auto']) {
      await segments.locator(`[data-ayah-status="${status}"]`).click();
      await page.waitForFunction(s => !editBusy && R.ayahs[1].status === s, status);
      assert.equal(await row.locator('dialog').isVisible(), true);
      assert.equal(await selected(), status);
      assert.equal(await row.getAttribute('data-timing-status'), status === 'verified' ? 'verified' : 'uncertain');
      const data = await (await page.request.get('http://127.0.0.1:8878/api/sura/114')).json();
      assert.equal(data.ayahs[1].status, status); assert.equal(data.ayahs[1].time, 2000);
    }
    await page.route('**/api/verify', r => r.fulfill({status: 409, json: {detail: 'test status failure'}}));
    await segments.locator('[data-ayah-status="verified"]').click();
    await page.waitForFunction(() => !editBusy && $('workspaceStatus').textContent.includes('test status failure'));
    assert.equal(await selected(), 'auto');
    assert.match(await row.locator('.ayah-time-error').textContent(), /test status failure/);
    await page.unroute('**/api/verify');
    await segments.locator('[data-ayah-status="verified"]').click();
    await page.waitForFunction(() => !editBusy && R.ayahs[1].status === 'verified');
    await row.locator('input').fill('00:02.100'); await row.locator('input').press('Enter');
    await page.waitForFunction(() => !editBusy && R.ayahs[1].time === 2100);
    assert.equal(await selected(), 'edited');
    await segments.locator('[data-ayah-status="verified"]').focus();
    await segments.locator('[data-ayah-status="verified"]').press('Space');
    await page.waitForFunction(() => !editBusy && R.ayahs[1].status === 'verified');
    await segments.locator('[data-ayah-status="edited"]').focus();
    await segments.locator('[data-ayah-status="edited"]').press('Enter');
    await page.waitForFunction(() => !editBusy && R.ayahs[1].status === 'edited');
    await segments.locator('[data-ayah-status="edited"]').press('Escape');
    assert.equal(await row.locator('.a-num').evaluate(e => e === document.activeElement), true);
    await page.reload(); await page.waitForFunction(() => dashboardSuras.length === 114);
    await page.locator('#grid button[data-surah="114"]').click();
    await page.waitForFunction(() => R.dataReady && R.audioReady && R.waveReady);
    row = page.locator('#ayahList > li').nth(1);
    await row.locator('.a-num').click(); assert.equal(await row.locator('[aria-pressed="true"][data-ayah-status]').getAttribute('data-ayah-status'), 'edited');
    await row.locator('dialog').screenshot({path: __dirname + '/status-light.png'});
    await page.evaluate(() => applyAppearance('dark', 30));
    await row.locator('dialog').screenshot({path: __dirname + '/status-dark.png'});
    await row.locator('input').press('Escape');
    await page.locator('#sidebarToggle').click();
    await page.waitForFunction(() => $('suraSidebar').hidden);
    for (const width of [820, 390, 320]) {
      await page.setViewportSize({width, height: 940});
      await row.locator('.a-num').click();
      const rect = await row.locator('dialog').boundingBox();
      assert.ok(rect.x >= 0 && rect.x + rect.width <= width);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      await row.locator('input').press('Escape');
    }
    const missing = page.locator('#ayahList > li').nth(3);
    await missing.locator('.a-num').click();
    assert.equal(await missing.locator('button[data-ayah-status="verified"]').isDisabled(), true);
    assert.equal(await missing.locator('button[data-ayah-status="edited"]').isDisabled(), true);
    await missing.locator('input').fill('00:06.000'); await missing.locator('input').press('Enter');
    await page.waitForFunction(() => !editBusy && R.ayahs[3].time === 6000);
    assert.equal(await missing.locator('button[data-ayah-status="verified"]').isDisabled(), false);
    assert.equal(await missing.locator('[aria-pressed="true"][data-ayah-status]').getAttribute('data-ayah-status'), 'edited');
    assert.deepEqual(errors, []);
    console.log('PASS: status round trips, preserved time, reload, failed save rollback, edit resets verification, missing timing, focus/Escape, themes and 1440/820/390/320 geometry.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
