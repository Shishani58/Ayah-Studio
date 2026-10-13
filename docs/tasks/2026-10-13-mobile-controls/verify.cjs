// Isolated fixture server; no generation or writes to user projects.
const { chromium } = require('/Users/ayub/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 820}, reducedMotion: 'reduce'});
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:8894');
    await page.waitForFunction(() => dashboardSuras.length === 114);
    await page.locator('#suraSearch').click();
    assert.equal(await page.locator('#suraSearch').evaluate(e => getComputedStyle(e).outlineStyle), 'none');
    await page.locator('#suraSearch').fill('114');
    await page.keyboard.press('Tab');
    assert.equal(await page.locator('#filterMenuToggle').evaluate(e => getComputedStyle(e).outlineStyle), 'solid');
    await page.locator('#suraSearch').fill('');
    await page.locator('#grid [data-surah="114"]').click();
    await page.waitForFunction(() => R.dataReady && R.audioReady && R.waveReady);
    for (const width of [1280, 820, 600, 390, 320]) {
      await page.setViewportSize({width, height: 820});
      for (const theme of ['light', 'dark']) {
        await page.evaluate(t => applyAppearance(t, 30), theme);
        await page.locator('#reviewToolsToggle').click();
        await page.waitForFunction(() => document.querySelector('.review-tools').open);
        await page.waitForTimeout(150); // details toggle event and opacity entrance
        const geometry = await page.locator('.review-actions').evaluate(e => {
          const r = e.getBoundingClientRect(); return {x:r.x, y:r.y, right:r.right, bottom:r.bottom};
        });
        assert.ok(geometry.x >= 0 && geometry.right <= width && geometry.y >= 0 && geometry.bottom <= 820, JSON.stringify({width,theme,geometry}));
        assert.equal(await page.locator('#reviewToolsToggle').evaluate(e => getComputedStyle(e).borderTopColor), 'rgba(0, 0, 0, 0)');
        assert.equal(await page.locator('#draftPausesBtn').evaluate(e => getComputedStyle(e).borderTopColor), 'rgba(0, 0, 0, 0)');
        if (width <= 600) assert.ok(geometry.bottom >= 807);
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        if ([1280,390].includes(width)) await page.screenshot({path: `${__dirname}/controls-${theme}-${width}.png`});
        await page.keyboard.press('Escape');
        assert.equal(await page.locator('#reviewToolsToggle').evaluate(e => e === document.activeElement), true);
        await page.keyboard.press('Tab');
        assert.equal(await page.locator('#showSuraLogBtn').evaluate(e => getComputedStyle(e).outlineStyle), 'solid');
        await page.locator('#settingsBtn').click();
        await page.waitForFunction(() => $('settingsDialog').open);
        assert.equal(await page.locator('#closeSettingsBtn').evaluate(e => getComputedStyle(e).outlineStyle), 'none');
        await page.locator('#closeSettingsBtn').click();
        await page.waitForFunction(() => !$('settingsDialog').open);
      }
    }
    const touch = await browser.newPage({viewport:{width:390,height:820}, hasTouch:true, isMobile:true});
    await touch.goto('http://127.0.0.1:8894');
    if (!await touch.locator('#suraSearch').isVisible()) await touch.locator('#sidebarToggle').tap();
    await touch.locator('#suraSearch').tap();
    assert.equal(await touch.locator('#suraSearch').evaluate(e => getComputedStyle(e).outlineStyle), 'none');
    await touch.locator('#filterMenuToggle').tap();
    await touch.waitForFunction(() => $('filterMenu').open);
    const bounds = await touch.locator('#filterPicker').boundingBox();
    assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 390);
    assert.deepEqual(errors, []);
    console.log('PASS: pointer/touch, keyboard focus, Escape, light/dark, menus within viewport at 1280/820/600/390/320; no JS errors.');
  } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exit(1);});
