/* Isolated browser checks. Reads production methods/CSS; never imports ComfyUI. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '../..');
// Set this to an older checkout to add the three baseline comparisons.
// The default run only exercises the current repository (12 checks).
const baselineRoot = process.env.CONTINUITY_BASELINE_DIR
  ? path.resolve(process.env.CONTINUITY_BASELINE_DIR) : null;
const sharedRoot = root;
const read = (file) => fs.readFileSync(file, 'utf8');
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const dom = read(path.join(sharedRoot, 'web/creator/dom.js'));
const helper = (name) => {
  const match = dom.match(new RegExp(`export function ${name}\\([^]*?^\\}`, 'm'));
  assert.ok(match, `Missing real DOM helper ${name}`);
  return match[0].replace(/^export /, '');
};
const atlas = read(path.join(sharedRoot, 'web/creator/presets/atlas.js'));
const categories = JSON.parse(atlas.match(/export const CATEGORIES = (\[[^]*?\]);/)[1].replace(/,\s*\]/, ']'));
const results = [];
const sourceHashes = {
  'shared/web/creator/dom.js': hash(path.join(sharedRoot, 'web/creator/dom.js')),
  'shared/web/creator/presets/atlas.js': hash(path.join(sharedRoot, 'web/creator/presets/atlas.js')),
};
let browser;
let page;
let requests = 0;

function method(source, name, optional = false) {
  const match = source.match(new RegExp(`^  ${name}\\([^]*?^  \\}`, 'm'));
  if (!optional) assert.ok(match, `Missing actual method ${name}`);
  return match?.[0] || '';
}

function fixture(version) {
  const sourceRoot = version === 'baseline' ? baselineRoot : root;
  const sourceLabel = version;
  const jsPath = path.join(sourceRoot, 'web/creator/presetlib.js');
  const source = read(jsPath);
  sourceHashes[`${sourceLabel}/web/creator/presetlib.js`] = hash(jsPath);
  const names = ['mount', 'pool', 'folders', 'renderShelves', 'visible'];
  const methods = [...names.map((name) => method(source, name)), ...['scrollShelves', 'keyShelves'].map((name) => method(source, name, true))].join('\n');
  const css = ['base', 'picker', 'presets'].map((name) => {
    const file = path.join(sourceRoot, `web/creator/styles/${name}.js`);
    sourceHashes[`${sourceLabel}/web/creator/styles/${name}.js`] = hash(file);
    const match = read(file).match(/export const css = `([^]*)`;\s*$/);
    assert.ok(match, `Missing actual CSS ${file}`);
    return match[1];
  }).join('\n');
  return { css, code: `
    ${helper('el')}
    ${helper('mountOverlay')}
    const t = (text) => text;
    const P = { SCOPES: ['style'], SCOPE_LABEL: { style: 'Style' } };
    const SHELF_ALL = 'all', SHELF_FAV = 'fav';
    class Subject {
      ${methods}
      renderBar() { this.bar.replaceChildren(this.search); }
      renderInspector() {}
      readAtlas() {}
      load() { this.renderShelves(); this.renderGrid(); }
      close() { this.unmount(); }
      renderGrid() {
        this.gridRenders++;
        this.filtered = this.visible().map((row) => row.name);
        this.grid.replaceChildren(...Array.from({ length: 40 }, (_, index) => el('div', {
          text: 'Fixture card ' + index, style: { height: '100px' }
        })));
      }
    }
    window.subject = Object.assign(new Subject(), {
      scope: 'style', query: '', shelf: 'all', gridRenders: 0, rows: [],
      styles: ${JSON.stringify(categories.map((folder, index) => ({ scope: 'style', folder, name: `Style ${index}`, starred: index === 0 })))}
    });
    window.wheelLog = [];
    const observeWheel = (event) => {
      const record = { deltaX: event.deltaX, deltaY: event.deltaY, deltaMode: event.deltaMode, ctrlKey: event.ctrlKey, target: event.target.className };
      record.prevented = event.defaultPrevented;
      window.wheelLog.push(record);
    };
    subject.mount();
    subject.shelfRow.addEventListener('wheel', observeWheel, { passive: true });
    subject.grid.addEventListener('wheel', observeWheel, { passive: true });
  ` };
}

async function load(version) {
  const data = fixture(version);
  new (require('node:vm').Script)(data.code, { filename: `${version}-extracted-fixture.js` });
  await page.goto('about:blank');
  await page.setContent(`<html><head><style>${data.css}</style></head><body></body></html>`);
  await page.evaluate(`(() => { ${data.code} })()`);
  await page.waitForFunction(() => subject.shelfRow.scrollWidth > subject.shelfRow.clientWidth);
}

async function state() {
  return page.evaluate(() => {
    const strip = subject.shelfRow;
    const focused = document.activeElement;
    const r = strip.getBoundingClientRect();
    const f = focused.getBoundingClientRect();
    return {
      left: strip.scrollLeft, room: strip.scrollWidth - strip.clientWidth, width: strip.clientWidth,
      active: [...strip.children].indexOf(focused), count: strip.children.length,
      focusVisible: focused.parentElement === strip && f.left >= r.left - 1 && f.right <= r.right + 1,
      shelf: subject.shelf, renders: subject.gridRenders, filtered: subject.filtered,
      gridTop: subject.grid.scrollTop, wheel: wheelLog.at(-1)
    };
  });
}

async function setLeft(left) {
  await page.evaluate((value) => subject.shelfRow.scrollTo({ left: value, behavior: 'instant' }), left);
}

async function waitForFocusedChip() {
  await page.waitForFunction(() => {
    const strip = subject.shelfRow;
    const active = document.activeElement;
    const r = strip.getBoundingClientRect();
    const f = active.getBoundingClientRect();
    return active.parentElement === strip && f.left >= r.left - 1 && f.right <= r.right + 1;
  }, null, { timeout: 3000 });
}

async function realWheel(x, y, target = '.mmc-shelf-strip') {
  const box = await page.locator(target).boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  const previous = await page.evaluate(() => wheelLog.length);
  await page.mouse.wheel(x, y);
  await page.waitForFunction((count) => wheelLog.length > count, previous);
  await page.waitForTimeout(180);
  return state();
}

async function syntheticWheel(values) {
  return page.evaluate((init) => {
    const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, ...init });
    const before = subject.shelfRow.scrollLeft;
    subject.shelfRow.dispatchEvent(event);
    return { before, after: subject.shelfRow.scrollLeft, prevented: event.defaultPrevented };
  }, values);
}

async function check(name, callback) {
  try {
    const evidence = await callback();
    results.push({ name, passed: true, evidence });
    process.stdout.write(`PASS ${name}\n`);
  } catch (error) {
    results.push({ name, passed: false, error: error.stack });
    process.stdout.write(`FAIL ${name}: ${error.message}\n`);
  }
}

(async () => {
  browser = await chromium.launch({ executablePath: process.env.BROWSER_EXECUTABLE_PATH || undefined, headless: true, args: ['--disable-background-networking', '--disable-default-apps', '--no-first-run'] });
  const context = await browser.newContext({ viewport: { width: 700, height: 700 } });
  await context.route('**/*', (route) => { requests++; return route.abort(); });
  page = await context.newPage();
  if (baselineRoot) {
  await load('baseline');
  await check('baseline: plain vertical wheel reproduces missing horizontal movement', async () => {
    const before = await state();
    const after = await realWheel(0, 120);
    assert.equal(after.left, before.left);
    assert.equal(after.wheel.prevented, false);
    return { before: before.left, after: after.left, overflow: after.room };
  });
  await check('baseline: native horizontal wheel already works', async () => {
    const after = await realWheel(120, 0);
    assert.ok(after.left > 0);
    assert.equal(after.wheel.prevented, false);
    return { left: after.left, prevented: after.wheel.prevented };
  });
  await check('baseline: native Tab can reach and reveal the last chip', async () => {
    await setLeft(0);
    await page.locator('.mmc-shelf').first().focus();
    for (let i = 1; i < categories.length + 2; i++) await page.keyboard.press('Tab');
    await waitForFocusedChip();
    const after = await state();
    assert.equal(after.active, after.count - 1);
    assert.ok(after.focusVisible);
    return { active: after.active, left: after.left, focusVisible: after.focusVisible };
  });

  }
  await load('candidate');
  await check('candidate: real vertical wheel moves right then left', async () => {
    const forward = await realWheel(0, 140);
    assert.ok(forward.left > 0);
    assert.equal(forward.wheel.prevented, true);
    const backward = await realWheel(0, -80);
    assert.ok(backward.left < forward.left);
    assert.equal(backward.wheel.prevented, true);
    return { forward: forward.left, backward: backward.left };
  });
  await check('candidate: native horizontal and diagonal trackpad input remains uncancelled', async () => {
    await setLeft(0);
    const horizontal = await realWheel(100, 0);
    assert.ok(horizontal.left > 0);
    assert.equal(horizontal.wheel.prevented, false);
    const diagonal = await realWheel(60, 30);
    assert.ok(diagonal.left > horizontal.left);
    assert.equal(diagonal.wheel.prevented, false);
    return { horizontal: horizontal.left, diagonal: diagonal.left };
  });
  await check('candidate: Ctrl/Meta/Alt wheels and already-cancelled events pass through', async () => {
    for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) {
      const out = await syntheticWheel({ deltaY: 50, [modifier]: true });
      assert.equal(out.after, out.before);
      assert.equal(out.prevented, false);
    }
    const cancelled = await page.evaluate(() => {
      const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 90 });
      event.preventDefault();
      const before = subject.shelfRow.scrollLeft;
      subject.shelfRow.dispatchEvent(event);
      return subject.shelfRow.scrollLeft === before;
    });
    assert.ok(cancelled);
    return { modifiers: ['ctrlKey', 'metaKey', 'altKey'], alreadyCancelledUnchanged: cancelled };
  });
  await check('candidate: line and page delta units are normalized', async () => {
    await setLeft(0);
    await page.evaluate(() => { subject.shelfRow.style.lineHeight = '24px'; });
    const line = await syntheticWheel({ deltaY: 2, deltaMode: 1 });
    assert.equal(line.after, 48);
    assert.ok(line.prevented);
    await setLeft(0);
    const initial = await state();
    const paged = await syntheticWheel({ deltaY: 1, deltaMode: 2 });
    assert.equal(paged.after, Math.min(initial.width, initial.room));
    assert.ok(paged.prevented);
    await page.evaluate(() => { subject.shelfRow.style.lineHeight = ''; });
    return { linePixels: line.after, pagePixels: paged.after, viewportWidth: initial.width };
  });
  await check('candidate: limits and nonoverflow do not consume wheel', async () => {
    await setLeft(0);
    const atStart = await syntheticWheel({ deltaY: -100 });
    assert.equal(atStart.prevented, false);
    await setLeft(100000);
    const atEnd = await syntheticWheel({ deltaY: 100 });
    assert.equal(atEnd.prevented, false);
    await page.evaluate(() => { window.savedStyles = subject.styles; subject.styles = []; subject.renderShelves(); });
    const fitting = await state();
    assert.equal(fitting.room, 0);
    const noOverflow = await syntheticWheel({ deltaY: 100 });
    assert.equal(noOverflow.prevented, false);
    await page.evaluate(() => { subject.styles = savedStyles; subject.renderShelves(); });
    return { startPrevented: atStart.prevented, endPrevented: atEnd.prevented, fittingPrevented: noOverflow.prevented };
  });
  await check('candidate enhancement: Home/End/Arrow keys reveal focused chips without filtering', async () => {
    await page.locator('.mmc-shelf').first().focus();
    const before = await state();
    await page.keyboard.press('End');
    const end = await state();
    assert.equal(end.active, end.count - 1);
    assert.ok(end.focusVisible);
    await page.keyboard.press('ArrowLeft');
    const left = await state();
    assert.equal(left.active, left.count - 2);
    assert.ok(left.focusVisible);
    await page.keyboard.press('Home');
    const home = await state();
    assert.equal(home.active, 0);
    assert.ok(home.focusVisible);
    await page.keyboard.press('ArrowRight');
    const right = await state();
    assert.equal(right.active, 1);
    assert.equal(right.shelf, before.shelf);
    assert.equal(right.renders, before.renders);
    return { endIndex: end.active, leftIndex: left.active, homeIndex: home.active, rightIndex: right.active, filter: right.shelf };
  });
  await check('candidate enhancement: native Enter/Space activate filters and retain focus', async () => {
    await page.keyboard.press('End');
    const beforeEnter = await state();
    await page.keyboard.press('Enter');
    const entered = await state();
    assert.equal(entered.active, beforeEnter.active);
    assert.equal(entered.renders, beforeEnter.renders + 1);
    assert.equal(entered.shelf, categories.slice().sort().at(-1));
    assert.equal(entered.filtered.length, 1);
    assert.equal(entered.left, beforeEnter.left);
    await page.keyboard.press('ArrowLeft');
    const beforeSpace = await state();
    await page.keyboard.press('Space');
    const spaced = await state();
    assert.equal(spaced.active, beforeSpace.active);
    assert.equal(spaced.renders, beforeSpace.renders + 1);
    assert.equal(spaced.shelf, categories.slice().sort().at(-2));
    assert.equal(spaced.filtered.length, 1);
    assert.equal(spaced.left, beforeSpace.left);
    return { enter: entered.shelf, space: spaced.shelf, focusIndices: [entered.active, spaced.active] };
  });
  await check('candidate: Tab and Shift+Tab keep their native route', async () => {
    await page.locator('.mmc-shelf').first().focus();
    await page.keyboard.press('Tab');
    assert.equal((await state()).active, 1);
    await page.keyboard.press('Shift+Tab');
    assert.equal((await state()).active, 0);
    for (let i = 1; i < categories.length + 2; i++) await page.keyboard.press('Tab');
    await waitForFocusedChip();
    const end = await state();
    assert.equal(end.active, end.count - 1);
    assert.ok(end.focusVisible);
    return { lastIndex: end.active, left: end.left };
  });
  await check('candidate: rerenders do not multiply the wheel listener', async () => {
    await page.evaluate(() => { for (let i = 0; i < 15; i++) subject.renderShelves(); });
    await setLeft(0);
    const after = await syntheticWheel({ deltaY: 23 });
    assert.equal(after.after, 23);
    assert.ok(after.prevented);
    return { rerenders: 15, deltaY: 23, scrollLeft: after.after };
  });
  await check('candidate: vertical wheel over the grid still scrolls the grid', async () => {
    const before = await state();
    const after = await realWheel(0, 180, '.mmc-preset-grid');
    assert.ok(after.gridTop > before.gridTop);
    assert.equal(after.left, before.left);
    assert.equal(after.wheel.prevented, false);
    return { gridBefore: before.gridTop, gridAfter: after.gridTop, shelfBefore: before.left, shelfAfter: after.left };
  });
  await check('candidate: RTL wheel and physical arrow directions', async () => {
    await page.evaluate(() => { subject.shelfRow.style.direction = 'rtl'; });
    await setLeft(0);
    const moved = await syntheticWheel({ deltaY: 100 });
    assert.equal(moved.after, -100);
    assert.ok(moved.prevented);
    await page.locator('.mmc-shelf').first().focus();
    await page.keyboard.press('ArrowLeft');
    assert.equal((await state()).active, 1);
    await page.keyboard.press('ArrowRight');
    assert.equal((await state()).active, 0);
    await page.keyboard.press('End');
    const end = await state();
    assert.equal(end.active, end.count - 1);
    assert.ok(end.focusVisible);
    await page.evaluate(() => { subject.shelfRow.style.direction = ''; });
    return { wheelLeft: moved.after, endIndex: end.active, endVisible: end.focusVisible };
  });
  await check('candidate: scrollbar exception is local to the preset strip', async () => {
    const styles = await page.evaluate(() => {
      const regular = document.createElement('div');
      regular.className = 'mmc-shelf-strip';
      subject.shelves.appendChild(regular);
      const result = { preset: getComputedStyle(subject.shelfRow).scrollbarWidth, regular: getComputedStyle(regular).scrollbarWidth };
      regular.remove();
      return result;
    });
    assert.equal(styles.preset, 'thin');
    assert.equal(styles.regular, 'none');
    return styles;
  });
  assert.equal(requests, 0, 'Fixture attempted a network request');
  await browser.close();
  const failed = results.filter((row) => !row.passed).length;
  const report = { generatedAt: new Date().toISOString(), candidateRoot: root, baselineRoot, sharedRoot, browser: 'Chromium-based browser headless, fresh Playwright context', requests, extraction: 'Actual mount/pool/folders/renderShelves/visible/scrollShelves/keyShelves and DOM helpers; all three CSS modules unchanged. Only catalogue loading and card rendering use local fixtures.', limitation: 'Isolated browser DOM test; no running ComfyUI, user profile, live trackpad hardware, or integration proof. Keyboard support is an enhancement beyond the confirmed vertical-wheel omission. Twelve checks exercise the candidate; an optional baseline checkout adds three baseline comparisons.', sourceHashes, passed: results.length - failed, failed, results };
  process.stdout.write(JSON.stringify(report, null, 2) + '\n');
  process.exitCode = failed ? 1 : 0;
})().catch(async (error) => {
  process.stderr.write(error.stack + '\n');
  if (browser) await browser.close();
  process.exitCode = 1;
});
