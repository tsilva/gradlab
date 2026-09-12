// Mounted integration checks. All interaction uses the production DOM and server.
// Fixture controls affect external preparation/transport only, never app callbacks.
const $ = selector => document.querySelector(selector);
const assert = (condition, message) => { if (!condition) throw Error(message); };
const wait = (predicate, message) => new Promise((resolve, reject) => {
  const deadline = performance.now() + 15000;
  function poll() {
    if (predicate()) return resolve();
    if (performance.now() > deadline) return reject(Error(message));
    requestAnimationFrame(poll);
  }
  poll();
});
const token = new URLSearchParams(location.hash.slice(1)).get('token');
const control = action => fetch(`/assets/fixture-control?action=${action}`, {
  headers: { Authorization: `Bearer ${token}` },
}).then(response => response.json());
const mask = () => !$('#checkpoint-loading-mask').hidden;
const rows = () => [...document.querySelectorAll('#source-browser tr[role="button"]')];
const click = selector => { const element = $(selector); assert(element, `Missing ${selector}`); element.click(); };
const waitStatus = async (predicate, message) => {
  const deadline = performance.now() + 20000;
  while (performance.now() < deadline) {
    const status = await control('status');
    if (predicate(status)) return status;
    await new Promise(requestAnimationFrame);
  }
  throw Error(message);
};
const ready = async () => {
  await waitStatus(status => status.phase === 'active' && !status.preparing, 'server activation did not settle');
  await wait(() => !mask() && $('#source-browser').hidden && $('#game-canvas')?.width === 3, 'frame not presented');
};
const list = () => wait(() => !$('#source-browser').hidden && rows().length === 2, 'checkpoint list not mounted');
const result = document.createElement('pre');
result.id = 'selection-test-results';
$('#fixture-controls').append(result);
const pass = text => { result.textContent += `PASS ${text}\n`; };

export async function runChecks() {
  result.textContent = '';
  try {
    await control('reset');
    await list();
    const path = location.pathname;
    await control('observer');
    await new Promise(requestAnimationFrame);
    rows()[0].click();
    assert(!mask() && location.pathname === path, 'observer refusal changed selection');
    assert((await control('status')).prepared.length === 0, 'observer dispatched preparation');
    pass('observer refusal preserves route and mask');
    await control('control');
    // A server status round trip fences the control update before selection.
    await wait(() => !rows()[0]?.classList.contains('disabled'), 'control not restored');
    await control('reject');
    rows()[0].click();
    await wait(() => $('#toast').textContent === 'Fixture command rejected' && !mask(), 'rejection did not clear mask');
    assert(location.pathname.includes('/checkpoints/checkpoint-100-'), 'rejection rolled route back');
    pass('command rejection releases loading and retains selected route');
    // Existing rejection leaves the table mounted; the list method allows another selection.
    rows()[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await wait(mask, 'keyboard did not initiate load');
    assert(document.body.getAttribute('aria-busy') === 'true', 'missing busy indication');
    assert($('#checkpoint-loading-mask').getAttribute('role') === 'status', 'missing status role');
    assert($('#checkpoint-loading-mask').getAttribute('aria-live') === 'polite', 'missing live status');
    await control('fail');
    await wait(() => !mask() && $('#source-browser').textContent.includes('Fixture preparation failed'), 'initial failure not visible');
    pass('keyboard selection and asynchronous failure expose accessible state');
    const retry = [...document.querySelectorAll('#source-browser button')].find(button => button.textContent === 'Retry');
    retry.click();
    await wait(() => $('#source-browser').textContent.includes('Waiting for fixture preparation'), 'retry not preparing');
    assert(!mask(), 'preparation Retry started tracked mask');
    await waitStatus(status => status.phase === 'loading', 'preparation not admitted');
    await control('succeed');
    await ready();
    assert((await control('status')).paused === 'paused', 'selection ran inference');
    pass('preparation Retry activates paused playback without tracked mask');
    await wait(() => !$('[data-checkpoint-next]').disabled, 'adjacent navigation unavailable');
    await control('hold-frames');
    const beforeAdjacent = (await control('status')).prepared.length;
    click('[data-checkpoint-next]');
    assert(mask() && $('[data-checkpoint-next]').disabled, 'adjacent pending guard missing');
    while ((await control('status')).prepared.length === beforeAdjacent) await new Promise(requestAnimationFrame);
    const count = beforeAdjacent + 1;
    click('[data-checkpoint-next]');
    assert((await control('status')).prepared.length === count, 'duplicate adjacent command');
    await waitStatus(status => status.phase === 'loading', 'preparation not admitted');
    await control('succeed');
    await wait(() => $('#source-browser').hidden && $('[data-checkpoint-position]').textContent === '2 / 2', 'activation did not reconcile navigation');
    assert(mask(), 'activation cleared mask before applicable frame');
    assert(!$('[data-checkpoint-previous]').disabled, 'navigation readiness tied to mask');
    await control('release-frames');
    await ready();
    assert(document.body.getAttribute('aria-busy') === null, 'busy indication not released');
    const pixel = $('#game-canvas').getContext('2d').getImageData(0, 0, 1, 1).data[0];
    assert(Math.abs(pixel - 203) < 4, `wrong checkpoint frame ${pixel}`);
    pass('adjacent guard, separate activation/presentation, exact checkpoint frame');
    click('#source-back');
    await list();
    const browsePath = location.pathname;
    const before = await control('status');
    await control('observer');
    await control('control');
    assert(location.pathname === browsePath && !$('#source-browser').hidden, 'background snapshot left browsing');
    assert((await control('status')).commands.filter(name => name === 'browse_sources').length === before.commands.filter(name => name === 'browse_sources').length, 'local browse closed runner');
    pass('local browsing survives background control snapshots without closing runner');
    rows()[0].click();
    await wait(mask, 'replacement not tracked');
    await control('fail');
    await wait(() => !mask(), 'replacement failure retained mask');
    const failed = await control('status');
    assert(failed.epoch === before.epoch, 'failed replacement discarded previous session');
    pass('failed replacement retains the previous Playback Session');
    result.dataset.status = 'passed';
  } catch (error) {
    result.textContent += `FAIL ${error.stack}\n`;
    result.dataset.status = 'failed';
  }
}

export async function runAdditionalChecks() {
  result.textContent = '';
  try {
    await control('reset');
    await list();
    rows()[0].click();
    await wait(mask, 'table selection not tracked');
    await waitStatus(status => status.phase === 'loading', 'preparation not admitted');
    await control('succeed');
    await ready();
    const activePath = location.pathname;
    click('#source-back');
    await list();
    const runPath = location.pathname;
    [...document.querySelectorAll('#source-breadcrumbs button')].find(button => button.textContent === 'Environments').click();
    await wait(() => location.pathname === '/', 'home route not restored');
    history.back();
    await wait(() => location.pathname === runPath, 'Back did not restore Run route');
    await list();
    history.forward();
    await wait(() => location.pathname === '/', 'Forward did not restore Environments');
    history.back();
    await list();
    assert((await control('status')).commands.filter(name => name === 'browse_sources').length === 0, 'history browsing closed active runner');
    pass('Back/Forward preserve local discovery routes and active session');
    rows()[0].click();
    await wait(mask, 'same checkpoint selection not tracked');
    await waitStatus(status => status.phase === 'loading', 'preparation not admitted');
    await control('succeed');
    await ready();
    assert(location.pathname === activePath, 'same checkpoint identity changed');
    pass('same checkpoint reselect activates a fresh paused session');
    const slider = $('[aria-label="Inspect an episode step"]');
    slider.value = '102';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.dispatchEvent(new Event('change', { bubbles: true }));
    await wait(() => document.body.textContent.includes('STEP 102'), 'inspection cursor did not move');
    assert((await control('status')).paused === 'paused', 'inspection ran inference');
    click('[data-checkpoint-next]');
    await wait(mask, 'selection during inspection not tracked');
    await waitStatus(status => status.phase === 'loading', 'preparation not admitted');
    await control('succeed');
    await ready();
    pass('inspection stays paused and replacement presents the new session');
    // Hide RGB through the real control; a fresh source can re-enable RGB according to server policy.
    click('[aria-label="Hide RGB and play at maximum speed"]');
    await wait(() => $('#game-canvas').hidden, 'RGB did not hide');
    assert((await control('status')).paused === 'paused', 'hiding RGB started inference');
    click('[data-checkpoint-previous]');
    await wait(mask, 'RGB-hidden selection not tracked');
    await waitStatus(status => status.phase === 'loading', 'preparation not admitted');
    await control('succeed');
    await ready();
    pass('selection begun with RGB hidden completes under server activation settings');
    const beforeImport = await control('status');
    await control('import');
    await wait(() => $('[data-checkpoint-position]').closest('nav').hidden && document.body.textContent.includes('inspection only'), 'imported playback not mounted');
    const imported = await control('status');
    assert(imported.mode === 'trajectory' && imported.activations === beforeImport.activations, 'import executed its checkpoint attachment');
    assert(!mask(), 'import retained checkpoint loading');
    pass('import activates recorded Playback without executing its attachment');
    result.dataset.status = 'passed';
  } catch (error) {
    result.textContent += `FAIL ${error.stack}\n`;
    result.dataset.status = 'failed';
  }
}
