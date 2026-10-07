const $ = selector => document.querySelector(selector);
const assert = (condition, message) => { if (!condition) throw Error(message); };
const wait = (predicate, label) => new Promise((resolve, reject) => {
  const deadline = performance.now() + 15000;
  function poll() { if (predicate()) resolve(); else if (performance.now() > deadline) reject(Error(label)); else requestAnimationFrame(poll); }
  poll();
});
const token = new URLSearchParams(location.hash.slice(1)).get('token');
const control = action => fetch(`/assets/fixture-control?action=${action}`, { headers: { Authorization: `Bearer ${token}` } }).then(response => response.json());
const button = name => [...$('#inspection-fixture').querySelectorAll('button')].find(node => node.textContent === name).click();
const gate = name => JSON.parse($('#inspection-gate-status').textContent)[name];
const seek = step => { $('#timeline-scrubber').value = String(step); $('#timeline-scrubber').dispatchEvent(new Event('input', { bubbles: true })); };
const selected = step => $('#timeline-label').textContent.includes(`STEP ${step}`) && $('#timeline').getAttribute('aria-busy') === 'false';
const pixel = () => $('#game-canvas').getContext('2d').getImageData(0, 0, 1, 1).data[0];
const result = () => $('#inspection-test-results');
const pass = text => { result().textContent += `PASS ${text}\n`; };

export async function runChecks() {
  result().textContent = ''; result().dataset.status = 'running';
  try {
    await control('reset');
    await wait(() => !$('#source-browser').hidden && $('#source-browser tr[role="button"]'), 'source list');
    $('#source-browser tr[role="button"]').click();
    await wait(() => !$('#checkpoint-loading-mask').hidden, 'loading mask');
    await control('succeed');
    await wait(() => selected(103) && $('#game-canvas')?.width === 3 && $('#checkpoint-loading-mask').hidden, 'initial frame');
    assert((await control('status')).paused === 'paused', 'load must pause inference');
    pass('Policy opens paused with its exact frame');

    // Exercise the mounted settings form through actual DOM events and observe
    // commands at the production host boundary, without advancing the episode.
    $('#playback-settings-toggle').click();
    await wait(() => !$('#playback-settings-menu').hidden, 'settings menu');
    const seed = $('#playback-settings-content [data-seed]');
    seed.value = '41414'; seed.dispatchEvent(new Event('input', {bubbles:true}));
    const fps = $('#playback-settings-content [data-fps]');
    fps.value = '30'; fps.dispatchEvent(new Event('input', {bubbles:true}));
    const sampling = $('#playback-settings-content [data-sampling]');
    sampling.value = 'deterministic'; sampling.dispatchEvent(new Event('change', {bubbles:true}));
    let settingsStatus;
    const commandDeadline = performance.now() + 15000;
    do {
      settingsStatus = await control('status');
      if (settingsStatus.commands.includes('set_fps') && settingsStatus.commands.includes('set_action_selection_mode')) break;
      assert(performance.now() < commandDeadline, 'settings events did not reach the host');
      await new Promise(requestAnimationFrame);
    } while (true);
    assert(settingsStatus.paused === 'paused' && selected(103), 'settings advanced the trajectory');
    assert(seed.value === '41414', 'snapshot update replaced the user seed');
    $('#playback-settings-close').click();
    const processing = $('[data-panel-enabled="value"]');
    processing.click();
    await wait(() => !processing.checked && $('.panel[data-panel="value"]').classList.contains('panel-disabled'), 'processing disabled');
    processing.click();
    await wait(() => processing.checked && !$('.panel[data-panel="value"]').classList.contains('panel-disabled'), 'processing restored');
    $('#panel-add').click();
    await wait(() => $('#panel-editor').open, 'panel editor');
    $('#panel-editor-cancel').click();
    await wait(() => !$('#panel-editor').open, 'panel editor close');
    pass('settings dispatch without inference, preserve edited seed, and panel controls toggle and edit');

    button('hold read'); seek(101);
    await wait(() => gate('read').pending === 1, 'first recorded read held');
    seek(102); seek(101); seek(102);
    assert($('#timeline').getAttribute('aria-busy') === 'true', 'latest seek loading missing');
    button('release read');
    await wait(() => selected(102) && pixel() === 102, 'latest selected frame');
    assert((await control('status')).paused === 'paused', 'scrubbing advanced inference');
    pass('held A and rapid selections settle on the latest step, frame and paused state');

    // Both game and observation decode real bitmaps through the gate. Release an
    // older decode after a newer cursor is already selected, with no next input.
    button('hold decode'); seek(101);
    await wait(() => gate('decode').pending > 0 && selected(101), 'old decode held');
    seek(102);
    await wait(() => selected(102), 'new selection while old decode held');
    button('release decode');
    await wait(() => gate('decode').pending === 0 && selected(102) && pixel() === 102, 'late bitmap replaced current frame');
    pass('late game and observation decoding cannot repaint an earlier selection');

    // Reward reference and chart range are owned by their production controls.
    const reference = $('.reward-table-block button');
    assert(reference, 'reward reference control missing');
    reference.click();
    const table = $('.reward-table-block table');
    await wait(() => table.getAttribute('aria-description')?.includes('from step 102'), 'reference step not selected');
    const referenceText = table.getAttribute('aria-description');
    seek(101); await wait(() => selected(101) && pixel() === 101, 'earlier inspection');
    assert(table.getAttribute('aria-description') === referenceText, 'seek changed reward reference');
    pass('scrubbing preserves the independent reward reference');

    button('hold decode');
    $('[data-checkpoint-next]').click();
    await wait(() => !$('#checkpoint-loading-mask').hidden, 'replacement mask');
    await control('succeed');
    await wait(() => gate('decode').pending > 0, 'replacement preparation held');
    assert(!$('#checkpoint-loading-mask').hidden, 'command acknowledgement cleared readiness');
    button('release decode');
    await wait(() => $('#checkpoint-loading-mask').hidden && selected(203) && pixel() === 203, 'replacement readiness');
    pass('Checkpoint loading waits for actual frame preparation and presents the replacement');
    result().dataset.status = 'passed';
  } catch (error) {
    result().textContent += `FAIL ${error.stack}\n`; result().dataset.status = 'failed';
    for (const name of ['read', 'decode', 'peer']) button(`release ${name}`);
  }
}
