import { mount } from '../../../../src/gradlab/web_player/panels/telemetry-panel.js';
import { chartHarness, full, flush } from '../helpers/chart-history.mjs';

const results = document.querySelector('#results');
const panels = [];
const snapshot = { sequence: 10, session: { episode: 1, step: 10 }, transition: { step: 10, episode: 1 } };
const exact = [1, 10].map(step => ({ step, episode: 1, reward_shaped: step, signals: { score: step } }));
const reference = { episode: 'episode-a', step: 1 };
const render = () => {
  if (!panels.length) return;
  const chart = h.history.read();
  panels.forEach(panel => panel.renderHistory(exact, snapshot, {
    chartHistory: chart.data, chartStatus: chart, chartRange: chart.range,
    selectedSequence: 10, rewardReference: reference,
  }));
};
const h = chartHarness(render);
const check = (condition, message) => { if (!condition) throw new Error(message); };
const status = panel => panel.element.querySelector('[role="status"]');
try {
  for (const [id, block] of [
    ['reward-line', { kind: 'line', metrics: ['reward/shaped'] }],
    ['signals', { kind: 'namespace-explorer', namespace: 'signal' }],
    ['rewards', { kind: 'reward-table' }],
  ]) {
    const panel = mount({ definition: { id, label: id, config: { blocks: [block] } },
      services: { retryChartHistory: () => h.history.retry(), setChartRange: range => h.history.selectRange(range) } });
    panels.push(panel); document.querySelector('#panels').append(panel.element);
  }
  h.history.setDemand(true);
  check(panels.every(panel => status(panel)?.textContent.includes('Loading')), 'All affected panels show loading');
  check(panels.every(panel => panel.element.dataset.chartStatus === 'loading'), 'Panel state is shared');
  h.requests[0].resolve(full()); await flush();
  check(panels.every(panel => status(panel).hidden), 'Ready panels hide status');
  h.history.selectRange({ first: 2, last: 5 });
  check(panels.every(panel => panel.element.querySelector('.telemetry-block').hidden), 'Changing selection hides obsolete charts and tables');
  check(h.requests.length === 2, 'Panels share one request per selection');
  h.requests[1].resolve(full([2, 5], 'b')); await flush();
  h.update({ lastStep: 11 }); await h.advance(1000);
  check(panels.every(panel => !panel.element.querySelector('.telemetry-block').hidden), 'Same-range refresh keeps valid data visible');
  h.requests[2].reject(new Error('History unavailable')); await flush();
  check(panels.every(panel => status(panel).textContent.includes('History unavailable')), 'Permanent error is visible in every panel');
  check(panels.every(panel => !status(panel).querySelector('button').hidden), 'Every affected panel offers Retry');
  status(panels[0]).querySelector('button').click();
  check(h.requests.length === 4, 'Retry in one panel starts one shared request');
  check(panels.every(panel => status(panel).querySelector('button').hidden), 'Retry clears errors in all panels');
  h.requests[3].resolve(full([2, 5], 'c')); await flush();
  check(panels.every(panel => status(panel).hidden), 'Successful Retry clears status');
  check(reference.step === 1 && snapshot.transition.step === 10, 'Chart navigation preserves cursor and reward reference');
  results.textContent = 'PASS: actual line, signal explorer and reward-table panels share loading, refresh, failure and Retry; obsolete plots are hidden; cursor and reference are unchanged.';
} catch (error) {
  results.textContent = `FAIL: ${error.message}`;
  console.error(error);
} finally { h.history.dispose(); }
