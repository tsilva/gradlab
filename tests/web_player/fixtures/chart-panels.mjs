import { mountPanel } from '../../../frontend/mount.ts';
import Telemetry from '../../../frontend/components/Telemetry.svelte';
const mount = options => mountPanel(Telemetry, options);
import { chartHarness, full, flush } from '../helpers/chart-history.mjs';

const results = document.querySelector('#results');
const panels = [];
let chartHoverStep = null;
// Observe the actual dashed cursor paths emitted by each canvas renderer.
const cursors = new Map();
const proto = CanvasRenderingContext2D.prototype;
const originalMoveTo = proto.moveTo;
const originalStroke = proto.stroke;
proto.moveTo = function(x, y) { this.cursorStart = { x, y }; return originalMoveTo.call(this, x, y); };
proto.stroke = function(...args) {
  if (this.getLineDash().join(',') === '4,3' && this.lineWidth === 1.5) cursors.set(this.canvas, this.cursorStart.x);
  return originalStroke.apply(this, args);
};
const snapshot = { sequence: 10, session: { episode: 1, step: 10 }, transition: { step: 10, episode: 1 } };
const exact = [1, 10].map(step => ({ step, episode: 1, reward_shaped: step, signals: { score: step } }));
const reference = { episode: 'episode-a', step: 1 };
const render = () => {
  if (!panels.length) return;
  const chart = h.history.read();
  panels.forEach(panel => panel.renderHistory(exact, snapshot, {
    chartHoverStep, chartHistory: chart.data, chartStatus: chart, chartRange: chart.range,
    selectedSequence: 10, rewardReference: reference,
  }));
};
const h = chartHarness(render);
const check = (condition, message) => { if (!condition) throw new Error(message); };
const status = panel => panel.element.querySelector('[role="status"]');
try {
  for (const [id, block] of [
    ['step-reward', { kind: 'line', metrics: ['reward/provider', 'reward/shaped'] }],
    ['signals', { kind: 'namespace-explorer', namespace: 'signal' }],
    ['rewards', { kind: 'reward-table' }],
  ]) {
    const panel = mount({ definition: { id, label: id, config: { blocks: [block] } },
      services: { setChartHoverStep: step => { chartHoverStep = step; render(); }, retryChartHistory: () => h.history.retry(), setChartRange: range => h.history.selectRange(range) } });
    panels.push(panel); document.querySelector('#panels').append(panel.element);
  }
  h.history.setDemand(true);
  check(panels.every(panel => status(panel)?.textContent.includes('Loading')), 'All affected panels show loading');
  check(panels.every(panel => panel.element.querySelector('.panel').dataset.chartStatus === 'loading'), 'Panel state is shared');
  h.requests[0].resolve(full()); await flush();
  check(panels.every(panel => status(panel).hidden), 'Ready panels hide status');
  h.history.selectRange({ first: 2, last: 5 });
  check(panels.every(panel => !panel.element.querySelector('.telemetry-block').getClientRects().length), 'Changing selection hides obsolete charts and tables');
  check(h.requests.length === 2, 'Panels share one request per selection');
  h.requests[1].resolve(full([2, 5], 'b')); await flush();
  h.update({ lastStep: 11 }); await h.advance(1000);
  check(panels.every(panel => !!panel.element.querySelector('.telemetry-block').getClientRects().length), 'Same-range refresh keeps valid data visible');
  h.requests[2].reject(new Error('History unavailable')); await flush();
  check(panels.every(panel => status(panel).textContent.includes('History unavailable')), 'Permanent error is visible in every panel');
  check(panels.every(panel => !status(panel).querySelector('button').hidden), 'Every affected panel offers Retry');
  status(panels[0]).querySelector('button').click();
  check(h.requests.length === 4, 'Retry in one panel starts one shared request');
  check(panels.every(panel => status(panel).querySelector('button').hidden), 'Retry clears errors in all panels');
  const hoverHistory = full([2, 5], 'c');
  hoverHistory.fields.push(['signals', 'score']);
  hoverHistory.rows.forEach(([step, values]) => values.push(step));
  h.requests[3].resolve(hoverHistory); await flush();
  check(panels.every(panel => status(panel).hidden), 'Successful Retry clears status');
  check(reference.step === 1 && snapshot.transition.step === 10, 'Chart navigation preserves cursor and reward reference');
  await new Promise(requestAnimationFrame);
  const canvases = panels.flatMap(panel => [...panel.element.querySelectorAll('canvas')]);
  for (const source of canvases) {
    const bounds = source.getBoundingClientRect();
    for (const fraction of [0.35, 0.7]) {
      cursors.clear();
      source.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: bounds.left + bounds.width * fraction }));
      await new Promise(requestAnimationFrame);
      for (const canvas of canvases) {
        const tooltip = canvas.parentElement.querySelector('[role="tooltip"]');
        check(tooltip && !tooltip.hidden, 'Every history chart shows its synchronized tooltip');
        const nearest = chartHoverStep < 3.5 ? 2 : 5;
        check(tooltip.querySelector('strong').textContent === `Step ${nearest}`, 'Tooltip labels the actual recorded step');
        check([...tooltip.querySelectorAll('.chart-tooltip-value')].some(value => value.textContent === String(nearest)), 'Tooltip shows the plotted sample value');
        const box = tooltip.getBoundingClientRect(), chartBox = canvas.getBoundingClientRect();
        check(box.left >= chartBox.left - 1 && box.right <= chartBox.right + 1, 'Tooltip stays inside the chart horizontally');
      }
      check(Number.isFinite(chartHoverStep), 'Hover publishes a shared step');
      check(cursors.size === canvases.length, 'Every chart draws a hover cursor, including reward and signals');
      // Both fixture canvases have the same plot size and episode domain.
      const positions = [...cursors.values()];
      check(Math.max(...positions) - Math.min(...positions) < 1, 'Dashed lines track the same step');
      check(reference.step === 1 && snapshot.transition.step === 10, 'Hover preserves playback and reference');
    }
    source.dispatchEvent(new PointerEvent('pointerleave', {bubbles: true}));
    await new Promise(requestAnimationFrame);
    check(chartHoverStep === null, 'Leaving a chart clears the shared hover');
    check(canvases.every(canvas => !canvas.parentElement.querySelector('[role="tooltip"]')), 'Leaving hides all synchronized tooltips');
  }
  results.textContent = 'PASS: actual line, signal explorer and reward-table panels share loading, refresh, failure and Retry; obsolete plots are hidden; cursor and reference are unchanged; hover cursors and tooltips synchronize, show recorded values, fit within charts, and clear on leave.';
} catch (error) {
  results.textContent = `FAIL: ${error.message}`;
  console.error(error);
} finally {
  h.history.dispose();
  proto.moveTo = originalMoveTo;
  proto.stroke = originalStroke;
}
