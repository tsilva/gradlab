import { createChartHistory } from '../../../src/gradlab/web_player/chart-history.js';

export const point = (step, episode = 1) => ({ step, episode, sequence: step, reward_shaped: step });
export const full = (steps = [1, 10], revision = 'a') => ({
  format: 'chart-columns-v1', revision, base: null, removed: [],
  fields: [['step'], ['sequence'], ['reward_shaped']], constants: [[['episode'], 1]],
  rows: steps.map(step => [step, [step, step, step]]),
});
export const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

export function chartHarness(onChange = () => {}) {
  let now = 0;
  let next = 0;
  const timers = new Map();
  const requests = [];
  const clock = {
    now: () => now,
    setTimeout(callback, delay) { const id = ++next; timers.set(id, { callback, at: now + delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
  };
  const history = createChartHistory({
    clock, onChange,
    request(selection) {
      return new Promise((resolve, reject) => requests.push({ ...selection, resolve, reject, at: now }));
    },
  });
  let context = { epoch: 1, episodeId: 'episode-a', episode: 1, lastStep: 10, liveHistory: [], throughStep: 10 };
  const update = (patch = {}) => { context = { ...context, ...patch }; history.updateContext(context); };
  update();
  return {
    history, requests, update,
    async advance(ms) {
      const end = now + ms;
      while (true) {
        const entry = [...timers].filter(([, timer]) => timer.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!entry) break;
        const [id, timer] = entry;
        timers.delete(id); now = timer.at; timer.callback(); await flush();
      }
      now = end; await flush();
    },
  };
}
