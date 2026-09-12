import { createPlaybackInspection } from '../../../src/gradlab/web_player/playback-inspection.js';
export const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
export const snapshot = (step, { epoch = 1, episode = 1, episodeId = 'episode-a', ...rest } = {}) => ({
  session_epoch: epoch, sequence: step, driver: 'policy', mode: 'playback', run_state: 'paused',
  transition: { step, episode }, session: { episode, target_fps: 30 },
  trajectory: { episode_id: episodeId, transitions: 101, first_step: 0, last_step: 100 },
  control: { has_control: true }, ...rest,
});
export const recorded = (step, options) => ({ snapshot: snapshot(step, options), points: [], frames: [] });
export const flush = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };

export function harness(options = {}) {
  const commands = [], messages = [], frames = [], errors = [], presentations = [];
  const timers = new Map(); let timerId = 0, now = 0;
  const inspection = createPlaybackInspection({
    windowId: 'main', fetchStep: async ({ step }) => recorded(step),
    command(name, payload) { commands.push({ name, payload }); return `command-${commands.length}`; },
    peer: message => messages.push(message),
    renderFrame: async (kind, blob, metadata) => frames.push({ kind, blob, ...metadata }),
    prepareFrame: async () => {}, resetFrames() {},
    presented: (ticket, value) => presentations.push({ ticket, snapshot: value }),
    onError: error => errors.push(error.message),
    setTimeout(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; },
    clearTimeout(id) { timers.delete(id); }, now: () => now,
    ...options,
  });
  return { inspection, commands, messages, frames, errors, presentations, timers,
    async tick(late = 0) { const [id, timer] = timers.entries().next().value; timers.delete(id); now += timer.delay + late; await timer.callback(); },
    elapse(ms) { now += ms; }, get now() { return now; },
  };
}
