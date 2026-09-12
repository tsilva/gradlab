// Fixture-only gates wrap external browser effects, never the inspection owner.
const gates = new Map(['read', 'decode', 'peer'].map(name => [name, { held: false, pending: [], seen: 0 }]));
let output;
function report() {
  if (output) output.textContent = JSON.stringify(Object.fromEntries([...gates].map(([name, gate]) => [name, { held: gate.held, pending: gate.pending.length, seen: gate.seen }])));
}
async function gate(name, value) {
  const state = gates.get(name);
  state.seen++;
  if (state.held) await new Promise(resolve => { state.pending.push(resolve); report(); });
  report();
  return value;
}
const fetchRead = window.fetch.bind(window);
window.fetch = async (...args) => {
  const response = await fetchRead(...args);
  if (String(args[0]).startsWith('/api/playback/recorded-step?')) await gate('read');
  return response;
};
const decode = window.createImageBitmap.bind(window);
window.createImageBitmap = async (...args) => gate('decode', await decode(...args));
const Channel = window.BroadcastChannel;
window.BroadcastChannel = class extends Channel {
  addEventListener(type, listener, ...options) {
    if (type !== 'message') return super.addEventListener(type, listener, ...options);
    return super.addEventListener(type, async event => {
      if (String(event.data?.type).startsWith('inspection-')) await gate('peer');
      listener(event);
    }, ...options);
  }
};
document.addEventListener('DOMContentLoaded', () => {
  const controls = document.createElement('aside'); controls.id = 'inspection-fixture';
  controls.style.cssText = 'position:fixed;left:8px;bottom:8px;z-index:10000;background:#121015;color:white;padding:8px;max-width:90vw;font-size:12px';
  for (const name of gates.keys()) for (const action of ['hold', 'release']) {
    const button = document.createElement('button'); button.textContent = `${action} ${name}`;
    button.onclick = () => {
      const state = gates.get(name); state.held = action === 'hold';
      if (!state.held) state.pending.splice(0).forEach(resolve => resolve());
      report();
    };
    controls.append(button);
  }
  const run = document.createElement('button'); run.textContent = 'Run inspection checks';
  run.onclick = async () => { run.disabled = true; await (await import('./inspection-checks.js')).runChecks(); run.disabled = false; };
  controls.append(run);
  const collapse = document.createElement('button'); collapse.textContent = 'Collapse inspection fixture';
  collapse.onclick = () => {
    for (const child of controls.children) if (child !== collapse) child.hidden = !child.hidden;
  };
  controls.append(collapse);
  output = document.createElement('output'); output.id = 'inspection-gate-status'; controls.append(output);
  const results = document.createElement('pre'); results.id = 'inspection-test-results'; controls.append(results);
  document.body.append(controls); report();
});
