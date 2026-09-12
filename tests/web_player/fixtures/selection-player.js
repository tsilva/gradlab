const controls = document.createElement('aside');
controls.id = 'fixture-controls';
const token = new URLSearchParams(location.hash.slice(1)).get('token');
for (const action of ['succeed', 'fail', 'status', 'hold-frames', 'release-frames', 'reject', 'observer', 'control', 'reset', 'import', 'hold-browser', 'release-browser', 'catalog-fail', 'catalog-recover', 'disconnect']) {
  const button = document.createElement('button');
  button.textContent = `Fixture ${action}`;
  button.onclick = async () => {
    const result = await fetch(`/assets/fixture-control?action=${action}`, {
      headers: { Authorization: `Bearer ${token}` },
    }).then(response => response.json());
    output.textContent = JSON.stringify(result);
  };
  controls.append(button);
}
const output = document.createElement('output');
output.id = 'fixture-output';
controls.append(output);
document.body.prepend(controls);
const run = document.createElement('button');
run.textContent = 'Run selection checks';
run.onclick = async () => { run.disabled = true; await (await import('./selection-checks.js')).runChecks(); };
controls.append(run);
const additional = document.createElement('button');
additional.textContent = 'Run history and window checks';
additional.onclick = async () => { await (await import('./selection-checks.js')).runAdditionalChecks(); };
controls.append(additional);
