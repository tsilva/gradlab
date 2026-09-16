// Tooltip values always come from one recorded sample, never interpolation.
export function createChartTooltip(canvas) {
  const element = document.createElement("div");
  element.className = "chart-tooltip";
  element.setAttribute("role", "tooltip");
  element.hidden = true;
  canvas.parentElement.append(element);
  let pointerY = null;
  canvas.addEventListener("pointermove", (event) => {
    const bounds = canvas.getBoundingClientRect();
    pointerY = (event.clientY - bounds.top) * canvas.clientHeight / bounds.height;
  });
  for (const event of ["pointerleave", "pointercancel"]) {
    canvas.addEventListener(event, () => { pointerY = null; });
  }
  return ({ steps, series, step, geometry }) => {
    const plot = geometry?.plot;
    element.hidden = !plot || !Number.isFinite(step) || !steps.length;
    if (element.hidden) return;
    const index = steps.reduce((best, value, i) => Math.abs(value - step) < Math.abs(steps[best] - step) ? i : best, 0);
    const title = document.createElement("strong");
    title.textContent = `Step ${steps[index]}`;
    element.replaceChildren(title);
    for (const item of series) {
      const row = document.createElement("div");
      row.className = "chart-tooltip-row";
      const swatch = document.createElement("span");
      swatch.className = "chart-tooltip-swatch";
      swatch.style.backgroundColor = item.color;
      const label = document.createElement("span");
      label.textContent = item.label;
      const value = document.createElement("span");
      value.className = "chart-tooltip-value";
      const number = item.values[index];
      value.textContent = Number.isFinite(number) ? Number(number.toPrecision(6)).toString() : "Unavailable";
      row.append(swatch, label, value);
      element.append(row);
    }
    const x = canvas.offsetLeft + plot.left + Math.max(0, Math.min(1,
      (step - steps[0]) / Math.max(1, steps.at(-1) - steps[0]))) * (plot.right - plot.left);
    const right = canvas.offsetLeft + canvas.clientWidth;
    const left = x + 12 + element.offsetWidth <= right ? x + 12 : x - element.offsetWidth - 12;
    element.style.left = `${Math.max(canvas.offsetLeft, left)}px`;
    const top = canvas.offsetTop + (pointerY ?? plot.top) + 12;
    element.style.top = `${Math.max(canvas.offsetTop, Math.min(top, canvas.offsetTop + canvas.clientHeight - element.offsetHeight))}px`;
  };
}
