import { bindChartRange } from "../chart-range.js";
import { createPanel } from "./shared.js";

const format = (value) => Number(value).toLocaleString(undefined, { maximumSignificantDigits: 6 });

export function mount({ definition, services }) {
  const element = createPanel({ id: definition.id, label: definition.label,
    body: '<div class="discounted-rewards-body"><p data-summary></p><p data-note></p><div data-timeline></div><table><thead><tr><th>Step / delay</th><th>Reward</th><th>Weight γᵏ</th><th>Contribution</th></tr></thead><tbody></tbody></table><button data-first type="button">First rewards</button><button data-more type="button">Later rewards</button></div>' });
  const summary = element.querySelector('[data-summary]');
  const note = element.querySelector('[data-note]');
  const timeline = element.querySelector('[data-timeline]');
  const body = element.querySelector('tbody');
  const firstPage = element.querySelector("[data-first]");
  firstPage.hidden = true;
  const more = element.querySelector('[data-more]');
  let key = null, current = null, revision = 0, pending = false, disposed = false, updated = 0;
  let cursor = null, next = null;
  let range = null, bounds = null;
  bindChartRange(timeline,
    () => ({ plot: { left: 0, right: timeline.clientWidth, top: 0, bottom: 48 } }),
    () => ({ history: bounds ? [{ step: bounds.first }, { step: bounds.last }] : [] }), services);
  more.hidden = true;
  async function load() {
    if (pending || !current || disposed) return;
    pending = true;
    const version = revision;
    try {
      const data = await services.loadRewardHistory(current.episode, current.step, cursor);
      if (disposed || version !== revision) return;
      summary.textContent = `From step ${data.first} · γ = ${format(data.discount)} · Reward sum ${format(data.reward_sum)} → discounted ${format(data.discounted_reward_sum)}`;
      const completion = data.complete ? `Complete recorded future. G(sₜ) = ${format(data.return_total)}.`
        : data.bootstrap ? `Truncated. Recorded rewards plus critic bootstrap ${format(data.bootstrap.contribution)} = ${format(data.return_total)}. This is a bootstrapped return, not realized G(sₜ).`
          : 'Partial recorded future. Unobserved rewards are unknown; this subtotal is not G(sₜ).';
      note.textContent = `${completion} The selected action’s reward has delay 0. Timeline bars show discount weight; drag to zoom, double-click to reset. Totals include rewards outside the chart window. Each later reward contributes r × γᵏ.${data.comparison_reasons.length ? ` Critic comparison unavailable: ${data.comparison_reasons.join('; ')}.` : ''}`;
      body.replaceChildren();
      timeline.replaceChildren();
      timeline.className = 'discounted-reward-timeline';
      bounds = { first: range?.first ?? data.first, last: range?.last ?? data.end };
      timeline.setAttribute('aria-label', `Reward timeline from step ${bounds.first} to ${bounds.last}; current reward page`);
      for (const point of data.points) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        const jump = document.createElement('button');
        jump.type = 'button';
        jump.textContent = `${point.step} / +${point.offset}`;
        jump.addEventListener('click', () => services.inspectStep(point.step));
        cell.append(jump); row.append(cell);
        for (const value of [point.reward, point.weight, point.contribution]) {
          const td = document.createElement('td'); td.textContent = format(value); row.append(td);
        }
        row.title = point.events.join(', ');
        body.append(row);
        if (point.step < bounds.first || point.step > bounds.last) continue;
        const marker = document.createElement('button');
        marker.type = 'button';
        marker.className = 'discounted-reward-marker';
        marker.style.left = `${100 * (point.step-bounds.first) / Math.max(1, bounds.last-bounds.first)}%`;
        marker.style.height = `${8 + 32 * point.weight}px`;
        marker.title = `Step ${point.step}: ${format(point.reward)} × ${format(point.weight)} = ${format(point.contribution)}`;
        marker.setAttribute('aria-label', marker.title);
        marker.addEventListener('click', () => services.inspectStep(point.step));
        timeline.append(marker);
      }
      if (!data.points.length) {
        const row = document.createElement('tr'); const td = document.createElement('td');
        td.colSpan = 4; td.textContent = 'No nonzero rewards in the recorded future.'; row.append(td); body.append(row);
      }
      firstPage.hidden = cursor === null;
      next = data.next_last; more.hidden = next === null;
      updated = Date.now();
    } catch (error) {
      if (version === revision && !disposed) {
        summary.textContent = error.message; note.textContent = ''; body.replaceChildren(); timeline.replaceChildren(); more.hidden = true; firstPage.hidden = true;
        updated = Date.now();
      }
    } finally {
      pending = false;
      if (version !== revision && !disposed) void load();
    }
  }
  firstPage.addEventListener("click", () => { if (!pending) { cursor = null; void load(); } });
  more.addEventListener('click', () => { if (!pending) { cursor = next; void load(); } });
  return {
    element,
    renderHistory(history, snapshot, view = {}) {
      const state = services.getState();
      const episode = snapshot?.trajectory?.episode_id || state.liveSnapshot?.trajectory?.episode_id;
      const step = snapshot?.transition?.step;
      range = view.chartRange;
      const identity = episode && Number.isInteger(step) ? `${view.sessionEpoch}:${episode}:${step}:${range?.first}:${range?.last}` : null;
      if (identity !== key) {
        key = identity; revision++; cursor = null; updated = 0;
        current = identity ? { episode, step } : null;
        summary.textContent = identity ? 'Loading recorded future rewards…' : 'Record a transition to inspect its future rewards.';
        note.textContent = ''; body.replaceChildren(); timeline.replaceChildren(); more.hidden = true; firstPage.hidden = true;
      }
      if (current && Date.now()-updated > 1000) void load();
    },
    destroy() { disposed = true; revision++; },
  };
}
