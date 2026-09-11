import { ChartVersions } from "./chart-transport.js";
import { chartWithLiveTail } from "./chart-live-tail.js";

// One instance owns the current selection in one window. Request identity, rather
// than cancellation completion, decides which response may publish its result.
export function createChartHistory({ token = "", request = chartRequest(token), onChange = () => {}, clock = {
  now: () => Date.now(),
  setTimeout: (callback, delay) => setTimeout(callback, delay),
  clearTimeout: (timer) => clearTimeout(timer),
} }) {
  const versions = new ChartVersions();
  let context = {};
  let range = null;
  let data = null;
  let status = "idle";
  let error = null;
  let demand = false;
  let active = null;
  let disposed = false;
  let timer = null;
  let retries = 0;
  let updated = -Infinity;
  let dirty = false;
  let fullRecovery = false;

  function cancel() {
    clock.clearTimeout(timer);
    timer = null;
    const previous = active;
    active = null;
    previous?.abort();
  }

  async function refresh() {
    if (disposed || !demand || !context.episodeId || !context.lastStep || active) return;
    dirty = false;
    const controller = new AbortController();
    active = controller;
    status = retries || fullRecovery ? "recovering" : data === null ? "loading" : "refreshing";
    onChange();
    try {
      const payload = await request({ epoch: context.epoch, episodeId: context.episodeId,
        range, base: versions.revision, signal: controller.signal });
      if (active !== controller) return;
      try {
        data = versions.accept(payload);
      } catch (failure) {
        // Decode failures must not be mistaken for network TypeErrors.
        versions.reset();
        error = failure.message;
        if (!fullRecovery) {
          fullRecovery = true;
          active = null;
          void refresh();
        } else status = "error";
        return;
      }
      status = "ready";
      updated = clock.now();
      error = null;
      retries = 0;
      fullRecovery = false;
    } catch (failure) {
      if (active !== controller) return;
      if (failure.name === "AbortError") { status = data === null ? "idle" : "ready"; return; }
      error = failure.message;
      const transient = failure.transient === true || failure instanceof TypeError;
      if (transient && retries < 3) {
        status = "recovering";
        const delay = [1000, 2000, 4000][retries++];
        timer = clock.setTimeout(() => { timer = null; void refresh(); }, delay);
      } else status = "error";
    } finally {
      if (active === controller) {
        active = null;
        if (status === "ready" && dirty) scheduleRefresh();
        onChange();
      }
    }
  }

  function scheduleRefresh() {
    if (disposed || !demand || status === "error") return;
    dirty = true;
    if (status === "recovering") return;
    if (active || timer !== null) return;
    const delay = 1000 - (clock.now() - updated);
    if (delay <= 0) void refresh();
    else timer = clock.setTimeout(() => { timer = null; void refresh(); }, delay);
  }

  function reset() {
    cancel();
    versions.reset();
    retries = 0;
    fullRecovery = false;
    updated = -Infinity;
    dirty = false;
    data = null;
    error = null;
    status = "idle";
    void refresh();
    onChange();
  }

  return {
    updateContext(next) {
      if (disposed) return;
      const replaced = next.epoch !== context.epoch || next.episodeId !== context.episodeId;
      const changed = next.lastStep !== context.lastStep || next.liveHistory !== context.liveHistory;
      const cursorChanged = next.throughStep !== context.throughStep;
      context = { ...next };
      if (replaced) { range = null; reset(); }
      else {
        if (changed) scheduleRefresh();
        if (changed || cursorChanged) onChange();
      }
    },
    setDemand(value) {
      if (disposed || demand === Boolean(value)) return;
      demand = Boolean(value);
      if (demand) { retries = 0; fullRecovery = false; void refresh(); }
      else cancel();
    },
    selectRange(next) {
      if (disposed || (next?.first === range?.first && next?.last === range?.last)) return;
      range = next ? Object.freeze({ first: next.first, last: next.last }) : null;
      reset();
    },
    read: () => ({
      data: data === null ? null : chartWithLiveTail(data, context.liveHistory || [], {
        episode: context.episode, range, throughStep: context.throughStep ?? Infinity,
      }),
      status, error, range,
    }),
    retry() { if (!disposed) { cancel(); retries = 0; fullRecovery = false; void refresh(); } },
    dispose() { disposed = true; cancel(); },
  };
}

function chartRequest(token) {
  return async ({ epoch, episodeId, range, base, signal }) => {
    const query = new URLSearchParams({ epoch, episode_id: episodeId, format: "chart-columns-v1" });
    if (base) query.set("base", base);
    if (range) { query.set("first", range.first); query.set("last", range.last); }
    const response = await fetch(`/api/playback/chart-history?${query}`, {
      signal, headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      const error = new Error(result.error || `Unable to load episode charts (HTTP ${response.status})`);
      error.transient = [408, 429].includes(response.status) || response.status >= 500;
      throw error;
    }
    return response.json();
  };
}
