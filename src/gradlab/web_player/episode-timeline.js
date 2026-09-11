// Keep the navigable episode independent of the bounded presentation cache.
export function episodeStepRange(trajectory, snapshots) {
  if (trajectory?.imported || (trajectory?.episode_id && trajectory?.transitions > 0)) {
    return { first: Number(trajectory.first_step), last: Number(trajectory.last_step) };
  }
  const steps = snapshots.map((snapshot) => Number(snapshot.transition?.step ?? snapshot.session?.step))
    .filter(Number.isFinite);
  return steps.length ? { first: Math.min(...steps), last: Math.max(...steps) } : null;
}

// Event positions belong to the episode, never to the chart's inspection page.
export class EventOverview {
  constructor(limit = 4096) {
    this.limit = limit;
    this.reset(null);
  }

  reset(episodeId) {
    this.episodeId = episodeId;
    this.width = 1;
    this.throughStep = -1;
    this.buckets = new Map();
  }

  merge(point) {
    const key = Math.floor(point.step / this.width);
    const previous = this.buckets.get(key);
    this.buckets.set(key, previous ? {
      step: Math.min(previous.step, point.step),
      last_step: Math.max(previous.last_step, point.last_step),
      count: previous.count + point.count,
      boundary: previous.boundary || point.boundary,
      events: [...new Set([...previous.events, ...point.events])].sort().slice(0, 32),
    } : { ...point, events: [...point.events] });
  }

  append(point) {
    const step = Number(point?.step);
    if (!Number.isInteger(step) || step <= this.throughStep) return;
    this.throughStep = step;
    if (!point.boundary && !point.events?.length) return;
    this.merge({ step, last_step: step, count: 1, boundary: Boolean(point.boundary),
      events: [...new Set(point.events || [])].sort().slice(0, 32) });
    while (this.buckets.size > this.limit) {
      const points = [...this.buckets.values()];
      this.width *= 2;
      this.buckets.clear();
      points.forEach((item) => this.merge(item));
    }
  }

  load(overview) {
    if (this.episodeId === overview.episode_id && this.throughStep > overview.through_step) return;
    this.reset(overview.episode_id);
    this.width = overview.bucket_size;
    this.throughStep = overview.through_step;
    overview.points.forEach((point) => this.merge(point));
  }
}

export function timelineEventMarkers(points, range, slots = 120) {
  if (!range) return [];
  const markers = [];
  for (const point of [...points].sort((a, b) => a.step - b.step)) {
    if (point.step < range.first || point.step > range.last) continue;
    const position = (point.step - range.first) / Math.max(1, range.last - range.first);
    const previous = markers.at(-1);
    if (previous && (position - previous.position < 1 / slots || markers.length >= slots)) {
      previous.count += point.count;
      previous.last_step = Math.max(previous.last_step, point.last_step);
      previous.boundary ||= point.boundary;
      previous.events = [...new Set([...previous.events, ...point.events])].sort();
    } else {
      markers.push({ ...point, position });
    }
  }
  return markers;
}

// At most one read is in flight. Scrubbing coalesces intermediate requests;
// replacing the episode or returning to latest invalidates outstanding results.
export class RecordedStepReader {
  constructor(fetchStep, { onInvalidate = () => {} } = {}) {
    this.fetchStep = fetchStep;
    this.onInvalidate = onInvalidate;
    this.version = 0;
    this.pending = null;
  }

  invalidate() {
    this.version += 1;
    this.onInvalidate();
  }

  async read(request) {
    const version = ++this.version;
    await this.pending?.catch(() => {});
    if (version !== this.version) return null;
    const pending = this.fetchStep(request);
    this.pending = pending;
    try {
      const result = await pending;
      return version === this.version ? result : null;
    } catch (error) {
      if (version === this.version) throw error;
      return null;
    } finally {
      if (this.pending === pending) this.pending = null;
    }
  }
}
