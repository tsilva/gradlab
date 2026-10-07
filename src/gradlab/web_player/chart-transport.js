// Each version belongs to one exact episode/range request. Missing bases reset.
export class ChartVersions {
  constructor() { this.reset(); }
  reset() { this.revision = null; this.rows = new Map(); }
  accept(payload) {
    if (payload.format !== "chart-columns-v1") throw new Error("Unsupported chart response");
    if (payload.base && payload.base !== this.revision) throw new Error("Chart revision is unavailable");
    const rows = payload.base ? new Map(this.rows) : new Map();
    payload.removed.forEach((step) => rows.delete(step));
    payload.rows.forEach(([step, values]) => rows.set(step, values));
    const assign = (target, path, value) => {
      if (value?.$absent === true) return;
      for (const key of path.slice(0, -1)) {
        if (!Object.hasOwn(target, key)) Object.defineProperty(target, key, { value: {}, enumerable: true, writable: true });
        target = target[key];
      }
      Object.defineProperty(target, path.at(-1), { value, enumerable: true, writable: true, configurable: true });
    };
    const points = [...rows].sort(([a], [b]) => a - b).map(([, values]) => {
      const point = {};
      payload.constants.forEach(([path, value]) => assign(point, path, value));
      payload.fields.forEach((path, index) => assign(point, path, values[index]));
      return point;
    });
    this.revision = payload.revision;
    this.rows = rows;
    return points;
  }
}

export function frameScheduler(render, schedule = requestAnimationFrame) {
  let pending = false;
  return () => {
    if (pending) return;
    pending = true;
    schedule(() => { pending = false; render(); });
  };
}
