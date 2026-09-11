const keyFor = ({ epoch, episode_id, step }) => JSON.stringify([epoch, episode_id, step]);

// A small sequential lookahead hides recorded-page reads without advancing the
// cursor or Policy. The byte budget counts serialized UTF-8 response bytes.
export class RecordedStepPrefetch {
  constructor(fetchStep, { limit = 4, maxBytes = 8 * 1024 * 1024 } = {}) {
    this.fetchStep = fetchStep;
    this.limit = limit;
    this.maxBytes = maxBytes;
    this.cache = new Map();
    this.bytes = 0;
    this.range = null;
    this.filling = null;
  }

  clear() {
    this.range = null;
    for (const entry of this.cache.values()) entry.controller.abort();
    this.cache.clear();
    this.bytes = 0;
  }

  read(request) {
    return this.cache.get(keyFor(request))?.promise ?? this.fetchStep(request);
  }

  ahead(request, lastStep) {
    const keys = new Set();
    for (let step = request.step; step <= lastStep && keys.size < this.limit; step++) {
      keys.add(keyFor({ ...request, step }));
    }
    this.range = { request, lastStep, keys };
    for (const [key, entry] of this.cache) {
      if (keys.has(key)) continue;
      entry.controller.abort();
      this.bytes -= entry.bytes;
      this.cache.delete(key);
    }
    if (!this.filling) {
      this.filling = this.fill().finally(() => { this.filling = null; });
    }
    return this.filling;
  }

  async fill() {
    while (this.range) {
      const { request, lastStep, keys } = this.range;
      let query = null;
      for (let step = request.step; step <= lastStep && step < request.step + this.limit; step++) {
        const candidate = { ...request, step };
        if (!this.cache.has(keyFor(candidate))) { query = candidate; break; }
      }
      if (!query) return;
      const key = keyFor(query);
      const entry = { controller: new AbortController(), bytes: 0, promise: null };
      this.cache.set(key, entry);
      try {
        entry.promise = this.fetchStep(query, { signal: entry.controller.signal });
        const result = await entry.promise;
        if (this.cache.get(key) !== entry) continue;
        const bytes = new TextEncoder().encode(JSON.stringify(result)).byteLength;
        if (this.bytes + bytes > this.maxBytes) {
          this.cache.delete(key);
          return;
        }
        entry.bytes = bytes;
        this.bytes += bytes;
      } catch {
        if (this.cache.get(key) !== entry) continue;
        this.cache.delete(key);
        // A speculative failure is retried by the ordinary demand read.
        if (this.range?.keys === keys) return;
      }
    }
  }
}
