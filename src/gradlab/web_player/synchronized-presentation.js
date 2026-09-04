export class SynchronizedPresentation {
  constructor({ isReady, prepare, present, limit = 4096 }) {
    this.isReady = isReady;
    this.prepare = prepare;
    this.present = present;
    this.limit = Math.max(1, Number(limit) || 1);
    this.snapshots = new Map();
    this.offered = 0;
    this.lastPresentedOrder = 0;
    this.generation = 0;
    this.running = null;
  }

  offer(snapshot) {
    const order = ++this.offered;
    this.snapshots.set(order, snapshot);
    while (this.snapshots.size > this.limit) {
      this.snapshots.delete(Math.min(...this.snapshots.keys()));
    }
    return this.#drain();
  }

  notifyReady() {
    return this.#drain();
  }

  reset() {
    this.generation += 1;
    this.snapshots.clear();
    this.lastPresentedOrder = this.offered;
  }

  #drain() {
    if (this.running) return this.running;
    const generation = this.generation;
    const running = this.#run(generation);
    this.running = running;
    return running.finally(() => {
      if (this.running === running) this.running = null;
    });
  }

  async #run(generation) {
    while (generation === this.generation) {
      const candidate = [...this.snapshots.entries()]
        .filter(([order, snapshot]) => (
          order > this.lastPresentedOrder && this.isReady(snapshot)
        ))
        .sort(([left], [right]) => right - left)[0];
      if (!candidate) return;
      const [order, snapshot] = candidate;
      await this.prepare(snapshot);
      if (generation !== this.generation) return;
      this.present(snapshot);
      this.lastPresentedOrder = order;
      [...this.snapshots.keys()]
        .filter((pendingOrder) => pendingOrder <= order)
        .forEach((pendingOrder) => this.snapshots.delete(pendingOrder));
    }
  }
}
