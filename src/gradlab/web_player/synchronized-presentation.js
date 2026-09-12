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
    this.requested = 0;
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
    const requested = ++this.requested;
    if (this.running) return this.running;
    const generation = this.generation;
    const running = this.#run(generation).finally(async () => {
      if (this.running !== running) return;
      this.running = null;
      // Offers received during preparation, including after reset, must drain
      // without relying on another socket/frame event to wake the queue.
      if (requested !== this.requested) await this.#drain();
    });
    this.running = running;
    return running;
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
      try {
        await this.prepare(snapshot);
      } catch (error) {
        if (generation === this.generation) throw error;
        return;
      }
      if (generation !== this.generation) return;
      this.present(snapshot);
      this.lastPresentedOrder = order;
      [...this.snapshots.keys()]
        .filter((pendingOrder) => pendingOrder <= order)
        .forEach((pendingOrder) => this.snapshots.delete(pendingOrder));
    }
  }
}
