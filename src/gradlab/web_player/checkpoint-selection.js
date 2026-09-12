// Local selection identity is deliberately separate from server source generations.
export class CheckpointSelection {
  #command;
  #pending = null;
  #generation = 0;
  #route = null;
  #navigationPending = false;
  #sourceMode = false;
  #background = null;
  #epoch = 0;
  #presentations = new WeakMap();
  #listeners = new Set();
  #disposed = false;

  constructor(command) { this.#command = command; }

  get view() {
    return Object.freeze({
      route: this.#route,
      loading: this.#pending !== null,
      navigationPending: this.#navigationPending,
      sourceMode: this.#sourceMode,
      backgroundSnapshot: this.#background,
    });
  }

  subscribe(listener) {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  #publish() { for (const listener of this.#listeners) listener(this.view); }

  select(source, route, { historyMode = 'push' } = {}) {
    if (this.#disposed) return false;
    const commandId = this.#command('select_source', { source, route: { ...route } });
    if (commandId === null) return false;
    this.#route = Object.freeze({ ...route });
    this.#pending = { commandId, checkpointId: String(route.checkpoint_id || ''), generation: ++this.#generation };
    this.#navigationPending = true;
    this.#publish();
    return { commandId, route: this.#route, historyMode };
  }

  browse(route, current, { historyMode = 'push' } = {}) {
    if (this.#disposed) return false;
    if (!current?.app?.has_active_runner) {
      if (this.#command('browse_sources', { route: { ...route } }) === null) return false;
    } else if (!this.#sourceMode) {
      this.#background = current;
    }
    this.#route = Object.freeze({ ...route });
    this.#sourceMode = true;
    this.#publish();
    return { route: this.#route, historyMode };
  }

  receive(message) {
    if (this.#disposed) return {};
    if (message.type === 'command_result') {
      if (message.id === this.#pending?.commandId && !message.ok) {
        this.#pending = null;
        this.#publish();
      }
      return {};
    }
    if (message.type === 'session_changed') {
      this.#epoch = Number(message.session_epoch || 0);
      this.#background = null;
      this.#presentations = new WeakMap();
      return {};
    }
    if (message.type !== 'snapshot') return {};
    const epoch = Number(message.session_epoch || 0);
    if (epoch !== this.#epoch) this.#background = null;
    this.#epoch = epoch;
    const app = message.app;
    let error;
    if (this.#pending && (app?.phase === 'error' || (app?.phase === 'active' && app.error))) {
      error = app.error || 'Could not open checkpoint';
      this.#pending = null;
    }
    const activates = Boolean(this.#pending && app?.phase === 'active'
      && String(app.route?.checkpoint_id || '') === this.#pending.checkpointId);
    if (this.#sourceMode && this.#background && app?.phase === 'active' && !activates) {
      this.#background = message;
      this.#publish();
      return { background: true, error };
    }
    if (activates) this.#background = null;
    this.#sourceMode = Boolean(app && app.phase !== 'active');
    if (app?.route) this.#route = Object.freeze({ ...app.route });
    if (app?.phase === 'active') {
      this.#navigationPending = false;
    }
    if (activates) this.#presentations.set(message, Object.freeze({
      generation: this.#pending.generation,
      epoch: this.#epoch,
      sequence: Number(message.sequence),
    }));
    if (message.mode === 'trajectory') this.#pending = null;
    this.#publish();
    return { error };
  }

  terminate() {
    this.#pending = null;
    this.#generation += 1;
    this.#presentations = new WeakMap();
    this.#publish();
  }

  dispose() {
    this.terminate();
    this.#disposed = true;
    this.#listeners.clear();
  }

  presentationFor(snapshot) { return this.#presentations.get(snapshot); }

  presented(ticket, snapshot) {
    if (!ticket || ticket !== this.#presentations.get(snapshot)
      || ticket.generation !== this.#pending?.generation
      || ticket.epoch !== this.#epoch
      || ticket.sequence !== Number(snapshot.sequence)) return;
    this.#pending = null;
    this.#publish();
  }
}
