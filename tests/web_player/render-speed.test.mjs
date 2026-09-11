import assert from "node:assert/strict";
import test from "node:test";

import { mountRenderSpeed } from "../../src/gradlab/web_player/panels/render-speed.js";
import { mount } from "../../src/gradlab/web_player/panels/game.js";

function browserClock(t) {
  let now = 0;
  let id = 0;
  const frames = new Map();
  const timers = new Map();
  const listeners = new Map();
  const nodes = new Map();
  const node = () => ({
    dataset: {}, style: {}, classList: { toggle() {} },
    clientWidth: 800, clientHeight: 600,
    querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, node());
      return nodes.get(selector);
    },
    attributes: {},
    addEventListener() {}, setAttribute(key, value) { this.attributes[key] = value; },
    getContext: () => ({ drawImage() { now += 2; } }),
  });
  const doc = {
    hidden: false,
    createElement: node,
    addEventListener(name, fn) {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(fn);
    },
    removeEventListener(name, fn) { listeners.get(name)?.delete(fn); },
  };
  const globals = {
    document: doc,
    ResizeObserver: class { observe() {} disconnect() {} },
    requestAnimationFrame: (fn) => { frames.set(++id, fn); return id; },
    cancelAnimationFrame: (key) => frames.delete(key),
    setInterval: (fn) => { timers.set(++id, fn); return id; },
    clearInterval: (key) => timers.delete(key),
    createImageBitmap: async () => {
      now += 4;
      return { width: 160, height: 210, close() {} };
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
    t.after(() => {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    });
  }
  t.mock.method(performance, "now", () => now);
  return {
    nodes, frames, timers,
    at(value) { now = value; },
    refresh() {
      const callbacks = [...frames.values()];
      frames.clear();
      callbacks.forEach((fn) => fn(now));
    },
    tick() { [...timers.values()].forEach((fn) => fn()); },
    visibility(hidden) {
      doc.hidden = hidden;
      listeners.get("visibilitychange")?.forEach((fn) => fn());
    },
  };
}

test("render speed counts refreshes, measures actual elapsed time, and expires idle readings", (t) => {
  const clock = browserClock(t);
  const element = document.createElement("div");
  const meter = mountRenderSpeed(element);
  assert.equal(element.querySelector("[data-fps-value]").textContent, "— FPS");
  meter.record(4, 2);
  meter.record(8, 4);
  assert.equal(clock.frames.size, 1);
  clock.refresh();
  clock.at(500);
  meter.record(6, 3);
  clock.refresh();
  clock.at(2000); // Delayed UI/timer: use two seconds, not the requested interval.
  clock.tick();
  assert.equal(element.querySelector("[data-fps-value]").textContent, "1.0 FPS (1–1)");
  clock.at(3000);
  clock.tick();
  assert.equal(element.querySelector("[data-fps-value]").textContent, "0.0 FPS (0–1)");
  meter.destroy();
  assert.equal(clock.timers.size, 0);
});

test("visibility and reset discard pending refreshes and old timings", (t) => {
  const clock = browserClock(t);
  const element = document.createElement("div");
  const meter = mountRenderSpeed(element);
  meter.record(5, 1);
  clock.visibility(true);
  assert.equal(clock.frames.size, 0);
  meter.record(10, 10);
  assert.equal(clock.frames.size, 0);
  clock.at(5000);
  clock.visibility(false);
  meter.record(2, 1);
  clock.refresh();
  clock.at(6000);
  clock.tick();
  assert.equal(element.querySelector("[data-fps-value]").textContent, "1.0 FPS (1–1)");
  meter.reset();
  assert.equal(element.querySelector("[data-fps-value]").textContent, "— FPS");
  meter.record(2, 1);
  meter.destroy();
  assert.equal(clock.frames.size, 0);
});

test("game panel measures only committed bitmaps, not repeated snapshots or missing frames", async (t) => {
  const clock = browserClock(t);
  const originalObserver = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  t.after(() => {
    if (originalObserver) globalThis.ResizeObserver = originalObserver;
    else delete globalThis.ResizeObserver;
  });
  const panel = mount({ definition: { id: "game" }, services: { getState: () => ({}) } });
  const element = clock.nodes.get("[data-render-speed]");
  await panel.prepareFrame(1, {}, { sequence: 1 });
  panel.render({ sequence: 1 });
  clock.refresh();
  panel.render({ sequence: 1 });
  await panel.renderFrame(1, {}, { sequence: 1 });
  assert.equal(clock.frames.size, 0);
  clock.at(1006);
  clock.tick();
  assert.equal(element.querySelector("[data-fps-value]").textContent, "1.0 FPS (1–1)");
  await panel.prepareFrame(1, null, { sequence: 2 });
  panel.render({ sequence: 2 });
  assert.equal(element.querySelector("[data-fps-value]").textContent, "— FPS");
  await panel.prepareFrame(1, {}, { sequence: 3 });
  panel.render({ sequence: 3 });
  panel.resetFrames();
  assert.equal(clock.frames.size, 0);
  assert.equal(element.querySelector("[data-fps-value]").textContent, "— FPS");
  panel.destroy();
  assert.equal(clock.timers.size, 0);
});

test("FPS chart bounds its history and clears it on reset", (t) => {
  const clock = browserClock(t);
  const element = document.createElement("div");
  const meter = mountRenderSpeed(element);
  meter.record(4, 2);
  clock.refresh();
  clock.at(1000);
  clock.tick();
  assert.match(element.title, /Decode 4.0 ms · Draw 2.0 ms/);
  assert.match(element.querySelector("[data-fps-area]").attributes.d, /^M118,32 L118,/);
  for (let second = 2; second <= 61; second += 1) {
    clock.at(second * 1000);
    clock.tick();
  }
  assert.equal(element.querySelector("[data-fps-value]").textContent, "0.0 FPS (0–0)");
  assert.equal((element.querySelector("[data-fps-area]").attributes.d.match(/L/g) || []).length, 121);
  meter.reset();
  assert.equal(element.querySelector("[data-fps-area]").attributes.d, "");
  meter.destroy();
});
