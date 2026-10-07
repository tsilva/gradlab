// Fixture-only instrumentation installed before either immutable frontend entry.
// The same external effects are measured in the leader and its companion view.
(() => {
  const now = () => performance.now();
  const raf = window.requestAnimationFrame.bind(window);
  const microtask = window.queueMicrotask.bind(window);
  const frame = () => new Promise((resolve) => raf(resolve));
  let active = null,
    pendingDecodes = 0,
    pendingReads = 0,
    lastActivity = now();
  const heap = () => performance.memory?.usedJSHeapSize ?? null;
  const empty = (name) => ({
    name,
    decodes: 0,
    decodeMs: 0,
    peakDecodes: 0,
    canvas: {},
    canvasMs: 0,
    scheduledUiMs: 0,
    inputDispatchMs: 0,
    domMutations: 0,
    domAllocations: 0,
    latencies: [],
    heapStart: heap(),
  });
  const finish = (run) => ({
    ...run,
    heapEnd: heap(),
    pendingDecodes,
    pendingReads,
  });
  for (const [name, schedule] of [
    ["requestAnimationFrame", raf],
    ["queueMicrotask", microtask],
  ]) {
    window[name] = (callback) =>
      schedule((...args) => {
        const started = now(),
          run = active;
        try {
          return callback(...args);
        } finally {
          if (run) run.scheduledUiMs += now() - started;
        }
      });
  }
  const decode = window.createImageBitmap.bind(window);
  window.createImageBitmap = async (...args) => {
    const run = active,
      started = now();
    pendingDecodes++;
    if (run) {
      run.decodes++;
      run.peakDecodes = Math.max(run.peakDecodes, pendingDecodes);
    }
    try {
      return await decode(...args);
    } finally {
      pendingDecodes--;
      lastActivity = now();
      if (run) run.decodeMs += now() - started;
    }
  };
  const fetchOriginal = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const read = String(args[0]).includes("/api/playback/");
    if (read) pendingReads++;
    try {
      return await fetchOriginal(...args);
    } finally {
      if (read) {
        pendingReads--;
        lastActivity = now();
      }
    }
  };
  for (const method of ["drawImage", "clearRect", "stroke", "fillText"]) {
    const original = CanvasRenderingContext2D.prototype[method];
    CanvasRenderingContext2D.prototype[method] = function (...args) {
      const started = now(),
        run = active;
      const value = original.apply(this, args);
      if (this.canvas.isConnected) {
        lastActivity = now();
        if (run) {
          const key = `${this.canvas.closest("[data-panel]")?.dataset.panel || "other"}:${method}`;
          run.canvas[key] = (run.canvas[key] || 0) + 1;
          run.canvasMs += now() - started;
        }
      } else if (run) {
        run.seriesRasterCalls = (run.seriesRasterCalls || 0) + 1;
        run.canvasMs += now() - started;
      }
      return value;
    };
  }
  for (const [prototype, methods] of [
    [
      Document.prototype,
      ["createElement", "createTextNode", "createDocumentFragment"],
    ],
    [Node.prototype, ["cloneNode"]],
  ])
    for (const method of methods) {
      const original = prototype[method];
      prototype[method] = function (...args) {
        if (active) active.domAllocations++;
        return original.apply(this, args);
      };
    }
  const wait = async (predicate, label = "selected transition") => {
    const deadline = now() + 20000;
    while (!predicate()) {
      if (now() > deadline) throw Error(`Timed out: ${label}`);
      await frame();
    }
  };
  const settled = () =>
    wait(
      () => !pendingDecodes && !pendingReads && now() - lastActivity > 1200,
      "settled external work",
    );
  const pixel = (step) => {
    const canvas = document.querySelector("#game-canvas");
    return (
      !canvas ||
      canvas.getContext("2d").getImageData(0, 0, 1, 1).data[0] === step % 256
    );
  };
  const channel = new BroadcastChannel("gradlab-fixture-performance");
  const replies = new Map();
  channel.onmessage = async ({ data }) => {
    if (data.type === "reply") {
      replies.set(data.id, data);
      return;
    }
    if (data.type === "begin") {
      await settled();
      active = empty(data.name);
    }
    if (data.type === "selected")
      await wait(() => pixel(data.step) && !pendingDecodes, "companion frame");
    const result = data.type === "end" && active ? finish(active) : null;
    if (data.type === "end") active = null;
    channel.postMessage({ type: "reply", id: data.id, result });
  };
  const peer = async (type, extra = {}) => {
    if (new URLSearchParams(location.search).get("workspace") !== "paired")
      return null;
    const id = crypto.randomUUID();
    channel.postMessage({ type, id, ...extra });
    await wait(() => replies.has(id), "companion response");
    const result = replies.get(id);
    replies.delete(id);
    return result.result;
  };
  const observer = new MutationObserver((records) => {
    if (active)
      active.domMutations += records.filter(
        (record) => !record.target.closest?.("#performance-fixture"),
      ).length;
  });
  document.addEventListener("DOMContentLoaded", () => {
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      characterData: true,
    });
    const controls = document.createElement("aside");
    controls.id = "performance-fixture";
    controls.style.cssText =
      "position:fixed;right:8px;bottom:8px;z-index:10001;max-width:55vw;max-height:45vh;overflow:auto;background:#121015;color:white;padding:8px;font-size:12px";
    const button = document.createElement("button");
    button.textContent = "Measure player workload";
    const output = document.createElement("pre");
    output.id = "performance-results";
    const metadata = document.createElement("pre");
    metadata.id = "performance-environment";
    metadata.hidden = true;
    metadata.textContent = JSON.stringify({
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      cores: navigator.hardwareConcurrency,
      devicePixelRatio,
      heapAvailable: heap() !== null,
      entryTypes: PerformanceObserver.supportedEntryTypes,
    });
    controls.append(button, output, metadata);
    document.body.append(controls);
    button.onclick = async () => {
      button.disabled = true;
      output.textContent = "";
      const measure = async (name, workload) => {
        await settled();
        await peer("begin", { name });
        active = empty(name);
        const run = active,
          started = now();
        await workload(run);
        await frame();
        active = null;
        run.durationMs = now() - started;
        const companion = await peer("end");
        output.textContent +=
          JSON.stringify({ ...finish(run), companion }) + "\n";
      };
      try {
        const scrubber = document.querySelector("#timeline-scrubber");
        if (!scrubber || scrubber.disabled)
          throw Error("Load a paused Checkpoint before measuring");
        const first = Math.max(1, Number(scrubber.min)),
          last = Number(scrubber.max);
        for (let repeat = 0; repeat < 3; repeat++) {
          await measure(`idle-${repeat}`, async () => {
            const until = now() + 1200;
            while (now() < until) await frame();
          });
          await measure(`hover-${repeat}`, async (run) => {
            const canvas = document.querySelector(".telemetry-chart");
            if (!canvas) throw Error("Measure in Stats with Player also open");
            const rect = canvas.getBoundingClientRect();
            for (let i = 0; i < 90; i++) {
              const started = now();
              canvas.dispatchEvent(
                new PointerEvent("pointermove", {
                  bubbles: true,
                  clientX: rect.left + rect.width * (0.1 + i / 112),
                  clientY: rect.top + rect.height / 2,
                }),
              );
              run.inputDispatchMs += now() - started;
              await frame();
              run.latencies.push(now() - started);
            }
            canvas.dispatchEvent(
              new PointerEvent("pointerleave", { bubbles: true }),
            );
          });
          await measure(`seek-${repeat}`, async (run) => {
            for (let i = 0; i < 12; i++) {
              const step = i % 2 ? last : first,
                started = now();
              scrubber.value = String(step);
              scrubber.dispatchEvent(new Event("input", { bubbles: true }));
              run.inputDispatchMs += now() - started;
              await wait(
                () =>
                  document
                    .querySelector("#timeline")
                    .getAttribute("aria-busy") === "false" &&
                  document
                    .querySelector("#timeline-label")
                    .textContent.includes(`STEP ${step}`) &&
                  pixel(step) &&
                  !pendingDecodes,
              );
              await peer("selected", { step });
              await frame();
              run.latencies.push(now() - started);
            }
          });
        }
        output.dataset.status = "complete";
      } catch (error) {
        active = null;
        output.textContent += `FAIL ${error.stack}`;
        output.dataset.status = "failed";
      } finally {
        button.disabled = false;
      }
    };
  });
})();
