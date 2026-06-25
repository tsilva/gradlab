from __future__ import annotations

import argparse
import json
import socket
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rlab.monitoring.state import MonitorOptions, collect_state


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>rlab Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f6f3;
      --surface: #ffffff;
      --surface-2: #fbfbfa;
      --text: #171717;
      --muted: #686a70;
      --border: #e5e2dc;
      --border-strong: #d4d0c8;
      --purple: #7c3aed;
      --teal: #0f9f8f;
      --amber: #c47a16;
      --red: #cf3d34;
      --shadow: 0 18px 45px rgba(24, 24, 27, 0.08);
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }
    button {
      font: inherit;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--border-strong); background: var(--surface-2); }
    button:disabled {
      color: #9a9a96;
      background: #f4f3ef;
      border-color: var(--border);
      cursor: not-allowed;
    }
    button:disabled:hover { background: #f4f3ef; border-color: var(--border); }
    .app { min-height: 100vh; padding: 18px 22px 22px; }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      height: 48px;
      margin-bottom: 14px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .mark {
      width: 28px;
      height: 28px;
      border-radius: 8px;
      background: #111827;
      color: #ffffff;
      display: grid;
      place-items: center;
      font-weight: 800;
      letter-spacing: 0;
    }
    h1 {
      margin: 0;
      font-size: 19px;
      line-height: 1.1;
      font-weight: 720;
      letter-spacing: 0;
    }
    .source {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 44vw;
    }
    .actions { display: flex; align-items: center; gap: 8px; }
    .primary {
      background: #171717;
      color: #ffffff;
      border-color: #171717;
    }
    .primary:hover { background: #2a2a2a; border-color: #2a2a2a; }
    .viewbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    .segments {
      display: inline-flex;
      background: #edeae3;
      border: 1px solid var(--border);
      padding: 3px;
      border-radius: 9px;
    }
    .segment {
      border: 0;
      background: transparent;
      padding: 7px 12px;
      color: var(--muted);
      border-radius: 7px;
    }
    .segment.active {
      background: #ffffff;
      color: var(--text);
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
    }
    .hint { color: var(--muted); font-size: 12px; }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 14px;
      align-items: start;
    }
    .table-shell {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .table-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--border);
    }
    .table-title { font-size: 15px; font-weight: 700; }
    .count { font-size: 12px; color: var(--muted); }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      vertical-align: middle;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #fbfaf8;
    }
    tr { cursor: pointer; }
    tbody tr:hover { background: #fbfbfa; }
    tbody tr.selected {
      background: #fbf9ff;
      box-shadow: inset 3px 0 0 var(--purple);
    }
    tbody tr.highlighted {
      background: #f1fffb;
      box-shadow: inset 3px 0 0 var(--teal);
    }
    tbody tr.selected.highlighted {
      background: #f7f4ff;
      box-shadow: inset 3px 0 0 var(--purple), inset 6px 0 0 var(--teal);
    }
    tbody tr:last-child td { border-bottom: 0; }
    .cell-link {
      display: inline-flex;
      align-items: center;
      border: 0;
      background: transparent;
      color: var(--text);
      padding: 0;
      font-weight: 650;
      text-decoration: underline;
      text-decoration-color: var(--border-strong);
      text-underline-offset: 3px;
      cursor: pointer;
    }
    .cell-link:hover {
      color: var(--teal);
      background: transparent;
      border: 0;
    }
    .mono {
      font-family:
        "SFMono-Regular", "Cascadia Code", "Liberation Mono", Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 22px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 650;
      line-height: 1;
      background: #f1f1ef;
      color: #525252;
    }
    .running, .available { background: #e9f8f4; color: #08796d; }
    .busy, .pending, .warning { background: #fff2df; color: #a35f00; }
    .failed, .offline, .unavailable { background: #fff0ee; color: #b7352d; }
    .progress {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .bar {
      height: 6px;
      width: 62px;
      background: #ece9e1;
      border-radius: 99px;
      overflow: hidden;
      flex: 0 0 auto;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: var(--teal);
      border-radius: inherit;
    }
    .usage-bars {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 5px 8px;
      min-width: 0;
    }
    .usage-mini {
      display: grid;
      grid-template-columns: 28px minmax(32px, 1fr);
      gap: 5px;
      align-items: center;
      min-width: 0;
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .meter {
      height: 6px;
      background: #ece9e1;
      border-radius: 99px;
      overflow: hidden;
      min-width: 0;
    }
    .meter > span {
      display: block;
      height: 100%;
      width: var(--pct);
      background: var(--teal);
      border-radius: inherit;
    }
    .meter.warn > span { background: var(--amber); }
    .meter.hot > span { background: var(--red); }
    .metric-value {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .metric-value .meter { height: 8px; }
    .metric-label { white-space: nowrap; }
    .resource-stack {
      padding: 12px 16px 4px;
      border-bottom: 1px solid var(--border);
    }
    .resource-row {
      display: grid;
      grid-template-columns: 48px minmax(0, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 7px 0;
    }
    .resource-name {
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .resource-row .meter { height: 9px; }
    .attention { color: var(--amber); font-weight: 650; }
    .attention.failed { color: var(--red); }
    .detail {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      position: sticky;
      top: 16px;
    }
    .detail-head {
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }
    .detail-title {
      font-size: 18px;
      font-weight: 720;
      margin-bottom: 6px;
    }
    .detail-subtitle {
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .fields { padding: 8px 16px 4px; }
    .field {
      display: grid;
      grid-template-columns: 94px minmax(0, 1fr);
      gap: 10px;
      padding: 9px 0;
      border-bottom: 1px solid var(--border);
    }
    .field:last-child { border-bottom: 0; }
    .key {
      color: var(--muted);
      font-size: 12px;
      text-transform: lowercase;
    }
    .value {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .value-link {
      display: block;
      width: 100%;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: #0b63ce;
      text-align: left;
      font: inherit;
      text-decoration: underline;
      text-underline-offset: 2px;
    }
    .value-link:hover {
      background: transparent;
      color: #074a99;
      border-color: transparent;
    }
    .detail-actions {
      display: flex;
      gap: 8px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--border);
      background: #fbfaf8;
    }
    .detail-actions button { flex: 1; padding: 8px 10px; }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(23, 23, 23, 0.38);
    }
    .modal-backdrop[hidden] { display: none; }
    .modal {
      width: min(760px, 100%);
      max-height: min(78vh, 720px);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
    }
    .modal-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }
    .modal-body {
      margin: 0;
      padding: 14px 16px;
      overflow: auto;
      background: #171717;
      color: #f7f7f4;
      font-family:
        "SFMono-Regular", "Cascadia Code", "Liberation Mono", Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre;
    }
    .empty {
      padding: 42px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .detail { position: static; }
      .source { display: none; }
      th:nth-child(4), td:nth-child(4),
      th:nth-child(7), td:nth-child(7) { display: none; }
    }
  </style>
</head>
<body>
  <main class="app">
    <header class="topbar">
      <div class="brand">
        <div class="mark">r</div>
        <div>
          <h1>rlab Monitor</h1>
          <div class="source" id="source">Loading...</div>
        </div>
      </div>
      <div class="actions">
        <button id="refresh">Refresh</button>
        <button class="primary" id="open-wandb">Open W&amp;B</button>
      </div>
    </header>
    <section class="viewbar">
      <div class="segments" role="tablist" aria-label="Monitor view">
        <button class="segment active" id="jobs-tab" role="tab">Jobs</button>
        <button class="segment" id="devices-tab" role="tab">Devices</button>
      </div>
      <div class="hint" id="hint">Switch to Devices to see target availability.</div>
    </section>
    <section class="layout">
      <div class="table-shell">
        <div class="table-head">
          <div class="table-title" id="table-title">Queue</div>
          <div class="count" id="count"></div>
        </div>
        <div id="table-wrap"></div>
      </div>
      <aside class="detail" id="detail"></aside>
    </section>
  </main>
  <div class="modal-backdrop" id="json-modal" hidden>
    <section class="modal" role="dialog" aria-modal="true" aria-labelledby="json-modal-title">
      <div class="modal-head">
        <div class="modal-title" id="json-modal-title"></div>
        <button id="json-modal-close" type="button">Close</button>
      </div>
      <pre class="modal-body" id="json-modal-body"></pre>
    </section>
  </div>
  <script>
    const params = new URLSearchParams(window.location.search);
    const initialGoal = params.get("goal") || "";
    const state = {
      view: params.get("view") === "devices" ? "devices" : "jobs",
      data: null,
      selectedId: null,
      highlightedJobIds: new Set(),
      loading: false,
    };
    const refreshMs = 5000;
    const columns = {
      jobs: ["Job", "Kind", "Target", "Where", "State", "Progress", "Attention"],
      devices: [
        "Host",
        "Launch target",
        "State",
        "Slots",
        "Usage",
        "Running jobs",
        "Queued jobs",
        "Health",
      ],
    };
    const columnWidths = {
      jobs: ["9%", "6%", "34%", "16%", "11%", "10%", "14%"],
      devices: ["10%", "13%", "9%", "8%", "16%", "17%", "16%", "11%"],
    };

    function cls(value) {
      return String(value || "").toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    }

    function progressCell(value) {
      if (!value) return "";
      const match = String(value).match(/^(\d+)(?:%|$)/);
      const pct = match ? Math.max(0, Math.min(100, Number(match[1]))) : 0;
      if (!pct) return `<span>${value}</span>`;
      return `<div class="progress"><div class="bar"><span style="width:${pct}%"></span></div><span>${value}</span></div>`;
    }

    function pct(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return 0;
      return Math.max(0, Math.min(100, number));
    }

    function meterClass(value) {
      const valuePct = pct(value);
      if (valuePct >= 90) return "meter hot";
      if (valuePct >= 70) return "meter warn";
      return "meter";
    }

    function metricBars(metrics) {
      const order = [["cpu", "CPU"], ["gpu", "GPU"], ["memory", "RAM"], ["vram", "VRAM"]];
      const bars = order.map(([key, label]) => {
        const metric = metrics && metrics[key];
        if (!metric) return "";
        const value = pct(metric.percent);
        return `
          <div class="usage-mini" title="${label} ${value.toFixed(0)}%">
            <span>${label}</span>
            <div class="${meterClass(value)}"><span style="--pct:${value}%"></span></div>
          </div>
        `;
      }).filter(Boolean).join("");
      return bars ? `<div class="usage-bars">${bars}</div>` : "";
    }

    function metricDetailValue(item, key, value) {
      const metric = item.metrics && item.metrics[key];
      if (!metric) return null;
      const valuePct = pct(metric.percent);
      return `
        <div class="metric-value" title="${escapeHtml(value)}">
          <div class="${meterClass(valuePct)}"><span style="--pct:${valuePct}%"></span></div>
          <span class="metric-label">${escapeHtml(value)}</span>
        </div>
      `;
    }

    function resourceStack(item) {
      if (state.view !== "devices" || !item.metrics) return "";
      const order = [["cpu", "CPU"], ["memory", "RAM"], ["gpu", "GPU"], ["vram", "VRAM"]];
      const rows = order.map(([key, label]) => {
        const metric = item.metrics[key];
        if (!metric) return "";
        const valuePct = pct(metric.percent);
        const metricLabel = metric.label || `${valuePct.toFixed(0)}%`;
        return `
          <div class="resource-row">
            <div class="resource-name">${label}</div>
            <div class="${meterClass(valuePct)}"><span style="--pct:${valuePct}%"></span></div>
            <div class="metric-label">${escapeHtml(metricLabel)}</div>
          </div>
        `;
      }).filter(Boolean).join("");
      return rows ? `<div class="resource-stack">${rows}</div>` : "";
    }

    function jobCountLabel(ids, noun) {
      const count = Array.isArray(ids) ? ids.length : 0;
      if (!count) return "";
      return `${count} ${noun}`;
    }

    function rowValues(item) {
      if (state.view === "jobs") {
        return [item.id, item.kind, item.target, item.where, item.state, item.progress, item.attention];
      }
      return [
        item.device,
        item.target,
        item.state,
        item.capacity,
        item.usage,
        jobCountLabel(item.current_jobs, "running"),
        jobCountLabel(item.queued_jobs, "queued"),
        item.last_check,
      ];
    }

    function itemId(item) {
      return state.view === "jobs" ? item.id : item.id;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function payloadJson(item) {
      return JSON.stringify(item.payload || {}, null, 2);
    }

    function resultPayloadKey(payload) {
      return Object.keys(payload || {}).find((key) => key.endsWith("_results"));
    }

    function pathValue(root, path) {
      let current = root;
      for (const part of path) {
        if (current === null || current === undefined || typeof current !== "object") {
          return { found: false, value: undefined };
        }
        if (!Object.prototype.hasOwnProperty.call(current, part)) {
          return { found: false, value: undefined };
        }
        current = current[part];
      }
      return { found: true, value: current };
    }

    function pathLabel(path) {
      return `payload${path.map((part) => {
        return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(part)
          ? `.${part}`
          : `[${JSON.stringify(part)}]`;
      }).join("")}`;
    }

    function payloadPathsForField(item, key) {
      const payload = item.payload || {};
      const configKey = payload.config_key || "";
      const resultsKey = resultPayloadKey(payload);
      const metricsBase = resultsKey ? [resultsKey, "metrics_json"] : null;
      const paths = {
        goal: [["context", "goal_slug"], ["job", "goal_slug"]],
        spec: [["context", "spec_slug"], ["job", "spec_slug"]],
        profile: [["job", "profile_id"]],
        run: [["job", "run_name"]],
        worker: [["job", "lease_owner"]],
        lease: [["job", "lease_expires_at"]],
        heartbeat: [["job", "heartbeat_at"]],
        wandb: resultsKey ? [[resultsKey, "wandb_url"]] : [],
        artifact: [
          ...(resultsKey ? [[resultsKey, "artifact_refs"], [resultsKey, "model_ref"]] : []),
          ...(configKey ? [["job", configKey, "artifact_ref"], ["job", configKey, "model_artifact"], ["job", configKey, "model_path"]] : []),
        ],
        episodes: configKey ? [["job", configKey, "episodes"]] : [],
        seed: configKey ? [["job", configKey, "seed"]] : [],
        n_envs: configKey ? [["job", configKey, "n_envs"]] : [],
        fps: metricsBase ? [[...metricsBase, "time/fps"], [...metricsBase, "throughput/loop_fps"]] : [],
        completion: metricsBase ? [[...metricsBase, "train/done/all"], [...metricsBase, "completion_rate"], [...metricsBase, "completion_count"]] : [],
        reward: metricsBase ? [[...metricsBase, "reward_mean"], [...metricsBase, "mean_reward"]] : [],
        max_x: metricsBase ? [[...metricsBase, "max_x_position_mean"], [...metricsBase, "max_x"]] : [],
      };
      const candidates = paths[key] ? [...paths[key]] : [];
      if (configKey) candidates.push(["job", configKey, key]);
      candidates.push(["job", key]);
      return candidates;
    }

    function payloadReferenceForField(item, key) {
      if (state.view !== "jobs" || !item.payload) return null;
      for (const path of payloadPathsForField(item, key)) {
        const found = pathValue(item.payload, path);
        if (found.found && found.value !== undefined && found.value !== null && found.value !== "") {
          return { path, value: found.value };
        }
      }
      return null;
    }

    function modalJson(value) {
      const json = JSON.stringify(value, null, 2);
      return json === undefined ? String(value) : json;
    }

    function openJsonModal(title, path, value) {
      document.getElementById("json-modal-title").textContent = `${title} · ${pathLabel(path)}`;
      document.getElementById("json-modal-body").textContent = modalJson(value);
      document.getElementById("json-modal").hidden = false;
    }

    function closeJsonModal() {
      document.getElementById("json-modal").hidden = true;
    }

    function urlForView(view) {
      const next = new URLSearchParams();
      if (initialGoal) next.set("goal", initialGoal);
      if (view === "devices") next.set("view", "devices");
      const query = next.toString();
      return query ? `/?${query}` : "/";
    }

    function stateApiUrl() {
      if (!initialGoal) return "/api/state";
      const next = new URLSearchParams();
      next.set("goal", initialGoal);
      return `/api/state?${next}`;
    }

    function showJobs(jobIds) {
      const ids = (jobIds || []).filter(Boolean);
      if (!ids.length) return;
      state.view = "jobs";
      state.highlightedJobIds = new Set(ids);
      state.selectedId = ids[0];
      history.replaceState(null, "", urlForView("jobs"));
      render();
    }

    function renderTable(items) {
      const tableWrap = document.getElementById("table-wrap");
      if (!items.length) {
        tableWrap.innerHTML = `<div class="empty">No rows in this view.</div>`;
        return;
      }
      const widths = columnWidths[state.view].map((width) => `<col style="width:${width}">`).join("");
      const head = columns[state.view].map((name) => `<th>${name}</th>`).join("");
      const body = items.map((item) => {
        const id = itemId(item);
        const selected = id === state.selectedId ? " selected" : "";
        const highlighted = state.view === "jobs" && state.highlightedJobIds.has(id) ? " highlighted" : "";
        const values = rowValues(item);
        const cells = values.map((value, index) => {
          const safeValue = escapeHtml(value || "");
          const title = safeValue ? ` title="${safeValue}"` : "";
          if (columns[state.view][index] === "State") {
            return `<td${title}><span class="pill ${cls(value)}">${safeValue}</span></td>`;
          }
          if (columns[state.view][index] === "Progress") {
            return `<td${title}>${progressCell(safeValue)}</td>`;
          }
          if (columns[state.view][index] === "Usage") {
            return `<td${title}>${metricBars(item.metrics)}</td>`;
          }
          if (state.view === "devices" && columns[state.view][index] === "Running jobs") {
            if (!value) return `<td></td>`;
            const ids = encodeURIComponent(JSON.stringify(item.current_jobs || []));
            return `<td${title}><button class="cell-link" type="button" data-running-jobs="${ids}">${safeValue}</button></td>`;
          }
          if (columns[state.view][index] === "Attention") {
            return `<td${title}><span class="attention ${cls(item.state)}">${safeValue}</span></td>`;
          }
          const mono = index === 0 ? " mono" : "";
          return `<td class="${mono}"${title}>${safeValue}</td>`;
        }).join("");
        return `<tr class="${selected}${highlighted}" data-id="${escapeHtml(id)}">${cells}</tr>`;
      }).join("");
      tableWrap.innerHTML = `<table><colgroup>${widths}</colgroup><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
      for (const button of tableWrap.querySelectorAll("[data-running-jobs]")) {
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          showJobs(JSON.parse(decodeURIComponent(button.dataset.runningJobs)));
        });
      }
      for (const row of tableWrap.querySelectorAll("tbody tr")) {
        row.addEventListener("click", () => {
          state.selectedId = row.dataset.id;
          render();
        });
      }
    }

    function renderDetail(item) {
      const detail = document.getElementById("detail");
      if (!item) {
        detail.innerHTML = `<div class="empty">Select a row.</div>`;
        return;
      }
      const title = state.view === "jobs" ? item.id : item.device;
      const subtitle = state.view === "jobs" ? `${item.target} · ${item.where}` : item.target;
      const details = item.details || {};
      const resourceKeys = new Set(["cpu", "memory", "gpu", "vram"]);
      const fields = Object.entries(details)
        .filter(([, value]) => value !== "" && value !== null && value !== undefined)
        .filter(([key]) => state.view !== "devices" || !resourceKeys.has(key))
        .map(([key, value]) => {
          const payloadReference = payloadReferenceForField(item, key);
          const safeKey = escapeHtml(key);
          const safeValue = escapeHtml(value);
          const metricValue = metricDetailValue(item, key, value);
          const valueHtml = payloadReference
            ? `<button class="value-link" type="button" data-payload-field="${safeKey}" title="${safeValue}">${safeValue}</button>`
            : metricValue
              ? metricValue
            : `<div class="value" title="${safeValue}">${safeValue}</div>`;
          return `
            <div class="field">
              <div class="key">${safeKey}</div>
              ${valueHtml}
            </div>
          `;
        }).join("");
      const hasJobPayload = state.view === "jobs" && item.payload;
      const hasDeviceJobs = state.view === "devices" && (item.current_job || item.queued_job);
      const actions = hasJobPayload
        ? `<div class="detail-actions"><button data-action="copy-payload">Copy JSON</button></div>`
        : hasDeviceJobs
          ? `<div class="detail-actions"><button data-action="open-jobs">Open jobs</button></div>`
          : "";
      detail.innerHTML = `
        <div class="detail-head">
          <div class="detail-title">${escapeHtml(title)}</div>
          <div class="detail-subtitle">${escapeHtml(subtitle)}</div>
        </div>
        ${resourceStack(item)}
        <div class="fields">${fields || `<div class="empty">No details.</div>`}</div>
        ${actions}
      `;
      for (const payloadButton of detail.querySelectorAll("[data-payload-field]")) {
        payloadButton.addEventListener("click", (event) => {
          event.stopPropagation();
          const field = payloadButton.dataset.payloadField;
          const payloadReference = payloadReferenceForField(item, field);
          if (payloadReference) openJsonModal(`${title} · ${field}`, payloadReference.path, payloadReference.value);
        });
      }
      const openJobs = detail.querySelector("[data-action='open-jobs']");
      if (openJobs) {
        openJobs.addEventListener("click", () => {
          const ids = item.current_jobs && item.current_jobs.length ? item.current_jobs : item.queued_jobs;
          showJobs(ids);
        });
      }
      const copyPayload = detail.querySelector("[data-action='copy-payload']");
      if (copyPayload) {
        copyPayload.addEventListener("click", async () => {
          await navigator.clipboard.writeText(payloadJson(item));
          copyPayload.textContent = "Copied";
          setTimeout(() => { copyPayload.textContent = "Copy JSON"; }, 1200);
        });
      }
    }

    function activeItems() {
      return state.view === "jobs" ? state.data.jobs : state.data.devices;
    }

    function render() {
      if (!state.data) return;
      const items = activeItems();
      if (!state.selectedId && items.length) state.selectedId = itemId(items[0]);
      const selected = items.find((item) => itemId(item) === state.selectedId) || items[0];
      document.getElementById("jobs-tab").classList.toggle("active", state.view === "jobs");
      document.getElementById("devices-tab").classList.toggle("active", state.view === "devices");
      document.getElementById("hint").textContent = state.view === "jobs"
        ? "Switch to Devices to see target availability."
        : "Switch to Jobs to see the queue first.";
      document.getElementById("table-title").textContent = state.view === "jobs" ? "Queue" : "Devices";
      document.getElementById("count").textContent = `${items.length} rows`;
      document.getElementById("source").textContent =
        `${state.data.source.campaign}: ${state.data.source.message} · ${state.data.refreshed_at}`;
      renderTable(items);
      renderDetail(selected);
    }

    async function loadState({ preserveSelection = true } = {}) {
      if (state.loading) return;
      state.loading = true;
      const selectedId = state.selectedId;
      try {
        const response = await fetch(stateApiUrl(), { cache: "no-store" });
        state.data = await response.json();
        state.selectedId = preserveSelection ? selectedId : null;
        render();
      } finally {
        state.loading = false;
      }
    }

    document.getElementById("jobs-tab").addEventListener("click", () => {
      state.view = "jobs";
      state.selectedId = null;
      state.highlightedJobIds = new Set();
      history.replaceState(null, "", urlForView("jobs"));
      render();
    });
    document.getElementById("devices-tab").addEventListener("click", () => {
      state.view = "devices";
      state.selectedId = null;
      state.highlightedJobIds = new Set();
      history.replaceState(null, "", urlForView("devices"));
      render();
    });
    document.getElementById("refresh").addEventListener("click", () => loadState());
    document.getElementById("open-wandb").addEventListener("click", () => {
      window.open("https://wandb.ai/tsilva/SuperMarioBros-NES", "_blank", "noopener");
    });
    document.getElementById("json-modal-close").addEventListener("click", closeJsonModal);
    document.getElementById("json-modal").addEventListener("click", (event) => {
      if (event.target.id === "json-modal") closeJsonModal();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeJsonModal();
    });
    loadState({ preserveSelection: false }).catch((error) => {
      document.getElementById("source").textContent = `monitor error: ${error}`;
    });
    setInterval(() => {
      loadState().catch((error) => {
        document.getElementById("source").textContent = `monitor error: ${error}`;
      });
    }, refreshMs);
  </script>
</body>
</html>
"""


class MonitorHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, options: MonitorOptions, **kwargs) -> None:
        self.options = options
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def send_bytes(self, body: bytes, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            goal = query.get("goal", [self.options.goal])[0] or self.options.goal
            options = MonitorOptions(
                repo_root=self.options.repo_root,
                goal=goal,
                direct=self.options.direct,
                sample=self.options.sample,
                limit=self.options.limit,
            )
            payload = json.dumps(collect_state(options), sort_keys=True).encode("utf-8")
            self.send_bytes(payload, content_type="application/json; charset=utf-8")
            return
        self.send_bytes(b"not found\n", content_type="text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)


def local_ip(host: str) -> str:
    if host not in {"0.0.0.0", "::"}:
        return host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def run_server(options: MonitorOptions, *, host: str, port: int, open_browser: bool = False) -> None:
    handler = partial(MonitorHandler, options=options)
    httpd = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = httpd.server_address[:2]
    url = f"http://{local_ip(str(actual_host))}:{actual_port}/"
    print(f"rlab monitor: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nrlab monitor stopped", flush=True)
    finally:
        httpd.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the read-only rlab monitoring UI.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--goal", help="Optional research_goals.slug filter.")
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample rows instead of connecting to the campaign database.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Use 0 for an automatic free port.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--open", action="store_true", help="Open the monitor in the default browser.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    options = MonitorOptions(
        repo_root=args.repo_root.expanduser().resolve(),
        goal=args.goal,
        direct=args.direct,
        sample=args.sample,
        limit=args.limit,
    )
    run_server(options, host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
