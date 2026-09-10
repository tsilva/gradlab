export function recordingDescription(trajectory = {}) {
  if (trajectory.imported) {
    const range = `${trajectory.first_step}–${trajectory.last_step}`;
    const completeness = trajectory.complete ? "Completed episode" : "Unfinished episode";
    const prefix = trajectory.first_step > 1 ? " · earlier transitions absent" : "";
    return `${completeness} · transitions ${range}${prefix} · ${trajectory.classification.replaceAll("_", " ")} Playback · inspection only`;
  }
  if (trajectory.error) return `Recording storage failed: ${trajectory.error}`;
  if (!trajectory.transitions) return trajectory.enabled ? "Recording current episode" : "Recording off";
  const prefix = trajectory.first_step > 1 ? ` · starts at transition ${trajectory.first_step}` : "";
  return `${trajectory.enabled ? "Recording" : "Recorded"} ${trajectory.transitions} transitions${prefix}`;
}

export function mountTrajectoryControls({ command, getState, request, toast }) {
  const retry = document.querySelector("#trajectory-retry");
  const download = document.querySelector("#trajectory-download");
  const importButton = document.querySelector("#trajectory-import");
  const file = document.querySelector("#trajectory-file");
  const seek = document.querySelector("#trajectory-seek");
  const previous = document.querySelector("#trajectory-previous");
  const next = document.querySelector("#trajectory-next");
  const dialog = document.querySelector("#trajectory-download-dialog");
  const confirm = document.querySelector("#trajectory-download-confirm");
  const cancel = document.querySelector("#trajectory-download-cancel");
  const progress = document.querySelector("#trajectory-download-progress");
  const downloadError = document.querySelector("#trajectory-download-error");
  let preparing = false;
  let importing = false;

  retry.addEventListener("click", () => command("set_recording", { enabled: true }));
  seek.addEventListener("change", () => command("seek", { step: Number(seek.value) }));
  seek.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      command("seek", { step: Number(seek.value) });
    }
  });
  previous.addEventListener("click", () => command("step_backward"));
  next.addEventListener("click", () => command("step", { count: 1 }));
  importButton.addEventListener("click", () => file.click());
  file.addEventListener("change", async () => {
    const archive = file.files?.[0];
    if (!archive) return;
    importing = true;
    importButton.disabled = true;
    importButton.setAttribute("aria-busy", "true");
    importButton.setAttribute("aria-label", "Importing episode");
    try {
      await request("/api/trajectory/import", archive);
      toast("Episode imported. Playback uses stored observations and images.");
    } catch (error) {
      toast(error.message, true);
    } finally {
      importing = false;
      file.value = "";
      importButton.removeAttribute("aria-busy");
      importButton.setAttribute("aria-label", "Import episode");
      render();
    }
  });
  download.addEventListener("click", () => {
    progress.hidden = true;
    downloadError.hidden = true;
    dialog.showModal();
  });
  cancel.addEventListener("click", () => dialog.close());
  dialog.addEventListener("cancel", (event) => {
    if (preparing) event.preventDefault();
  });
  confirm.addEventListener("click", async () => {
    if (preparing) return;
    preparing = true;
    download.disabled = true;
    confirm.disabled = true;
    cancel.disabled = true;
    progress.hidden = false;
    downloadError.hidden = true;
    dialog.setAttribute("aria-busy", "true");
    try {
      const result = await request("/api/trajectory/download");
      const link = document.createElement("a");
      link.href = result.url;
      link.download = result.filename;
      document.body.append(link);
      link.click();
      link.remove();
      dialog.close();
    } catch (error) {
      downloadError.textContent = error.message;
      downloadError.hidden = false;
    } finally {
      preparing = false;
      confirm.disabled = false;
      cancel.disabled = false;
      progress.hidden = true;
      dialog.removeAttribute("aria-busy");
      render();
    }
  });

  function render() {
    const state = getState();
    const snapshot = state.liveSnapshot || state.snapshot;
    const trajectory = snapshot?.trajectory || {};
    const imported = Boolean(trajectory.imported);
    const available = trajectory.available || imported;
    document.querySelector("#trajectory-controls").hidden = !available || (!imported && !trajectory.error);
    document.querySelector("#trajectory-navigation").hidden = !imported;
    retry.hidden = !trajectory.error;
    retry.disabled = !state.hasControl;
    download.hidden = !available || imported;
    download.disabled = preparing || !trajectory.transitions;
    importButton.disabled = importing || !state.hasControl;
    const status = document.querySelector("#trajectory-status");
    status.hidden = !imported && !trajectory.error;
    status.textContent = status.hidden ? "" : recordingDescription(trajectory);
    if (!preparing) confirm.disabled = imported || !trajectory.transitions;
    if (imported) {
      seek.min = String(trajectory.first_step);
      seek.max = String(trajectory.last_step);
      if (document.activeElement !== seek) seek.value = String(Math.max(trajectory.first_step, trajectory.current_step));
      seek.disabled = !state.hasControl;
      previous.disabled = !state.hasControl || trajectory.current_step <= trajectory.first_step;
      next.disabled = !state.hasControl || trajectory.current_step >= trajectory.last_step;
    }
  }
  return { render };
}
