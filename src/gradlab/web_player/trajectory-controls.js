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
  const toggle = document.querySelector("#restoration-toggle");
  const discard = document.querySelector("#trajectory-discard");
  const add = document.querySelector("#bookmark-add");
  const bookmarksRoot = document.querySelector("#trajectory-bookmarks");
  const confirmation = document.querySelector("#bookmark-confirmation");
  let pendingCut = null;
  let bookmarkKey = "";
  let resampleControls = [];
  const currentTrajectory = () => (getState().liveSnapshot || getState().snapshot)?.trajectory || {};
  const identity = (trajectory) => ({ episode_id: trajectory.episode_id, trajectory_revision: trajectory.trajectory_revision });
  const canRestore = (trajectory, step) => Boolean(trajectory.restoration?.supported
    && trajectory.restoration.ranges?.some(([first, last]) => step >= first && step <= last));
  function cut(name, step, bookmarkId) {
    const trajectory = currentTrajectory();
    const payload = { ...identity(trajectory), step, bookmark_id: bookmarkId };
    if (trajectory.bookmarks?.some((bookmark) => bookmark.step > step)) {
      pendingCut = { name, payload: { ...payload, confirmed_bookmark_revision: trajectory.bookmark_revision } };
      confirmation.showModal();
    } else command(name, payload);
  }
  confirmation.addEventListener("close", () => {
    if (confirmation.returnValue === "confirm" && pendingCut) command(pendingCut.name, pendingCut.payload);
    pendingCut = null;
  });
  toggle.addEventListener("change", () => command("set_restoration_capture", { enabled: toggle.checked }));
  discard.addEventListener("click", () => cut("discard_future", currentTrajectory().current_step));
  add.addEventListener("click", () => {
    const trajectory = currentTrajectory();
    command("bookmark_add", { ...identity(trajectory), step: trajectory.current_step, name: document.querySelector("#bookmark-name").value });
  });
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
    importButton.textContent = "Importing…";
    try {
      await request("/api/trajectory/import", archive);
      toast("Episode imported. Playback uses stored observations and images.");
    } catch (error) {
      toast(error.message, true);
    } finally {
      importing = false;
      file.value = "";
      importButton.textContent = "Import episode";
      render();
    }
  });
  download.addEventListener("click", async () => {
    preparing = true;
    download.disabled = true;
    download.textContent = "Preparing download…";
    try {
      const result = await request("/api/trajectory/download");
      const link = document.createElement("a");
      link.href = result.url;
      link.download = "episode.gradtraj";
      document.body.append(link);
      link.click();
      link.remove();
    } catch (error) {
      toast(error.message, true);
    } finally {
      preparing = false;
      download.textContent = "Download episode";
      render();
    }
  });

  function render() {
    const state = getState();
    const snapshot = state.liveSnapshot || state.snapshot;
    const trajectory = snapshot?.trajectory || {};
    const imported = Boolean(trajectory.imported);
    const available = trajectory.available || imported;
    document.querySelector("#trajectory-controls").hidden = !available;
    document.querySelector("#trajectory-navigation").hidden = !available;
    retry.hidden = !trajectory.error;
    retry.disabled = !state.hasControl;
    download.hidden = imported;
    download.disabled = preparing || !trajectory.transitions;
    importButton.disabled = importing || !state.hasControl;
    document.querySelector("#trajectory-status").textContent = recordingDescription(trajectory);
    const restore = trajectory.restoration || {};
    toggle.checked = Boolean(restore.enabled);
    toggle.disabled = imported || !state.hasControl || !restore.supported;
    document.querySelector("#restoration-toggle-label").hidden = imported;
    discard.hidden = imported;
    discard.disabled = !state.hasControl || !canRestore(trajectory, trajectory.current_step);
    add.hidden = imported;
    add.disabled = !state.hasControl;
    document.querySelector("#bookmark-name").parentElement.hidden = imported;
    document.querySelector("#restoration-status").textContent = imported
      ? "Bookmarks navigate stored data only. Live restoration is unavailable in imported episodes."
      : restore.error ? `Capture paused: ${restore.error}. Disable capture or retry before continuing.`
      : !restore.supported ? restore.reason || "Exact restoration unavailable."
      : `Resumable positions: ${restore.ranges?.map(([first, last]) => first === last ? first : `${first}–${last}`).join(", ") || "none"}. ${restore.explanation}`;
    if (trajectory.classification === "counterfactual" && !imported) {
      document.querySelector("#trajectory-status").textContent += " · Counterfactual Playback · inspection only";
    }
    const nextKey = JSON.stringify([trajectory.bookmarks, trajectory.trajectory_revision, state.hasControl, imported]);
    if (bookmarkKey !== nextKey) {
      bookmarkKey = nextKey;
      bookmarksRoot.replaceChildren();
      resampleControls = [];
      const markers = document.querySelector("#trajectory-bookmark-steps");
      markers.replaceChildren();
      for (const bookmark of trajectory.bookmarks || []) {
        const marker = document.createElement("option");
        marker.value = String(bookmark.step);
        marker.label = bookmark.name;
        markers.append(marker);
        const item = document.createElement("div");
        item.className = "trajectory-bookmark";
        const navigate = document.createElement("button");
        navigate.type = "button";
        navigate.className = "quiet";
        navigate.textContent = `${bookmark.name} · episode ${bookmark.episode}, step ${bookmark.step}`;
        navigate.disabled = !state.hasControl;
        navigate.addEventListener("click", () => command("seek", { step: bookmark.step }));
        if (bookmark.thumbnail) {
          const image = document.createElement("img");
          image.src = `data:image/png;base64,${bookmark.thumbnail}`;
          image.alt = "";
          navigate.prepend(image);
        }
        item.append(navigate);
        if (!imported) {
          const rename = document.createElement("input");
          rename.value = bookmark.name;
          rename.maxLength = 120;
          rename.setAttribute("aria-label", `Rename ${bookmark.name}`);
          rename.disabled = !state.hasControl;
          const saveName = () => {
            if (rename.value !== bookmark.name) command("bookmark_rename", { ...identity(trajectory), bookmark_id: bookmark.id, name: rename.value });
          };
          rename.addEventListener("change", saveName);
          rename.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              saveName();
            }
          });
          const remove = document.createElement("button");
          remove.type = "button";
          remove.className = "quiet";
          remove.textContent = "Delete";
          remove.disabled = !state.hasControl;
          remove.addEventListener("click", () => command("bookmark_delete", { ...identity(trajectory), bookmark_id: bookmark.id }));
          const resample = document.createElement("button");
          resample.type = "button";
          resample.className = "quiet";
          resample.textContent = "Resample from bookmark";
          resample.disabled = !state.hasControl || !canRestore(trajectory, bookmark.step);
          resampleControls.push({ button: resample, step: bookmark.step });
          resample.addEventListener("click", () => cut("resample_bookmark", bookmark.step, bookmark.id));
          item.append(rename, resample, remove);
        }
        bookmarksRoot.append(item);
      }
    }
    for (const { button, step } of resampleControls) {
      button.disabled = !state.hasControl || !canRestore(trajectory, step);
    }
    if (available) {
      seek.min = String(Math.max(0, trajectory.first_step - 1));
      seek.max = String(trajectory.last_step);
      if (document.activeElement !== seek) seek.value = String(Math.max(trajectory.first_step - 1, trajectory.current_step));
      seek.disabled = !state.hasControl;
      previous.disabled = !state.hasControl || trajectory.current_step <= trajectory.first_step - 1;
      next.disabled = !state.hasControl || trajectory.current_step >= trajectory.last_step;
    }
  }
  return { render };
}
