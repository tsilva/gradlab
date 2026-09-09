"""Serialized live trajectory navigation, capture and replacement commands."""

from __future__ import annotations

from copy import deepcopy

from gradlab.play_bookmarks import bookmark_name, create_bookmark
from gradlab.play_restoration import LiveRestoration

COMMANDS = frozenset(
    {
        "set_restoration_capture",
        "bookmark_add",
        "bookmark_rename",
        "bookmark_delete",
        "discard_future",
        "resample_bookmark",
        "seek",
        "step_backward",
    }
)


class LiveTrajectory:
    def __init__(self, runner, restoration_factory=LiveRestoration):
        self.runner = runner
        self.restoration_factory = restoration_factory
        self.adapter = restoration_factory(runner.session)
        self.enabled = False
        self.capture_preference = None
        self.activation = "explicit"
        self.error = None
        self.ranges = []
        self.inspection_step = None
        self.initial_frame = None
        self.initial_frames = ()
        self.resampled = False
        self.pending_history = None

    def begin(self):
        runner = self.runner
        self.adapter = self.restoration_factory(runner.session)
        self.error = None
        self.ranges = []
        self.inspection_step = None
        self.resampled = False
        self.pending_history = None
        self.initial_frame = deepcopy(runner.session.current_frame)
        self.initial_frames = tuple(deepcopy(runner.session.frames or ()))
        if (
            self.capture_preference is None
            and self.adapter.capability.get("automatic_candidate")
            and 0 < runner.target_fps <= 60
        ):
            # Native Breakout measurements cover payloads up to 256 KiB at 60 Hz.
            # Larger, slower and unmeasured combinations retain the explicit toggle.
            import time
            from gradlab.play_trajectory import pack_record

            try:
                started = time.perf_counter()
                state = self.adapter.capture()
                data = pack_record(state)
                if len(data) <= 256 * 1024 and time.perf_counter() - started <= 0.003:
                    runner.recording.set_initial_restore(state)
                    self.enabled = True
                    self.activation = "automatic"
                    self._add_range(runner.session.step_index)
                    return
            except Exception:
                pass  # Automatic admission failed; show the explicit capture control.
        if self.enabled and self.adapter.capability["supported"]:
            self.capture_initial()

    def _add_range(self, step):
        if self.ranges and self.ranges[-1][1] + 1 == step:
            self.ranges[-1][1] = step
        elif not self.ranges or self.ranges[-1][1] < step:
            self.ranges.append([step, step])

    def capture_initial(self):
        recording = self.runner.recording
        if recording is None:
            raise ValueError("restoration requires a current episode recording")
        if self.runner.session.step_index != recording.metadata["first_step"] - 1:
            return  # Enabling midway captures future boundaries; never invent a prefix.
        try:
            recording.set_initial_restore(self.adapter.capture())
            self._add_range(self.runner.session.step_index)
        except Exception as exc:
            self.error = str(exc)

    def capture(self, transition):
        if not self.enabled or transition.boundary:
            return None
        try:
            state = self.adapter.capture()
            return state
        except Exception as exc:
            self.error = str(exc)
            return None

    def status(self):
        return {
            **self.adapter.capability,
            "enabled": self.enabled,
            "activation": self.activation,
            "error": self.error,
            "ranges": deepcopy(self.ranges),
            "explanation": "Restores environment and Policy memory; sampling keeps advancing. Deterministic actions may produce the same future.",
        }

    def require_identity(self, payload):
        recording = self.runner.recording
        if recording is None:
            raise ValueError("no current episode recording")
        if (
            payload.get("episode_id") != recording.metadata["episode_id"]
            or payload.get("trajectory_revision") != recording.metadata["trajectory_revision"]
        ):
            raise ValueError("stale episode or trajectory revision; refresh the current position")
        return recording

    def apply(self, command):
        runner = self.runner
        payload = command.payload
        name = command.name
        if name == "set_restoration_capture":
            if not self.adapter.capability["supported"]:
                raise ValueError(self.adapter.capability["reason"])
            if runner.recording is None or not runner.recording_enabled:
                raise ValueError("enable episode recording before restoration capture")
            if type(payload.get("enabled")) is not bool:
                raise ValueError("capture enabled must be boolean")
            self.enabled = payload["enabled"]
            self.capture_preference = self.enabled
            self.activation = "explicit"
            self.error = None
            if self.enabled and not runner.awaiting_next_episode:
                self.capture_initial()
            runner._set_state(
                "paused",
                message=self.error
                or "Restoration capture "
                + ("on" if self.enabled else "off; captured points retained"),
            )
            return
        if name in {"seek", "step_backward"}:
            recording = runner.recording
            if recording is None:
                raise ValueError("no current episode recording")
            current = (
                self.inspection_step
                if self.inspection_step is not None
                else recording.status()["last_step"]
            )
            self.seek(current - 1 if name == "step_backward" else payload.get("step"))
            runner._set_state("paused")
            return
        recording = self.require_identity(payload)
        bookmarks = recording.metadata["bookmarks"]
        if name.startswith("bookmark_"):
            if name == "bookmark_add":
                bookmark = create_bookmark(recording, payload.get("step"), payload.get("name"))
                bookmarks.append(bookmark)
            else:
                bookmark = next(
                    (b for b in bookmarks if b["id"] == payload.get("bookmark_id")), None
                )
                if bookmark is None:
                    raise ValueError("bookmark no longer exists")
                if name == "bookmark_rename":
                    bookmark["name"] = bookmark_name(payload.get("name"))
                else:
                    bookmarks.remove(bookmark)
            recording.metadata["bookmark_revision"] += 1
            runner.revision += 1
            runner._publish(runner.session.last_transition)
            return
        if name == "resample_bookmark":
            bookmark = next((b for b in bookmarks if b["id"] == payload.get("bookmark_id")), None)
            if bookmark is None:
                raise ValueError("bookmark no longer exists")
            step = bookmark["step"]
        else:
            step = payload.get("step")
        if type(step) is not int:
            raise ValueError("restoration step must be an integer")
        if not self.adapter.capability["supported"]:
            raise ValueError(self.adapter.capability["reason"])
        if any(b["step"] > step for b in bookmarks):
            if (
                payload.get("confirmed_bookmark_revision")
                != recording.metadata["bookmark_revision"]
            ):
                raise ValueError(
                    "confirm deletion of later bookmarks at the current bookmark revision"
                )
        if len(recording.metadata["resampling"]) >= 10000:
            raise ValueError("current episode reached its 10,000 restoration limit")
        state = recording.restore_point(step)
        replacement = recording.replacement(
            step, sampling_mode=runner.sampling_mode, next_sequence=runner.session.sequence + 1
        )
        try:
            from gradlab.play_web import HISTORY_LIMIT, history_point_payload

            history = [
                history_point_payload(replacement.transition(position)["presentation"])
                for position in range(
                    max(recording.metadata["first_step"], step - HISTORY_LIMIT + 1), step + 1
                )
            ]
            self.adapter.restore(state)
        except BaseException:
            replacement.close()
            raise
        # No fallible disk preparation remains between successful restoration and
        # replacement. Frozen downloads own the previous store's bytes and metadata.
        runner.recording = replacement
        runner.recording_enabled = True
        runner.session.trajectory_recording = True
        self.ranges = [[start, min(end, step)] for start, end in self.ranges if start <= step]
        self.resampled = True
        self.error = None
        runner.awaiting_next_episode = False
        runner.remaining_steps = 0
        runner.continue_target = None
        runner.driver = "policy"
        runner.clear_input()
        runner.capture.abort("resampled Counterfactual Playback")
        runner.history.clear()
        runner.history.extend(history)
        self.pending_history = history
        runner.encoder.clear_trajectory()
        with runner._snapshot_lock:
            runner._snapshot_updates.clear()
            runner._episode_start_snapshot = {}
            runner._episode_start_frames = {}
        self.inspection_step = step
        runner._set_state(
            "playing" if name == "resample_bookmark" else "paused",
            message="Counterfactual Playback · " + self.status()["explanation"],
        )
        recording.close()

    def seek(self, step):
        recording = self.runner.recording
        if (
            type(step) is not int
            or not recording.metadata["first_step"] - 1 <= step <= recording.status()["last_step"]
        ):
            raise ValueError("position is outside the current episode")
        self.inspection_step = step
        self.runner.remaining_steps = 0
        self.runner.continue_target = None

    def inspection(self):
        if self.inspection_step is None:
            return None
        from gradlab.play_web import FRAME_GAME, FRAME_OBSERVATION, history_point_payload
        from gradlab.play_session import render_obs_stack

        runner = self.runner
        recording = runner.recording
        initial = self.inspection_step == recording.metadata["first_step"] - 1
        row = None if initial else recording.transition(self.inspection_step)
        payload = runner._snapshot_payload(None)
        if self.pending_history is not None:
            payload["trajectory_history"] = self.pending_history
            self.pending_history = None
        payload["sequence"] = (
            recording.metadata["initial_snapshot"]["sequence"] if initial else row["sequence"]
        )
        payload["transition"] = None if initial else row["presentation"]
        if not initial:
            payload["session"].update(deepcopy(row["presentation"].get("recorded_session", {})))
            payload["session"]["next_sampling_mode"] = runner.sampling_mode
            payload["policy"]["action_selection"].update(
                requested_mode=runner.sampling_mode,
                effective_mode=(row["presentation"].get("decision") or {}).get(
                    "action_selection_mode"
                ),
            )
        payload["history_point"] = None if initial else history_point_payload(row["presentation"])
        payload["session"].update(
            step=self.inspection_step,
            episode=recording.metadata["episode"],
            total_reward=0.0 if initial else row["return"],
        )
        frame = self.initial_frame if initial else row["after_image"]
        obs = self.initial_frames if initial else row["observation_frames"]
        return payload, {
            FRAME_GAME: frame,
            FRAME_OBSERVATION: render_obs_stack(obs, 1) if obs else None,
        }

    def replay_step(self):
        if self.inspection_step is None:
            return False
        runner = self.runner
        last = runner.recording.status()["last_step"]
        if self.inspection_step >= last:
            self.inspection_step = None
            return False
        self.inspection_step += 1
        runner.revision += 1
        if runner.run_state == "stepping":
            runner.remaining_steps -= 1
            if runner.remaining_steps <= 0:
                runner.run_state = "paused"
        if self.inspection_step == last:
            runner.run_state = "paused"
        runner._publish()
        return True
