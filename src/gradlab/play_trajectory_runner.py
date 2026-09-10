"""Stored-transition navigation using the Player's recorded transport and frames."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from gradlab.play_session import render_obs_stack
from gradlab.play_trajectory import ImportedTrajectory
from gradlab.play_web import (
    DatasetPlaybackRunner,
    FRAME_GAME,
    FRAME_OBSERVATION,
    PROTOCOL_VERSION,
    _frame_packet,
    history_point_payload,
)


class TrajectoryPlaybackRunner(DatasetPlaybackRunner):
    def __init__(self, archive: Path, args: Any):
        self.recording = ImportedTrajectory(Path(archive))
        self.metadata = self.recording.metadata
        self.checkpoint_path = self.recording.bundle.checkpoint_path
        self._init_protocol(thread_name="gradlab-trajectory-playback")
        self.args = args
        self.rows = range(self.metadata["transition_count"] + 1)
        self.transition_index = 0
        self.first_step = self.metadata["first_step"]
        self.sequence = 0
        self.run_state = "paused"
        self.target_fps = max(0.0, float(getattr(args, "fps", 30)))
        self.remaining_steps = 0
        self.continue_target = None
        self.continue_count = 0
        self._status_message = "Imported recording ready"
        self._transition = None
        first = self.recorded_transition(self.first_step)
        self.current_frame = first["before_image"]
        self.observation_frames = first["observation_frames"]

    def chart_history(self, episode_id, first=None, last=None):
        from gradlab.play_chart_history import chart_history

        with self._snapshot_lock:
            return chart_history(self, episode_id, first, last)

    def stop(self) -> None:
        super().stop()
        self.recording.close()

    def recorded_transition(self, step: int) -> dict[str, Any]:
        return self.recording.transition(step)

    def _snapshot_payload(self) -> dict[str, Any]:
        payload = deepcopy(self.metadata["initial_snapshot"])
        current = self._transition
        payload.update(
            protocol=PROTOCOL_VERSION,
            mode="trajectory",
            revision=self.revision,
            sequence=self.sequence,
            run_state=self.run_state,
            driver="recorded",
            interactive=False,
            status_message=self._status_message,
            transition=current,
            episode_rewards=getattr(self, "_episode_rewards", None) if current else None,
            history_point=dict(self.history[-1]) if self.history else None,
            trajectory={
                "imported": True,
                "scientific_evidence": False,
                "complete": self.metadata["complete"],
                "episode_id": self.metadata["episode_id"],
                "first_step": self.first_step,
                "last_step": self.first_step + self.metadata["transition_count"] - 1,
                "current_step": self.first_step + self.transition_index - 1,
                "classification": self.metadata["classification"],
            },
        )
        payload.pop("publication_capture", None)
        session = payload["session"]
        if current:
            session.update(deepcopy(current.get("recorded_session", {})))
            payload["policy"]["action_selection"].update(
                requested_mode=session["sampling_mode"],
                effective_mode=(current.get("decision") or {}).get("action_selection_mode")
                or session["sampling_mode"],
            )
        session.update(
            step=self.first_step + self.transition_index - 1,
            episode=self.metadata["episode"],
            target_fps=self.target_fps,
            total_reward=current["reward"]["return"] if current else session.get("total_reward", 0),
            awaiting_next_episode=self.transition_index >= len(self.rows) - 1,
            can_start_next_episode=False,
            history_size=len(self.history),
        )
        for key, capability_key in (("attribution", "supported_modes"), ("cnn", "layers")):
            capability = (payload.get("policy") or {}).get(key) or {}
            session[key] = {
                "status": "not-recorded" if capability.get(capability_key) else "unsupported",
                "mode": "none",
                "enabled": False,
                "generation": 0,
                "unavailable_reason": "not recorded in this archive"
                if capability.get(capability_key)
                else capability.get("unavailable_reason") or "unsupported by the recorded Policy",
            }
        # Keep scientific incomparability separate from diagnostics we did not record.
        session.setdefault(
            "critic_comparison",
            {"available": False, "reasons": [], "discount": self.metadata.get("discount")},
        )
        return payload

    def _publish(self) -> None:
        frames = {
            FRAME_GAME: self.current_frame,
            FRAME_OBSERVATION: render_obs_stack(self.observation_frames, 1)
            if self.observation_frames
            else None,
        }
        self.encoder.submit_batch(self.sequence, frames)
        payload = self._snapshot_payload()
        with self._snapshot_lock:
            self._latest_snapshot = payload
            self._snapshot_updates.append(payload)
            if self.transition_index == 0:
                self._episode_start_snapshot = payload
                self._episode_start_frames = {
                    kind: (
                        self.sequence,
                        _frame_packet(kind, self.sequence, frame, session_epoch=self.encoder.epoch),
                    )
                    for kind, frame in frames.items()
                    if frame is not None
                }

    def _load_step(self, step: int) -> None:
        row = self.recorded_transition(step)
        self.transition_index = step - self.first_step + 1
        self.sequence = row["sequence"]
        self.current_frame = row["after_image"]
        self.observation_frames = row["observation_frames"]
        self._transition = row["presentation"]
        self._episode_rewards = row["presentation"].get("episode_rewards")

    def _apply(self, command) -> None:
        if command.name == "replay":
            first = self.recorded_transition(self.first_step)
            self.transition_index = 0
            self.sequence = 0
            self._transition = None
            self.current_frame = first["before_image"]
            self.observation_frames = first["observation_frames"]
            self.history.clear()
            self.remaining_steps = 0
            self.continue_target = None
            self._set_state("playing")
            self._response(command, ok=True)
            return
        if command.name in {"seek", "step_backward"}:
            try:
                step = (
                    self.first_step + self.transition_index - 2
                    if command.name == "step_backward"
                    else command.payload.get("step")
                )
                if type(step) is not int:
                    raise ValueError("transition number must be an integer")
                self._load_step(step)
                assert self._transition is not None
                self.history.clear()
                for index in range(max(self.first_step, step - 63), step + 1):
                    self.history.append(
                        history_point_payload(self.recorded_transition(index)["presentation"])
                    )
                self.remaining_steps = 0
                self.continue_target = None
                self._set_state("paused", message=f"Recorded transition {step}")
                self._response(command, ok=True)
            except (ValueError, TypeError) as exc:
                self._response(command, ok=False, error=str(exc))
            return
        super()._apply(command)

    def _step_once(self) -> None:
        self._load_step(self.first_step + self.transition_index)
        assert self._transition is not None
        self.history.append(history_point_payload(self._transition))
        self.revision += 1
        self._status_message = None
        if self.transition_index == len(self.rows) - 1:
            self.run_state = "paused"
            self.remaining_steps = 0
            self.continue_target = None
            self._status_message = (
                "Recorded episode complete"
                if self.metadata["complete"]
                else "Recording cutoff reached; episode was unfinished"
            )
        elif self.run_state == "stepping":
            self.remaining_steps -= 1
            if self.remaining_steps <= 0:
                self.run_state = "paused"
        elif self.run_state == "continuing":
            self.continue_count += 1
            events = self._transition.get("events", [])
            target = self.continue_target or "any"
            if (
                bool(events) if target == "any" else target in events
            ) or self.continue_count >= 10_000:
                self.run_state = "paused"
        self._publish()
