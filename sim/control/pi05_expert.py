"""Composable scripted experts for the first Pi0.5 dataset."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from sim.control.probe_controller import ProbeController, ProbeOutcome


@dataclass(frozen=True)
class ExpertStep:
    kind: str  # "probe" or "grasp"
    region: int
    position: tuple[float, float, float]
    tool_dir: tuple[float, float, float]
    finger_dir: tuple[float, float, float]
    pre_dist: float
    force_n: float
    hold_s: float
    force_band_n: float = 1.0
    force_width_step_m: float = 0.00002
    force_fast_width_step_m: float = 0.0002
    force_fast_threshold_n: float = 2.0
    grasp_timeout_s: float = 10.0
    lift_height_m: float = 0.12
    lift_success_height_m: float = 0.06
    lift_timeout_s: float = 8.0
    touch_force_n: float | None = None
    touch_hold_s: float = 0.0


class ForceGraspLiftController(ProbeController):
    """Grasp, regulate bilateral force, lift, and hold it."""

    def __init__(
        self,
        bundle,
        spec,
        *,
        grasp_force_n: float,
        force_band_n: float,
        force_width_step_m: float,
        force_fast_width_step_m: float,
        force_fast_threshold_n: float,
        grasp_hold_s: float,
        lift_height_m: float,
        lift_success_height_m: float,
        grasp_timeout_s: float,
        lift_timeout_s: float,
    ):
        super().__init__(bundle, spec)
        self.grasp_force_n = float(grasp_force_n)
        self.force_band_n = float(force_band_n)
        self.force_width_step_m = float(force_width_step_m)
        self.force_fast_width_step_m = float(force_fast_width_step_m)
        self.force_fast_threshold_n = float(force_fast_threshold_n)
        self.grasp_hold_steps = max(1, int(round(grasp_hold_s / self.dt)))
        self.lift_success_height_m = float(lift_success_height_m)
        self.grasp_timeout_s = float(grasp_timeout_s)
        self.lift_timeout_s = float(lift_timeout_s)
        self.initial_cup_z = float(bundle.cup_pose()[0][2])
        self.lift_target = np.asarray(self.pose.pos, dtype=np.float64).copy()
        self.lift_target[2] += float(lift_height_m)
        self.force_hold_count = 0
        self.lift_hold_count = 0

    @property
    def termination_reason(self) -> str:
        if self.phase == "DONE":
            return "grasp_lift_complete"
        return self.failure_reason or "running"

    def _command(self) -> None:
        if self.phase in ("HOLD", "GRASP"):
            self.controller.set_pose_target(self.pose.pos, self.pose.quat)
        elif self.phase in ("LIFT", "HOLD_LIFT"):
            self.controller.set_pose_target(self.lift_target, self.pose.quat)
        else:
            super()._command()
            return
        self.last_command = self.controller.apply()

    def _regulate_force(
        self,
        normal: np.ndarray,
        touching: np.ndarray,
        *,
        target_force_n: float,
    ) -> bool:
        bilateral = bool(np.all(touching))
        control_force = float(np.min(normal)) if bilateral else 0.0
        error = control_force - float(target_force_n)

        # Negative error means "close more"; positive error opens slightly.
        max_step = (
            self.force_fast_width_step_m
            if abs(error) > self.force_fast_threshold_n
            else self.force_width_step_m
        )
        adjustment = np.clip(
            2.0 * float(self.contact_cfg["gain_m_per_n"]) * error,
            -max_step,
            max_step,
        )
        self.controller.set_width_target(self.controller.width_target + adjustment)
        return bilateral and bool(np.all(np.abs(normal - float(target_force_n)) <= self.force_band_n))

    def _update(self, normal: np.ndarray, touching: np.ndarray) -> None:
        self.peak_force = max(self.peak_force, float(np.max(normal, initial=0.0)))

        if self.phase in ("APPROACH", "SEEK_CONTACT"):
            super()._update(normal, touching)
            if self.phase == "HOLD":
                # SEEK_CONTACT ends with a fully-closed width target. Rebase the
                # force controller at the command that actually established
                # contact instead of continuing to drive toward zero.
                self.controller.set_width_target(self.controller.width_cmd)
                self.force_hold_count = 0
            return

        if self.phase == "HOLD":
            self._regulate_force(
                normal,
                touching,
                target_force_n=float(self.spec.target_force_n),
            )
            if self.phase_steps * self.dt >= float(self.spec.contact_hold_s):
                self.force_hold_count = 0
                self._set_phase("GRASP")
            return

        if self.phase == "GRASP":
            stable = self._regulate_force(
                normal,
                touching,
                target_force_n=self.grasp_force_n,
            )
            self.force_hold_count = self.force_hold_count + 1 if stable else 0
            if self.force_hold_count >= self.grasp_hold_steps:
                self.lift_hold_count = 0
                self._set_phase("LIFT")
            elif self.phase_steps * self.dt >= self.grasp_timeout_s:
                self._fail("grasp_force_timeout")
            return

        if self.phase == "LIFT":
            self._regulate_force(
                normal,
                touching,
                target_force_n=self.grasp_force_n,
            )
            if self.controller.at_pose(
                self.lift_target,
                self.pose.quat,
                pos_tol=0.008,
                rot_tol=0.10,
            ):
                self.lift_hold_count = 0
                self._set_phase("HOLD_LIFT")
            elif self.phase_steps * self.dt >= self.lift_timeout_s:
                self._fail("lift_timeout")
            return

        if self.phase == "HOLD_LIFT":
            self._regulate_force(
                normal,
                touching,
                target_force_n=self.grasp_force_n,
            )
            lifted = self.bundle.cup_pose()[0][2] - self.initial_cup_z
            self.lift_hold_count = self.lift_hold_count + 1 if lifted >= self.lift_success_height_m else 0
            if self.lift_hold_count >= self.grasp_hold_steps:
                self.contact_end_s = self.step_index * self.dt
                self._set_phase("DONE")
            elif self.phase_steps * self.dt >= self.lift_timeout_s:
                self._fail("cup_not_lifted")


class Pi05ExpertController:
    """Run a configured probe/grasp sequence without resetting the episode."""

    def __init__(self, bundle, spec, steps: Sequence[ExpertStep]):
        if not steps:
            raise ValueError("Pi05ExpertController needs at least one step")
        self.bundle = bundle
        self.spec = spec
        self.steps = tuple(steps)
        self.last_command: tuple[np.ndarray, float] | None = None
        self.phase = "RESET"
        self.current_step = 0
        self.current_controller: ProbeController | None = None

    def _child_spec(self, step: ExpertStep):
        force_n = (
            float(step.touch_force_n)
            if step.kind == "grasp" and step.touch_force_n is not None
            else float(step.force_n)
        )
        hold_s = (
            float(step.touch_hold_s)
            if step.kind == "grasp"
            else float(step.hold_s)
        )
        return replace(
            self.spec,
            target_region=int(step.region),
            mixed=False,
            contact_order=(),
            contact_hold_s_by_region=(),
            contact_hold_s=hold_s,
            target_force_n=force_n,
            contact_quality="stable",
            probe_pos=tuple(float(v) for v in step.position),
            probe_tool_dir=tuple(float(v) for v in step.tool_dir),
            probe_finger_dir=tuple(float(v) for v in step.finger_dir),
            probe_pre_dist=float(step.pre_dist),
            probe_positions=(),
            probe_tool_dirs=(),
            probe_finger_dirs=(),
        )

    def run(
        self,
        on_sample: Callable[[int], None] | None = None,
        on_control: Callable[[int], None] | None = None,
    ) -> ProbeOutcome:
        global_sim_step = 0
        total_policy_steps = 0
        peak_force = 0.0
        first_contact: float | None = None
        last_contact: float | None = None
        failures: list[str] = []
        sim_dt = float(self.bundle.cfg["controller"]["sim_dt"])

        for sequence_index, step in enumerate(self.steps):
            self.current_step = sequence_index
            child_spec = self._child_spec(step)
            if step.kind == "probe":
                controller = ProbeController(self.bundle, child_spec)
            elif step.kind == "grasp":
                controller = ForceGraspLiftController(
                    self.bundle,
                    child_spec,
                    grasp_timeout_s=float(step.grasp_timeout_s),
                    grasp_force_n=float(step.force_n),
                    force_band_n=float(step.force_band_n),
                    force_width_step_m=float(step.force_width_step_m),
                    force_fast_width_step_m=float(step.force_fast_width_step_m),
                    force_fast_threshold_n=float(step.force_fast_threshold_n),
                    grasp_hold_s=float(step.hold_s),
                    lift_height_m=float(step.lift_height_m),
                    lift_success_height_m=float(step.lift_success_height_m),
                    lift_timeout_s=float(step.lift_timeout_s),
                )
            else:
                raise ValueError(f"unknown expert step kind: {step.kind}")
            self.current_controller = controller

            def sample_callback(local_step: int) -> None:
                if sequence_index > 0 and local_step == 0:
                    return
                if on_sample is not None:
                    on_sample(global_sim_step + local_step)

            def control_callback(local_step: int) -> None:
                self.last_command = controller.last_command
                self.phase = controller.phase
                if on_control is not None:
                    on_control(global_sim_step + local_step)

            outcome = controller.run(on_sample=sample_callback, on_control=control_callback)
            if outcome.contact_start_s is not None:
                offset = global_sim_step * sim_dt
                start = offset + outcome.contact_start_s
                end = offset + float(outcome.contact_end_s or outcome.contact_start_s)
                first_contact = start if first_contact is None else first_contact
                last_contact = end

            expected = "contact_stable" if step.kind == "probe" else "grasp_lift_complete"
            if outcome.termination_reason != expected:
                failures.append(f"{step.kind}_{sequence_index}:{outcome.termination_reason}")
            global_sim_step += int(controller.sim_step)
            total_policy_steps += int(outcome.policy_steps)
            peak_force = max(peak_force, float(outcome.peak_normal_force_n))
            if failures:
                break

        self.phase = "DONE" if not failures else "FAILED"
        return ProbeOutcome(
            termination_reason="pi05_expert_complete" if not failures else "pi05_expert_partial:" + ",".join(failures),
            contact_detected=first_contact is not None,
            contact_start_s=first_contact,
            contact_end_s=last_contact,
            policy_steps=total_policy_steps,
            peak_normal_force_n=peak_force,
        )


__all__ = ["ExpertStep", "ForceGraspLiftController", "Pi05ExpertController"]
