"""Thermal-aware scripted probe/grasp demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence
import time
import zlib

import numpy as np

from sim.control.grasp_controller import GraspLiftController, GraspPolicyConfig
from sim.control.probe_controller import ProbeController, ProbeOutcome
from sim.episode import EpisodeSpec
from sim.generation.episode_runner import run_episode
from sim.recording import write_episode
from sim.scene import build_scene

WALL_REGIONS = ("lower", "middle", "upper")
POLICIES = (
    {
        "id": "A_wall_handle_grasp_handle",
        "regime": "A",
        "sequence": ("probe_wall", "probe_handle", "grasp_handle"),
    },
    {
        "id": "A_handle_grasp_handle",
        "regime": "A",
        "sequence": ("probe_handle", "grasp_handle"),
    },
    {
        "id": "B_wall_grasp_wall",
        "regime": "B",
        "sequence": ("probe_wall", "grasp_wall"),
    },
    {
        "id": "B_handle_wall_grasp_wall",
        "regime": "B",
        "sequence": ("probe_handle", "probe_wall", "grasp_wall"),
    },
)


@dataclass(frozen=True)
class ThermalStep:
    kind: str
    region: int
    position: tuple[float, float, float]
    tool_dir: tuple[float, float, float]
    finger_dir: tuple[float, float, float]
    pre_dist: float
    hold_s: float
    grasp_timeout_s: float = 10.0
    lift_height_m: float = 0.12
    lift_success_height_m: float = 0.06
    touch_hold_s: float = 0.0


class ThermalSequenceController:
    """Run probe and grasp steps continuously in one episode."""

    def __init__(self, bundle, spec: EpisodeSpec, steps: Sequence[ThermalStep]):
        if not steps:
            raise ValueError("thermal sequence needs at least one step")
        self.bundle = bundle
        self.spec = spec
        self.steps = tuple(steps)
        self.last_command = None
        self.phase = "RESET"
        self.current_step = 0
        self.current_controller = None

    def _child_spec(self, step: ThermalStep) -> EpisodeSpec:
        hold_s = float(step.touch_hold_s if step.kind == "grasp" else step.hold_s)
        return replace(
            self.spec,
            target_region=int(step.region),
            mixed=False,
            contact_order=(),
            contact_hold_s_by_region=(),
            contact_hold_s=hold_s,
            contact_quality="stable",
            probe_pos=step.position,
            probe_tool_dir=step.tool_dir,
            probe_finger_dir=step.finger_dir,
            probe_pre_dist=float(step.pre_dist),
            probe_positions=(),
            probe_tool_dirs=(),
            probe_finger_dirs=(),
        )

    def _controller_for(self, step: ThermalStep):
        child_spec = self._child_spec(step)
        if step.kind == "probe":
            return ProbeController(self.bundle, child_spec)
        if step.kind == "grasp":
            policy = GraspPolicyConfig(
                policy_id="thermal_grasp",
                instruction=self.spec.instruction,
                grasp_region=self.spec.region_order[int(step.region)],
                lift_height_m=float(step.lift_height_m),
                lift_success_height_m=float(step.lift_success_height_m),
                grasp_timeout_s=float(step.grasp_timeout_s),
                lift_hold_s=float(step.hold_s),
                touch_hold_s=float(step.touch_hold_s),
                grasp_hold_s=float(step.hold_s),
            )
            return GraspLiftController(self.bundle, child_spec, policy)
        raise ValueError(f"unknown thermal step kind: {step.kind}")

    def run(
        self,
        on_sample: Callable[[int], None] | None = None,
        on_control: Callable[[int], None] | None = None,
    ) -> ProbeOutcome:
        global_sim_step = 0
        total_policy_steps = 0
        peak_force = 0.0
        first_contact = None
        last_contact = None
        failure = ""
        sim_dt = float(self.bundle.cfg["controller"]["sim_dt"])

        for sequence_index, step in enumerate(self.steps):
            self.current_step = sequence_index
            controller = self._controller_for(step)
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

            outcome = controller.run(
                on_sample=sample_callback,
                on_control=control_callback,
            )
            if outcome.contact_start_s is not None:
                offset_s = global_sim_step * sim_dt
                start = offset_s + outcome.contact_start_s
                end = offset_s + float(outcome.contact_end_s or outcome.contact_start_s)
                first_contact = start if first_contact is None else first_contact
                last_contact = end

            expected = "contact_stable" if step.kind == "probe" else "grasp_lift_complete"
            if outcome.termination_reason != expected:
                failure = f"{step.kind}_{sequence_index}:{outcome.termination_reason}"
            global_sim_step += int(controller.sim_step)
            total_policy_steps += int(outcome.policy_steps)
            peak_force = max(peak_force, float(outcome.peak_normal_force_n))
            if failure:
                break

        self.phase = "DONE" if not failure else "FAILED"
        return ProbeOutcome(
            termination_reason=(
                "thermal_sequence_complete"
                if not failure
                else f"thermal_sequence_partial:{failure}"
            ),
            contact_detected=first_contact is not None,
            contact_start_s=first_contact,
            contact_end_s=last_contact,
            policy_steps=total_policy_steps,
            peak_normal_force_n=peak_force,
        )


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * float(axis @ vector) * (1.0 - np.cos(angle))
    )


def _sample_probe(rng: np.random.Generator, task: dict, region_name: str):
    cfg = task["regions"][region_name]
    xyz = np.asarray(cfg["pos"], dtype=np.float64).copy()
    xy_jitter = float(cfg.get("xy_jitter_m", 0.0))
    xyz[:2] += rng.uniform(-xy_jitter, xy_jitter, size=2)
    z_jitter = float(cfg.get("z_jitter_m", 0.0))
    xyz[2] += float(rng.uniform(-z_jitter, z_jitter))

    azimuth = 0.0
    if region_name != "handle":
        limit = float(cfg.get("azimuth_jitter_rad", 0.0))
        azimuth = float(rng.uniform(-limit, limit))
    c, s = np.cos(azimuth), np.sin(azimuth)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tool = rotation @ np.asarray(cfg["tool_dir"], dtype=np.float64)
    finger = rotation @ np.asarray(cfg["finger_dir"], dtype=np.float64)
    tilt_limit = float(cfg.get("tilt_jitter_rad", 0.0))
    tool = _rotate_about_axis(tool, finger, float(rng.uniform(-tilt_limit, tilt_limit)))
    tool /= np.linalg.norm(tool)
    finger -= float(finger @ tool) * tool
    finger /= np.linalg.norm(finger)
    return tuple(float(v) for v in xyz), tuple(float(v) for v in tool), tuple(float(v) for v in finger)


def _sample_grasp(rng: np.random.Generator, grasp_cfg: dict):
    xyz = np.asarray(
        [grasp_cfg["x"], grasp_cfg["y"], rng.uniform(*grasp_cfg["z_range"])],
        dtype=np.float64,
    )
    jitter = float(grasp_cfg["xy_jitter_m"])
    xyz[:2] += rng.uniform(-jitter, jitter, size=2)
    return (
        tuple(float(v) for v in xyz),
        tuple(float(v) for v in grasp_cfg["tool_dir"]),
        tuple(float(v) for v in grasp_cfg["finger_dir"]),
        float(grasp_cfg["pre_dist"]),
    )


def _sample_sensor(
    rng: np.random.Generator,
    nominal: dict,
    randomization: dict,
) -> tuple[float, float, float, float]:
    return (
        nominal["rc_s"] * float(rng.uniform(*randomization["rc_multiplier_range"])),
        nominal["noise_c"] * float(rng.uniform(*randomization["noise_multiplier_range"])),
        float(rng.uniform(*randomization["bias_absolute_range"])),
        nominal["quantization_c"]
        * float(rng.uniform(*randomization["quantization_multiplier_range"])),
    )


def _make_episode(
    task: dict,
    dataset: dict,
    policy: dict,
    success_slot: int,
    attempt: int,
    global_attempt: int,
    *,
    split: str,
    instruction_index: int,
) -> tuple[EpisodeSpec, tuple[ThermalStep, ...], dict]:
    policy_id = str(policy["id"])
    sequence = tuple(str(v) for v in policy["sequence"])
    instruction = str(task["instructions"][int(instruction_index)])
    seed = int(dataset["seed"])
    seq = np.random.SeedSequence(
        [
            seed,
            int(zlib.crc32(split.encode())),
            int(zlib.crc32(instruction.encode())),
            int(zlib.crc32(policy_id.encode())),
            int(success_slot),
            int(attempt),
        ]
    )
    rng = np.random.default_rng(seq)

    regime_name = str(policy["regime"])
    regime = dataset["regimes"][regime_name]
    region_order = tuple(str(v) for v in task["object"]["region_order"])
    temps = tuple(float(rng.uniform(*regime[name])) for name in region_order)
    wall_name = str(WALL_REGIONS[int(rng.integers(0, len(WALL_REGIONS)))])
    wall_region = region_order.index(wall_name)
    handle_region = region_order.index("handle")
    wall_probe = _sample_probe(rng, task, wall_name)
    handle_probe = _sample_probe(rng, task, "handle")
    wall_grasp = _sample_grasp(rng, dataset["grasp_targets"]["wall"])
    handle_grasp = _sample_grasp(rng, dataset["grasp_targets"]["handle"])

    execution = dataset["execution"]
    hold_s = float(execution["contact_hold_s"])
    steps: list[ThermalStep] = []
    for action in sequence:
        if action == "probe_wall":
            pos, tool, finger = wall_probe
            steps.append(
                ThermalStep("probe", wall_region, pos, tool, finger, task["regions"][wall_name]["pre_dist"], hold_s)
            )
        elif action == "probe_handle":
            pos, tool, finger = handle_probe
            steps.append(
                ThermalStep("probe", handle_region, pos, tool, finger, task["regions"]["handle"]["pre_dist"], hold_s)
            )
        elif action in {"grasp_wall", "grasp_handle"}:
            pos, tool, finger, pre_dist = wall_grasp if action == "grasp_wall" else handle_grasp
            region = wall_region if action == "grasp_wall" else handle_region
            steps.append(
                ThermalStep(
                    "grasp",
                    region,
                    pos,
                    tool,
                    finger,
                    pre_dist,
                    hold_s,
                    grasp_timeout_s=float(execution["grasp_timeout_s"]),
                    lift_height_m=float(execution["lift_height_m"]),
                    lift_success_height_m=float(task["success"]["min_lift_height_m"]),
                    touch_hold_s=hold_s,
                )
            )
        else:
            raise ValueError(f"unknown policy action {action!r}")

    merged: list[ThermalStep] = []
    for step in steps:
        if (
            step.kind == "grasp"
            and merged
            and merged[-1].kind == "probe"
            and merged[-1].region == step.region
        ):
            merged.pop()
        merged.append(step)
    steps = merged

    sensor_rc, sensor_noise, sensor_bias, sensor_quant = _sample_sensor(
        rng,
        task["sensor"]["nominal"],
        dataset["sensor_randomization"],
    )
    jitter_xy = float(task["object"]["xy_jitter_m"])
    first = steps[0]
    spec = EpisodeSpec(
        episode_id=f"{dataset['dataset_version']}_{split}_ins{instruction_index:02d}_{policy_id}_{success_slot:06d}",
        dataset_version=str(dataset["dataset_version"]),
        split=split,
        random_seed=int(seq.generate_state(1)[0]),
        region_order=region_order,
        target_region=-2,
        mixed=False,
        regime=regime_name,
        finger_mode="ambient",
        link_temps_c=temps,
        fingertip_temps_c=(25.0, 25.0),
        contact_quality="stable",
        contact_hold_s=hold_s,
        sensor_rc_s=sensor_rc,
        sensor_noise_c=sensor_noise,
        sensor_bias_c=sensor_bias,
        sensor_quantization_c=sensor_quant,
        probe_pos=first.position,
        probe_tool_dir=first.tool_dir,
        probe_finger_dir=first.finger_dir,
        probe_pre_dist=first.pre_dist,
        cup_xy_jitter_m=(
            float(rng.uniform(-jitter_xy, jitter_xy)),
            float(rng.uniform(-jitter_xy, jitter_xy)),
        ),
        cup_yaw_jitter_rad=float(
            rng.uniform(-float(task["object"]["yaw_jitter_rad"]), float(task["object"]["yaw_jitter_rad"]))
        ),
        material_parameters=tuple(
            sorted(
                (str(k), float(v))
                for k, v in task["object"]["material_properties"][task["object"]["material"]].items()
            )
        ),
        instruction=instruction,
    )
    metadata = {
        "global_index": int(global_attempt),
        "success_slot": int(success_slot),
        "attempt": int(attempt),
        "policy_id": policy_id,
        "instruction_index": int(instruction_index),
        "policy_sequence": list(sequence),
        "execution_sequence": [
            {
                "kind": step.kind,
                "region": region_order[step.region],
                "position": list(step.position),
                "hold_s": float(step.touch_hold_s if step.kind == "grasp" else step.hold_s),
            }
            for step in steps
        ],
        "regime": regime_name,
        "wall_region": wall_name,
        "contact_hold_s": hold_s,
        "grasp_points": [
            {"kind": step.kind, "region": region_order[step.region], "position": list(step.position)}
            for step in steps
            if step.kind == "grasp"
        ],
    }
    return spec, tuple(steps), metadata


def generate_thermal_grasp(
    *,
    task: dict,
    dataset: dict,
    split: str,
    start: int,
    count: int,
    instruction_index: int,
    out_dir: Path,
    seed: int | None,
    skip_existing: bool,
    max_attempts_per_episode: int,
) -> dict[str, int]:
    if seed is not None:
        dataset = {**dataset, "seed": int(seed)}
    if max_attempts_per_episode < 1:
        raise ValueError("max_attempts_per_episode must be >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_scene(task, seed=0)
    written = skipped = discarded = 0
    started_at = time.time()

    for offset, global_index in enumerate(range(start, start + count), start=1):
        policy = POLICIES[global_index % len(POLICIES)]
        success_slot = global_index // len(POLICIES)
        attempt = 0
        while True:
            if attempt >= max_attempts_per_episode:
                raise SystemExit(
                    f"{policy['id']} slot={success_slot} failed {attempt} attempts"
                )
            spec, steps, expert_metadata = _make_episode(
                task,
                dataset,
                policy,
                success_slot,
                attempt,
                global_index * max_attempts_per_episode + attempt,
                split=split,
                instruction_index=instruction_index,
            )
            path = out_dir / f"{spec.episode_id}.hdf5"
            if skip_existing and path.exists():
                skipped += 1
                break

            bundle.reset_episode(spec)
            controller = ThermalSequenceController(bundle, spec, steps)
            buffers, metadata, static = run_episode(bundle, spec, controller=controller)
            metadata.update(expert_metadata)
            metadata["success"] = bool(metadata["task_success"]["success"])
            metadata["thermal_exploration"] = True
            if metadata["success"]:
                write_episode(path, buffers, metadata, static)
                written += 1
                print(
                    f"[{offset}/{count}] {spec.episode_id} "
                    f"attempt={attempt} {metadata['control_frames']} control frames"
                )
                break
            discarded += 1
            print(
                f"discard {policy['id']} slot={success_slot} attempt={attempt}: "
                f"{metadata['termination_reason']}"
            )
            attempt += 1

    print(f"thermal_grasp finished in {time.time() - started_at:.1f}s")
    return {"written": written, "skipped": skipped, "discarded": discarded}


__all__ = [
    "POLICIES",
    "ThermalSequenceController",
    "ThermalStep",
    "generate_thermal_grasp",
]
