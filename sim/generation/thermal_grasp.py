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
from sim.obs import get_policy_observation
from sim.recording import write_episode
from sim.sampler import sample_initial_link_temperatures
from sim.scene import build_scene

POLICY_ID = "nearest_neighbor_temperature"
PROBE_ORDERS = (
    ("lower", "middle", "upper", "handle"),
    ("middle", "upper", "lower", "handle"),
    ("middle", "lower", "upper", "handle"),
    ("upper", "middle", "lower", "handle"),
    ("handle", "lower", "middle", "upper"),
    ("handle", "middle", "upper", "lower"),
    ("handle", "middle", "lower", "upper"),
    ("handle", "upper", "middle", "lower"),
)


def measured_temperature_decision(
    observation: dict,
    *,
    max_temperature_c: float,
) -> tuple[bool, float | None, np.ndarray, np.ndarray]:
    """Judge thermal safety from policy-visible fingertip measurements only.

    Returns ``(safe, decision_temperature, fingertip_temperatures, contact_mask)``.
    The hottest currently contacting finger is used conservatively.  Missing
    contact or non-finite measurements can never produce a safe decision.
    """
    temperatures = np.asarray(
        observation["fingertip_temp"], dtype=np.float64
    ).reshape(2)
    contact_mask = np.asarray(observation["contact_mask"], dtype=bool).reshape(2)
    if not np.any(contact_mask) or not np.all(np.isfinite(temperatures[contact_mask])):
        return False, None, temperatures, contact_mask
    decision_temperature = float(np.max(temperatures[contact_mask]))
    return (
        decision_temperature <= float(max_temperature_c),
        decision_temperature,
        temperatures,
        contact_mask,
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
    orientation: str = ""


class ThermalSequenceController:
    """Probe until a safe region is found, then grasp and lift there."""

    def __init__(
        self,
        bundle,
        spec: EpisodeSpec,
        steps: Sequence[ThermalStep],
        *,
        max_grasp_temperature_c: float,
        execution: dict,
    ):
        if not steps:
            raise ValueError("thermal sequence needs at least one step")
        self.bundle = bundle
        self.spec = spec
        self.steps = tuple(steps)
        self.max_grasp_temperature_c = float(max_grasp_temperature_c)
        self.execution = dict(execution)
        self.last_command = None
        self.phase = "RESET"
        self.current_step = 0
        self.current_controller = None
        self.safe_region: str | None = None
        self.decision_temperature_c: float | None = None
        self.decision_fingertip_temperatures_c: tuple[float, float] | None = None
        self.decision_contact_mask: tuple[bool, bool] | None = None
        self.execution_sequence: list[dict] = []

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

    def _run_controller(
        self,
        controller,
        global_sim_step: int,
        on_sample: Callable[[int], None] | None,
        on_control: Callable[[int], None] | None,
        *,
        start_at_contact: bool = False,
    ) -> ProbeOutcome:
        def sample_callback(local_step: int) -> None:
            if global_sim_step > 0 and local_step == 0:
                return
            if on_sample is not None:
                on_sample(global_sim_step + local_step)

        def control_callback(local_step: int) -> None:
            self.last_command = controller.last_command
            self.phase = controller.phase
            if on_control is not None:
                on_control(global_sim_step + local_step)

        return controller.run(
            on_sample=sample_callback,
            on_control=control_callback,
            start_at_contact=start_at_contact,
        )

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
            if step.kind != "probe":
                raise ValueError("thermal sequences contain probe steps only")
            self.current_step = sequence_index
            child_spec = self._child_spec(step)
            region_name = self.spec.region_order[int(step.region)]
            wall_regions = {"lower", "middle", "upper"}
            previous_name = (
                None
                if sequence_index == 0
                else self.spec.region_order[int(self.steps[sequence_index - 1].region)]
            )
            if previous_name in wall_regions and region_name in wall_regions:
                if {previous_name, region_name} == {"middle", "upper"}:
                    upper_steps = [
                        self.steps[index]
                        for index in (sequence_index - 1, sequence_index)
                        if self.spec.region_order[int(self.steps[index].region)] == "upper"
                    ]
                    approach_mode = (
                        "rim"
                        if any(step.orientation == "vertical" for step in upper_steps)
                        else "direct"
                    )
                else:
                    approach_mode = "direct"
            else:
                approach_mode = "global"
            decision: dict[str, object] = {
                "temperature": None,
                "fingertip_temperatures": None,
                "contact_mask": None,
                "safe": False,
            }

            def hold_decision() -> bool:
                # The expert must make the same decision that an online policy can
                # make: use only measured fingertip temperatures and contact state.
                # ``privileged/link_temp`` remains a label/audit signal and must not
                # control the expert trajectory.
                observation = get_policy_observation(self.bundle)
                safe, temperature, temperatures, contact_mask = (
                    measured_temperature_decision(
                        observation,
                        max_temperature_c=self.max_grasp_temperature_c,
                    )
                )
                decision["fingertip_temperatures"] = temperatures.copy()
                decision["contact_mask"] = contact_mask.copy()
                decision["temperature"] = temperature
                decision["safe"] = safe
                return safe

            controller = ProbeController(
                self.bundle,
                child_spec,
                retract_to_precontact=False,
                hold_decision=hold_decision,
                approach_mode=approach_mode,
            )
            self.current_controller = controller
            outcome = self._run_controller(
                controller, global_sim_step, on_sample, on_control
            )
            if outcome.contact_start_s is not None:
                offset_s = global_sim_step * sim_dt
                start = offset_s + outcome.contact_start_s
                end = offset_s + float(outcome.contact_end_s or outcome.contact_start_s)
                first_contact = start if first_contact is None else first_contact
                last_contact = end

            self.execution_sequence.append(
                {
                    "kind": "probe",
                    "region": region_name,
                    "temperature_c": decision["temperature"],
                    "fingertip_temperatures_c": (
                        None
                        if decision["fingertip_temperatures"] is None
                        else np.asarray(
                            decision["fingertip_temperatures"], dtype=np.float64
                        ).tolist()
                    ),
                    "contact_mask": (
                        None
                        if decision["contact_mask"] is None
                        else np.asarray(decision["contact_mask"], dtype=bool).tolist()
                    ),
                    "temperature_source": "tactile/fingertip_temp",
                    "safe": bool(decision["safe"]),
                    "approach_mode": approach_mode,
                }
            )
            global_sim_step += int(controller.sim_step)
            total_policy_steps += int(outcome.policy_steps)
            peak_force = max(peak_force, float(outcome.peak_normal_force_n))
            if outcome.termination_reason != "contact_stable":
                failure = f"probe_{region_name}:{outcome.termination_reason}"
            if failure:
                break
            if decision["safe"]:
                self.safe_region = region_name
                self.decision_temperature_c = float(decision["temperature"])
                self.decision_fingertip_temperatures_c = tuple(
                    float(value)
                    for value in np.asarray(
                        decision["fingertip_temperatures"], dtype=np.float64
                    )
                )
                self.decision_contact_mask = tuple(
                    bool(value)
                    for value in np.asarray(decision["contact_mask"], dtype=bool)
                )
                grasp_policy = GraspPolicyConfig(
                    policy_id=POLICY_ID,
                    instruction=self.spec.instruction,
                    grasp_region=region_name,
                    lift_height_m=float(self.execution["lift_height_m"]),
                    lift_success_height_m=float(
                        self.bundle.cfg["success"]["min_lift_height_m"]
                    ),
                    grasp_timeout_s=float(self.execution["grasp_timeout_s"]),
                    lift_hold_s=float(self.execution["contact_hold_s"]),
                    touch_hold_s=0.0,
                    grasp_hold_s=0.0,
                )
                grasp_controller = GraspLiftController(
                    self.bundle, child_spec, grasp_policy
                )
                self.current_controller = grasp_controller
                grasp_outcome = self._run_controller(
                    grasp_controller,
                    global_sim_step,
                    on_sample,
                    on_control,
                    start_at_contact=True,
                )
                if grasp_outcome.contact_start_s is not None:
                    offset_s = global_sim_step * sim_dt
                    start = offset_s + grasp_outcome.contact_start_s
                    end = offset_s + float(
                        grasp_outcome.contact_end_s or grasp_outcome.contact_start_s
                    )
                    first_contact = start if first_contact is None else first_contact
                    last_contact = end
                self.execution_sequence.append(
                    {"kind": "grasp", "region": region_name}
                )
                global_sim_step += int(grasp_controller.sim_step)
                total_policy_steps += int(grasp_outcome.policy_steps)
                peak_force = max(
                    peak_force, float(grasp_outcome.peak_normal_force_n)
                )
                if grasp_outcome.termination_reason != "grasp_lift_complete":
                    failure = f"grasp_{region_name}:{grasp_outcome.termination_reason}"
                break
            if sequence_index == len(self.steps) - 1:
                failure = "unsafe_regions_exhausted"
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


def _sample_probe(
    rng: np.random.Generator,
    task: dict,
    region_name: str,
    dataset: dict,
):
    cfg = dict(task["regions"][region_name])
    orientation = ""
    orientation_cfg = dataset.get("upper_probe_orientations")
    if region_name == "upper" and orientation_cfg:
        names = list(orientation_cfg)
        orientation = str(names[int(rng.integers(0, len(names)))])
        cfg.update(dict(orientation_cfg[orientation]))
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
    tool = _rotate_about_axis(
        tool,
        finger,
        float(rng.uniform(-tilt_limit, tilt_limit)),
    )
    tool /= np.linalg.norm(tool)
    finger -= float(finger @ tool) * tool
    finger /= np.linalg.norm(finger)
    return (
        tuple(float(v) for v in xyz),
        tuple(float(v) for v in tool),
        tuple(float(v) for v in finger),
        orientation,
        float(cfg["pre_dist"]),
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


def _probe_order(
    task: dict,
    dataset: dict,
    instruction: str,
    order_index: int,
) -> tuple[tuple[int, str], ...]:
    """Return one order while guaranteeing full coverage in every block of eight."""
    names = [str(v) for v in task["object"]["region_order"]]
    block, slot = divmod(int(order_index), len(PROBE_ORDERS))
    sequence = np.random.SeedSequence(
        [
            int(dataset["seed"]),
            int(zlib.crc32(str(task["task_id"]).encode())),
            int(zlib.crc32(instruction.encode())),
            block,
        ]
    )
    permutation = np.random.default_rng(sequence).permutation(len(PROBE_ORDERS))
    order = PROBE_ORDERS[int(permutation[slot])]
    return tuple((names.index(name), name) for name in order)


def _make_episode(
    task: dict,
    dataset: dict,
    success_slot: int,
    attempt: int,
    global_attempt: int,
    order_index: int,
    *,
    split: str,
    instruction_index: int,
) -> tuple[EpisodeSpec, tuple[ThermalStep, ...], dict]:
    policy_id = POLICY_ID
    instruction = str(task["instructions"][int(instruction_index)])
    seed = int(dataset["seed"])
    seq = np.random.SeedSequence(
        [
            seed,
            int(zlib.crc32(split.encode())),
            int(zlib.crc32(instruction.encode())),
            int(success_slot),
            int(attempt),
        ]
    )
    rng = np.random.default_rng(seq)

    region_order = tuple(str(v) for v in task["object"]["region_order"])
    temps = sample_initial_link_temperatures(task, rng)
    execution = dataset["execution"]
    hold_s = float(execution["contact_hold_s"])
    probe_order = _probe_order(task, dataset, instruction, order_index)
    steps: list[ThermalStep] = []
    for region, name in probe_order:
        pos, tool, finger, orientation, pre_dist = _sample_probe(
            rng, task, name, dataset
        )
        steps.append(
            ThermalStep(
                "probe",
                region,
                pos,
                tool,
                finger,
                pre_dist,
                hold_s,
                orientation=orientation,
            )
        )

    sensor_rc, sensor_noise, sensor_bias, sensor_quant = _sample_sensor(
        rng,
        task["sensor"]["nominal"],
        dataset["sensor_randomization"],
    )
    jitter_xy = float(task["object"]["xy_jitter_m"])
    first = steps[0]
    spec = EpisodeSpec(
        episode_id=(
            f"{task['task_id']}_ins{instruction_index:02d}_{success_slot:06d}"
        ),
        dataset_version=str(dataset["dataset_version"]),
        split=split,
        random_seed=int(seq.generate_state(1)[0]),
        region_order=region_order,
        target_region=-2,
        mixed=False,
        regime="object_config",
        finger_mode="ambient",
        link_temps_c=temps,
        fingertip_temps_c=(
            float(task["sensor"]["ambient_temp_c"]),
            float(task["sensor"]["ambient_temp_c"]),
        ),
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
        "probe_schedule_slot": int(order_index % len(PROBE_ORDERS)),
        "first_region": probe_order[0][1],
        "probe_order": [name for _, name in probe_order],
        "probe_orientation": [step.orientation for step in steps],
        "policy_sequence": [f"probe_{name}" for _, name in probe_order],
        "regime": "object_config",
        "initial_temperature_c": list(temps),
        "contact_hold_s": hold_s,
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
    if count < len(PROBE_ORDERS):
        print(
            f"warning: count={count} < {len(PROBE_ORDERS)}; "
            "this instruction does not cover all probe orders"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_scene(task, seed=0)
    written = skipped = discarded = 0
    started_at = time.time()

    for offset, global_index in enumerate(range(start, start + count), start=1):
        success_slot = global_index
        attempt = 0
        while True:
            if attempt >= max_attempts_per_episode:
                raise SystemExit(
                    f"slot={success_slot} failed {attempt} attempts"
                )
            spec, steps, expert_metadata = _make_episode(
                task,
                dataset,
                success_slot,
                attempt,
                global_index * max_attempts_per_episode + attempt,
                global_index - start,
                split=split,
                instruction_index=instruction_index,
            )
            path = out_dir / f"{spec.episode_id}.hdf5"
            if skip_existing and path.exists():
                skipped += 1
                break

            bundle.reset_episode(spec)
            controller = ThermalSequenceController(
                bundle,
                spec,
                steps,
                max_grasp_temperature_c=float(
                    task["success"]["max_grasp_temperature_c"]
                ),
                execution=dataset["execution"],
            )
            buffers, metadata, static = run_episode(bundle, spec, controller=controller)
            metadata.update(expert_metadata)
            metadata["execution_sequence"] = controller.execution_sequence
            metadata["safe_region"] = controller.safe_region
            metadata["decision_temperature_c"] = controller.decision_temperature_c
            metadata["decision_fingertip_temperatures_c"] = (
                None
                if controller.decision_fingertip_temperatures_c is None
                else list(controller.decision_fingertip_temperatures_c)
            )
            metadata["decision_contact_mask"] = (
                None
                if controller.decision_contact_mask is None
                else list(controller.decision_contact_mask)
            )
            metadata["temperature_decision_source"] = "tactile/fingertip_temp"
            metadata["temperature_decision_rule"] = "max_contacting_finger"
            metadata["temperature_decision_threshold_c"] = float(
                task["success"]["max_grasp_temperature_c"]
            )
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
                f"discard slot={success_slot} attempt={attempt}: "
                f"{metadata['termination_reason']}"
            )
            attempt += 1

    print(f"thermal_grasp finished in {time.time() - started_at:.1f}s")
    return {"written": written, "skipped": skipped, "discarded": discarded}


__all__ = [
    "POLICY_ID",
    "ThermalSequenceController",
    "ThermalStep",
    "generate_thermal_grasp",
    "measured_temperature_decision",
]
