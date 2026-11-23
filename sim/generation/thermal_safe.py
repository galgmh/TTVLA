"""Thermal-safe scripted expert pipeline used by the unified generator."""

from __future__ import annotations

from pathlib import Path
import time
import zlib

import numpy as np

from sim.control.pi05_expert import ExpertStep, Pi05ExpertController
from sim.control.cup_expert_defaults import (
    GRASP_TARGETS,
    THERMAL_SAFE_EXECUTION,
    THERMAL_SAFE_POLICIES,
)
from sim.episode import EpisodeSpec
from sim.generation.episode_runner import run_episode
from sim.recording import write_episode
from sim.scene import build_scene

WALL_REGIONS = ("lower", "middle", "upper")


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
    xyz[2] += float(
        rng.uniform(
            -float(cfg.get("z_jitter_m", 0.0)),
            float(cfg.get("z_jitter_m", 0.0)),
        )
    )

    azimuth = 0.0
    if region_name != "handle":
        limit = float(cfg.get("azimuth_jitter_rad", 0.0))
        azimuth = float(rng.uniform(-limit, limit))
    c, s = np.cos(azimuth), np.sin(azimuth)
    rz = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tool = rz @ np.asarray(cfg["tool_dir"], dtype=np.float64)
    finger = rz @ np.asarray(cfg["finger_dir"], dtype=np.float64)

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
    rng: np.random.Generator, nominal: dict, randomization: dict
) -> tuple[float, float, float, float]:
    rc = float(nominal["rc_s"]) * float(
        rng.uniform(*randomization["rc_multiplier_range"])
    )
    noise = float(nominal["noise_c"]) * float(
        rng.uniform(*randomization["noise_multiplier_range"])
    )
    quantization = float(nominal["quantization_c"]) * float(
        rng.uniform(*randomization["quantization_multiplier_range"])
    )
    bias = float(rng.uniform(*randomization["bias_absolute_range"]))
    return rc, noise, bias, quantization


def _make_episode(
    task: dict,
    dataset: dict,
    policy: dict,
    success_slot: int,
    attempt: int,
    global_attempt: int,
    split: str | None = None,
    instruction_index: int = 0,
) -> tuple[EpisodeSpec, tuple[ExpertStep, ...], dict]:
    policy_id = str(policy["id"])
    sequence = tuple(str(v) for v in policy["sequence"])
    regime_name = str(policy["regime"])
    split_name = str(split or dataset.get("split", "train"))
    instruction = str(
        task.get("instruction")
        or task["instructions"][int(instruction_index)]
    )
    seed = int(dataset["seed"])
    seq = np.random.SeedSequence([
        seed,
        int(zlib.crc32(split_name.encode())),
        int(zlib.crc32(instruction.encode("utf-8"))),
        int(zlib.crc32(policy_id.encode())),
        int(success_slot),
        int(attempt),
    ])
    rng = np.random.default_rng(seq)
    random_seed = int(seq.generate_state(1)[0])

    regime = dataset["regimes"][regime_name]
    region_order = tuple(str(v) for v in task["object"]["region_order"])
    temps = tuple(float(rng.uniform(*regime[name])) for name in region_order)

    wall_name = str(WALL_REGIONS[int(rng.integers(0, len(WALL_REGIONS)))])
    wall_region = region_order.index(wall_name)
    handle_region = region_order.index("handle")
    wall_probe = _sample_probe(rng, task, wall_name)
    handle_probe = _sample_probe(rng, task, "handle")
    wall_grasp = _sample_grasp(rng, GRASP_TARGETS["wall"])
    handle_grasp = _sample_grasp(rng, GRASP_TARGETS["handle"])

    execution = THERMAL_SAFE_EXECUTION
    probe_force = float(execution["probe_force_n"])
    grasp_force = float(execution["grasp_force_n"])
    hold_s = float(execution["contact_hold_s"])
    grasp_options = {
        "force_band_n": float(execution["force_band_n"]),
        "force_width_step_m": float(execution["force_width_step_m"]),
        "force_fast_width_step_m": float(execution["force_fast_width_step_m"]),
        "force_fast_threshold_n": float(execution["force_fast_threshold_n"]),
        "grasp_timeout_s": float(execution["grasp_timeout_s"]),
        "lift_height_m": float(execution["lift_height_m"]),
        "lift_success_height_m": float(task["success"]["min_lift_height_m"]),
        "lift_timeout_s": float(execution["lift_timeout_s"]),
        "touch_force_n": probe_force,
        "touch_hold_s": hold_s,
    }
    steps: list[ExpertStep] = []
    for action in sequence:
        if action == "probe_wall":
            pos, tool, finger = wall_probe
            steps.append(ExpertStep(
                "probe", wall_region, pos, tool, finger,
                float(task["regions"][wall_name]["pre_dist"]), probe_force, hold_s,
            ))
        elif action == "probe_handle":
            pos, tool, finger = handle_probe
            steps.append(ExpertStep(
                "probe", handle_region, pos, tool, finger,
                float(task["regions"]["handle"]["pre_dist"]), probe_force, hold_s,
            ))
        elif action == "grasp_wall":
            pos, tool, finger, pre_dist = wall_grasp
            steps.append(ExpertStep(
                "grasp", wall_region, pos, tool, finger,
                pre_dist, grasp_force, hold_s, **grasp_options,
            ))
        elif action == "grasp_handle":
            pos, tool, finger, pre_dist = handle_grasp
            steps.append(ExpertStep(
                "grasp", handle_region, pos, tool, finger,
                pre_dist, grasp_force, hold_s, **grasp_options,
            ))
        else:
            raise ValueError(f"unknown policy action {action!r}")

    # The final touch and grasp use the same point. Keep TCP there, hold the
    # probe force, then increase force to the grasp target without retracting.
    merged_steps: list[ExpertStep] = []
    for step in steps:
        if (
            step.kind == "grasp"
            and merged_steps
            and merged_steps[-1].kind == "probe"
            and merged_steps[-1].region == step.region
        ):
            merged_steps.pop()
        merged_steps.append(step)
    steps = merged_steps

    sensor_rc, sensor_noise, sensor_bias, sensor_quant = _sample_sensor(
        rng, task["sensor"]["nominal"], dataset["sensor_randomization"]
    )
    object_cfg = task["object"]
    jitter_xy = float(object_cfg["xy_jitter_m"])
    cup_xy = (
        float(rng.uniform(-jitter_xy, jitter_xy)),
        float(rng.uniform(-jitter_xy, jitter_xy)),
    )
    cup_yaw = float(rng.uniform(-float(object_cfg["yaw_jitter_rad"]), float(object_cfg["yaw_jitter_rad"])))
    first = steps[0]
    spec = EpisodeSpec(
        episode_id=(
            f"{dataset['dataset_version']}_{split_name}_ins{instruction_index:02d}_"
            f"{policy_id}_{success_slot:06d}"
        ),
        dataset_version=str(dataset["dataset_version"]),
        split=split_name,
        random_seed=random_seed,
        region_order=region_order,
        target_region=-2,
        mixed=False,
        regime=regime_name,
        finger_mode="ambient",
        link_temps_c=temps,
        fingertip_temps_c=(25.0, 25.0),
        force_level=1.0,
        target_force_n=probe_force,
        contact_quality="stable",
        contact_hold_s=hold_s,
        fail_retreat_m=0.0,
        sensor_rc_s=sensor_rc,
        sensor_noise_c=sensor_noise,
        sensor_bias_c=sensor_bias,
        sensor_quantization_c=sensor_quant,
        probe_pos=first.position,
        probe_tool_dir=first.tool_dir,
        probe_finger_dir=first.finger_dir,
        probe_pre_dist=first.pre_dist,
        cup_xy_jitter_m=cup_xy,
        cup_yaw_jitter_rad=cup_yaw,
        material_parameters=tuple(
            sorted(
                (str(k), float(v))
                for k, v in task["object"]["material_properties"][task["object"]["material"]].items()
            )
        ),
        instruction=instruction,
        contact_order=(),
        contact_hold_s_by_region=(),
        probe_positions=(),
        probe_tool_dirs=(),
        probe_finger_dirs=(),
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
                "force_n": float(step.touch_force_n if step.kind == "grasp" and step.touch_force_n is not None else step.force_n),
                "hold_s": float(step.touch_hold_s if step.kind == "grasp" else step.hold_s),
            }
            for step in steps
        ],
        "regime": regime_name,
        "wall_region": wall_name,
        "probe_force_n": probe_force,
        "grasp_force_n": grasp_force,
        "contact_hold_s": hold_s,
        "grasp_points": [
            {"kind": step.kind, "region": region_order[step.region], "position": list(step.position)}
            for step in steps
            if step.kind == "grasp"
        ],
    }
    return spec, tuple(steps), metadata


def generate_thermal_safe(
    *, task: dict, dataset: dict, split: str, start: int, count: int,
    instruction_index: int,
    out_dir: Path, seed: int | None, skip_existing: bool,
    max_attempts_per_episode: int,
) -> dict[str, int]:
    """Generate ``count`` total episodes, balanced round-robin over experts."""
    if seed is not None:
        dataset = {**dataset, "seed": int(seed)}
    policies = [dict(policy) for policy in THERMAL_SAFE_POLICIES]
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_scene(task, seed=0)
    written = skipped = discarded = 0
    started_at = time.time()
    if max_attempts_per_episode < 1:
        raise SystemExit("--max-attempts-per-episode must be >= 1")

    for offset, global_index in enumerate(range(start, start + count), start=1):
        policy = policies[global_index % len(policies)]
        success_slot = global_index // len(policies)
        policy_dir = out_dir / str(policy["id"])
        policy_dir.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            if attempt >= max_attempts_per_episode:
                raise SystemExit(
                    f"{policy['id']} slot={success_slot} 连续失败 {attempt} 次；"
                    "请提高 --max-attempts-per-episode 或检查专家控制。"
                )
            spec, steps, expert_meta = _make_episode(
                task, dataset, policy, success_slot, attempt,
                global_index * max_attempts_per_episode + attempt,
                split=split,
                instruction_index=instruction_index,
            )
            path = policy_dir / f"{spec.episode_id}.hdf5"
            if skip_existing and path.exists():
                skipped += 1
                break

            bundle.reset_episode(spec)
            controller = Pi05ExpertController(bundle, spec, steps)
            buffers, metadata, static = run_episode(bundle, spec, controller=controller)
            metadata.update(expert_meta)
            metadata["success"] = bool(metadata["task_success"]["success"])
            if metadata["success"]:
                write_episode(path, buffers, metadata, static)
                written += 1
                print(
                    f"[{offset}/{count}] {spec.episode_id} "
                    f"attempt={attempt} {metadata['control_frames']} control frames"
                )
                break
            else:
                discarded += 1
                print(
                    f"丢弃 {policy['id']} success_slot={success_slot} attempt={attempt}: "
                    f"{metadata['termination_reason']}"
                )
                attempt += 1

    print(f"热安全专家用时 {time.time() - started_at:.1f}s")
    return {"written": written, "skipped": skipped, "discarded": discarded}


__all__ = ["_make_episode", "generate_thermal_safe"]
