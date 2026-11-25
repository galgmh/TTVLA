"""Direct-grasp expert pipeline used by the unified generator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import zlib

from sim.recording import write_episode
from sim.generation.episode_runner import run_episode
from sim.control.grasp_controller import (
    GraspLiftController,
    grasp_policy_from_task,
)
from sim.sampler import LiftSampler
from sim.scene import build_scene


def generate_direct_grasp(
    *, task: dict, dataset: dict, split: str, start: int, count: int,
    instruction_index: int,
    out_dir: Path, seed: int | None, skip_existing: bool,
) -> dict[str, int]:
    policy = grasp_policy_from_task(task)
    instruction = str(task["instruction"])
    base_seed = int(dataset["seed"] if seed is None else seed)
    instruction_seed = (
        base_seed + int(zlib.crc32(instruction.encode("utf-8")))
    ) % (2**32)
    try:
        region = tuple(task["object"]["region_order"]).index(policy.grasp_region)
    except ValueError as exc:
        raise SystemExit(f"未知 grasp_region: {policy.grasp_region}") from exc
    sampler = LiftSampler(
        task,
        dataset,
        split,
        seed=instruction_seed,
        count=start + count,
        region=region,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"专家 {policy.policy_id}：{count} 集，instruction={instruction!r}")
    bundle = build_scene(task, seed=0)
    written = skipped = 0
    for offset, index in enumerate(range(start, start + count), start=1):
        base = sampler.sample(index)
        spec = replace(
            base,
            episode_id=(
                f"{policy.policy_id}_ins{instruction_index:02d}_{index:06d}"
            ),
            instruction=instruction,
        )
        path = out_dir / f"{spec.episode_id}.hdf5"
        if skip_existing and path.exists():
            skipped += 1
            continue
        bundle.reset_episode(spec)
        controller = GraspLiftController(bundle, spec, policy)
        buffers, metadata, static = run_episode(bundle, spec, controller=controller)
        metadata.update({
            "task_id": policy.policy_id,
            "instruction_index": int(instruction_index),
            "grasp_region": policy.grasp_region,
            "success": bool(metadata["task_success"]["success"]),
            "thermal_exploration": False,
        })
        write_episode(path, buffers, metadata, static)
        written += 1
        print(
            f"[{offset}/{count}] {spec.episode_id} "
            f"{metadata['termination_reason']} {metadata['control_frames']} 控制帧"
        )
    return {"written": written, "skipped": skipped, "discarded": 0}


__all__ = ["generate_direct_grasp"]

