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
from sim.sampler import ThermalContactSampler
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
    sampler = ThermalContactSampler(
        task, dataset, split, seed=instruction_seed, count=start + count
    )
    try:
        region = tuple(task["object"]["region_order"]).index(policy.grasp_region)
    except ValueError as exc:
        raise SystemExit(f"未知 grasp_region: {policy.grasp_region}") from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"专家 {policy.policy_id}：{count} 集，instruction={instruction!r}")
    bundle = build_scene(task, seed=0)
    written = skipped = 0
    for offset, index in enumerate(range(start, start + count), start=1):
        base = sampler.sample(index)
        # multi-contact 采样器已为四个区域生成各自的抖动位姿；只取本策略的抓取区。
        if len(base.probe_positions) != len(task["object"]["region_order"]):
            raise RuntimeError("纯抓取生成器需要 multi_contact 数据配置提供四区域抖动位姿")
        spec = replace(
            base,
            episode_id=(
                f"{policy.policy_id}_{split}_ins{instruction_index:02d}_{index:06d}"
            ),
            target_region=region,
            mixed=False,
            contact_quality="stable",
            contact_order=(),
            contact_hold_s_by_region=(),
            probe_positions=(),
            probe_tool_dirs=(),
            probe_finger_dirs=(),
            instruction=instruction,
            probe_pos=base.probe_positions[region],
            probe_tool_dir=base.probe_tool_dirs[region],
            probe_finger_dir=base.probe_finger_dirs[region],
            probe_pre_dist=float(task["regions"][policy.grasp_region]["pre_dist"]),
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

