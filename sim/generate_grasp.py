"""生成不探测温度的纯抓取/抬升 π0.5 基线轨迹。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.recording import write_episode  # noqa: E402
from sim.generate import run_episode  # noqa: E402
from sim.config import REPO_ROOT, load_dataset_config, load_task_config  # noqa: E402
from sim.control.grasp_controller import (  # noqa: E402
    GraspLiftController,
    load_grasp_policy_config,
)
from sim.sampler import ThermalContactSampler  # noqa: E402
from sim.scene import build_scene  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Genesis 纯抓取 π0.5 基线数据")
    parser.add_argument("--policy", required=True, help="抓取策略 YAML")
    parser.add_argument("--config", default="sim/configs/tasks/thermal_contact_probe.yaml")
    parser.add_argument("--dataset", default="sim/configs/datasets/thermal_contact_v1.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--out", default="datasets/grasp_baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--instruction", default=None, help="覆盖 policy 中的 instruction")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task = load_task_config(args.config)
    dataset = load_dataset_config(args.dataset)
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    policy = load_grasp_policy_config(policy_path)
    instruction = str(args.instruction or policy.instruction)
    sampler = ThermalContactSampler(task, dataset, args.split, seed=args.seed)
    if args.start < 0 or args.count < 1 or args.start + args.count > sampler.count:
        raise SystemExit(f"范围 [{args.start}, {args.start + args.count}) 超出 {args.split} 的 {sampler.count} 集")
    try:
        region = tuple(task["object"]["region_order"]).index(policy.grasp_region)
    except ValueError as exc:
        raise SystemExit(f"未知 grasp_region: {policy.grasp_region}") from exc

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_dir = out_root / policy.policy_id / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"策略 {policy.policy_id}：{args.count} 集，instruction={instruction!r}")
    print(f"复用场景 {args.config}，输出 {out_dir}")
    bundle = build_scene(task, seed=0)
    written = skipped = 0
    t0 = time.time()
    for offset, index in enumerate(range(args.start, args.start + args.count), start=1):
        base = sampler.sample(index)
        # multi-contact 采样器已为四个区域生成各自的抖动位姿；只取本策略的抓取区。
        if len(base.probe_positions) != len(task["object"]["region_order"]):
            raise RuntimeError("纯抓取生成器需要 multi_contact 数据配置提供四区域抖动位姿")
        spec = replace(
            base,
            episode_id=f"{policy.policy_id}_{args.split}_{index:06d}",
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
        if args.skip_existing and path.exists():
            skipped += 1
            continue
        bundle.reset_episode(spec)
        controller = GraspLiftController(bundle, spec, policy)
        buffers, metadata, static = run_episode(bundle, spec, controller=controller)
        metadata.update({
            "task_id": policy.policy_id,
            "grasp_region": policy.grasp_region,
            "success": metadata["termination_reason"] == "grasp_lift_complete",
            "thermal_exploration": False,
        })
        write_episode(path, buffers, metadata, static)
        written += 1
        print(
            f"[{offset}/{args.count}] {spec.episode_id} "
            f"{metadata['termination_reason']} {metadata['control_frames']} 控制帧"
        )
    print(f"完成：写出 {written} 集，跳过 {skipped} 集，用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

