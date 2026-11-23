"""Unified command-line entry point for all simulator data generation.

Example::

    python -m sim.generate \
      --task sim/configs/tasks/thermal_safe_grasp.yaml \
      --scene sim/configs/scene/cup_on_table.yaml \
      --split train --start 0 --num_per_ins 100 --out datasets/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from sim.config import REPO_ROOT, load_dataset_config, load_task_config
from sim.generation.direct_grasp import generate_direct_grasp
from sim.generation.thermal_safe import generate_thermal_safe

DIRECT_GRASP_GOALS = {"cup_wall", "handle", "cup_rim"}
PRESET_BY_GOAL = {
    "cup_wall": "sim/generation/presets/thermal_contact_v1.yaml",
    "handle": "sim/generation/presets/thermal_contact_v1.yaml",
    "cup_rim": "sim/generation/presets/thermal_contact_v1.yaml",
    "thermally_safe": "sim/generation/presets/pi05_v1.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一生成 Genesis 专家示范 HDF5")
    parser.add_argument("--task", required=True, help="任务 YAML")
    parser.add_argument("--scene", required=True, help="场景 YAML")
    parser.add_argument("--split", required=True, help="train/validation/test")
    parser.add_argument("--start", type=int, default=0, help="每条 instruction 的起始轨迹编号")
    parser.add_argument(
        "--num_per_ins", type=int, required=True,
        help="每条 instruction 生成的 episode 数量",
    )
    parser.add_argument("--out", required=True, help="输出根目录")
    parser.add_argument("--seed", type=int, default=None, help="覆盖生成预设随机种子")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-attempts-per-episode", type=int, default=20)
    return parser


def _output_directory(root: str | Path, task_id: str, split: str) -> Path:
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / task_id / split


def main() -> int:
    args = build_parser().parse_args()
    if args.start < 0:
        raise SystemExit("--start 必须 >= 0")
    if args.num_per_ins < 1:
        raise SystemExit("--num_per_ins 必须 >= 1")
    if not args.split.strip():
        raise SystemExit("--split 不能为空")

    task = load_task_config(args.task, scene_path=args.scene)
    goal = str(task["success"]["required_grasp_region"])
    try:
        preset_path = PRESET_BY_GOAL[goal]
    except KeyError as exc:
        raise SystemExit(f"任务 {task['task_id']} 没有注册生成管线：{goal}") from exc
    preset = load_dataset_config(preset_path)
    out_dir = _output_directory(args.out, str(task["task_id"]), args.split)

    instructions = list(task["instructions"])
    print(f"任务：{task['task_id']}，{len(instructions)} 条 instruction")
    print(f"场景：{task['scene_id']}")
    print(
        f"每条范围：{args.split}[{args.start}:{args.start + args.num_per_ins}]，"
        f"总计 {len(instructions) * args.num_per_ins} 集"
    )
    print(f"输出：{out_dir}")
    started_at = time.time()

    summary = {"written": 0, "skipped": 0, "discarded": 0}
    for instruction_index, instruction in enumerate(instructions):
        episode_task = {
            **task,
            "instruction": str(instruction),
            "instruction_index": int(instruction_index),
        }
        print(f"instruction[{instruction_index}]: {instruction}")
        common = dict(
            task=episode_task,
            dataset=preset,
            split=args.split,
            start=args.start,
            count=args.num_per_ins,
            instruction_index=instruction_index,
            out_dir=out_dir,
            seed=args.seed,
            skip_existing=args.skip_existing,
        )
        if goal in DIRECT_GRASP_GOALS:
            result = generate_direct_grasp(**common)
        else:
            result = generate_thermal_safe(
                **common,
                max_attempts_per_episode=args.max_attempts_per_episode,
            )
        for key in summary:
            summary[key] += result[key]

    print(
        f"完成：写出 {summary['written']}，跳过 {summary['skipped']}，"
        f"丢弃失败 {summary['discarded']}，用时 {time.time() - started_at:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
