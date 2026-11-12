"""批量生成 ThermalContact-v1 原始数据（一集一个 HDF5）。

用法::

    python scripts/generate_thermal_contact.py --split pilot --count 20

每集流程：采样 ``EpisodeSpec`` → 复位场景 → 受控触碰控制器跑完一集 →
记录器逐帧收 ``get_policy_observation`` / ``get_privileged_observation``
与指令 → 按 ``data.schema`` 契约写出。派生量（ΔT、斜率、前缀窗口）一律
不落盘。

场景只构建一次，逐集复位；输出目录默认取数据集配置的 ``output_dir``。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 允许直接以脚本方式运行（仓库根目录在 sys.path 里）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.schema.thermal_contact_v1 import metadata_for_spec  # noqa: E402
from data.writer.hdf5_writer import write_episode  # noqa: E402
from data.writer.recorder import EpisodeRecorder  # noqa: E402
from sim.config import load_dataset_config, load_task_config  # noqa: E402
from sim.control.probe_controller import MultiProbeController, ProbeController  # noqa: E402
from sim.obs import (  # noqa: E402
    get_policy_observation,
    get_privileged_observation,
    get_robot_observation,
)
from sim.sampling.thermal_contact_sampler import ThermalContactSampler  # noqa: E402
from sim.scene import build_scene  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 ThermalContact-v1 原始数据")
    parser.add_argument("--config", default="configs/tasks/thermal_contact_probe.yaml", help="任务配置")
    parser.add_argument("--dataset", default="configs/datasets/thermal_contact_v1.yaml", help="数据集配置")
    parser.add_argument("--split", default="pilot", help="划分名（train/validation/test_*/pilot）")
    parser.add_argument("--start", type=int, default=0, help="起始序号（含）")
    parser.add_argument("--count", type=int, default=None, help="生成集数；省略表示到划分末尾")
    parser.add_argument("--out", default=None, help="输出根目录（默认取数据集配置 output_dir）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖数据集种子")
    parser.add_argument("--skip-existing", action="store_true", help="已存在的文件跳过")
    parser.add_argument("--quiet", action="store_true", help="只打印汇总")
    return parser


def _resolve_indices(sampler: ThermalContactSampler, start: int, count: int | None) -> list[int]:
    end = sampler.count if count is None else min(sampler.count, start + count)
    if start < 0 or start >= sampler.count:
        raise SystemExit(f"--start {start} 超出划分 {sampler.split} 的 [0, {sampler.count})")
    return list(range(start, end))


def run_episode(bundle, spec) -> tuple[dict, dict, dict]:
    """跑一集，返回 ``(多时间轴缓冲, 元数据, 静态标定)``。"""
    probe = MultiProbeController(bundle, spec) if spec.contact_order else ProbeController(bundle, spec)
    recorder = EpisodeRecorder(
        target_region=int(spec.target_region),
        link_count=len(bundle.regions),
    )
    recorder.set_camera_calibration(bundle.camera_calibration())
    sim_dt = float(bundle.cfg["controller"]["sim_dt"])

    def on_sensor_sample(sim_step: int) -> None:
        recorder.add_sensor(
            sim_time=float(sim_step) * sim_dt,
            tactile=get_policy_observation(bundle),
            privileged=get_privileged_observation(bundle),
        )

    def on_control_sample(sim_step: int) -> None:
        command = probe.last_command
        if command is None:
            raise RuntimeError("控制器尚未下发任何指令")
        recorder.add_control(
            sim_time=float(sim_step) * sim_dt,
            sensor_index=recorder.sensor_frames - 1,
            command=command,
            target_force_n=float(spec.target_force_n),
            robot=get_robot_observation(bundle),
            camera=bundle.camera_observation(),
        )

    outcome = probe.run(on_sample=on_sensor_sample, on_control=on_control_sample)
    recorder.finalize()
    cup_pos, cup_quat = bundle.cup_pose()
    import genesis as gs

    metadata = metadata_for_spec(
        spec,
        genesis_version=str(getattr(gs, "__version__", "unknown")),
        cup_pose=np.concatenate([cup_pos, cup_quat]),
        cup_asset_version=str(bundle.cfg["object"]["asset_version"]),
        outcome=outcome,
    )
    metadata.update(recorder.summary())
    metadata["config_paths"] = dict(bundle.cfg["config_paths"])
    return recorder.buffers, metadata, recorder.static


def main() -> int:
    args = build_parser().parse_args()
    task = load_task_config(args.config)
    dataset = load_dataset_config(args.dataset)
    sampler = ThermalContactSampler(task, dataset, args.split, seed=args.seed)
    indices = _resolve_indices(sampler, args.start, args.count)

    out_root = Path(args.out) if args.out else Path(dataset["output_dir"])
    if not out_root.is_absolute():
        from sim.config import REPO_ROOT

        out_root = REPO_ROOT / out_root
    out_dir = out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"划分 {args.split}：{len(indices)} 集，输出 {out_dir}")
        print("构建场景 ...")
    bundle = build_scene(task, seed=0)

    written = skipped = 0
    t0 = time.time()
    for n, index in enumerate(indices, start=1):
        spec = sampler.sample(index)
        path = out_dir / f"{spec.episode_id}.hdf5"
        if args.skip_existing and path.exists():
            skipped += 1
            continue
        bundle.reset_episode(spec)
        buffers, metadata, static = run_episode(bundle, spec)
        write_episode(path, buffers, metadata, static)
        written += 1
        if not args.quiet:
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0.0
            print(
                f"[{n}/{len(indices)}] {spec.episode_id} "
                f"{spec.target_region_name:>6} {spec.contact_quality:>6} "
                f"{metadata['termination_reason']:>16} "
                f"{metadata['sensor_frames']:4d}/{metadata['control_frames']:4d} 帧 "
                f"{metadata['peak_normal_force_n']:.2f} N "
                f"({rate:.2f} 集/s)"
            )
    elapsed = time.time() - t0
    print(f"完成：写出 {written} 集，跳过 {skipped} 集，用时 {elapsed:.1f} s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
