"""并行全量生成 ThermalContact-v1：按划分分块，多进程共享 GPU，可断点续跑。

用法::

    python scripts/generate_all.py --workers 4                # 本地 4 进程
    python scripts/generate_all.py --workers 12 --chunk 200   # 云端 12 进程
    python scripts/generate_all.py --splits train --chunk 20 --limit 240   # 测速

每个分块调用一次 ``scripts/generate_thermal_contact.py``；已存在的文件由
``--skip-existing`` 跳过，中断后重跑同一条命令即可续跑。

集数以数据集配置的 ``splits`` 为准（生成脚本本身也只读它）；清单文件只做
一致性提示。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENERATOR = REPO / "scripts" / "generate_thermal_contact.py"
MANIFEST_DIR = REPO / "data" / "manifests"
SPLITS = (
    "train",
    "validation",
    "test_id",
    "test_finger_history",
    "test_sensor_shift",
    "test_mixed_contact",
    "pilot",
)


def load_dataset(dataset_path: Path) -> tuple[dict[str, int], str, str]:
    """读数据集配置：返回 (各划分集数, 数据集版本, 默认输出目录)。"""
    sys.path.insert(0, str(REPO))
    from sim.config import load_dataset_config

    cfg = load_dataset_config(dataset_path)
    counts = {str(name): int(value) for name, value in cfg["splits"].items()}
    return counts, str(cfg["dataset_version"]), str(cfg["output_dir"])


def build_jobs(
    splits: list[str],
    counts: dict[str, int],
    chunk: int,
    limit: int | None,
) -> list[tuple[str, int, int]]:
    """把每个划分切成 ``chunk`` 大小的块；``limit`` 限制总集数（测速用）。"""
    jobs: list[tuple[str, int, int]] = []
    remaining = limit
    for split in splits:
        total = counts.get(split, 0)
        if total <= 0:
            print(f"警告：划分 {split} 在数据集配置里没有集数，跳过")
            continue
        for start in range(0, total, chunk):
            if remaining is not None and remaining <= 0:
                return jobs
            count = min(chunk, total - start)
            if remaining is not None:
                count = min(count, remaining)
                remaining -= count
            jobs.append((split, start, count))
    return jobs


def manifest_lines(dataset_version: str, split: str) -> int | None:
    """清单行数；清单不存在时返回 None。"""
    path = MANIFEST_DIR / f"{dataset_version}_{split}.jsonl"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def run_job(
    job: tuple[str, int, int],
    out: Path,
    log_dir: Path,
) -> tuple[tuple[str, int, int], int, float, Path]:
    """跑一个分块（阻塞），返回 (任务, 退出码, 墙钟耗时, 日志路径)。"""
    split, start, count = job
    log_path = log_dir / f"{split}_{start:06d}_{count:04d}.log"
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--split",
        split,
        "--start",
        str(start),
        "--count",
        str(count),
        "--out",
        str(out),
        "--skip-existing",
        "--quiet",
    ]
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    return job, code, time.time() - t0, log_path


def tail(path: Path, lines: int = 5) -> str:
    """日志末尾若干行（失败时打印用）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return " | ".join(text[-lines:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="并行生成 ThermalContact-v1（分块、可续跑）")
    parser.add_argument("--dataset", default="configs/datasets/thermal_contact_v1.yaml", help="数据集配置")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--workers", type=int, default=4, help="并行进程数；GPU 计算饱和后收益递减")
    parser.add_argument("--chunk", type=int, default=200, help="每个进程一次生成的集数")
    parser.add_argument("--limit", type=int, default=None, help="总共只生成 N 集（测速用）")
    parser.add_argument("--out", default=None, help="输出根目录（默认取数据集配置 output_dir）")
    parser.add_argument("--log-dir", default="logs/generate_all", help="每个分块的日志目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印分块计划")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.chunk < 1:
        raise SystemExit("--workers 和 --chunk 必须 >= 1")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO / dataset_path
    counts, version, output_dir = load_dataset(dataset_path)

    # 清单只做提示：生成脚本以配置里的集数为准。
    for split in args.splits:
        lines = manifest_lines(version, split)
        if lines is not None and lines != counts.get(split, 0):
            print(
                f"警告：{split} 的清单有 {lines} 行，配置是 {counts.get(split, 0)} 集；"
                "请先重新生成清单（scripts/generate_contact_manifests.py）"
            )

    jobs = build_jobs(args.splits, counts, args.chunk, args.limit)
    total_eps = sum(job[2] for job in jobs)
    print(
        f"划分 {len(args.splits)} 个，分块 {len(jobs)} 个，共 {total_eps} 集；"
        f"{args.workers} 进程，块大小 {args.chunk}"
    )
    if args.dry_run:
        for split, start, count in jobs:
            print(f"  {split:>22} start={start:>6} count={count:>4}")
        return 0

    out = Path(args.out) if args.out else Path(output_dir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出 {out}")
    print(f"日志 {log_dir}")

    t_all = time.time()
    done_eps = 0
    failures: list[tuple[tuple[str, int, int], Path]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job, out, log_dir) for job in jobs]
        for n, future in enumerate(as_completed(futures), start=1):
            (split, start, count), code, elapsed, log_path = future.result()
            done_eps += count
            used = time.time() - t_all
            eta = used / done_eps * (total_eps - done_eps) if done_eps else 0.0
            status = "OK" if code == 0 else f"失败({code})"
            print(
                f"[{n}/{len(jobs)}] {split:>22} start={start:>6} {count:>4} 集 "
                f"{elapsed:7.1f} s {status} | 已完成 {done_eps}/{total_eps} 集，"
                f"已用 {used / 3600:.2f} h，预计剩余 {eta / 3600:.2f} h"
            )
            if code != 0:
                failures.append(((split, start, count), log_path))
                print(f"    日志尾部：{tail(log_path)}")

    elapsed = time.time() - t_all
    print(f"全部结束：{len(jobs)} 个分块，{done_eps} 集，用时 {elapsed / 3600:.2f} h")
    if failures:
        print(f"有 {len(failures)} 个分块失败；重跑同一条命令会自动跳过已生成的文件：")
        for (split, start, count), log_path in failures:
            print(f"  {split} start={start} count={count} -> {log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
