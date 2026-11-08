"""生成各划分的 episode 清单（每条 = 一份完整参数组合的快照）。

清单是“划分一旦生成就不再变”的冻结层：生成脚本可以只读清单跑集，校验
脚本也能对照清单复查分布。每行一个 JSON 对象：``{"index": i, "spec": {...}}``。

用法::

    python scripts/generate_contact_manifests.py            # 全部划分
    python scripts/generate_contact_manifests.py --splits pilot train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import REPO_ROOT, load_dataset_config, load_task_config  # noqa: E402
from sim.sampling.thermal_contact_sampler import SPLITS, ThermalContactSampler  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 ThermalContact-v1 划分清单")
    parser.add_argument("--config", default="configs/tasks/thermal_contact_probe.yaml")
    parser.add_argument("--dataset", default="configs/datasets/thermal_contact_v1.yaml")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--out", default="data/manifests")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def manifest_path(out_dir: Path, version: str, split: str) -> Path:
    return out_dir / f"{version}_{split}.jsonl"


def main() -> int:
    args = build_parser().parse_args()
    task = load_task_config(args.config)
    dataset = load_dataset_config(args.dataset)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    version = str(dataset["dataset_version"])
    for split in args.splits:
        sampler = ThermalContactSampler(task, dataset, split, seed=args.seed)
        path = manifest_path(out_dir, version, split)
        regions: Counter[str] = Counter()
        regimes: Counter[str] = Counter()
        qualities: Counter[str] = Counter()
        forces: Counter[float] = Counter()
        with path.open("w", encoding="utf-8") as handle:
            for index in range(sampler.count):
                spec = sampler.sample(index)
                handle.write(json.dumps({"index": index, "spec": spec.to_dict()}, ensure_ascii=False))
                handle.write("\n")
                regions[spec.target_region_name] += 1
                regimes[spec.regime] += 1
                qualities[spec.contact_quality] += 1
                forces[spec.force_level] += 1
        print(f"{split}: {sampler.count} 集 -> {path.name}")
        print(f"  区域 {dict(regions)}")
        print(f"  情景 {dict(regimes)}")
        print(f"  质量 {dict(qualities)}")
        print(f"  力档 {dict(forces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
