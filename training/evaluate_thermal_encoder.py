"""评估热编码器 checkpoint：逐划分报温度/有效性指标。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.thermal_model import ThermalModel  # noqa: E402
from sim.config import load_dataset_config  # noqa: E402
from training.train_thermal_encoder import (  # noqa: E402
    dataset_for,
    evaluate,
    format_metrics,
    resolve_device,
)

# 默认评估的划分顺序。
DEFAULT_SPLITS = (
    "validation",
    "test_id",
    "test_finger_history",
    "test_sensor_shift",
    "test_mixed_contact",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="评估热编码器")
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", default="configs/datasets/thermal_contact_v1.yaml")
    parser.add_argument("--data-root", default=None, help="覆盖 checkpoint 里的数据根目录")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--limit", type=int, default=None, help="每个划分最多用多少集")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # checkpoint 只含张量与普通字典/列表，按纯权重模式加载即可。
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    stats = payload["stats"]
    device = resolve_device(args.device or str(cfg["training"]["device"]))
    model = ThermalModel(cfg).to(device)
    model.load_state_dict(payload["model"])

    dataset_cfg = load_dataset_config(args.dataset)
    data_cfg = {
        "prefixes_s": [float(v) for v in dataset_cfg["prefixes_s"]],
        "sample_hz": float(dataset_cfg["sample_hz"]),
        "training_hz": float(dataset_cfg["training_hz"]),
    }
    root = Path(args.data_root) if args.data_root else Path(cfg["data"]["root"])
    weights = {k: float(v) for k, v in cfg["training"]["weights"].items()}

    print(f"checkpoint: {args.checkpoint}  epoch={payload.get('epoch')}")
    for split in args.splits:
        name = cfg["data"].get(split, split)
        files = sorted((root / name).glob("*.hdf5"))
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"{split:>20}: 没有文件（{root / name}）")
            continue
        dataset = dataset_for(files, data_cfg, stats)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate(model, loader, device, weights)
        print(f"{split:>20} 窗口={len(dataset):>6}  " + format_metrics("", metrics).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
