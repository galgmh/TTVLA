"""评估热编码器 checkpoint：逐划分报温度/有效性指标。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from thermal_model.data.loader import PreparedThermalEncoderDataset  # noqa: E402
from thermal_model.encoder_pretraining.config import build_encoder  # noqa: E402
from thermal_model.encoder_pretraining.train import (  # noqa: E402
    evaluate,
    format_metrics,
    resolve_device,
)

# 默认评估的划分顺序。
DEFAULT_SPLITS = (
    "validation",
    "test",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="评估热编码器")
    parser.add_argument("checkpoint")
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
    model = build_encoder(cfg).to(device)
    model.load_state_dict(payload["model"])

    root = Path(args.data_root) if args.data_root else Path(cfg["data"]["root"])
    weights = {k: float(v) for k, v in cfg["training"]["weights"].items()}

    print(f"checkpoint: {args.checkpoint}  epoch={payload.get('epoch')}")
    for split in args.splits:
        name = cfg["data"].get(split, split)
        prepared_path = root / f"{name}.hdf5"
        if prepared_path.is_file():
            dataset = PreparedThermalEncoderDataset(prepared_path, stats=stats)
            if args.limit:
                dataset = Subset(dataset, range(min(args.limit, len(dataset))))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            metrics = evaluate(model, loader, device, weights)
            print(f"{split:>20} 窗口={len(dataset):>6}  " + format_metrics("", metrics).strip())
        else:
            print(f"{split:>20}: 没有文件（{prepared_path}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
