"""训练热编码器。

数据来自各划分目录下的 HDF5（一集一个文件）；归一化统计只在训练集上算，
第一次训练时流式计算一次并写进 checkpoint，评估时复用。

用法::

    python training/train_thermal_encoder.py --config configs/models/thermal_encoder.yaml \
        --data-root /tmp/thermal_smoke --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets.window_dataset import ThermalWindowDataset, compute_stats  # noqa: E402
from models.config import load_model_config  # noqa: E402
from models.losses import compute_losses  # noqa: E402
from models.thermal_model import ThermalModel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练热编码器")
    parser.add_argument("--config", default="configs/models/thermal_encoder.yaml")
    parser.add_argument("--dataset", default="configs/datasets/thermal_contact_v1.yaml")
    parser.add_argument("--data-root", default=None, help="覆盖配置里的数据根目录")
    parser.add_argument("--train-split", default=None, help="覆盖训练划分目录名")
    parser.add_argument("--val-split", default=None, help="覆盖验证划分目录名")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None, help="只用前 N 个训练文件")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--stats", default=None, help="已有的归一化统计 JSON")
    parser.add_argument("--out", default=None, help="checkpoint 路径")
    parser.add_argument("--device", default=None)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def split_dir(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def dataset_for(files: list[Path], data_cfg: dict, stats: dict | None) -> ThermalWindowDataset:
    sample_hz = float(data_cfg["sample_hz"])
    training_hz = float(data_cfg["training_hz"])
    stride = int(round(sample_hz / training_hz))
    return ThermalWindowDataset(
        files,
        prefixes_s=data_cfg["prefixes_s"],
        training_hz=training_hz,
        stride=stride,
        stats=stats,
    )


@torch.no_grad()
def evaluate(model, loader, device, weights) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    temp_abs = []
    valid_correct = 0
    valid_total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["features"], batch["mask"])
        terms = compute_losses(out, batch)
        total_loss += float(terms.total(weights))
        batches += 1
        keep = torch.isfinite(batch["temperature"])
        if keep.any():
            temp_abs.append((out.temperature[keep] - batch["temperature"][keep]).abs())
        pred_v = out.valid_logit > 0
        label_v = batch["contact_valid"] > 0.5
        valid_correct += int((pred_v == label_v).sum())
        valid_total += int(label_v.numel())
    result = {"loss": total_loss / max(batches, 1)}
    if temp_abs:
        abs_err = torch.cat(temp_abs)
        result["temp_mae"] = float(abs_err.mean())
        result["temp_rmse"] = float(torch.sqrt((abs_err**2).mean()))
    if valid_total:
        result["valid_acc"] = valid_correct / valid_total
    return result


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    parts = [f"{prefix} loss={metrics['loss']:.4f}"]
    for key in ("temp_mae", "temp_rmse", "valid_acc"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return "  ".join(parts)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_model_config(args.config)
    train_cfg = dict(cfg["training"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.device is not None:
        train_cfg["device"] = args.device

    root = Path(args.data_root) if args.data_root else Path(cfg["data"]["root"])
    train_name = args.train_split or cfg["data"]["train"]
    val_name = args.val_split or cfg["data"]["validation"]
    train_files = sorted(split_dir(root, train_name).glob("*.hdf5"))
    val_files = sorted(split_dir(root, val_name).glob("*.hdf5"))
    if args.limit_train:
        train_files = train_files[: args.limit_train]
    if args.limit_val:
        val_files = val_files[: args.limit_val]
    if not train_files:
        raise SystemExit(f"训练目录没有 HDF5 文件：{split_dir(root, train_name)}")

    # 数据侧的前缀与频率口径来自数据集配置，模型配置只存模型/训练超参。
    from sim.config import load_dataset_config

    dataset_cfg = load_dataset_config(args.dataset)
    data_cfg = {
        "prefixes_s": [float(v) for v in dataset_cfg["prefixes_s"]],
        "sample_hz": float(dataset_cfg["sample_hz"]),
        "training_hz": float(dataset_cfg["training_hz"]),
    }
    if args.stats:
        stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    else:
        print(f"计算归一化统计（{len(train_files)} 个训练文件）...")
        stats = compute_stats(
            train_files,
            prefixes_s=data_cfg["prefixes_s"],
            training_hz=data_cfg["training_hz"],
            stride=int(round(data_cfg["sample_hz"] / data_cfg["training_hz"])),
        )
    train_set = dataset_for(train_files, data_cfg, stats)
    val_set = dataset_for(val_files, data_cfg, stats) if val_files else None
    print(f"训练窗口 {len(train_set)}，验证窗口 {len(val_set) if val_set else 0}")

    device = resolve_device(str(train_cfg["device"]))
    model = ThermalModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    weights = {k: float(v) for k, v in train_cfg["weights"].items()}

    generator = torch.Generator().manual_seed(int(train_cfg["seed"]))
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
        )
        if val_set
        else None
    )

    out_dir = Path(train_cfg["output_dir"])
    if not out_dir.is_absolute():
        from sim.config import REPO_ROOT

        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "thermal_encoder.pt"
    # --out 可能指向配置目录之外的路径，先保证父目录存在。
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    history: list[dict] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["features"], batch["mask"])
            terms = compute_losses(out, batch)
            loss = terms.total(weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        train_loss = running / max(batches, 1)
        row = {"epoch": epoch, "train_loss": train_loss, "time_s": time.time() - t0}
        print(f"epoch {epoch}: train loss={train_loss:.4f}  ({row['time_s']:.1f} s)")
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, weights)
            row.update({f"val_{k}": v for k, v in metrics.items()})
            print("  " + format_metrics("val", metrics))
            score = metrics["loss"]
        else:
            score = train_loss
        history.append(row)
        if score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "stats": stats,
                    "prefixes_s": data_cfg["prefixes_s"],
                    "training_hz": data_cfg["training_hz"],
                    "epoch": epoch,
                    "metrics": row,
                },
                out_path,
            )
            print(f"  已保存 {out_path}")

    (out_path.with_suffix(".history.json")).write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"完成：最佳分数 {best:.4f}，checkpoint {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
