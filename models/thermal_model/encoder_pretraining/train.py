"""训练热编码器。

只读取已经生成的 `thermal-encoder-v3` 数据。频率等元数据来自 HDF5 属性。

用法::

    python -m thermal_model.encoder_pretraining.train \
        --data-root datasets/thermal_encoder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from thermal_model.data.loader import (  # noqa: E402
    PreparedThermalEncoderDataset,
    compute_prepared_stats,
)
from thermal_model.encoder_pretraining.config import build_encoder, load_model_config  # noqa: E402
from thermal_model.encoder_pretraining.losses import compute_losses  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练热编码器")
    parser.add_argument("--config", default="thermal_model/configs/encoder_pretraining.yaml")
    parser.add_argument("--data-root", default=None, help="覆盖配置里的数据根目录")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
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


@torch.no_grad()
def evaluate(model, loader, device, weights) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    temp_abs = []
    temp_std = []
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
            temp_abs.append((out.temp_mean_c[keep, 0] - batch["temperature"][keep]).abs())
            temp_std.append(torch.exp(0.5 * out.temp_logvar[keep, 0]))
        pred_v = out.valid_logit[:, 0] > 0
        label_v = batch["contact_valid"] > 0.5
        valid_correct += int((pred_v == label_v).sum())
        valid_total += int(label_v.numel())
    result = {"loss": total_loss / max(batches, 1)}
    if temp_abs:
        abs_err = torch.cat(temp_abs)
        result["temp_mae"] = float(abs_err.mean())
        result["temp_rmse"] = float(torch.sqrt((abs_err**2).mean()))
        std = torch.cat(temp_std)
        result["sigma_mean"] = float(std.mean())
        result["sigma_median"] = float(std.median())
        result["coverage_90"] = float((abs_err <= 1.644854 * std).float().mean())
    if valid_total:
        result["valid_acc"] = valid_correct / valid_total
    return result


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    parts = [f"{prefix} loss={metrics['loss']:.4f}"]
    for key in (
        "temp_mae",
        "temp_rmse",
        "sigma_mean",
        "sigma_median",
        "coverage_90",
        "valid_acc",
    ):
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
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.device is not None:
        train_cfg["device"] = args.device

    root = Path(args.data_root) if args.data_root else Path(cfg["data"]["root"])
    prepared_train = root / f"{cfg['data']['train']}.hdf5"
    prepared_val = root / f"{cfg['data']['validation']}.hdf5"
    if not prepared_train.is_file():
        raise SystemExit(f"训练数据不存在：{prepared_train}")

    if args.stats:
        stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    elif (root / "stats.json").is_file():
        stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
        print(f"读取归一化统计 {root / 'stats.json'}")
    else:
        print(f"计算归一化统计（{prepared_train}）...")
        stats = compute_prepared_stats(prepared_train)

    train_set = PreparedThermalEncoderDataset(prepared_train, stats=stats)
    val_set = (
        PreparedThermalEncoderDataset(prepared_val, stats=stats)
        if prepared_val.is_file()
        else None
    )
    if args.limit_train is not None or args.limit_val is not None:
        from torch.utils.data import Subset

        if args.limit_train is not None:
            train_set = Subset(train_set, range(min(args.limit_train, len(train_set))))
        if val_set is not None and args.limit_val is not None:
            val_set = Subset(val_set, range(min(args.limit_val, len(val_set))))
    if len(train_set) == 0:
        raise ValueError("训练数据集没有可用窗口")
    base_train_set = getattr(train_set, "dataset", train_set)
    training_hz = float(base_train_set.training_hz)
    configured_channels = int(cfg["model"]["signal_dim"])
    dataset_channels = int(base_train_set[0]["features"].shape[-1])
    if dataset_channels != configured_channels:
        raise ValueError(
            f"数据集特征通道 {dataset_channels} 与模型配置 {configured_channels} 不一致；"
            "请重建 12 通道派生数据，不要继续使用旧 checkpoint/stats"
        )
    print(f"训练窗口 {len(train_set)}，验证窗口 {len(val_set) if val_set else 0}")

    device = resolve_device(str(train_cfg["device"]))
    model = build_encoder(cfg).to(device)
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
        # 每个 epoch 都落盘，训练中途也能画曲线。
        out_path.with_suffix(".history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "stats": stats,
                    "dataset_format": str(base_train_set.format),
                    "training_hz": training_hz,
                    "epoch": epoch,
                    "metrics": row,
                },
                out_path,
            )
            print(f"  已保存 {out_path}")

    print(f"完成：最佳分数 {best:.4f}，checkpoint {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
