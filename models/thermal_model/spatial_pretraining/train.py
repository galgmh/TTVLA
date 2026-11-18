"""Train spatial thermal representation on complete episode sequences."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.config import REPO_ROOT  # noqa: E402
from thermal_model.data.loader import compute_prepared_stats  # noqa: E402
from thermal_model.memory_bank import ThermalMemoryBank  # noqa: E402
from thermal_model.spatial_pretraining.config import (  # noqa: E402
    build_model,
    load_encoder_checkpoint,
    load_model_config,
)

MEMORY_SIZE = 8
CONTACT_ON_THRESHOLD = 0.7
CONTACT_OFF_THRESHOLD = 0.3
CONTACT_ON_STEPS = 2
CONTACT_OFF_STEPS = 2
ACTIVE_TOKEN_MOMENTUM = 0.8


class EpisodeDataset(Dataset):
    """按 ``episode_offsets`` 返回一条完整的空间热觉序列。"""

    def __init__(self, path: str | Path, *, stats: dict[str, list[float]] | None = None):
        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            self.format = str(handle.attrs.get("format", ""))
            if self.format != "thermal-spatial-v2":
                raise ValueError(
                    f"{self.path} 不是 thermal-spatial-v2；"
                    "请用 spatial_pretraining/prepare_dataset.py --overwrite 重建"
                )
            if "episode_offsets" not in handle:
                raise ValueError(
                    f"{self.path} 没有 episode_offsets；请用 spatial_pretraining/"
                    "prepare_dataset.py --overwrite 重建"
                )
            self.episode_offsets = np.asarray(handle["episode_offsets"], dtype=np.int64)
            self.episode_index = np.asarray(handle["episode_index"], dtype=np.int32)
            self.training_hz = float(handle.attrs["training_hz"])
            self.window_size = int(handle["features"].shape[1])
            self.channels = int(handle["features"].shape[2])
        if self.episode_offsets[0] != 0:
            raise ValueError("episode_offsets 必须从0开始")
        if self.episode_offsets[-1] != len(self.episode_index):
            raise ValueError("episode_offsets 最后一个值必须等于窗口数量")
        self.stats = stats
        self._handle = None

    def __len__(self) -> int:
        return len(self.episode_offsets) - 1

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"episode 下标 {index} 超出 [0, {len(self)})")
        handle = self._file()
        start = int(self.episode_offsets[index])
        stop = int(self.episode_offsets[index + 1])
        features = np.asarray(handle["features"][start:stop], dtype=np.float32)
        mask = np.asarray(handle["mask"][start:stop], dtype=bool)
        if self.stats is not None:
            mean = np.asarray(self.stats["mean"], dtype=np.float32)
            std = np.asarray(self.stats["std"], dtype=np.float32)
            features = (features - mean) / std
            features[~mask] = 0.0
        steps = stop - start
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "positions": torch.from_numpy(
                np.asarray(handle["positions"][start:stop], dtype=np.float32)
            ),
            "normals": torch.from_numpy(
                np.asarray(handle["normals"][start:stop], dtype=np.float32)
            ),
            "query_xyz": torch.from_numpy(
                np.asarray(handle["query_xyz"][start:stop], dtype=np.float32)
            ),
            "query_temperature": torch.from_numpy(
                np.asarray(handle["query_temperature"][start:stop], dtype=np.float32)
            ),
            "query_mask": torch.from_numpy(
                np.asarray(handle["query_mask"][start:stop], dtype=bool)
            ),
            "query_region": torch.from_numpy(
                np.asarray(handle["query_region"][start:stop], dtype=np.int64)
            ),
            "contact_valid": torch.from_numpy(
                np.asarray(handle["contact_valid"][start:stop], dtype=bool)
            ),
            "contact_region": torch.from_numpy(
                np.asarray(handle["contact_region"][start:stop], dtype=np.int64)
            ),
            "sequence_mask": torch.ones(steps, dtype=torch.bool),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass


def collate_episodes(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """把不同长度 episode 补齐到 batch 内最长序列。"""
    batch = len(items)
    max_steps = max(int(item["sequence_mask"].shape[0]) for item in items)
    first = items[0]
    output = {
        "features": torch.zeros(
            batch, max_steps, *first["features"].shape[1:], dtype=first["features"].dtype
        ),
        "mask": torch.zeros(
            batch, max_steps, *first["mask"].shape[1:], dtype=torch.bool
        ),
        "positions": torch.zeros(
            batch, max_steps, *first["positions"].shape[1:], dtype=first["positions"].dtype
        ),
        "normals": torch.zeros(
            batch, max_steps, *first["normals"].shape[1:], dtype=first["normals"].dtype
        ),
        "query_xyz": torch.zeros(
            batch, max_steps, *first["query_xyz"].shape[1:], dtype=first["query_xyz"].dtype
        ),
        "query_temperature": torch.zeros(
            batch,
            max_steps,
            *first["query_temperature"].shape[1:],
            dtype=first["query_temperature"].dtype,
        ),
        "query_mask": torch.zeros(
            batch, max_steps, *first["query_mask"].shape[1:], dtype=torch.bool
        ),
        "query_region": torch.zeros(
            batch,
            max_steps,
            *first["query_region"].shape[1:],
            dtype=first["query_region"].dtype,
        ),
        "contact_valid": torch.zeros(batch, max_steps, dtype=torch.bool),
        "contact_region": torch.zeros(
            batch, max_steps, dtype=first["contact_region"].dtype
        ),
        "sequence_mask": torch.zeros(batch, max_steps, dtype=torch.bool),
    }
    for row, item in enumerate(items):
        steps = int(item["sequence_mask"].shape[0])
        for name in (
            "features",
            "mask",
            "positions",
            "normals",
            "query_xyz",
            "query_temperature",
            "query_mask",
            "query_region",
            "contact_valid",
            "contact_region",
            "sequence_mask",
        ):
            output[name][row, :steps] = item[name]
    return output


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {name: value.to(device) for name, value in batch.items()}


def make_memory_bank(hidden_dim: int) -> ThermalMemoryBank:
    return ThermalMemoryBank(
        MEMORY_SIZE,
        hidden_dim,
        on_threshold=CONTACT_ON_THRESHOLD,
        off_threshold=CONTACT_OFF_THRESHOLD,
        on_steps=CONTACT_ON_STEPS,
        off_steps=CONTACT_OFF_STEPS,
        momentum=ACTIVE_TOKEN_MOMENTUM,
    )


def run_episode_batch(
    model,
    memory_bank: ThermalMemoryBank,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    *,
    backward: bool = False,
    memory_mode: str = "full",
    field_loss: str = "mse",
    huber_delta: float = 2.0,
) -> tuple[float, int, float, float]:
    """按时间顺序运行一个 episode batch，返回加权 loss 和误差累计。"""
    sequence_mask = batch["sequence_mask"]
    batch_size, steps = sequence_mask.shape
    parameter = next(model.parameters())
    contact_valid = batch["contact_valid"] & sequence_mask
    contact_region = batch["contact_region"]
    safe_region = contact_region.clamp(0, batch["query_xyz"].shape[2] - 1)
    current_query = batch["query_xyz"].gather(
        2,
        safe_region[..., None, None].expand(-1, -1, 1, 3),
    )
    current_target = batch["query_temperature"].gather(
        2,
        safe_region[..., None, None].expand(-1, -1, 1, 1),
    )
    current_query_mask = batch["query_mask"].gather(
        2,
        safe_region[..., None],
    )
    all_finite = torch.isfinite(batch["query_temperature"]).squeeze(-1)
    memory_bank.reset(
        batch_size,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    stored_regions = torch.full(
        (batch_size, memory_bank.capacity),
        -1,
        dtype=torch.long,
        device=parameter.device,
    )
    active_region = torch.full(
        (batch_size,), -1, dtype=torch.long, device=parameter.device
    )
    seen_regions = torch.zeros(
        (batch_size, batch["query_xyz"].shape[2]),
        dtype=torch.bool,
        device=parameter.device,
    )

    weighted_error_sum = 0.0
    count = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    for step in range(steps):
        if not bool(sequence_mask[:, step].any()):
            continue
        contact = model.contact_encoder(batch["features"][:, step], batch["mask"][:, step])
        token = model.contact_builder(
            contact.latent,
            batch["positions"][:, step],
            batch["normals"][:, step],
        )
        probability = torch.sigmoid(contact.valid_logit)
        step_contact = contact_valid[:, step]
        step_region = contact_region[:, step]
        active_region = torch.where(
            step_contact & (step_region >= 0),
            step_region,
            active_region,
        )
        if memory_mode == "full":
            before_count = memory_bank.mask.sum(dim=1)
            memory_bank.update(token, probability)
            after_count = memory_bank.mask.sum(dim=1)
            for row in range(batch_size):
                before = int(before_count[row])
                after = int(after_count[row])
                if after <= before:
                    continue
                if before >= memory_bank.capacity:
                    stored_regions[row, :-1] = stored_regions[row, 1:].clone()
                    slot = memory_bank.capacity - 1
                else:
                    slot = before
                stored_regions[row, slot] = active_region[row]
                active_region[row] = -1
            tokens = torch.cat([memory_bank.tokens, token[:, None]], dim=1)
            token_mask = torch.cat(
                [memory_bank.mask, step_contact[:, None]],
                dim=1,
            )
        elif memory_mode == "current":
            tokens = token[:, None]
            token_mask = step_contact[:, None]
        else:
            raise ValueError(f"unknown memory_mode: {memory_mode}")
        current_seen = (
            step_contact
            & (step_region >= 0)
            & (step_region < seen_regions.shape[1])
        )
        if current_seen.any():
            seen_regions |= (
                torch.nn.functional.one_hot(
                    step_region.clamp_min(0),
                    num_classes=seen_regions.shape[1],
                ).bool()
                & current_seen[:, None]
            )
        stored_known = stored_regions >= 0
        if stored_known.any():
            seen_regions |= (
                torch.nn.functional.one_hot(
                    stored_regions.clamp_min(0),
                    num_classes=seen_regions.shape[1],
                ).bool()
                & stored_known[:, :, None]
            ).any(dim=1)

        spatial = model.spatial_memory(tokens, token_mask)
        if memory_mode == "full":
            field = model.field_decoder(
                spatial.spatial_latents,
                batch["query_xyz"][:, step],
            )
            target = batch["query_temperature"][:, step]
            available = sequence_mask[:, step] & (
                step_contact | memory_bank.mask.any(dim=1)
            )
            query_valid = (
                available[:, None]
                & batch["query_mask"][:, step]
                & all_finite[:, step]
            )
            region_weights = torch.where(
                seen_regions,
                torch.ones_like(seen_regions, dtype=field.temp_mean_c.dtype),
                torch.full_like(
                    seen_regions,
                    1.2,
                    dtype=field.temp_mean_c.dtype,
                ),
            )
            rows = torch.arange(batch_size, device=parameter.device)
            current_rows = step_contact & (step_region >= 0)
            region_weights[
                rows[current_rows],
                step_region[current_rows],
            ] = 0.5
            sample_weights = query_valid.to(region_weights.dtype) * region_weights
        else:
            field = model.field_decoder(
                spatial.spatial_latents,
                current_query[:, step],
            )
            target = current_target[:, step]
            keep = (
                sequence_mask[:, step, None]
                & step_contact[:, None]
                & (step_region >= 0)[:, None]
                & current_query_mask[:, step]
                & torch.isfinite(target).squeeze(-1)
            )
            sample_weights = keep.to(field.temp_mean_c.dtype)

        keep = sample_weights > 0
        step_count = int(keep.sum().item())
        if step_count == 0:
            continue
        residual = field.temp_mean_c[keep] - target[keep]
        if field_loss == "mse":
            sample_loss = residual.square()
        elif field_loss == "huber":
            sample_loss = F.huber_loss(
                field.temp_mean_c[keep],
                target[keep],
                delta=float(huber_delta),
                reduction="none",
            )
        elif field_loss == "nll":
            logvar = field.temp_logvar[keep]
            sample_loss = 0.5 * (
                logvar + residual.square() / (torch.exp(logvar) + 1e-6)
            )
        else:
            raise ValueError(f"unknown field_loss: {field_loss}")
        active_weights = sample_weights[keep]
        weighted_error = (sample_loss * active_weights).sum()
        weighted_error_sum += float(
            (weighted_error * float(weights["field_temp"])).detach()
        )
        count += step_count
        with torch.no_grad():
            absolute_sum += float(residual.abs().sum())
            squared_sum += float(residual.square().sum())
        if backward:
            step_loss = weighted_error / active_weights.sum().clamp_min(1.0)
            (step_loss * float(weights["field_temp"]) / steps).backward()

    if count == 0:
        return 0.0, 0, 0.0, 0.0
    return weighted_error_sum / count, count, absolute_sum, squared_sum


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    weights: dict[str, float],
    memory_bank: ThermalMemoryBank,
    *,
    memory_mode: str = "full",
    field_loss: str = "mse",
    huber_delta: float = 2.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        loss, batch_count, batch_abs, batch_sq = run_episode_batch(
            model,
            memory_bank,
            batch,
            weights,
            memory_mode=memory_mode,
            field_loss=field_loss,
            huber_delta=huber_delta,
        )
        total_loss += loss * batch_count
        count += batch_count
        absolute_sum += batch_abs
        squared_sum += batch_sq
    result = {"loss": total_loss / max(count, 1)}
    if count:
        result["temp_mae"] = absolute_sum / count
        result["temp_rmse"] = float(np.sqrt(squared_sum / count))
    return result


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    parts = [f"{prefix} loss={metrics['loss']:.4f}"]
    for key in ("temp_mae", "temp_rmse"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return "  ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 episode 训练空间热觉表示")
    parser.add_argument(
        "--config",
        default="thermal_model/configs/spatial_pretraining.yaml",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="每个 batch 的 episode 数")
    parser.add_argument(
        "--memory-mode",
        choices=("full", "current"),
        default="full",
        help="full=使用历史MemoryBank，current=只看当前token",
    )
    parser.add_argument(
        "--field-loss",
        choices=("mse", "huber", "nll"),
        default=None,
        help="覆盖配置中的 field loss",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="加载已有 spatial checkpoint 后继续训练；不恢复optimizer和epoch",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit-train", type=int, default=None, help="只使用前 N 条训练 episode")
    parser.add_argument("--limit-val", type=int, default=None, help="只使用前 N 条验证 episode")
    parser.add_argument("--stats", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_model_config(args.config)
    train_cfg = dict(cfg["training"])
    for name, value in (
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("learning_rate", args.lr),
        ("device", args.device),
    ):
        if value is not None:
            train_cfg[name] = value
    if args.field_loss is not None:
        train_cfg["field_loss"] = args.field_loss

    root = Path(args.data_root) if args.data_root else REPO_ROOT / cfg["data"]["root"]
    train_path = root / f"{cfg['data']['train']}.hdf5"
    val_path = root / f"{cfg['data']['validation']}.hdf5"
    if not train_path.is_file():
        raise SystemExit(f"空间预训练数据不存在：{train_path}")
    if not val_path.is_file():
        raise SystemExit(f"空间验证数据不存在：{val_path}")

    if args.stats:
        stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    elif (root / "stats.json").is_file():
        stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
        print(f"读取归一化统计 {root / 'stats.json'}")
    else:
        stats = compute_prepared_stats(train_path)

    train_set = EpisodeDataset(train_path, stats=stats)
    val_set = EpisodeDataset(val_path, stats=stats)
    if args.limit_train is not None:
        train_set = Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_val is not None:
        val_set = Subset(val_set, range(min(args.limit_val, len(val_set))))
    print(f"训练 episode {len(train_set)}，验证 episode {len(val_set)}")
    print(f"memory mode: {args.memory_mode}")
    print(f"field loss: {train_cfg['field_loss']}")

    device = resolve_device(str(train_cfg["device"]))
    model = build_model(cfg)
    encoder_checkpoint = train_cfg["encoder_checkpoint"]
    if not encoder_checkpoint:
        raise SystemExit("spatial_pretraining.training.encoder_checkpoint 不能为空")
    loaded = load_encoder_checkpoint(model, encoder_checkpoint)
    print(f"加载接触编码器 {loaded}")
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser()
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
        init_payload = torch.load(init_path, map_location="cpu", weights_only=True)
        if init_payload.get("dataset_format") != "thermal-spatial-v2":
            raise SystemExit(f"初始化 checkpoint 不是 thermal-spatial-v2：{init_path}")
        model.load_state_dict(init_payload["model"])
        print(f"初始化空间模型 {init_path}")
    if bool(train_cfg["freeze_encoder"]):
        for parameter in model.contact_encoder.parameters():
            parameter.requires_grad_(False)
    model.to(device)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("空间预训练没有可训练参数")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    weights = {name: float(value) for name, value in train_cfg["weights"].items()}
    memory_bank = make_memory_bank(int(cfg["model"]["hidden_dim"]))
    generator = torch.Generator().manual_seed(int(train_cfg["seed"]))
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_episodes,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_episodes,
    )

    out_dir = Path(train_cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "thermal_spatial.pt"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        if bool(train_cfg["freeze_encoder"]):
            model.contact_encoder.eval()
        start = time.time()
        total_loss = 0.0
        count = 0
        absolute_sum = 0.0
        squared_sum = 0.0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, batch_count, batch_abs, batch_sq = run_episode_batch(
                model,
                memory_bank,
                batch,
                weights,
                backward=True,
                memory_mode=args.memory_mode,
                field_loss=str(train_cfg["field_loss"]),
                huber_delta=float(train_cfg["huber_delta"]),
            )
            if batch_count == 0:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += loss * batch_count
            count += batch_count
            absolute_sum += batch_abs
            squared_sum += batch_sq

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(count, 1),
            "temp_mae": absolute_sum / max(count, 1),
            "temp_rmse": float(np.sqrt(squared_sum / max(count, 1))),
            "time_s": time.time() - start,
        }
        print(
            f"epoch {epoch}: train loss={row['train_loss']:.4f}"
            f"  temp_mae={row['temp_mae']:.4f}"
            f"  temp_rmse={row['temp_rmse']:.4f}"
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            weights,
            memory_bank,
            memory_mode=args.memory_mode,
            field_loss=str(train_cfg["field_loss"]),
            huber_delta=float(train_cfg["huber_delta"]),
        )
        row.update({f"val_{name}": value for name, value in metrics.items()})
        print("  " + format_metrics("val", metrics))
        score = metrics["loss"]
        history.append(row)
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
                    "dataset_format": "thermal-spatial-v2",
                    "sequence_training": True,
                    "memory_mode": args.memory_mode,
                    "field_loss": str(train_cfg["field_loss"]),
                    "huber_delta": float(train_cfg["huber_delta"]),
                    "episode_offsets": "episode_offsets",
                    "training_hz": float(
                        getattr(train_set, "dataset", train_set).training_hz
                    ),
                    "encoder_checkpoint": str(encoder_checkpoint),
                    "init_checkpoint": str(args.init_checkpoint or ""),
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
