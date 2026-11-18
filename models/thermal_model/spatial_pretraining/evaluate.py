"""Evaluate a spatial thermal pretraining checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.config import REPO_ROOT  # noqa: E402
from thermal_model.spatial_pretraining.config import build_model  # noqa: E402
from thermal_model.spatial_pretraining.train import (  # noqa: E402
    EpisodeDataset,
    collate_episodes,
    evaluate,
    format_metrics,
    make_memory_bank,
    resolve_device,
    _to_device,
)

DEFAULT_SPLITS = ("validation", "test")
REGION_ROLES = ("current", "seen", "unseen")


@torch.no_grad()
def evaluate_regions(model, loader, device, memory_mode: str, memory_bank) -> dict:
    """在相同状态上统计 current/seen/unseen 区域误差。"""
    model.eval()
    counts = {name: 0 for name in REGION_ROLES}
    absolute = {name: 0.0 for name in REGION_ROLES}
    squared = {name: 0.0 for name in REGION_ROLES}

    for batch in loader:
        batch = _to_device(batch, device)
        sequence_mask = batch["sequence_mask"]
        batch_size, steps = sequence_mask.shape
        number = int(batch["query_xyz"].shape[2])
        parameter = next(model.parameters())
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
            (batch_size, number), dtype=torch.bool, device=parameter.device
        )

        for step in range(steps):
            if not bool(sequence_mask[:, step].any()):
                continue
            contact = model.contact_encoder(
                batch["features"][:, step],
                batch["mask"][:, step],
            )
            token = model.contact_builder(
                contact.latent,
                batch["positions"][:, step],
                batch["normals"][:, step],
            )
            probability = torch.sigmoid(contact.valid_logit)
            step_contact = batch["contact_valid"][:, step] & sequence_mask[:, step]
            step_region = batch["contact_region"][:, step]
            region_valid = (
                step_contact & (step_region >= 0) & (step_region < number)
            )
            active_region = torch.where(
                region_valid,
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
            else:
                tokens = token[:, None]
                token_mask = step_contact[:, None]

            if region_valid.any():
                seen_regions |= (
                    torch.nn.functional.one_hot(
                        step_region.clamp_min(0),
                        num_classes=number,
                    ).bool()
                    & region_valid[:, None]
                )
            stored_known = stored_regions >= 0
            if stored_known.any():
                seen_regions |= (
                    torch.nn.functional.one_hot(
                        stored_regions.clamp_min(0),
                        num_classes=number,
                    ).bool()
                    & stored_known[:, :, None]
                ).any(dim=1)

            available = sequence_mask[:, step, None] & (
                step_contact[:, None] | seen_regions
            )
            if not available.any():
                continue
            spatial = model.spatial_memory(tokens, token_mask)
            field = model.field_decoder(
                spatial.spatial_latents,
                batch["query_xyz"][:, step],
            )
            residual = (
                field.temp_mean_c[:, :, 0]
                - batch["query_temperature"][:, step, :, 0]
            )
            valid = (
                available
                & batch["query_mask"][:, step]
                & torch.isfinite(residual)
            )
            query_index = torch.arange(number, device=parameter.device)[None]
            current = region_valid[:, None] & (
                query_index == step_region[:, None]
            )
            roles = {
                "current": valid & current,
                "seen": valid & ~current & seen_regions,
                "unseen": valid & ~seen_regions,
            }
            for name, mask in roles.items():
                if not mask.any():
                    continue
                values = residual[mask]
                counts[name] += int(values.numel())
                absolute[name] += float(values.abs().sum())
                squared[name] += float(values.square().sum())

    result = {}
    for name in REGION_ROLES:
        count = counts[name]
        result[name] = {
            "count": count,
            "mae": absolute[name] / max(count, 1),
            "rmse": float(np.sqrt(squared[name] / max(count, 1))),
        }
    return result


def format_region_metrics(metrics: dict) -> str:
    return "  ".join(
        f"{name}:n={metrics[name]['count']},mae={metrics[name]['mae']:.4f},"
        f"rmse={metrics[name]['rmse']:.4f}"
        for name in REGION_ROLES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="评测空间热觉表示")
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="每个 batch 的 episode 数")
    parser.add_argument(
        "--memory-mode",
        choices=("full", "current"),
        default=None,
        help="默认使用训练时的模式；可覆盖为 full 或 current",
    )
    parser.add_argument(
        "--field-loss",
        choices=("mse", "huber", "nll"),
        default=None,
        help="默认使用训练时的 field loss",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    stats = payload["stats"]
    device = resolve_device(args.device or str(cfg["training"]["device"]))
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model"])
    root = Path(args.data_root) if args.data_root else REPO_ROOT / cfg["data"]["root"]
    weights = {name: float(value) for name, value in cfg["training"]["weights"].items()}
    memory_bank = make_memory_bank(int(cfg["model"]["hidden_dim"]))
    memory_mode = args.memory_mode or str(payload.get("memory_mode", "full"))
    field_loss = (
        args.field_loss
        or payload.get("field_loss")
        or cfg["training"].get("field_loss", "mse")
    )
    huber_delta = float(
        payload.get("huber_delta", cfg["training"].get("huber_delta", 2.0))
    )

    print(
        f"checkpoint: {args.checkpoint}  epoch={payload.get('epoch')}"
        f"  memory_mode={memory_mode}  field_loss={field_loss}"
    )
    for split in args.splits:
        name = cfg["data"].get(split, split)
        path = root / f"{name}.hdf5"
        if not path.is_file():
            print(f"{split:>20}: 没有文件（{path}）")
            continue
        dataset = EpisodeDataset(path, stats=stats)
        if args.limit:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_episodes,
        )
        metrics = evaluate(
            model,
            loader,
            device,
            weights,
            memory_bank,
            memory_mode=memory_mode,
            field_loss=str(field_loss),
            huber_delta=huber_delta,
        )
        print(f"{split:>20} episode={len(dataset):>4}  " + format_metrics("", metrics).strip())
        region_metrics = evaluate_regions(
            model,
            loader,
            device,
            memory_mode,
            memory_bank,
        )
        print(f"{split:>20} regions  " + format_region_metrics(region_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
