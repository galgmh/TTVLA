"""Smoke test for `data.data_loader.create_data_loader`.
"""
import pathlib
import numpy as np
import torch
from PIL import Image

from data.data_loader import (
    DataConfig,
    PaligemmaTokenizer,
    RepackTransform,
    create_data_loader,
)
from pi05_model.config import Pi0Config
from pi05_model.observation import Observation


TestDataPath = "/media/sunhao/T71/svla_so101_pickplace"
TokenizerPath = "/media/sunhao/T71/Pi0.5/paligemma_tokenizer/paligemma_tokenizer.model"
NormStatsPath = "/media/sunhao/T71/Assets/svla_so101_pickplace_norm_stats+action_horizon_10"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Configs
    # ------------------------------------------------------------------    
    batch_size = 4
    num_workers = 4

    # pi05 defaults: action_horizon=10, max_token_len=200, action_dim=32
    model_config = Pi0Config(pi05=True)

    # The dataset feature is `action` (singular) and `observation.state`.
    # Map them into the canonical {images, state, actions, task_index} schema.
    data_config = DataConfig(
        repo_id="lerobot/svla_so101_pickplace+my_own_label",  # any string; local_root overrides Hub lookup
        local_root=TestDataPath,
        asset_id="svla_so101_pickplace_norm_stats+action_horizon_10",  # used for norm_stats asset directory
        
        norm_stats=None,
        norm_stats_dir=NormStatsPath,
        use_quantile_norm = True,
        
        repack_transforms=(
            RepackTransform({
                "images": {
                    "base_0_rgb":       "observation.images.up",
                    "left_wrist_0_rgb": "observation.images.side",
                },
                "state":      "observation.state",
                "actions":    "action",
                "task_index": "task_index",
            }),
        ),
        data_transforms=(),  # no data transforms for this smoke test; just the repack
        
        action_sequence_keys=("action",),         # LeRobot uses "action" here, not "actions"
        prompt_from_task=True,                    # resolve task_index via meta.tasks
        default_prompt=None,
        
        tokenizer_path=TokenizerPath,
    )

    # ------------------------------------------------------------------
    # 2. Build loader
    # ------------------------------------------------------------------
    loader = create_data_loader(
        data_config=data_config,
        model_config=model_config,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=0,
        shuffle=True,
        skip_norm_stats=False,  # no norm_stats computed for this smoke test
    )

    print(f"\nlen(loader) = {len(loader)} batches/epoch (batch_size={batch_size})")
    assert len(loader) > 0, "loader produced zero batches"

    # ------------------------------------------------------------------
    # 3. Pull one batch and validate
    # ------------------------------------------------------------------
    obs, actions = next(iter(loader))

    assert isinstance(obs, Observation), f"expected Observation, got {type(obs)}"
    assert isinstance(actions, torch.Tensor)

    # --- actions: (B, action_horizon, action_dim) ---------------------
    raw_action_dim = 6  # so100 follower
    assert actions.shape == (batch_size, model_config.action_horizon, model_config.action_dim), (
        f"actions shape mismatch: {tuple(actions.shape)} "
        f"expected ({batch_size}, {model_config.action_horizon}, {model_config.action_dim})"
    )
    assert actions.dtype == torch.float32
    assert torch.all(actions[..., raw_action_dim:] == 0), "padded action region must be zero"

    # --- state: (B, action_dim) ---------------------------------------
    assert obs.state.shape == (batch_size, model_config.action_dim), (
        f"state shape mismatch: {tuple(obs.state.shape)}"
    )
    assert obs.state.dtype == torch.float32
    assert torch.all(obs.state[..., raw_action_dim:] == 0), "padded state region must be zero"

    # --- tokens / mask: (B, max_token_len) ----------------------------
    assert obs.tokenized_prompt.shape == (batch_size, model_config.max_token_len)
    assert obs.tokenized_prompt.dtype == torch.long
    assert obs.tokenized_prompt_mask.shape == (batch_size, model_config.max_token_len)
    assert obs.tokenized_prompt_mask.dtype == torch.bool
    # Every sample must have at least one real token (mask not all-False).
    assert bool(obs.tokenized_prompt_mask.any(dim=-1).all()), "found a sample with empty token mask"

    # --- images -------------------------------------------------------
    expected_image_keys = {"base_0_rgb", "left_wrist_0_rgb"}
    assert set(obs.images) == expected_image_keys, f"image keys: {set(obs.images)}"
    for k, img in obs.images.items():
        assert img.ndim == 4, f"{k}: expected 4D (B,H,W,C) or (B,C,H,W), got {tuple(img.shape)}"
        assert img.shape[0] == batch_size, f"{k}: batch dim mismatch ({tuple(img.shape)})"

    # --- image_masks: auto-filled when absent in the batch -----------
    assert set(obs.image_masks) == expected_image_keys
    for k, m in obs.image_masks.items():
        assert m.shape == (batch_size,)
        assert m.dtype == torch.bool
        assert bool(m.all()), f"{k}: auto-filled image_mask should be all True"
    

    # ------------------------------------------------------------------
    # 4. Summary + decode the first prompt for a visual sanity check
    # ------------------------------------------------------------------
    print("\nFirst batch summary:")
    print(f"  actions:                {tuple(actions.shape)} {actions.dtype}")
    print(f"  state:                  {tuple(obs.state.shape)} {obs.state.dtype}")
    print(f"  tokenized_prompt:       {tuple(obs.tokenized_prompt.shape)} {obs.tokenized_prompt.dtype}")
    print(f"  tokenized_prompt_mask:  {tuple(obs.tokenized_prompt_mask.shape)} {obs.tokenized_prompt_mask.dtype}")
    for k, img in obs.images.items():
        print(f"  images[{k!s}]: {tuple(img.shape)} {img.dtype}  "
              f"min={img.min().item():.3f} max={img.max().item():.3f}")

    tok = PaligemmaTokenizer(TokenizerPath, max_len=model_config.max_token_len)
    for i in range(batch_size):
        n = int(obs.tokenized_prompt_mask[i].sum().item())
        decoded = tok._sp.decode([int(x) for x in obs.tokenized_prompt[i, :n].tolist()])
        print(f"\nDecoded prompt[{i}] (valid_len={n}/{model_config.max_token_len}):")
        print(f"  {decoded!r}")
        print(f"  state[{i}, :{raw_action_dim}] = "
              f"{np.array2string(obs.state[i, :raw_action_dim].numpy(), precision=3)}")
        print(f"  actions[{i}, 0, :{raw_action_dim}] = "
              f"{np.array2string(actions[i, 0, :raw_action_dim].numpy(), precision=3)}")

    # ------------------------------------------------------------------
    # 4b. Save the batch's images as one merged grid for visual inspection.
    # Rows = batch samples, columns = camera views.
    # ------------------------------------------------------------------
    def to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
        # Accepts (H,W,C) or (C,H,W) — detect channel dim by size <= 4.
        arr = img.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
        if arr.dtype == np.uint8:
            return arr
        # Float images live in [-1, 1] (see preprocessing.resize_with_pad_torch).
        arr = (arr + 1.0) * 0.5  # [-1,1] -> [0,1]
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    camera_keys = sorted(obs.images.keys())
    rows = []
    for i in range(batch_size):
        row_imgs = [to_uint8_hwc(obs.images[k][i]) for k in camera_keys]
        # Pad to same height in case different cameras have different resolutions.
        max_h = max(im.shape[0] for im in row_imgs)
        padded = []
        for im in row_imgs:
            if im.shape[0] < max_h:
                pad = np.zeros((max_h - im.shape[0], im.shape[1], im.shape[2]), dtype=np.uint8)
                im = np.concatenate([im, pad], axis=0)
            padded.append(im)
        rows.append(np.concatenate(padded, axis=1))

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], r.shape[2]), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded_rows.append(r)
    grid = np.concatenate(padded_rows, axis=0)

    out_path = pathlib.Path(__file__).parent / "TestData_batch_preview.png"
    Image.fromarray(grid).save(out_path)
    print(f"\nSaved merged image preview ({grid.shape[1]}x{grid.shape[0]}, "
          f"rows=batch_size={batch_size}, cols={camera_keys}) -> {out_path}")

    # ------------------------------------------------------------------
    # 5. Pull a second batch to confirm the iterator is reusable.
    # ------------------------------------------------------------------
    it = iter(loader)
    obs_a, actions_a = next(it)
    obs_b, actions_b = next(it)
    assert obs_a.state.shape == obs_b.state.shape
    assert actions_a.shape == actions_b.shape
    print("\nSecond / third batches OK — iterator yields consistent shapes.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
