"""End-to-end smoke test for the Pi0.5 PyTorch model.

Covers:
  1. Model construction (+ optional safetensors checkpoint load).
  2. Training forward pass / flow-matching loss.
  3. Inference via `sample_actions` (flow-matching denoising).
"""

import pathlib
import time
from pi05_model._patches import install as install_patches
install_patches()
import torch

from pi05_model.config import Pi0Config
from pi05_model.model import PI0Pytorch
from pi05_model.observation import Observation
from pi05_model.preprocessing import IMAGE_KEYS, IMAGE_RESOLUTION

CHECKPOINT_PATH = "/media/sunhao/T71/Pi0.5/pi05_base_torch/model.safetensors"


def make_dummy_observation_and_actions(config: Pi0Config, batch_size: int, device: torch.device) -> tuple[Observation, torch.Tensor]:
    """Random-tensor batch shaped exactly like real training inputs."""
    h, w = IMAGE_RESOLUTION
    images = {k: torch.empty(batch_size, 3, h, w, device=device).uniform_(-1.0, 1.0) for k in IMAGE_KEYS}
    image_masks = {k: torch.ones(batch_size, dtype=torch.bool, device=device) for k in IMAGE_KEYS}
    state = torch.randn(batch_size, config.action_dim, device=device)
    tokenized_prompt = torch.randint(0, 257_152, (batch_size, 15), dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones(batch_size, 15, dtype=torch.bool, device=device)
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim, device=device)

    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    return obs, actions


def test_load(config: Pi0Config, device: torch.device) -> PI0Pytorch:
    print("=" * 60)
    print("[1/3] Model load")
    print("=" * 60)
    t0 = time.time()
    model = PI0Pytorch(config).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  built PI0Pytorch in {time.time() - t0:.1f}s — {n_total / 1e6:.1f} M params, pi05={config.pi05}")

    ckpt_path = pathlib.Path(CHECKPOINT_PATH)
    if ckpt_path.exists():
        import safetensors.torch

        t0 = time.time()
        sd = safetensors.torch.load_file(str(ckpt_path))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded {ckpt_path} in {time.time() - t0:.1f}s")
        print(f"    missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"    missing[:3]={missing[:3]}")
        if unexpected[:3]:
            print(f"    unexpected[:3]={unexpected[:3]}")
    else:
        print(f"  checkpoint not found at {ckpt_path} — using random init")
        raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")

    return model


def test_loss(model: PI0Pytorch, config: Pi0Config, device: torch.device) -> None:
    print("=" * 60)
    print("[2/3] Loss computation (training forward)")
    print("=" * 60)
    model.train()
    obs, actions = make_dummy_observation_and_actions(config, batch_size=4, device=device)

    t0 = time.time()
    per_elem_loss = model(obs, actions)
    loss = per_elem_loss.mean()
    print(
        f"  forward: {time.time() - t0:.2f}s | per-elem shape={tuple(per_elem_loss.shape)} | "
        f"loss={loss.item():.4f}"
    )

    # Sanity-check gradient flow.
    t0 = time.time()
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  backward: {time.time() - t0:.2f}s | params with grad: {n_with_grad}/{n_trainable}")

    model.zero_grad(set_to_none=True)


@torch.no_grad()
def test_inference(model: PI0Pytorch, config: Pi0Config, device: torch.device) -> None:
    print("=" * 60)
    print("[3/3] Inference (sample_actions)")
    print("=" * 60)
    model.eval()
    obs, _ = make_dummy_observation_and_actions(config, batch_size=2, device=device)

    t0 = time.time()
    actions = model.sample_actions(device, obs, num_steps=10)
    print(
        f"  sample_actions: {time.time() - t0:.2f}s | "
        f"shape={tuple(actions.shape)} | dtype={actions.dtype} | "
        f"mean={actions.mean().item():+.4f} std={actions.std().item():.4f}"
    )
    expected = (obs.state.shape[0], config.action_horizon, config.action_dim)
    assert tuple(actions.shape) == expected, f"expected {expected}, got {tuple(actions.shape)}"
    assert torch.isfinite(actions).all(), "non-finite values in sampled actions"
    print("  ✓ shape + finiteness checks passed")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    config = Pi0Config(pi05=False, dtype="bfloat16" if device.type == "cuda" else "float32")
    print(f"config: {config}")

    model = test_load(config, device)
    test_loss(model, config, device)
    test_inference(model, config, device)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
