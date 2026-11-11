"""π0.5 + Thermal 集成测试（dummy variant，CPU）。

覆盖：
1. thermal_enabled=False：state_dict 没有 thermal_* 参数，prefix 长度不变；
2. thermal_enabled=True：prefix 追加 M+1 个 token，compute_loss_dict/backward 正常；
3. sample_actions 与训练使用同一 prefix 构造，并返回诊断字段；
4. 权重加载只容忍 thermal_encoder.* 缺失。
"""

from __future__ import annotations

import pytest
import torch

from models.pi05.observation import IMAGE_KEYS, Observation
from models.pi05 import pi0_config
from models.pi05 import pi0_pytorch
from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.outputs import ThermalSupervision
from training.pi05.thermal.step import thermal_train_step


def _make_config(*, thermal: ThermalPi0Config | None) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_dim=7,
        action_horizon=4,
        discrete_state_input=False,
        pytorch_compile_mode=None,
        thermal=thermal,
    )


def _small_thermal(**overrides) -> ThermalPi0Config:
    base = {
        "enabled": True,
        "history_size": 2,
        "window_size": 6,
        "signal_dim": 14,
        "num_surface_points": 8,
        "hidden_dim": 32,
        "thermal_latent_dim": 16,
        "num_spatial_tokens": 4,
    }
    base.update(overrides)
    return ThermalPi0Config(**base)


def _observation(cfg: pi0_config.Pi0Config, batch: int = 2) -> Observation:
    images = {key: torch.rand(batch, 3, 32, 32) * 2 - 1 for key in IMAGE_KEYS}
    image_masks = {key: torch.ones(batch, dtype=torch.bool) for key in IMAGE_KEYS}
    fields = {
        "images": images,
        "image_masks": image_masks,
        "state": torch.randn(batch, cfg.action_dim),
        "tokenized_prompt": torch.zeros(batch, 16, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(batch, 16, dtype=torch.bool),
    }
    if cfg.thermal is not None and cfg.thermal.enabled:
        t = cfg.thermal
        k, length, c, n = t.history_size, t.window_size, t.signal_dim, t.num_surface_points
        fields.update(
            thermal_history=torch.randn(batch, k, length, c),
            thermal_history_time_mask=torch.ones(batch, k, length, dtype=torch.bool),
            contact_mask=torch.ones(batch, k, dtype=torch.bool),
            contact_xyz_obj=torch.randn(batch, k, 3),
            contact_normal_obj=torch.nn.functional.normalize(torch.randn(batch, k, 3), dim=-1),
            contact_age_s=torch.rand(batch, k, 1),
            surface_xyz_obj=torch.randn(batch, n, 3),
            surface_normal_obj=torch.nn.functional.normalize(torch.randn(batch, n, 3), dim=-1),
            surface_mask=torch.ones(batch, n, dtype=torch.bool),
            surface_reachable=torch.ones(batch, n, dtype=torch.bool),
            surface_grasp_score=torch.rand(batch, n, 1),
            ee_xyz_obj=torch.randn(batch, 3),
            probe_budget_remaining=torch.ones(batch, 1),
        )
    return Observation(**fields)


def _num_image_tokens() -> int:
    # dummy 变体的视觉塔仍是 224/14 patch，每张图 256 个 token。
    return (224 // 14) ** 2


def test_thermal_disabled_keeps_original_prefix_and_parameters():
    torch.manual_seed(0)
    cfg = _make_config(thermal=None)
    model = pi0_pytorch.PI0Pytorch(cfg).eval()
    assert not any(name.startswith("thermal_encoder.") for name in model.state_dict())

    observation = _observation(cfg)
    processed = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    embs, pad_masks, att_masks, thermal_output = model.embed_prefix(
        list(processed.images.values()),
        list(processed.image_masks.values()),
        processed.tokenized_prompt,
        processed.tokenized_prompt_mask,
        observation=processed,
    )
    assert thermal_output is None
    expected_len = len(IMAGE_KEYS) * _num_image_tokens() + 16
    assert embs.shape[1] == expected_len
    assert pad_masks.shape[1] == expected_len
    assert att_masks.shape[1] == expected_len

    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    loss = model(
        observation,
        actions,
        noise=torch.randn_like(actions),
        time=torch.rand(2),
    )
    assert loss.shape == (2, cfg.action_horizon, cfg.action_dim)


def test_thermal_enabled_prefix_losses_and_gradients():
    torch.manual_seed(0)
    thermal = _small_thermal()
    cfg = _make_config(thermal=thermal)
    model = pi0_pytorch.PI0Pytorch(cfg)
    assert any(name.startswith("thermal_encoder.") for name in model.state_dict())

    observation = _observation(cfg)
    processed = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    embs, pad_masks, _, thermal_output = model.embed_prefix(
        list(processed.images.values()),
        list(processed.image_masks.values()),
        processed.tokenized_prompt,
        processed.tokenized_prompt_mask,
        observation=processed,
    )
    expected_len = len(IMAGE_KEYS) * _num_image_tokens() + 16 + thermal.num_spatial_tokens + 1
    assert embs.shape[1] == expected_len
    assert pad_masks.shape[1] == expected_len
    assert thermal_output is not None
    assert thermal_output.prefix_tokens.shape[1] == thermal.num_spatial_tokens + 1

    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    supervision = ThermalSupervision(
        contact_temp_c=torch.randn(2, thermal.history_size, 1),
        contact_valid=torch.ones(2, thermal.history_size, 1),
        surface_temp_c=torch.randn(2, thermal.num_surface_points, 1),
        surface_temp_mask=torch.ones(2, thermal.num_surface_points, dtype=torch.bool),
        surface_safe=torch.randint(0, 2, (2, thermal.num_surface_points, 1)).float(),
        strategy_type=torch.tensor([1, 2]),
        target_surface_index=torch.tensor([0, 3]),
    )
    losses = model.compute_loss_dict(
        observation,
        actions,
        supervision,
        noise=torch.randn_like(actions),
        time=torch.rand(2),
    )
    for name in ("flow", "contact_temp", "contact_valid", "field_temp", "field_safe", "skill", "target", "aux_total", "total"):
        assert name in losses
        assert torch.isfinite(losses[name]).all()

    losses["total"].backward()
    assert model.thermal_encoder.type_embedding.grad is not None
    assert model.thermal_encoder.contact_encoder.project[0].weight.grad is not None
    assert model.action_in_proj.weight.grad is not None


def test_forward_with_supervision_matches_compute_loss_dict():
    torch.manual_seed(0)
    thermal = _small_thermal()
    cfg = _make_config(thermal=thermal)
    model = pi0_pytorch.PI0Pytorch(cfg)
    observation = _observation(cfg)
    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    noise = torch.randn_like(actions)
    time = torch.rand(2)
    supervision = ThermalSupervision(
        contact_temp_c=torch.randn(2, thermal.history_size, 1),
        contact_valid=torch.ones(2, thermal.history_size, 1),
        surface_temp_c=torch.randn(2, thermal.num_surface_points, 1),
        surface_temp_mask=torch.ones(2, thermal.num_surface_points, dtype=torch.bool),
        surface_safe=torch.zeros(2, thermal.num_surface_points, 1),
        strategy_type=torch.zeros(2, dtype=torch.long),
        target_surface_index=torch.full((2,), -1, dtype=torch.long),
    )
    torch.manual_seed(123)
    loss_tensor = model(observation, actions, noise=noise, time=time, supervision=supervision)
    torch.manual_seed(123)
    losses = model.compute_loss_dict(observation, actions, supervision, noise=noise, time=time)
    torch.testing.assert_close(loss_tensor.mean(), losses["total"])


def test_thermal_train_step_returns_grad_total_and_detached_metrics():
    torch.manual_seed(0)
    thermal = _small_thermal()
    cfg = _make_config(thermal=thermal)
    model = pi0_pytorch.PI0Pytorch(cfg)
    observation = _observation(cfg)
    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    supervision = ThermalSupervision(
        contact_temp_c=torch.randn(2, thermal.history_size, 1),
        contact_valid=torch.ones(2, thermal.history_size, 1),
        surface_temp_c=torch.randn(2, thermal.num_surface_points, 1),
        surface_temp_mask=torch.ones(2, thermal.num_surface_points, dtype=torch.bool),
        surface_safe=torch.zeros(2, thermal.num_surface_points, 1),
        strategy_type=torch.tensor([1, 2]),
        target_surface_index=torch.tensor([0, 3]),
    )

    total, metrics = thermal_train_step(
        model,
        observation,
        actions,
        supervision,
        noise=torch.randn_like(actions),
        time=torch.rand(2),
    )
    assert total.requires_grad
    assert {"flow", "aux_total", "total"} <= set(metrics)
    assert all(not value.requires_grad for value in metrics.values())
    total.backward()
    assert model.thermal_encoder.type_embedding.grad is not None


def test_sample_actions_returns_diagnostics():
    torch.manual_seed(0)
    thermal = _small_thermal()
    cfg = _make_config(thermal=thermal)
    model = pi0_pytorch.PI0Pytorch(cfg).eval()
    observation = _observation(cfg)
    noise = torch.randn(2, cfg.action_horizon, cfg.action_dim)

    actions, diagnostics = model.sample_actions(
        "cpu", observation, noise=noise, num_steps=2, return_diagnostics=True
    )
    assert actions.shape == (2, cfg.action_horizon, cfg.action_dim)
    assert set(diagnostics) == {
        "skill_probability",
        "selected_surface_index",
        "field_temp_mean_c",
        "field_temp_std_c",
    }
    assert diagnostics["skill_probability"].shape == (2, 3)
    assert diagnostics["selected_surface_index"].shape == (2,)
    assert diagnostics["field_temp_mean_c"].shape == (2, thermal.num_surface_points, 1)
    assert torch.isfinite(diagnostics["field_temp_std_c"]).all()


def test_weight_loading_allows_only_thermal_missing():
    torch.manual_seed(0)
    base_cfg = _make_config(thermal=None)
    base_model = pi0_pytorch.PI0Pytorch(base_cfg)

    torch.manual_seed(0)
    thermal_cfg = _make_config(thermal=_small_thermal())
    thermal_model = pi0_pytorch.PI0Pytorch(thermal_cfg)

    # 原 π0.5 权重 -> 热觉模型：只缺 thermal_encoder.*，允许
    pi0_pytorch.load_state_dict_compat(thermal_model, base_model.state_dict())

    # 热觉权重 -> 非热觉模型：多出 thermal_encoder.*，必须报错
    with pytest.raises(RuntimeError):
        pi0_pytorch.load_state_dict_compat(base_model, thermal_model.state_dict())
