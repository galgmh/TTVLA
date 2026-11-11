"""热觉模块单元测试（CPU、tiny 配置、不依赖 π0.5 预训练权重）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.contact_encoder import ThermalContactEncoder
from models.pi05.thermal.integrated_encoder import IntegratedThermalEncoder
from models.pi05.thermal.losses import compute_thermal_losses
from models.pi05.thermal.outputs import IntegratedThermalOutput
from models.pi05.thermal.outputs import ThermalSupervision
from models.pi05.thermal.point_encoder import SurfacePointEncoder
from models.pi05.thermal.spatial_memory import ContactTokenBuilder
from models.pi05.thermal.spatial_memory import ThermalSpatialMemory
from models.pi05.thermal.strategy import ActiveStrategyModule


def _cfg(**overrides) -> ThermalPi0Config:
    base = {
        "enabled": True,
        "history_size": 4,
        "window_size": 8,
        "signal_dim": 14,
        "num_surface_points": 16,
        "hidden_dim": 32,
        "thermal_latent_dim": 16,
        "num_spatial_tokens": 4,
        "min_grasp_score": 0.3,
    }
    base.update(overrides)
    return ThermalPi0Config(**base)


def _fake_observation(cfg: ThermalPi0Config, batch: int = 2) -> SimpleNamespace:
    b, k, length, c, n = batch, cfg.history_size, cfg.window_size, cfg.signal_dim, cfg.num_surface_points
    return SimpleNamespace(
        images={"base_0_rgb": torch.zeros(b, 3, 32, 32)},
        image_masks={"base_0_rgb": torch.ones(b, dtype=torch.bool)},
        state=torch.randn(b, 32),
        tokenized_prompt=torch.zeros(b, 8, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(b, 8, dtype=torch.bool),
        thermal_history=torch.randn(b, k, length, c),
        thermal_history_time_mask=torch.ones(b, k, length, dtype=torch.bool),
        contact_mask=torch.ones(b, k, dtype=torch.bool),
        contact_xyz_obj=torch.randn(b, k, 3),
        contact_normal_obj=torch.nn.functional.normalize(torch.randn(b, k, 3), dim=-1),
        contact_age_s=torch.rand(b, k, 1),
        surface_xyz_obj=torch.randn(b, n, 3),
        surface_normal_obj=torch.nn.functional.normalize(torch.randn(b, n, 3), dim=-1),
        surface_mask=torch.ones(b, n, dtype=torch.bool),
        surface_reachable=torch.ones(b, n, dtype=torch.bool),
        surface_grasp_score=torch.rand(b, n, 1),
        ee_xyz_obj=torch.randn(b, 3),
        probe_budget_remaining=torch.ones(b, 1),
    )


def test_config_rejects_bad_selection_mode():
    with pytest.raises(ValueError, match="selection_mode"):
        ThermalPi0Config(selection_mode="hard")


def test_contact_encoder_shapes_and_empty_k():
    torch.manual_seed(0)
    encoder = ThermalContactEncoder(signal_dim=14, latent_dim=16).eval()
    history = torch.randn(2, 3, 8, 14)
    time_mask = torch.ones(2, 3, 8, dtype=torch.bool)
    contact_mask = torch.ones(2, 3, dtype=torch.bool)
    out = encoder(history, time_mask, contact_mask)
    assert out.latent.shape == (2, 3, 16)
    assert out.temp_mean_c.shape == (2, 3, 1)
    assert torch.isfinite(out.latent).all()

    empty = encoder(torch.zeros(2, 0, 8, 14), torch.zeros(2, 0, 8, dtype=torch.bool), torch.zeros(2, 0, dtype=torch.bool))
    assert empty.latent.shape == (2, 0, 16)
    assert torch.isfinite(empty.temp_mean_c).all()


def test_contact_encoder_ignores_masked_future_frames():
    torch.manual_seed(0)
    encoder = ThermalContactEncoder(signal_dim=14, latent_dim=16).eval()
    history_a = torch.randn(1, 1, 8, 14)
    history_b = history_a.clone()
    history_b[:, :, 6:] = torch.randn(1, 1, 2, 14)  # 只改“未来”帧
    time_mask = torch.zeros(1, 1, 8, dtype=torch.bool)
    time_mask[..., :6] = True
    contact_mask = torch.ones(1, 1, dtype=torch.bool)
    out_a = encoder(history_a, time_mask, contact_mask)
    out_b = encoder(history_b, time_mask, contact_mask)
    torch.testing.assert_close(out_a.latent, out_b.latent, atol=1e-6, rtol=1e-6)


def test_point_encoder_shape():
    encoder = SurfacePointEncoder(hidden_dim=32)
    out = encoder(
        torch.randn(2, 16, 3),
        torch.nn.functional.normalize(torch.randn(2, 16, 3), dim=-1),
        torch.ones(2, 16, dtype=torch.bool),
        torch.rand(2, 16, 1),
        torch.randn(2, 3),
    )
    assert out.shape == (2, 16, 32)


def test_contact_token_builder_shape():
    builder = ContactTokenBuilder(latent_dim=16, hidden_dim=32)
    tokens = builder(
        torch.randn(2, 3, 16),
        torch.randn(2, 3, 3),
        torch.randn(2, 3, 3),
        torch.randn(2, 3, 1),
        torch.zeros(2, 3, 1),
        torch.rand(2, 3, 1),
        torch.rand(2, 3, 1),
    )
    assert tokens.shape == (2, 3, 32)


def test_spatial_memory_permutation_invariance():
    torch.manual_seed(0)
    memory = ThermalSpatialMemory(hidden_dim=32, num_spatial_tokens=4).eval()
    contact_tokens = torch.randn(2, 5, 32)
    contact_mask = torch.ones(2, 5, dtype=torch.bool)
    surface = torch.randn(2, 16, 32)
    surface_mask = torch.ones(2, 16, dtype=torch.bool)
    language = torch.randn(2, 32)

    out_a = memory(contact_tokens, contact_mask, surface, surface_mask, language)
    perm = torch.tensor([4, 0, 3, 1, 2])
    out_b = memory(contact_tokens[:, perm], contact_mask[:, perm], surface, surface_mask, language)
    torch.testing.assert_close(out_a.surface_features, out_b.surface_features, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_a.temp_mean_c, out_b.temp_mean_c, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_a.spatial_latents, out_b.spatial_latents, atol=1e-5, rtol=1e-5)


def test_spatial_memory_handles_empty_contacts_and_masked_surface():
    torch.manual_seed(0)
    memory = ThermalSpatialMemory(hidden_dim=32, num_spatial_tokens=4).eval()
    language = torch.randn(2, 32)
    surface = torch.randn(2, 16, 32)
    surface_mask = torch.zeros(2, 16, dtype=torch.bool)

    # 所有 contact 无效（K 维仍存在）
    out = memory(torch.randn(2, 4, 32), torch.zeros(2, 4, dtype=torch.bool), surface, surface_mask, language)
    for tensor in (out.surface_features, out.temp_mean_c, out.temp_logvar, out.safe_logit, out.spatial_latents):
        assert torch.isfinite(tensor).all()

    # 字面 K=0
    out = memory(torch.zeros(2, 0, 32), torch.zeros(2, 0, dtype=torch.bool), surface, surface_mask, language)
    assert out.spatial_latents.shape == (2, 4, 32)
    assert torch.isfinite(out.spatial_latents).all()


def _strategy_inputs(batch: int = 1):
    surface = torch.randn(batch, 16, 32)
    surface_mask = torch.ones(batch, 16, dtype=torch.bool)
    reachable = torch.ones(batch, 16, dtype=torch.bool)
    grasp_score = torch.full((batch, 16, 1), 0.9)
    image = torch.randn(batch, 32)
    language = torch.randn(batch, 32)
    state = torch.randn(batch, 32)
    budget = torch.ones(batch, 1)
    return surface, surface_mask, reachable, grasp_score, image, language, state, budget


def test_strategy_masks_unreachable_low_grasp_and_budget():
    torch.manual_seed(0)
    strategy = ActiveStrategyModule(hidden_dim=32, min_grasp_score=0.3).eval()
    surface, surface_mask, reachable, grasp_score, image, language, state, budget = _strategy_inputs()
    reachable[0, 3] = False
    grasp_score[0, 5] = 0.1
    budget[0, 0] = 0.0

    out = strategy(surface, surface_mask, reachable, grasp_score, image, language, state, budget)
    neg = torch.finfo(out.probe_logits.dtype).min
    assert out.probe_logits[0, 3].item() == neg
    assert out.grasp_logits[0, 3].item() == neg
    assert out.grasp_logits[0, 5].item() == neg
    assert out.probe_logits[0, 5].item() != neg
    # budget=0 屏蔽 PROBE 技能
    assert out.skill_logits[0, 1].item() == neg

    # 评测模式（argmax）给出合法索引且分布为 one-hot 混合
    assert out.selected_index is not None
    assert int(out.selected_index[0]) in range(16)
    assert torch.isfinite(out.selected_distribution).all()


def test_strategy_goal_feature_gets_gradient():
    torch.manual_seed(0)
    strategy = ActiveStrategyModule(hidden_dim=32).train()
    surface = torch.randn(1, 16, 32, requires_grad=True)
    _, surface_mask, reachable, grasp_score, image, language, state, budget = _strategy_inputs()
    out = strategy(surface, surface_mask, reachable, grasp_score, image, language, state, budget)
    out.goal_feature.sum().backward()
    assert surface.grad is not None
    assert surface.grad.abs().sum().item() > 0


def _dummy_output(batch: int = 2, k: int = 3, n: int = 5) -> IntegratedThermalOutput:
    return IntegratedThermalOutput(
        prefix_tokens=torch.zeros(batch, 1, 8),
        prefix_mask=torch.ones(batch, 1, dtype=torch.bool),
        contact=SimpleNamespace(
            latent=torch.zeros(batch, k, 16),
            temp_mean_c=torch.randn(batch, k, 1),
            temp_logvar=torch.zeros(batch, k, 1),
            valid_logit=torch.randn(batch, k, 1),
        ),
        field_temp_mean_c=torch.randn(batch, n, 1),
        field_temp_logvar=torch.zeros(batch, n, 1),
        field_safe_logit=torch.randn(batch, n, 1),
        strategy=SimpleNamespace(
            skill_logits=torch.randn(batch, 3),
            probe_logits=torch.randn(batch, n),
            grasp_logits=torch.randn(batch, n),
            selected_distribution=torch.zeros(batch, n),
            goal_feature=torch.zeros(batch, 32),
        ),
    )


def test_thermal_losses_respect_masks():
    torch.manual_seed(0)
    output = _dummy_output()
    supervision = ThermalSupervision(
        contact_temp_c=torch.randn(2, 3, 1),
        contact_valid=torch.tensor([[[1.0], [0.0], [1.0]], [[0.0], [1.0], [1.0]]]),
        surface_temp_c=torch.randn(2, 5, 1),
        surface_temp_mask=torch.tensor([[1, 1, 0, 0, 1], [1, 0, 0, 0, 0]], dtype=torch.bool),
        surface_safe=torch.randint(0, 2, (2, 5, 1)).float(),
        strategy_type=torch.tensor([1, 0]),
        target_surface_index=torch.tensor([2, -1]),
    )
    weights = {
        "contact_temp": 0.2,
        "contact_valid": 0.1,
        "field_temp": 0.3,
        "field_safe": 0.1,
        "skill": 0.2,
        "target": 0.3,
    }
    terms = compute_thermal_losses(output, supervision, weights)
    for name in ("contact_temp", "contact_valid", "field_temp", "field_safe", "skill", "target", "aux_total"):
        assert name in terms
        assert terms[name].shape == (2,)
        assert torch.isfinite(terms[name]).all()

    expected = sum(weights[name] * terms[name] for name in weights)
    torch.testing.assert_close(terms["aux_total"], expected)


def test_integrated_encoder_prefix_and_gradients():
    torch.manual_seed(0)
    cfg = _cfg()
    encoder = IntegratedThermalEncoder(cfg, paligemma_width=64, state_dim=32)
    observation = _fake_observation(cfg)
    image_embeddings = [torch.randn(2, 64, 64)]
    language_embeddings = torch.randn(2, 8, 64)

    out = encoder(observation, image_embeddings, language_embeddings)
    assert out.prefix_tokens.shape == (2, cfg.num_spatial_tokens + 1, 64)
    assert out.prefix_mask.shape == (2, cfg.num_spatial_tokens + 1)
    assert out.prefix_mask.all()
    assert out.contact.latent.shape == (2, cfg.history_size, cfg.thermal_latent_dim)
    assert out.field_temp_mean_c.shape == (2, cfg.num_surface_points, 1)

    out.prefix_tokens.sum().backward()
    assert encoder.strategy.skill_embeddings.grad is not None
    assert encoder.contact_encoder.project[0].weight.grad is not None
    assert encoder.type_embedding.grad is not None
