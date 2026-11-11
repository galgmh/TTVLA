"""分阶段冻结规则与训练 wrapper 测试（dummy variant，CPU）。"""

from __future__ import annotations

import io

import pytest
import torch

from models.pi05.observation import IMAGE_KEYS, Observation
from models.pi05 import pi0_config
from models.pi05 import pi0_pytorch
from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.outputs import ThermalSupervision
from training.pi05.thermal.finetune import configure_stage
from training.pi05.thermal.finetune import parameter_groups
from training.pi05.thermal.finetune import stage_loss_keys
from training.pi05.thermal.finetune import trainable_parameter_names
from training.pi05.thermal.step import thermal_train_step


def _small_thermal() -> ThermalPi0Config:
    return ThermalPi0Config(
        enabled=True,
        history_size=2,
        window_size=6,
        signal_dim=14,
        num_surface_points=8,
        hidden_dim=32,
        thermal_latent_dim=16,
        num_spatial_tokens=4,
    )


def _make_model() -> tuple[pi0_config.Pi0Config, pi0_pytorch.PI0Pytorch]:
    torch.manual_seed(0)
    cfg = pi0_config.Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_dim=7,
        action_horizon=4,
        discrete_state_input=False,
        pytorch_compile_mode=None,
        thermal=_small_thermal(),
    )
    torch.manual_seed(0)
    return cfg, pi0_pytorch.PI0Pytorch(cfg)


def _observation(cfg: pi0_config.Pi0Config, batch: int = 2) -> Observation:
    thermal = cfg.thermal
    assert thermal is not None
    k, length, c, n = thermal.history_size, thermal.window_size, thermal.signal_dim, thermal.num_surface_points
    fields = {
        "images": {key: torch.rand(batch, 3, 32, 32) * 2 - 1 for key in IMAGE_KEYS},
        "image_masks": {key: torch.ones(batch, dtype=torch.bool) for key in IMAGE_KEYS},
        "state": torch.randn(batch, cfg.action_dim),
        "tokenized_prompt": torch.zeros(batch, 16, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(batch, 16, dtype=torch.bool),
        "thermal_history": torch.randn(batch, k, length, c),
        "thermal_history_time_mask": torch.ones(batch, k, length, dtype=torch.bool),
        "contact_mask": torch.ones(batch, k, dtype=torch.bool),
        "contact_xyz_obj": torch.randn(batch, k, 3),
        "contact_normal_obj": torch.nn.functional.normalize(torch.randn(batch, k, 3), dim=-1),
        "contact_age_s": torch.rand(batch, k, 1),
        "surface_xyz_obj": torch.randn(batch, n, 3),
        "surface_normal_obj": torch.nn.functional.normalize(torch.randn(batch, n, 3), dim=-1),
        "surface_mask": torch.ones(batch, n, dtype=torch.bool),
        "surface_reachable": torch.ones(batch, n, dtype=torch.bool),
        "surface_grasp_score": torch.rand(batch, n, 1),
        "ee_xyz_obj": torch.randn(batch, 3),
        "probe_budget_remaining": torch.ones(batch, 1),
    }
    return Observation(**fields)


def _supervision(cfg: pi0_config.Pi0Config, batch: int = 2) -> ThermalSupervision:
    thermal = cfg.thermal
    assert thermal is not None
    return ThermalSupervision(
        contact_temp_c=torch.randn(batch, thermal.history_size, 1),
        contact_valid=torch.ones(batch, thermal.history_size, 1),
        surface_temp_c=torch.randn(batch, thermal.num_surface_points, 1),
        surface_temp_mask=torch.ones(batch, thermal.num_surface_points, dtype=torch.bool),
        surface_safe=torch.randint(0, 2, (batch, thermal.num_surface_points, 1)).float(),
        strategy_type=torch.tensor([1, 2]),
        target_surface_index=torch.tensor([0, 3]),
    )


def test_stage1_only_trains_contact_and_spatial_modules():
    _, model = _make_model()
    counts = configure_stage(model, 1)
    names = trainable_parameter_names(model)

    assert counts["thermal"] > 0
    assert counts["action_expert"] == 0
    assert counts["paligemma"] == 0
    assert all(name.startswith("thermal_encoder.") for name in names)
    for prefix in (
        "thermal_encoder.contact_encoder.",
        "thermal_encoder.point_encoder.",
        "thermal_encoder.spatial_memory.",
    ):
        assert any(name.startswith(prefix) for name in names)
    # 策略与 prefix 投影在阶段 1 必须冻结。
    assert not any("thermal_encoder.strategy." in name for name in names)
    assert not any("thermal_encoder.spatial_proj." in name for name in names)
    assert not any("thermal_encoder.goal_proj." in name for name in names)
    assert not any("thermal_encoder.type_embedding" in name for name in names)


def test_stage2_only_trains_strategy():
    _, model = _make_model()
    counts = configure_stage(model, 2)
    names = trainable_parameter_names(model)

    assert counts["thermal"] > 0
    assert counts["action_expert"] == 0
    assert names
    assert all(name.startswith("thermal_encoder.strategy.") for name in names)


def test_stage3_trains_thermal_and_action_expert_but_not_vlm():
    _, model = _make_model()
    counts = configure_stage(model, 3)
    names = trainable_parameter_names(model)

    assert counts["thermal"] > 0
    assert counts["action_expert"] > 0
    assert counts["paligemma"] == 0
    assert any(name.startswith("paligemma_with_expert.gemma_expert.") for name in names)
    assert any(name.startswith("action_in_proj.") for name in names)
    assert not any(name.startswith("paligemma_with_expert.paligemma.") for name in names)


def test_stage4_unfreezes_last_paligemma_layers_only():
    _, model = _make_model()
    configure_stage(model, 4, paligemma_last_layers=2)
    names = trainable_parameter_names(model)
    layer_names = [name for name in names if ".language_model.layers." in name]

    assert any(".language_model.layers.2." in name for name in layer_names)
    assert any(".language_model.layers.3." in name for name in layer_names)
    assert not any(".language_model.layers.0." in name for name in layer_names)
    assert not any(".language_model.layers.1." in name for name in layer_names)
    assert not any("vision_tower" in name for name in names)
    assert not any("multi_modal_projector" in name for name in names)


def test_parameter_groups_follow_stage_learning_rates():
    _, model = _make_model()

    configure_stage(model, 1)
    groups = parameter_groups(model, 1)
    assert [group["name"] for group in groups] == ["thermal"]
    assert groups[0]["lr"] == pytest.approx(1e-4)

    configure_stage(model, 3)
    groups = {group["name"]: group for group in parameter_groups(model, 3)}
    assert set(groups) == {"thermal", "action_expert"}
    assert groups["thermal"]["lr"] == pytest.approx(1e-4)
    assert groups["action_expert"]["lr"] == pytest.approx(5e-5)

    configure_stage(model, 4, paligemma_last_layers=2)
    groups = {group["name"]: group for group in parameter_groups(model, 4)}
    assert set(groups) == {"thermal", "action_expert", "paligemma"}
    assert groups["paligemma"]["lr"] == pytest.approx(1e-5)

    # 三组参数互不重叠，且覆盖全部可训练参数。
    total_grouped = sum(len(group["params"]) for group in groups.values())
    assert total_grouped == sum(1 for parameter in model.parameters() if parameter.requires_grad)


def test_stage_loss_keys():
    assert "flow" not in stage_loss_keys(1)
    assert set(stage_loss_keys(1)) == {"contact_temp", "contact_valid", "field_temp", "field_safe"}
    assert set(stage_loss_keys(2)) == {"skill", "target"}
    assert "flow" in stage_loss_keys(3)
    assert "flow" in stage_loss_keys(4)
    with pytest.raises(ValueError, match="stage"):
        stage_loss_keys(5)


def test_stage1_optimizer_step_and_checkpoint_reload():
    cfg, model = _make_model()
    configure_stage(model, 1)
    optimizer = torch.optim.AdamW(parameter_groups(model, 1))

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    frozen_before = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.detach().clone()

    observation = _observation(cfg)
    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    supervision = _supervision(cfg)
    total, metrics = thermal_train_step(
        model,
        observation,
        actions,
        supervision,
        noise=torch.randn_like(actions),
        time=torch.rand(2),
        loss_keys=stage_loss_keys(1),
    )
    assert total.requires_grad
    assert "flow" in metrics
    total.backward()
    optimizer.step()

    changed = [name for name, parameter in model.named_parameters() if name in before and not torch.equal(before[name], parameter)]
    assert changed, "阶段 1 优化器步后应至少有一个热觉参数更新"
    assert torch.equal(
        frozen_before,
        model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight,
    ), "阶段 1 不允许改动 Action Expert"

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    _, reloaded = _make_model()
    pi0_pytorch.load_state_dict_compat(reloaded, torch.load(buffer, weights_only=True))
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key]), f"checkpoint 重载不一致: {key}"
