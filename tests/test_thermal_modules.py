"""无候选点Thermal模块单元测试。"""

from types import SimpleNamespace

import pytest
import torch

from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.contact_encoder import ThermalContactEncoder
from models.pi05.thermal.field_decoder import SpatialFieldDecoder
from models.pi05.thermal.integrated_encoder import IntegratedThermalEncoder
from models.pi05.thermal.losses import compute_thermal_losses
from models.pi05.thermal.outputs import ThermalSupervision
from models.pi05.thermal.spatial_memory import ContactTokenBuilder, ThermalSpatialMemory
from models.pi05.thermal.strategy_aux import GlobalSkillAuxHead, SKILL_PROBE


def _cfg(**overrides):
    values = dict(enabled=True, memory_size=4, window_size=8, signal_dim=14,
                  hidden_dim=32, thermal_latent_dim=16, num_spatial_tokens=4,
                  prefix_adapter_hidden_dim=64)
    values.update(overrides)
    return ThermalPi0Config(**values)


def _observation(cfg, batch=2):
    return SimpleNamespace(
        state=torch.randn(batch, 32),
        image_masks={"cam": torch.ones(batch, dtype=torch.bool)},
        tokenized_prompt_mask=torch.ones(batch, 8, dtype=torch.bool),
        thermal_window=torch.randn(batch, cfg.window_size, cfg.signal_dim),
        thermal_window_time_mask=torch.ones(batch, cfg.window_size, dtype=torch.bool),
        sensor_xyz_base=torch.randn(batch, 3),
        sensor_normal_base=torch.nn.functional.normalize(torch.randn(batch, 3), dim=-1),
    )


def _supervision(cfg, batch=2, queries=7):
    return ThermalSupervision(
        contact_temp_c=torch.randn(batch, 1),
        contact_valid=torch.ones(batch, 1),
        field_query_xyz_base=torch.randn(batch, queries, 3),
        field_query_temp_c=torch.randn(batch, queries, 1),
        field_query_mask=torch.ones(batch, queries, dtype=torch.bool),
        strategy_type=torch.tensor([0, 1]),
    )


def test_config_rejects_bad_gate():
    with pytest.raises(ValueError, match="prefix_gate_init"):
        ThermalPi0Config(prefix_gate_init=0.0)


def test_contact_encoder_masks_input_and_shapes():
    torch.manual_seed(0)
    encoder = ThermalContactEncoder(signal_dim=14, latent_dim=16).eval()
    history = torch.randn(1, 8, 14)
    changed = history.clone()
    changed[:, :3] = 100
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, 3:] = True
    a = encoder(history, mask)
    b = encoder(changed, mask)
    torch.testing.assert_close(a.latent, b.latent)
    assert a.temp_mean_c.shape == (1, 1)


def test_contact_token_and_memory_are_set_based():
    torch.manual_seed(0)
    builder = ContactTokenBuilder(latent_dim=16, hidden_dim=32)
    tokens = builder(torch.randn(10, 16), torch.randn(10, 3), torch.randn(10, 3),
                     torch.randn(10, 1), torch.zeros(10, 1), torch.rand(10, 1)).view(2, 5, 32)
    memory = ThermalSpatialMemory(hidden_dim=32, num_spatial_tokens=4).eval()
    mask = torch.ones(2, 5, dtype=torch.bool)
    a = memory(tokens, mask).spatial_latents
    permutation = torch.tensor([4, 0, 3, 1, 2])
    b = memory(tokens[:, permutation], mask[:, permutation]).spatial_latents
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    empty = memory(torch.zeros(2, 0, 32), torch.zeros(2, 0, dtype=torch.bool)).spatial_latents
    assert empty.shape == (2, 4, 32) and torch.isfinite(empty).all()


def test_field_decoder_is_training_query_only():
    decoder = SpatialFieldDecoder(hidden_dim=32)
    memory = torch.randn(2, 4, 32, requires_grad=True)
    output = decoder(memory, torch.randn(2, 7, 3))
    assert output.temp_mean_c.shape == (2, 7, 1)
    assert output.temp_logvar.shape == (2, 7, 1)
    output.temp_mean_c.sum().backward()
    assert memory.grad is not None


def test_global_skill_head_has_no_point_output():
    head = GlobalSkillAuxHead(hidden_dim=32).eval()
    output = head(torch.randn(1, 4, 32), torch.randn(1, 32), torch.randn(1, 32),
                  torch.randn(1, 32))
    assert output.skill_logits.shape == (1, 4)
    assert not hasattr(output, "probe_utility")


def test_integrated_encoder_prefix_has_only_memory_tokens_and_losses_work():
    torch.manual_seed(0)
    cfg = _cfg()
    encoder = IntegratedThermalEncoder(cfg, paligemma_width=64, state_dim=32)
    observation = _observation(cfg)
    output = encoder(observation, [torch.randn(2, 10, 64)], torch.randn(2, 8, 64))
    assert output.prefix_tokens.shape == (2, cfg.num_spatial_tokens, 64)
    assert output.field is None
    assert output.strategy.skill_logits.shape == (2, 4)
    supervision = _supervision(cfg)
    encoder.decode_field(output, supervision.field_query_xyz_base)
    terms = compute_thermal_losses(output, supervision, {
        "contact_temp": .2, "contact_valid": .1, "field_temp": .3, "skill": .2,
    })
    assert set(terms) == {"contact_temp", "contact_valid", "field_temp", "skill", "aux_total"}
    terms["aux_total"].mean().backward()
    assert encoder.contact_encoder.project[0].weight.grad is not None
    assert encoder.field_decoder.temp_mean_head.weight.grad is not None


def test_physical_forward_does_not_run_strategy_or_prefix():
    cfg = _cfg()
    encoder = IntegratedThermalEncoder(cfg, paligemma_width=64)
    output = encoder.forward_physical(_observation(cfg))
    assert output.prefix_tokens.shape == (2, 0, 64)
    assert output.strategy is None and output.field is None
