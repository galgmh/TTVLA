"""CPU smoke test for the native PyTorch LoRA implementation."""

from __future__ import annotations

import tempfile

import torch
from torch import nn

from pi05_model.lora import (
    LoRATrainingConfig,
    configure_lora_training,
    inject_lora,
    load_lora_adapter,
    save_lora_adapter,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.gate_proj = nn.Linear(8, 16, bias=False)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _PaliGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = _PaliGemma()
        self.gemma_expert = _LanguageModel()
        self.action_in_proj = nn.Linear(8, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.paligemma.language_model.layers[0].q_proj(inputs)


def main() -> None:
    torch.manual_seed(7)
    model = _Policy()
    inputs = torch.randn(2, 8)
    base_output = model(inputs).detach().clone()
    config = LoRATrainingConfig(enabled=True, apply_to="all")
    replaced = inject_lora(model, config)
    counts = configure_lora_training(model, config)

    assert len(replaced) == 4
    assert torch.equal(base_output, model(inputs))
    assert counts["lora"] > 0
    assert not model.paligemma.language_model.layers[0].q_proj.weight.requires_grad

    with tempfile.TemporaryDirectory() as directory:
        with torch.no_grad():
            model.paligemma.language_model.layers[0].q_proj.lora_b.fill_(0.1)
        expected = model(inputs).detach().clone()
        save_lora_adapter(model, directory, config)

        restored = _Policy()
        # Both policies represent the same loaded dense base checkpoint.
        dense = {
            key: value
            for key, value in model.state_dict().items()
            if not key.endswith(("lora_a", "lora_b"))
        }
        inject_lora(restored, config)
        configure_lora_training(restored, config)
        restored.load_state_dict(dense, strict=False)
        load_lora_adapter(restored, directory)
        assert torch.allclose(expected, restored(inputs))

    print(f"LoRA smoke OK: layers={len(replaced)}, trainable_lora={counts['lora']}")


if __name__ == "__main__":
    main()
