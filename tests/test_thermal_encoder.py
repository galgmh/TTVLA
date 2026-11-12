"""共享单触点编码器兼容包装测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.thermal_encoder import ThermalEncoder, load_contact_encoder_checkpoint  # noqa: E402
from models.pi05.thermal.contact_encoder import ThermalContactEncoder  # noqa: E402


class EncoderTest(unittest.TestCase):
    def test_receptive_field(self) -> None:
        model = ThermalEncoder(in_channels=14, hidden=8, levels=4, kernel=3)
        # 四个残差块各有两层卷积：1 + 2*2*(1+2+4+8) = 61。
        self.assertEqual(model.receptive_field, 61)

    def test_wrapper_uses_shared_core(self) -> None:
        torch.manual_seed(0)
        model = ThermalEncoder(in_channels=1, hidden=4, levels=1, kernel=1, dropout=0.0, out_dim=3)
        model.eval()
        features = torch.randn(1, 8, 1)
        mask = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            z = model(features, mask)
        shared = model.core(features, mask).latent
        torch.testing.assert_close(z, shared)

    def test_padded_frames_do_not_leak(self) -> None:
        """mask 外的帧内容不得影响输出（偏置/前向都已被清零）。"""
        torch.manual_seed(0)
        model = ThermalEncoder(in_channels=1, hidden=4, levels=1, kernel=3, dropout=0.0, out_dim=3)
        model.eval()
        base = torch.randn(1, 8, 1)
        other = base.clone()
        other[:, :5] = 100.0  # 只改被 mask 掉的前 5 帧
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[:, -3:] = True
        with torch.no_grad():
            a = model(base, mask)
            b = model(other, mask)
        torch.testing.assert_close(a, b)

    def test_checkpoint_converts_strictly_to_vla_encoder(self) -> None:
        source = ThermalEncoder()
        target = ThermalContactEncoder()
        state = {f"encoder.core.{key}": value for key, value in source.core.state_dict().items()}
        load_contact_encoder_checkpoint(target, {"model": state})
        for key, value in source.core.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
