"""编码器测试：右对齐 mask 下必须取最后一个有效帧的隐状态。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.thermal_encoder import ThermalEncoder  # noqa: E402


class EncoderTest(unittest.TestCase):
    def test_receptive_field(self) -> None:
        model = ThermalEncoder(in_channels=14, hidden=8, levels=4, kernel=3)
        # kernel 3、膨胀 1/2/4/8：1 + 2 + 4 + 8 + 16 = 31 帧。
        self.assertEqual(model.receptive_field, 31)

    def test_last_valid_frame_is_gathered(self) -> None:
        """短前缀（前端补零）与完整输入在同一末帧上必须给出相同 z。"""
        torch.manual_seed(0)
        model = ThermalEncoder(in_channels=1, hidden=4, levels=1, kernel=1, dropout=0.0, out_dim=3)
        model.eval()
        full = torch.randn(1, 8, 1)
        # 只保留最后 3 帧，前端补零并把 mask 置 False。
        short = torch.zeros(1, 8, 1)
        short[:, -3:] = full[:, -3:]
        short_mask = torch.zeros(1, 8, dtype=torch.bool)
        short_mask[:, -3:] = True
        full_mask = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            z_full = model(full, full_mask)
            z_short = model(short, short_mask)
        # kernel=1 时末帧的隐状态只取决于末帧本身，两者必须一致。
        torch.testing.assert_close(z_short, z_full)

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


if __name__ == "__main__":
    unittest.main()
