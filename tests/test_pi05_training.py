"""Thermal-π0.5 训练层测试：配置、数据、collate、冒烟训练与断点续训。

运行：``PYTHONPATH=. .venv/bin/python -m pytest tests/test_pi05_training.py -q``
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# π0.5 依赖的 transformers/h5py/sentencepiece 只在项目根 .venv 里；在 tempvla
# 环境跑 ``unittest discover`` 时整个模块跳过，而不是让收集阶段报错。
try:
    import h5py
    import torch

    from models.pi05.thermal.config import ThermalPi0Config
    from training.pi05.checkpoints import latest_checkpoint_step
    from training.pi05.config import load_train_config
    from training.pi05.data import (
        SUPERVISION_FIELDS,
        FakeThermalVlaDataset,
        Hdf5ThermalVlaDataset,
        collate_samples,
    )
    from training.pi05.eval import evaluate
    from training.pi05.tokenizer import DEFAULT_TOKENIZER_PATH, PaligemmaTokenizer
    from training.pi05.train import train_loop
    from training.pi05.transforms import Normalizer, pad_or_trim, prepare_sample
except ImportError as exc:  # pragma: no cover - 取决于运行环境
    raise unittest.SkipTest(f"π0.5 训练测试需要项目根 .venv：{exc}") from exc

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "training"
SMOKE_CONFIG = CONFIG_DIR / "pi05_thermal_smoke.yaml"

THERMAL = ThermalPi0Config(enabled=True, history_size=2, window_size=4, signal_dim=6, num_surface_points=5)


class ConfigTest(unittest.TestCase):
    def test_stage_configs_load(self) -> None:
        for stage in (1, 2, 3, 4):
            config = load_train_config(CONFIG_DIR / f"pi05_thermal_stage{stage}.yaml")
            self.assertEqual(config.thermal_stage, stage)
            self.assertEqual(config.model.thermal.enabled, True)  # noqa: E712
            self.assertEqual(config.data.action_dim, config.model.action_dim)
            self.assertEqual(config.data.action_horizon, config.model.action_horizon)

    def test_unknown_key_rejected(self) -> None:
        text = SMOKE_CONFIG.read_text(encoding="utf-8") + "\nunknown_section: 1\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_train_config(path)


class DataTest(unittest.TestCase):
    def test_fake_dataset_shapes(self) -> None:
        dataset = FakeThermalVlaDataset(
            size=3,
            image_keys=("base_0_rgb",),
            image_size=32,
            action_dim=7,
            action_horizon=4,
            prompt="pick the cold block",
            thermal=THERMAL,
        )
        sample = dataset[0]
        self.assertEqual(sample["image"]["base_0_rgb"].shape, (32, 32, 3))
        self.assertEqual(sample["actions"].shape, (4, 7))
        self.assertEqual(sample["thermal_history"].shape, (2, 4, 6))
        self.assertEqual(sample["surface_xyz_obj"].shape, (5, 3))
        self.assertEqual(sample["thermal_supervision"]["strategy_type"].shape, ())
        self.assertTrue(all(key in sample["thermal_supervision"] for key in SUPERVISION_FIELDS))

    @unittest.skipUnless(DEFAULT_TOKENIZER_PATH.exists(), "缺少 paligemma tokenizer")
    def test_prepare_and_collate(self) -> None:
        dataset = FakeThermalVlaDataset(
            size=2,
            image_keys=("base_0_rgb",),
            image_size=32,
            action_dim=7,
            action_horizon=6,
            prompt="pick the cold block",
            thermal=THERMAL,
        )
        tokenizer = PaligemmaTokenizer(max_len=16)
        normalizer = Normalizer.identity(7, 7)
        samples = [
            prepare_sample(
                dataset[index],
                normalizer=normalizer,
                tokenizer=tokenizer,
                image_keys=("base_0_rgb",),
                action_dim=7,
                action_horizon=6,
            )
            for index in range(2)
        ]
        batch = collate_samples(samples)
        self.assertEqual(batch["image"]["base_0_rgb"].shape, (2, 32, 32, 3))
        self.assertEqual(batch["image"]["base_0_rgb"].dtype, torch.uint8)
        self.assertEqual(batch["tokenized_prompt"].shape, (2, 16))
        self.assertEqual(batch["thermal_history"].shape, (2, 2, 4, 6))
        self.assertEqual(batch["thermal_history_time_mask"].dtype, torch.bool)
        self.assertEqual(batch["thermal_supervision"]["strategy_type"].shape, (2,))
        self.assertEqual(batch["thermal_supervision"]["surface_temp_mask"].dtype, torch.bool)

    def test_hdf5_dataset_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._write_episode(pathlib.Path(tmp) / "episode.hdf5")
            dataset = Hdf5ThermalVlaDataset(tmp, prompt="fallback prompt", thermal=THERMAL)
            self.assertEqual(len(dataset), 3)
            sample = dataset[1]
            self.assertEqual(sample["image"]["base_0_rgb"].shape, (32, 32, 3))
            self.assertEqual(sample["actions"].shape, (6, 7))
            self.assertEqual(sample["prompt"], "pick the cold block")
            self.assertEqual(sample["thermal_history"].shape, (2, 4, 6))
            self.assertEqual(sample["thermal_supervision"]["strategy_type"].shape, ())
            states, actions = dataset.stats()
            self.assertEqual(states.shape, (3, 8))
            self.assertEqual(actions.shape, (3, 6, 7))

    @staticmethod
    def _write_episode(path: pathlib.Path) -> None:
        rng = np.random.default_rng(0)
        with h5py.File(path, "w") as f:
            f.attrs["language_instruction"] = "pick the cold block"
            images = f.create_group("images")
            images.create_dataset("base_0_rgb", data=rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8))
            f.create_dataset("state", data=rng.uniform(-1, 1, (3, 8)).astype(np.float32))
            f.create_dataset("actions", data=rng.uniform(-1, 1, (3, 6, 7)).astype(np.float32))
            thermal = f.create_group("thermal")
            thermal.create_dataset("thermal_history", data=rng.uniform(-5, 45, (3, 2, 4, 6)).astype(np.float32))
            thermal.create_dataset("thermal_history_time_mask", data=np.ones((3, 2, 4), dtype=bool))
            thermal.create_dataset("contact_mask", data=np.ones((3, 2), dtype=bool))
            thermal.create_dataset("contact_xyz_obj", data=rng.uniform(-1, 1, (3, 2, 3)).astype(np.float32))
            thermal.create_dataset("contact_normal_obj", data=rng.uniform(-1, 1, (3, 2, 3)).astype(np.float32))
            thermal.create_dataset("contact_age_s", data=rng.uniform(0, 5, (3, 2, 1)).astype(np.float32))
            thermal.create_dataset("surface_xyz_obj", data=rng.uniform(-1, 1, (3, 5, 3)).astype(np.float32))
            thermal.create_dataset("surface_normal_obj", data=rng.uniform(-1, 1, (3, 5, 3)).astype(np.float32))
            thermal.create_dataset("surface_mask", data=np.ones((3, 5), dtype=bool))
            thermal.create_dataset("surface_reachable", data=np.ones((3, 5), dtype=bool))
            thermal.create_dataset("surface_grasp_score", data=rng.uniform(0, 1, (3, 5, 1)).astype(np.float32))
            thermal.create_dataset("ee_xyz_obj", data=rng.uniform(-1, 1, (3, 3)).astype(np.float32))
            thermal.create_dataset("probe_budget_remaining", data=np.ones((3, 1), dtype=np.float32))
            supervision = f.create_group("supervision")
            supervision.create_dataset("contact_temp_c", data=rng.uniform(-5, 45, (3, 2, 1)).astype(np.float32))
            supervision.create_dataset("contact_valid", data=np.ones((3, 2, 1), dtype=np.float32))
            supervision.create_dataset("surface_temp_c", data=rng.uniform(-5, 45, (3, 5, 1)).astype(np.float32))
            supervision.create_dataset("surface_temp_mask", data=np.ones((3, 5), dtype=bool))
            supervision.create_dataset("surface_safe", data=rng.uniform(0, 1, (3, 5, 1)).astype(np.float32))
            supervision.create_dataset("strategy_type", data=np.zeros((3,), dtype=np.int64))
            supervision.create_dataset("target_surface_index", data=np.zeros((3,), dtype=np.int64))

    def test_pad_or_trim(self) -> None:
        values = np.arange(6, dtype=np.float32).reshape(3, 2)
        padded = pad_or_trim(values, 5)
        self.assertEqual(padded.shape, (5, 2))
        np.testing.assert_array_equal(padded[3:], np.repeat(values[-1:], 2, axis=0))
        trimmed = pad_or_trim(values, 2)
        np.testing.assert_array_equal(trimmed, values[:2])


@unittest.skipUnless(DEFAULT_TOKENIZER_PATH.exists(), "缺少 paligemma tokenizer")
class TrainLoopTest(unittest.TestCase):
    def test_smoke_train_and_resume(self) -> None:
        config = load_train_config(SMOKE_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = pathlib.Path(tmp) / "run"
            config = dataclasses.replace(
                config,
                checkpoint_dir=str(checkpoint_dir),
                num_train_steps=2,
                save_interval=1,
                overwrite=True,
            )
            train_loop(config)
            self.assertEqual(latest_checkpoint_step(checkpoint_dir), 2)
            self.assertTrue((checkpoint_dir / "2" / "model.safetensors").exists())
            self.assertTrue((checkpoint_dir / "norm_stats.json").exists())

            resumed = dataclasses.replace(config, num_train_steps=3, resume=True)
            train_loop(resumed)
            self.assertEqual(latest_checkpoint_step(checkpoint_dir), 3)


@unittest.skipUnless(DEFAULT_TOKENIZER_PATH.exists(), "缺少 paligemma tokenizer")
class EvalTest(unittest.TestCase):
    def test_evaluate_smoke_checkpoint(self) -> None:
        config = load_train_config(SMOKE_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = pathlib.Path(tmp) / "run"
            config = dataclasses.replace(
                config,
                checkpoint_dir=str(checkpoint_dir),
                num_train_steps=1,
                save_interval=1,
                overwrite=True,
            )
            train_loop(config)
            metrics = evaluate(config, checkpoint_dir=checkpoint_dir, num_batches=1, device="cpu")
            self.assertEqual(int(metrics["step"]), 1)
            self.assertEqual(int(metrics["batches"]), 1)
            self.assertTrue(math.isfinite(metrics["total"]))


if __name__ == "__main__":
    unittest.main()
