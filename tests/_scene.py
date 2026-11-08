"""测试共用的仿真夹具：任务配置与场景只构建一次。

Genesis 初始化需要 GPU 与 WSL 的 CUDA 库（运行前导出
``LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH``）。初始化失败时
抛 ``unittest.SkipTest``，让不依赖仿真的测试照常跑完。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TASK_CONFIG = "configs/tasks/thermal_contact_probe.yaml"
DATASET_CONFIG = "configs/datasets/thermal_contact_v1.yaml"

_BUNDLE = None


def task_config() -> dict:
    from sim.config import load_task_config

    return load_task_config(str(REPO_ROOT / TASK_CONFIG))


def dataset_config() -> dict:
    from sim.config import load_dataset_config

    return load_dataset_config(str(REPO_ROOT / DATASET_CONFIG))


def get_bundle():
    """构建（或复用）场景。"""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    try:
        import genesis  # noqa: F401
    except Exception as exc:  # 环境里没有 Genesis：跳过
        raise unittest.SkipTest(f"Genesis 不可用：{exc}")
    from sim.scene import build_scene

    try:
        _BUNDLE = build_scene(task_config(), seed=0)
    except Exception as exc:
        raise unittest.SkipTest(f"场景构建失败（需要 GPU/WSL CUDA 库）：{exc}")
    return _BUNDLE


__all__ = ["DATASET_CONFIG", "REPO_ROOT", "TASK_CONFIG", "dataset_config", "get_bundle", "task_config"]
