"""π0.5 的观测数据结构（纯 PyTorch / NumPy，不依赖 JAX）。

字段与官方 openpi ``model.Observation`` 一致；热觉字段是
Thermal-π0.5 追加的可选输入，热觉关闭时全部为 ``None``，原 π0.5 输入不变。

数据变换产出的嵌套字典（``image`` / ``image_mask`` / ``state`` / token 及热觉字段）
通过 :meth:`Observation.from_dict` 转成该结构。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

#: 模型固定使用的三路相机键。
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

#: 模型输入图像分辨率（H, W）。
IMAGE_RESOLUTION = (224, 224)


@dataclasses.dataclass
class Observation:
    """模型观测输入。

    ``images`` 为 ``[B, H, W, C]`` 或 ``[B, C, H, W]`` 的 float32（[-1, 1]）；
    ``from_dict`` 会把 uint8 图像转成 ``[B, C, H, W]`` 的 [-1, 1] float32。
    """

    # 图像（[-1, 1] float32）与逐视图有效 mask。
    images: dict[str, Any]
    image_masks: dict[str, Any]
    # 低维机器人状态。
    state: Any

    # 分词后的语言指令及其 mask。
    tokenized_prompt: Any | None = None
    tokenized_prompt_mask: Any | None = None

    # π0-fast 专用字段；PyTorch π0.5 路径不使用。
    token_ar_mask: Any | None = None
    token_loss_mask: Any | None = None

    # 当前时刻及前L-1帧的在线热触觉窗口；历史接触由模型内部MemoryBank维护。
    thermal_window: Any | None = None  # [B,L,C]
    thermal_window_time_mask: Any | None = None  # [B,L]
    # 当前传感器位置/表面法向，由机器人状态/FK提供，不是预先枚举的候选点。
    sensor_xyz_base: Any | None = None  # [B,3]
    sensor_normal_base: Any | None = None  # [B,3]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        """把数据变换产出的嵌套字典转成 Observation。

        与官方实现一致：uint8 图像就地转成 [-1, 1] float32；
        ``tokenized_prompt`` 与 ``tokenized_prompt_mask`` 必须成对出现。
        热觉字段通过 ``data.get`` 读取，缺省为 None。
        """
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            thermal_window=data.get("thermal_window"),
            thermal_window_time_mask=data.get("thermal_window_time_mask"),
            sensor_xyz_base=data.get("sensor_xyz_base"),
            sensor_normal_base=data.get("sensor_normal_base"),
        )

    def to_dict(self) -> dict[str, Any]:
        """转回嵌套字典（不深拷贝张量，避免复制大图像批次）。"""
        result = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


__all__ = [
    "IMAGE_KEYS",
    "IMAGE_RESOLUTION",
    "Observation",
]
