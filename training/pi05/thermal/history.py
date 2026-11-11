"""因果接触历史缓存（规格第 3 节，Task G）。

一个训练/推理样本只能看到**此前已经完成**的 K 次接触窗口；正在执行的 probe
数据不得进入 memory。本模块把这个约束固化成唯一入口：只有
``add_completed_contact`` 会写入记录，``snapshot`` 只读。

缓存不依赖仿真器：环境在 probe 结束后把窗口、物体坐标系锚点/法向与结束时间
交给本模块；控制时刻 t 调用 ``snapshot(t)`` 得到与 ``Observation`` 字段同名的
numpy 数组，交给 ``thermal.transforms`` 转成模型输入。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

#: snapshot 返回的键（与 Observation 的热觉字段同名，可直接并入数据批次）。
SNAPSHOT_KEYS = (
    "thermal_history",
    "thermal_history_time_mask",
    "contact_mask",
    "contact_xyz_obj",
    "contact_normal_obj",
    "contact_age_s",
)


class ContactHistoryBuffer:
    """固定容量的已完成接触记录；超出 K 时丢弃最旧记录。

    Args:
        history_size: 历史接触上限 K。
        window_size: 每次接触的窗口帧数 L（规格默认 25，即 0.5 s × 50 Hz）。
        signal_dim: 每帧信号维数 C（规格默认 14）。
    """

    def __init__(self, *, history_size: int = 8, window_size: int = 25, signal_dim: int = 14):
        if history_size < 1 or window_size < 1 or signal_dim < 1:
            raise ValueError("history_size / window_size / signal_dim 必须为正整数")
        self.history_size = int(history_size)
        self.window_size = int(window_size)
        self.signal_dim = int(signal_dim)
        self._signals: list[np.ndarray] = []
        self._xyz: list[np.ndarray] = []
        self._normals: list[np.ndarray] = []
        self._end_times: list[float] = []

    def __len__(self) -> int:
        return len(self._signals)

    @property
    def contact_count(self) -> int:
        return len(self._signals)

    def clear(self) -> None:
        """开始新 episode 时清空（网络本身不保存跨 episode 状态）。"""
        self._signals.clear()
        self._xyz.clear()
        self._normals.clear()
        self._end_times.clear()

    def add_completed_contact(
        self,
        signal: np.ndarray,
        xyz_obj: np.ndarray,
        normal_obj: np.ndarray,
        end_time_s: float,
    ) -> None:
        """登记一次**已完成**的接触。

        ``signal`` 形状必须严格为 ``[L, C]``；``xyz_obj`` / ``normal_obj`` 为物体
        坐标系下的接触锚点与法向。调用方只能在 probe 结束后调用本方法——这是
        防止未来信息泄漏的唯一写入口。
        """
        signal = np.asarray(signal, dtype=np.float32)
        xyz = np.asarray(xyz_obj, dtype=np.float32).reshape(-1)
        normal = np.asarray(normal_obj, dtype=np.float32).reshape(-1)
        if signal.shape != (self.window_size, self.signal_dim):
            raise ValueError(
                f"signal 形状必须为 ({self.window_size}, {self.signal_dim})，得到 {signal.shape}"
            )
        if xyz.shape != (3,) or normal.shape != (3,):
            raise ValueError(f"xyz_obj / normal_obj 必须为 3 维，得到 {xyz.shape} / {normal.shape}")
        if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(normal)):
            raise ValueError("接触窗口或锚点含非有限值")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-8:
            raise ValueError("normal_obj 不能为零向量")
        end_time_s = float(end_time_s)
        if not np.isfinite(end_time_s):
            raise ValueError("end_time_s 必须是有限数值")

        self._signals.append(signal.copy())
        self._xyz.append(xyz.copy())
        self._normals.append((normal / normal_norm).copy())
        self._end_times.append(end_time_s)
        # 只保留最近 K 次；列表按时间从旧到新。
        overflow = len(self._signals) - self.history_size
        if overflow > 0:
            del self._signals[:overflow]
            del self._xyz[:overflow]
            del self._normals[:overflow]
            del self._end_times[:overflow]

    def snapshot(self, now_s: float) -> dict[str, np.ndarray]:
        """返回控制时刻 ``now_s`` 可见的历史（右对齐、旧→新、不足补零）。"""
        now_s = float(now_s)
        if not np.isfinite(now_s):
            raise ValueError("now_s 必须是有限数值")

        k, length, channels = self.history_size, self.window_size, self.signal_dim
        history = np.zeros((k, length, channels), dtype=np.float32)
        time_mask = np.zeros((k, length), dtype=bool)
        contact_mask = np.zeros((k,), dtype=bool)
        xyz = np.zeros((k, 3), dtype=np.float32)
        normal = np.zeros((k, 3), dtype=np.float32)
        age = np.zeros((k, 1), dtype=np.float32)

        count = len(self._signals)
        start = k - count
        for offset in range(count):
            slot = start + offset
            history[slot] = self._signals[offset]
            time_mask[slot] = True
            contact_mask[slot] = True
            xyz[slot] = self._xyz[offset]
            normal[slot] = self._normals[offset]
            age[slot, 0] = max(0.0, now_s - self._end_times[offset])

        return {
            "thermal_history": history,
            "thermal_history_time_mask": time_mask,
            "contact_mask": contact_mask,
            "contact_xyz_obj": xyz,
            "contact_normal_obj": normal,
            "contact_age_s": age,
        }

    def snapshot_into(self, target: dict, now_s: float) -> dict:
        """把 snapshot 合并进一个已有数据字典（便于拼 Observation 字段）。"""
        target.update(self.snapshot(now_s))
        return target


def merge_snapshot(base: Mapping[str, np.ndarray], snapshot: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """返回 ``base`` 的浅拷贝并合入 snapshot（不修改输入）。"""
    merged = dict(base)
    merged.update(snapshot)
    return merged


__all__ = ["SNAPSHOT_KEYS", "ContactHistoryBuffer", "merge_snapshot"]
