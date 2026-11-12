"""数据集加载：HDF5 原始帧 -> 编码器输入窗口与标签。

只在这里做派生：

* 100 Hz 原始帧按 2 取 1 重采样到 50 Hz（``sample_hz`` / ``training_hz``）；
* 逐帧 14 维特征（温差、温度斜率、切向力、接触时间等，见
  ``ENCODER_INPUT_CHANNELS``）；
* 前缀窗口 ``prefixes_s``：窗口末端取接触期内的多个时刻，长度不同的历史
  用 ``mask`` 表示；
* 归一化统计只在训练集上算（:func:`compute_stats`）。

标签全部来自 ``privileged/``：接触 link 的当前温度和接触有效性。区域
编号、link 名、真实温度都不进模型输入。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from data.writer.hdf5_writer import read_episode

# 温度斜率的平滑时间常数（秒）：原始 50 Hz 序列上的一阶低通。
RATE_SMOOTH_S = 0.1


@dataclass(frozen=True)
class Episode:
    """一集的加载结果（原始数组 + 元数据 + 标量）。"""

    arrays: dict[str, np.ndarray]
    metadata: dict
    target_region: int

    @property
    def split(self) -> str:
        return str(self.metadata["split"])


@dataclass(frozen=True)
class WindowSet:
    """一集的全部训练窗口（形状统一，末尾用 mask 屏蔽）。"""

    features: np.ndarray  # (N, H, 14) float32
    mask: np.ndarray  # (N, H) bool
    temperature: np.ndarray  # (N,) 接触 link 当前温度 (°C)
    contact_valid: np.ndarray  # (N,) 本帧杯子是否有接触（有效性头标签）
    prefix_s: np.ndarray  # (N,) 本窗口使用的历史长度
    times: np.ndarray  # (N,) 窗口末端时间 (s)


def load_episode(path: str | Path) -> Episode:
    arrays, metadata, _ = read_episode(path)
    return Episode(
        arrays=arrays,
        metadata=metadata,
        target_region=int(metadata["target_region"]),
    )


def resample(arrays: dict[str, np.ndarray], *, stride: int = 2) -> dict[str, np.ndarray]:
    """按整数步长抽取（100 Hz -> 50 Hz 即 ``stride=2``）。"""
    if stride <= 1:
        return arrays
    sensor_prefixes = ("sensor/", "tactile/", "privileged/")
    return {
        name: np.asarray(arr)[::stride] if name.startswith(sensor_prefixes) else np.asarray(arr)
        for name, arr in arrays.items()
    }


def _rate(values: np.ndarray, dt: float) -> np.ndarray:
    """一阶低通 + 中心差分得到温度斜率 (°C/s)。"""
    alpha = min(1.0, dt / max(RATE_SMOOTH_S, dt))
    smooth = np.empty_like(values)
    smooth[0] = values[0]
    for i in range(1, len(values)):
        smooth[i] = smooth[i - 1] + alpha * (values[i] - smooth[i - 1])
    rate = np.zeros_like(values)
    if len(values) > 2:
        rate[1:-1] = (smooth[2:] - smooth[:-2]) / (2.0 * dt)
        rate[0] = (smooth[1] - smooth[0]) / dt
        rate[-1] = (smooth[-1] - smooth[-2]) / dt
    return rate


def frame_features(arrays: dict[str, np.ndarray], *, dt: float) -> np.ndarray:
    """逐帧 14 维特征矩阵 (L, 14)，通道顺序见 ``ENCODER_INPUT_CHANNELS``。"""
    temp = np.asarray(arrays["tactile/fingertip_temp"], dtype=np.float64)
    normal = np.asarray(arrays["tactile/contact_normal_force"], dtype=np.float64)
    force = np.asarray(arrays["tactile/contact_force_link"], dtype=np.float64)
    area = np.asarray(arrays["tactile/contact_area"], dtype=np.float64)
    time = np.asarray(arrays["sensor/time"], dtype=np.float64)
    control_time = np.asarray(arrays["control/time"], dtype=np.float64)
    control_width = np.asarray(arrays["robot/gripper_width"], dtype=np.float64)
    control_index = np.searchsorted(control_time, time, side="right") - 1
    control_index = np.clip(control_index, 0, len(control_width) - 1)
    width = control_width[control_index].reshape(-1, 1)

    # 切向力 = 合力里扣掉法向分量后的模长。
    tangential = np.sqrt(np.maximum(np.sum(force**2, axis=2) - normal**2, 0.0))

    # 每次接触事件都从 0 重新计时；离开杯子后回到 0。
    event_id = np.asarray(arrays["tactile/contact_event_id"], dtype=np.int64)
    contact_time = np.zeros((len(time), 1), dtype=np.float64)
    for event in range(1, int(event_id.max(initial=0)) + 1):
        indices = np.where(event_id == event)[0]
        if indices.size:
            contact_time[indices, 0] = time[indices] - time[indices[0]]

    dtemp = temp - temp[:1]
    rate = np.stack([_rate(temp[:, 0], dt), _rate(temp[:, 1], dt)], axis=1)

    return np.concatenate(
        [temp, dtemp, rate, normal, tangential, area, width, contact_time], axis=1
    ).astype(np.float32)


def _contact_start(arrays: dict[str, np.ndarray]) -> float | None:
    """首次有接触的时刻；整集没有接触时返回 ``None``。"""
    mask = np.asarray(arrays["tactile/contact_mask"], dtype=bool).reshape(-1, 2)
    hit = np.where(mask.any(axis=1))[0]
    if hit.size == 0:
        return None
    time = np.asarray(arrays["sensor/time"], dtype=np.float64)
    return float(time[hit[0]])


def window_end_times(
    arrays: dict[str, np.ndarray],
    *,
    min_contact_s: float,
    window_stride_s: float,
    no_contact_window_s: float = 0.0,
) -> np.ndarray:
    """接触期内窗口末端的时刻（秒）。

    正常集从“接触开始后 ``min_contact_s``”开始，每 ``window_stride_s`` 取
    一个末端，直到最后一次接触。

    没有接触（失败档）或接触段太短时：``no_contact_window_s > 0`` 就在序列
    末尾取这么长的一段，把“正在找接触”的帧作为接触有效性的负样本；否则
    返回空数组。
    """
    time = np.asarray(arrays["sensor/time"], dtype=np.float64)
    event_id = np.asarray(arrays["tactile/contact_event_id"], dtype=np.int64)
    ends_per_event: list[np.ndarray] = []
    for event in range(1, int(event_id.max(initial=0)) + 1):
        hit = np.where(event_id == event)[0]
        if hit.size == 0:
            continue
        t_start, t_end = float(time[hit[0]]), float(time[hit[-1]])
        if t_end - t_start < min_contact_s:
            continue
        values = np.arange(t_start + window_stride_s, t_end + 1e-9, window_stride_s)
        ends_per_event.append(values if values.size else np.asarray([t_end]))
    if ends_per_event:
        return np.concatenate(ends_per_event)

    if len(time) == 0:
        return np.zeros(0, dtype=np.float64)
    if no_contact_window_s <= 0.0:
        return np.zeros(0, dtype=np.float64)
    else:
        t_end = float(time[-1])
        t_start = max(float(time[0]), t_end - float(no_contact_window_s))
    ends = np.arange(t_start + window_stride_s, t_end + 1e-9, window_stride_s)
    if ends.size == 0:
        ends = np.array([t_end], dtype=np.float64)
    return ends


def build_windows(
    episode: Episode,
    *,
    prefixes_s: Sequence[float],
    training_hz: float,
    stride: int,
    window_stride_s: float = 0.05,
    min_contact_s: float = 0.05,
    no_contact_window_s: float = 0.5,
) -> WindowSet:
    """把一集切成训练窗口：接触期内的多个末端时刻 × 多个前缀长度。"""
    arrays = resample(episode.arrays, stride=stride)
    dt = 1.0 / float(training_hz)
    features = frame_features(arrays, dt=dt)

    time = np.asarray(arrays["sensor/time"], dtype=np.float64)
    link_temp = np.asarray(arrays["privileged/link_temp"], dtype=np.float64)
    link_mask = np.asarray(arrays["privileged/contact_link_mask"], dtype=bool)
    contact_valid = np.asarray(arrays["privileged/contact_valid"], dtype=bool).any(axis=1)

    touched_regions = link_mask.any(axis=1)
    pure = touched_regions.sum(axis=1) == 1
    ends = window_end_times(
        arrays,
        min_contact_s=min_contact_s,
        window_stride_s=window_stride_s,
        no_contact_window_s=no_contact_window_s,
    )
    if ends.size == 0:
        return _empty_windows(0, features.shape[1])

    max_len = int(round(max(prefixes_s) * training_hz)) + 1

    feats, masks, temps, valids, prefix_col, times = [], [], [], [], [], []
    for end_t in ends:
        end_idx = int(np.searchsorted(time, end_t, side="right") - 1)
        if end_idx < 0:
            continue
        for prefix in prefixes_s:
            length = int(round(float(prefix) * training_hz)) + 1
            length = min(length, max_len, end_idx + 1)
            window = np.zeros((max_len, features.shape[1]), dtype=np.float32)
            window_mask = np.zeros(max_len, dtype=bool)
            window[-length:] = features[end_idx + 1 - length : end_idx + 1]
            window_mask[-length:] = True
            feats.append(window)
            masks.append(window_mask)
            # 温度标签只在“纯目标区域接触”时成立：碰到别的 link 时，接触
            # link 的温度不是目标区域的温度，标签无意义。
            if pure[end_idx]:
                label_region = int(np.flatnonzero(touched_regions[end_idx])[0])
                temps.append(link_temp[end_idx, label_region])
            else:
                # 没有接触（失败档的负样本）、混杂接触或越界接触：温度标签留 NaN。
                temps.append(np.nan)
            # 有效性头的标签是“本帧杯子有任何接触”，与纯接触分开。
            valids.append(contact_valid[end_idx])
            prefix_col.append(float(prefix))
            times.append(float(time[end_idx]))

    if not feats:
        return _empty_windows(0, features.shape[1])
    return WindowSet(
        features=np.stack(feats),
        mask=np.stack(masks),
        temperature=np.asarray(temps, dtype=np.float32),
        contact_valid=np.asarray(valids, dtype=bool),
        prefix_s=np.asarray(prefix_col, dtype=np.float32),
        times=np.asarray(times, dtype=np.float32),
    )


def _empty_windows(n: int, channels: int) -> WindowSet:
    return WindowSet(
        features=np.zeros((n, 1, channels), dtype=np.float32),
        mask=np.zeros((n, 1), dtype=bool),
        temperature=np.zeros(n, dtype=np.float32),
        contact_valid=np.zeros(n, dtype=bool),
        prefix_s=np.zeros(n, dtype=np.float32),
        times=np.zeros(n, dtype=np.float32),
    )


def concatenate(sets: Iterable[WindowSet]) -> WindowSet:
    """把多集窗口拼成一份。"""
    sets = [s for s in sets if s.features.shape[0] > 0]
    if not sets:
        return _empty_windows(0, 14)
    return WindowSet(
        features=np.concatenate([s.features for s in sets]),
        mask=np.concatenate([s.mask for s in sets]),
        temperature=np.concatenate([s.temperature for s in sets]),
        contact_valid=np.concatenate([s.contact_valid for s in sets]),
        prefix_s=np.concatenate([s.prefix_s for s in sets]),
        times=np.concatenate([s.times for s in sets]),
    )


def compute_stats(windows: WindowSet) -> dict[str, list[float]]:
    """训练集上的逐通道均值/标准差（只用有 mask 的帧）。"""
    valid = windows.mask
    if not valid.any():
        raise ValueError("训练窗口为空，无法计算归一化统计")
    data = windows.features[valid]  # (M, C)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_stats(windows: WindowSet, stats: dict[str, list[float]]) -> WindowSet:
    """按统计量归一化特征（mask 之外的帧保持 0）。"""
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    features = (windows.features - mean) / std
    features = np.where(windows.mask[..., None], features, 0.0)
    return WindowSet(
        features=features.astype(np.float32),
        mask=windows.mask,
        temperature=windows.temperature,
        contact_valid=windows.contact_valid,
        prefix_s=windows.prefix_s,
        times=windows.times,
    )


__all__ = [
    "Episode",
    "WindowSet",
    "apply_stats",
    "build_windows",
    "compute_stats",
    "concatenate",
    "frame_features",
    "load_episode",
    "resample",
]
