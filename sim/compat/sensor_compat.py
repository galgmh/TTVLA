"""针对固定版本 Genesis 传感器实现的两个小补丁。

``install_sensor_reset_fix`` 抄自 tactile-genesis/dexterous-hands 的同名补丁。

``install_contact_area_policy`` 是本仓库自己的补充：改写引擎接触传热核所
消费的接触面积。
"""

from functools import wraps

import torch

# 引擎原始核函数，首次安装时捕获。
_ENGINE_CONTACT_AREA_KERNEL = None


def install_sensor_reset_fix():
    """就地清零指定环境的漂移行——张量高级索引返回的是副本。

    Genesis 的 ``SimpleSensor.reset`` 用 ``_cur_random_walk[envs_idx].fill_(0)``：
    当 ``envs_idx`` 是张量/列表时，实际改的是副本，原状态不变。保留上游的
    reset 链，再显式对行赋值。
    """
    from genesis.engine.sensors.base_sensor import SimpleSensor

    original = SimpleSensor.reset.__func__
    if getattr(original, "_dex_indexed_reset", False):
        return

    @wraps(original)
    def reset(cls, metadata, ground_truth, envs_idx):
        original(cls, metadata, ground_truth, envs_idx)
        metadata._cur_random_walk[envs_idx] = 0.0

    reset._dex_indexed_reset = True
    SimpleSensor.reset = classmethod(reset)


def _as_torch(value):
    """quadrants 张量（碰撞器状态）-> torch 视图；torch 张量原样返回。"""
    if torch.is_tensor(value):
        return value
    return value.to_torch(copy=None)


def _rewrite_finger_contact_areas(contact_area, collider_state, finger_faces):
    """对每个触及指尖 link 的接触，就地改写 A_i = A_face / sqrt(n_i)。

    这样 sum(A_i²) = A_face² 在任何接触点数下都成立：2 点/3 点接触阶段的
    接触热流不再差 24 倍，等价于一整块垫面面积的接触。
    """
    n_rows, n_envs = contact_area.shape
    if n_rows == 0 or n_envs == 0:
        return
    contact_data = collider_state.contact_data
    link_a = _as_torch(contact_data.link_a)
    link_b = _as_torch(contact_data.link_b)
    sort_idx = _as_torch(collider_state.contact_sort_idx)
    n_contacts = _as_torch(collider_state.n_contacts).reshape(-1)[:n_envs]
    n_slots = int(link_a.shape[0])
    rows = min(n_rows, n_slots, int(sort_idx.shape[0]))
    if rows == 0:
        return

    device = contact_area.device
    # 拷贝后再 to(int64)（会分配内存）；碰撞器状态本身从不写入。
    # 超过 n_contacts 的槽位存放的是未初始化的排序下标。
    phys = sort_idx[:rows].to(torch.int64).clamp_(0, n_slots - 1)
    envs = torch.arange(n_envs, device=device)
    la = link_a[phys, envs]
    lb = link_b[phys, envs]
    active = torch.arange(rows, device=device).unsqueeze(1) < n_contacts.unsqueeze(0)
    view = contact_area[:rows]
    for link_idx, a_face_m2 in finger_faces.items():
        touching = active & ((la == link_idx) | (lb == link_idx))
        count = touching.sum(dim=0)
        per_contact = a_face_m2 / count.clamp_(min=1).to(contact_area.dtype).sqrt()
        view[touching] = per_contact.unsqueeze(0).expand(rows, n_envs)[touching]


def install_contact_area_policy(finger_faces=None):
    """让指尖接触面积走 pad-face 分摊策略。

    参数
    ----
    finger_faces : mapping[int, float] | None
        全局 link 下标 -> 垫面面积（m²），每根指尖 link 一项。传 None 则
        恢复引擎原始行为。

    幂等：首次调用捕获引擎原始核函数，之后每次调用只是换上/换下同一个
    包装。
    """
    from genesis.engine.sensors import temperature as _temperature

    global _ENGINE_CONTACT_AREA_KERNEL
    if _ENGINE_CONTACT_AREA_KERNEL is None:
        _ENGINE_CONTACT_AREA_KERNEL = _temperature._kernel_compute_contact_areas
    original = _ENGINE_CONTACT_AREA_KERNEL

    if not finger_faces:
        _temperature._kernel_compute_contact_areas = original
        return

    faces = {int(k): float(v) for k, v in finger_faces.items()}
    if any(a <= 0.0 for a in faces.values()):
        raise ValueError(f"垫面面积必须为正，实际 {faces}")

    @wraps(original)
    def contact_areas(contact_area, scratch, dyn_state, collider_state, eps):
        original(contact_area, scratch, dyn_state, collider_state, eps)
        _rewrite_finger_contact_areas(contact_area, collider_state, faces)

    contact_areas._dex_pad_face_areas = faces
    _temperature._kernel_compute_contact_areas = contact_areas
