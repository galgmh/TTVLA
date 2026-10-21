"""Small compatibility fixes for the pinned Genesis sensor implementation.

``install_sensor_reset_fix`` is copied from tactile-genesis/dexterous-hands/src/sensor_compat.py

``install_contact_area_policy`` is this repo's own addition: 
it rewrites the contact areas the engine's contact-heat kernel consumes.
"""

from functools import wraps

import torch

# Pristine engine kernel, captured on the first install.
_ENGINE_CONTACT_AREA_KERNEL = None


def install_sensor_reset_fix():
    """Reset indexed drift rows in-place; tensor advanced indexing returns a copy.

    Genesis SimpleSensor.reset uses `_cur_random_walk[envs_idx].fill_(0)`.
    For a tensor/list of environment indices that leaves the original state
    unchanged. Preserve the upstream reset chain and explicitly assign the rows.
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
    """
    quadrants Tensor (collider state) -> torch view; torch tensors pass through.
    """
    if torch.is_tensor(value):
        return value
    return value.to_torch(copy=None)


def _rewrite_finger_contact_areas(contact_area, collider_state, finger_faces):
    """
    A_i = A_face / sqrt(n_i) for every contact touching a finger link, in place.
    (sum(A_i^2) = A_face^2 holds for any contact)
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
    # Clamp the copy(to(int64) allocates); the collider state itself is never written.
    # Slots past n_contacts hold uninitialised sort indices.
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
    """Route finger contact areas through the pad-face split.

    Parameters
    finger_faces : mapping[int, float] | None
        Global link index -> pad face area (m^2) for every finger link.

    Idempotent: the pristine engine kernel is captured on the first call and
    every later call swaps a single wrapper in or out.
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
        raise ValueError(f"pad face areas must be positive, got {faces}")

    @wraps(original)
    def contact_areas(contact_area, scratch, dyn_state, collider_state, eps):
        original(contact_area, scratch, dyn_state, collider_state, eps)
        _rewrite_finger_contact_areas(contact_area, collider_state, faces)

    contact_areas._dex_pad_face_areas = faces
    _temperature._kernel_compute_contact_areas = contact_areas
