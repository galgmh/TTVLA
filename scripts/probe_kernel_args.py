"""Wrap the engine's contact-heat kernel and log what it really receives.

The engine's ``TemperatureGridSensor._update_raw_data`` calls the module-global
``_kernel_contact_heat``. Replacing that global with a Python wrapper lets us

  1. run the real kernel against a *scratch copy* of ``raw_data_T`` and diff it: the
     delta the kernel computes from the arguments it was handed, and
  2. run it against the real buffer and diff *that*: the delta that actually lands.

If (1) >> (2) the heat is destroyed after the kernel (clamp / transform / ring);
if (1) == (2) the kernel itself is injecting far less than the documented formula
``dt*k_eff*A^2/(V^2*rho_cp)`` implies, and the discrepancy is in the arguments.

Run: python scripts/probe_kernel_args.py [--substeps 40] [--report 3]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate1_sensors import _drive, _np  # noqa: E402
from gate_common import load_config  # noqa: E402

STATE: dict = {"on": False, "done": True}


def install() -> None:
    import genesis.engine.sensors.temperature as T

    orig = T._kernel_contact_heat

    def wrapper(*call_args):
        if not STATE["on"] or STATE["done"]:
            return orig(*call_args)
        STATE["done"] = True
        (
            links_idx, sensor_cache_start, link_to_material_idx, aabb_min, grid_size,
            voxel_size, voxel_volume, depth_weight, link_temps, link_volume,
            link_base_temperature, link_conductivity, link_rho_cp, contact_area,
            output, dyn_state, collider_state, dt, eps,
        ) = call_args

        before = output.clone()
        scratch = output.clone()
        # Link temps are also written by the kernel: give the scratch pass its own copy.
        orig(
            links_idx, sensor_cache_start, link_to_material_idx, aabb_min, grid_size,
            voxel_size, voxel_volume, depth_weight, link_temps.clone(), link_volume,
            link_base_temperature, link_conductivity, link_rho_cp, contact_area,
            scratch, dyn_state, collider_state, dt, eps,
        )
        orig(*call_args)

        n_c = int(_np(collider_state.n_contacts).reshape(-1)[0])
        STATE["info"] = {
            "n_c": n_c,
            "dt": float(dt),
            "eps": float(eps),
            "voxel_volume": _np(voxel_volume).tolist(),
            "depth_weight": _np(depth_weight).tolist(),
            "grid_size": _np(grid_size).astype(int).tolist(),
            "starts": _np(sensor_cache_start).astype(int).tolist(),
            "rcp": _np(link_rho_cp).tolist(),
            "k": _np(link_conductivity).tolist(),
            "link_to_mat": _np(link_to_material_idx).astype(int).tolist(),
            "link_temps": _np(link_temps)[0].tolist(),
            "area": _np(contact_area)[:n_c, 0].tolist(),
            "scratch_delta": _np(scratch)[:, 0] - _np(before)[:, 0],
            "real_delta": _np(output)[:, 0] - _np(before)[:, 0],
        }
        return None

    T._kernel_contact_heat = wrapper
    print("installed kernel wrapper", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--report", type=int, default=3, help="report this many substeps that have a nonzero delta")
    ap.add_argument("--hot", type=float, default=130.0)
    ap.add_argument("--config", default="configs/pick_hot_cylinder.yaml")
    args = ap.parse_args()

    install()

    from tempvla.actions import DeltaPoseController
    from tempvla.expert import TOP_DOWN_QUAT
    from tempvla.scene import build_scene, scripted_episode

    cfg = load_config(args.config)
    bundle = build_scene(cfg, material="aluminum", seed=cfg.seed)
    bundle.reset_episode(scripted_episode(cfg, xy_offset=np.zeros(2), yaw=0.0, temp_c=args.hot))
    bundle.step()

    ctrl = DeltaPoseController(bundle)
    ctrl.reset()
    obj_pos, _ = bundle.object_pose()
    _drive(bundle, ctrl, obj_pos + np.array([0.0, 0.0, 0.08]), TOP_DOWN_QUAT, -1.0, max_steps=40)
    _drive(bundle, ctrl, obj_pos, TOP_DOWN_QUAT, -1.0, max_steps=40)

    ctrl.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    shown = 0
    for sub in range(args.substeps):
        STATE["on"] = True
        STATE["done"] = False
        bundle.step()
        STATE["on"] = False
        info = STATE.get("info")
        if not info:
            continue
        sd = np.asarray(info["scratch_delta"])
        rd = np.asarray(info["real_delta"])
        if not np.any(np.abs(sd) > 1e-9):
            continue
        print(f"sub {sub:>3} n_c {info['n_c']} dt {info['dt']} eps {info['eps']:.1e}")
        print(f"        voxel_volume {info['voxel_volume']} depth_weight {info['depth_weight']}")
        print(f"        grid {info['grid_size']} starts {info['starts']}")
        print(f"        rcp {info['rcp']}")
        print(f"        k {info['k']}")
        print(f"        link_to_mat {info['link_to_mat']}")
        print(f"        link_temps[0] {[round(v, 2) for v in info['link_temps']]}")
        print(f"        areas[:n_c] {[round(v * 1e4, 3) for v in info['area']]} cm^2")
        print(f"        scratch delta [0:16]  {np.round(sd[:16], 4).tolist()}")
        print(f"        scratch delta [16:32] {np.round(sd[16:32], 4).tolist()}")
        print(f"        scratch delta [32:48] {np.round(sd[32:48], 4).tolist()}")
        print(f"        real    delta [0:16]  {np.round(rd[:16], 4).tolist()}")
        print(f"        real    delta [16:32] {np.round(rd[16:32], 4).tolist()}")
        print(f"        real    delta [32:48] {np.round(rd[32:48], 4).tolist()}", flush=True)
        shown += 1
        if shown >= args.report:
            break
    if shown == 0:
        print("no substep produced a nonzero scratch delta", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
