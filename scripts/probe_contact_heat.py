"""Per-substep probe of the engine's contact-heat update on the finger pads.

Dumps, for every substep of the first few close steps: each finger's 16 cells,
the raw ``contact_area_buffer`` rows with their link pair and penetration, and
the link indices, plus an inversion of the engine's own update,

    dmean*N = sum_j dt*k_eff*(T_other_j - T_cell)*A_j^2 / (V^2*rho_cp)

solved for the area the engine *behaves* as if it used (``A_eff``).

Kept because it is how §9-R16 was pinned down: ``A_eff/A_buf`` came out as a
constant 1.17e-2 = 1/sqrt(7160), and 7160 = ((voxel_volume + gs.EPS)/voxel_volume)^2
— the kernel divides by ``voxel_volume + gs.EPS``, and a pad's voxel_volume
(1.4e-9 m^3) is smaller than gs.EPS (1.19e-7), so the epsilon dominates. With
that term included the formula matches the engine to ~1% (see
``scripts/probe_kernel_args.py`` for the direct check).

The ``--scale-v`` / ``--scale-rcp`` flags multiply the finger's kernel arguments
in place, which is what showed the gain is linear in 1/rho_cp and (misleadingly)
only ~V^-0.33 while eps dominates.

Run: python scripts/probe_contact_heat.py [--substeps 40] [--hot 130]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate1_sensors import _drive, _np, _tactile_summary, _temps  # noqa: E402
from gate_common import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--hot", type=float, default=130.0)
    ap.add_argument("--config", default="configs/pick_hot_cylinder.yaml")
    ap.add_argument("--scale-v", type=float, default=1.0, help="multiply the finger voxel_volume (kernel-arg probe)")
    ap.add_argument("--scale-rcp", type=float, default=1.0, help="multiply the finger material rho*cp (kernel-arg probe)")
    args = ap.parse_args()

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

    md = bundle.temperatures[0].sensor._shared_metadata
    links_idx = _np(md.links_idx).astype(np.int64)
    names = {int(l.idx): l.name for e in md.solver._entities for l in e.links}
    print("sensors: " + ", ".join(f"i_s={i} link={li} {names.get(int(li))}" for i, li in enumerate(links_idx)))
    dt = float(md.solver._sim.dt)
    print(f"dt={dt} grid={_np(md.grid_size).astype(int).tolist()} voxel_volume={_np(md.voxel_volume).tolist()}")
    print(f"voxel_size={_np(md.voxel_size).tolist()}")
    print(f"aabb_min={_np(md.aabb_min).tolist()} aabb_extent={_np(md.aabb_extent).tolist()}")
    print(f"link_volume={_np(md.link_volume).tolist()}")
    print(f"types: voxel_volume={type(md.voxel_volume).__name__} link_volume={type(md.link_volume).__name__}")
    if args.scale_rcp != 1.0:
        props_t = md.link_material_properties
        mat_f = int(_np(md.link_to_material_idx).astype(np.int64)[int(links_idx[0])])
        old = _np(props_t)[3, mat_f]
        row = props_t[3]
        row[mat_f] = old * args.scale_rcp
        print(f"PATCHED finger rho_cp x{args.scale_rcp}: {old} -> {_np(props_t)[3, mat_f]}")
    if args.scale_v != 1.0:
        md.voxel_volume[:2] *= args.scale_v
        print(f"PATCHED finger voxel_volume x{args.scale_v} -> {_np(md.voxel_volume).tolist()}")

    props = _np(md.link_material_properties)
    to_mat = _np(md.link_to_material_idx).astype(np.int64)
    link_temps = md.link_temps
    vol = _np(md.voxel_volume)
    prev = None

    ctrl.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    for sub in range(args.substeps):
        bundle.step()
        fingers, obj = _temps(bundle)
        cs = md.solver.collider.collider_state
        n_c = int(_np(cs.n_contacts).reshape(-1)[0])
        order = _np(cs.contact_sort_idx)[:n_c, 0].astype(np.int64)
        la = _np(cs.contact_data.link_a)[order, 0].astype(np.int64)
        lb = _np(cs.contact_data.link_b)[order, 0].astype(np.int64)
        pen = _np(cs.contact_data.penetration)[order, 0]
        area = _np(md.contact_area_buffer)[:n_c, 0]
        depth, force, contact = _tactile_summary(bundle)
        rows = " | ".join(
            f"s{i}:{names.get(int(a), a)[:9]}-{names.get(int(b), b)[:9]}"
            f" A={ar * 1e4:.2f}cm2 p={pn * 1e3:.2f}mm"
            for i, (a, b, ar, pn) in enumerate(zip(la, lb, area, pen, strict=True))
        )
        cells = {k: _np(v).reshape(-1) for k, v in fingers.items()}
        grid = " ".join(f"{k.split('_')[-1][:5]}[{' '.join(f'{c:.2f}' for c in v)}]" for k, v in cells.items())
        print(f"sub {sub:>3} | n_c {n_c} | obj {obj:6.2f} | depth {depth * 1e3:.2f}mm | {grid}")
        print(f"        {rows}", flush=True)

        # Invert the engine's own update for the area it *behaves* as if it used:
        #   dmean*N = sum_j dt*k_eff*(T_other_j - T_cell)*A_j^2/(V^2*rho_cp)
        if prev is not None:
            for i_s in range(2):
                li = int(links_idx[i_s])
                mat = int(to_mat[li])
                if mat < 0:
                    continue
                k_s = float(props[1, mat])
                rcp = float(props[3, mat])
                arr = _np(link_temps[0])
                dmean = float(cells[list(cells)[i_s]].mean() - prev[i_s].mean())
                t_hot = float(cells[list(cells)[i_s]].max())
                drive = 0.0
                a_buf = 0.0
                for slot in np.flatnonzero((la == li) | (lb == li)):
                    other = int(lb[slot]) if int(la[slot]) == li else int(la[slot])
                    if not (0 <= other < len(to_mat)) or int(to_mat[other]) < 0:
                        continue
                    k_o = float(props[1, int(to_mat[other])])
                    k_eff = 2.0 * k_s * k_o / (k_s + k_o + 1e-12)
                    drive += k_eff * (float(arr[other]) - t_hot)
                    a_buf = max(a_buf, float(area[slot]))
                v = vol[i_s]
                a_eff = np.sqrt(dmean * 16 * v**2 * rcp / (dt * drive)) if drive > 0 else 0.0
                name = names.get(li, li)
                print(
                    f"        {name}: dmean {dmean:+.2e} C | A_buf {a_buf * 1e4:.3f} cm^2 | "
                    f"A_eff {a_eff * 1e4:.4f} cm^2 | ratio {a_eff / a_buf if a_buf else 0:.2e}",
                    flush=True,
                )
        prev = [_np(cells[list(cells)[i]]).copy() for i in range(len(cells))]

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
