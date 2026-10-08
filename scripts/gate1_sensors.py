#!/usr/bin/env python
"""Gate 1 — sensors, layouts, conventions, TCP calibration, contact + heat.

Builds the real Franka + gripper scene once and checks everything the
generation code assumes, in escalating order:

  1. scene build (engine, URDF, sensors) + memory/time cost
  2. robot introspection: joints, dofs, hand/finger links
  3. tactile layouts derived from the URDF (probe grids, pad boxes, normals)
  4. live sensor readouts and their shapes
  5. quaternion convention cross-check against genesis.utils.geom
  6. TCP (grip-center) calibration -> recommended robot.tcp_offset
  7. press test: drive the TCP onto the object, close the gripper, and verify
     that contact, tactile depth/force, and heat flow from the hot object into
     the finger pads all register

    python scripts/gate1_sensors.py [--config configs/x.yaml] [--material aluminum] [--viewer]

The press test is the expensive part (~5-10 s of sim, a few hundred steps).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import Gate, Timer, fmt_floats, load_config, meminfo  # noqa: E402


def _np(value) -> np.ndarray:
    # ``collider_state`` fields are quadrants Tensors (no ``.detach``), engine
    # buffers (contact_area_buffer, link_temps, ...) are plain torch tensors.
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=np.float64)
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64)


# --------------------------------------------------------------------- sections


def check_build(gate: Gate, cfg, material: str, viewer: bool):
    from tempvla.scene import build_scene

    mem0 = meminfo().get("MemAvailable", 0.0)
    try:
        with Timer() as t:
            bundle = build_scene(cfg, material=material, seed=cfg.seed, show_viewer=viewer)
    except Exception as exc:  # noqa: BLE001
        gate.fail("scene build", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None
    mem1 = meminfo().get("MemAvailable", 0.0)
    spec = bundle.object_spec
    gate.ok(
        "scene build",
        f"{t} | material={material} r={spec.radius_m:.4f} h={spec.height_m:.4f} m={spec.mass_kg:.4f} kg "
        f"| RAM -{max(mem0 - mem1, 0.0):.2f} GiB",
    )
    return bundle


def check_robot(gate: Gate, bundle) -> None:
    gate.section("2. robot introspection")
    gate.ok("joints", ", ".join(bundle.joint_names))
    gate.ok("links", ", ".join(bundle.link_names))
    gate.ok("arm dofs", f"{bundle.arm_dofs.tolist()} (local idx)")
    gate.ok("finger dofs", f"{bundle.finger_dofs.tolist()}")
    expected = len(bundle.cfg.robot.arm_joint_names) == len(bundle.arm_dofs) and len(bundle.cfg.robot.finger_joint_names) == len(bundle.finger_dofs)
    gate.check("configured joint names resolved", expected)
    gate.ok("hand link", bundle.hand_link_name)
    gate.ok("finger links", ", ".join(bundle.finger_link_names))


def check_layouts(gate: Gate, bundle) -> None:
    from tempvla.sensors import summarize

    gate.section("3. tactile layouts (URDF-derived)")
    grids = {m: bundle.cfg.tactile.grid_for(m) for m in bundle.cfg.tactile.modalities}
    for modality, lays in bundle.tactile_layouts.items():
        gate.ok(
            "per-modality grid",
            f"{modality}: {grids[modality][0]}x{grids[modality][1]} = {lays[0].num_probes} probes/finger",
        )
    # Pad geometry is grid-independent (same pad, denser sampling), so both the
    # summary and the face/extent checks run once per *distinct grid*.
    for grid in dict.fromkeys(grids.values()):
        modality = next(m for m, g in grids.items() if g == grid)
        lays = bundle.tactile_layouts[modality]
        for line in summarize(lays).splitlines():
            gate.ok(f"layout[{modality}]", line)
    src = bundle.layout_info.get("closure_source")
    if src:
        detail = f"closure axis from {src}"
        if "fallback" in src:
            gate.warn("closure axis", detail + " — pad geometry missing, verify carefully")
        else:
            gate.ok("closure axis", detail)
    pad_tol = float(bundle.layout_info.get("pad_tol_m", 1.5e-3))
    probe_offset = float(bundle.cfg.tactile.probe_offset_m)
    for grid in dict.fromkeys(grids.values()):
        modality = next(m for m, g in grids.items() if g == grid)
        lays = bundle.tactile_layouts[modality]
        n_expected = int(np.prod(grid))
        for lay in lays:
            gate.check(f"{modality}/{lay.link_name}: probes == grid", lay.num_probes == n_expected, f"{lay.num_probes}")
            # Each probe sits at (local pad surface + standoff). pad_box_origin is
            # the reference plane through the pad's outermost mesh vertices, so a
            # probe whose nearest slab vertex lies pad_tol below that plane lands
            # up to pad_tol *behind* the reference — never in front of the standoff.
            offset = lay.probe_local_pos - lay.pad_box_origin
            along = (offset * lay.inner_dir).sum(axis=1)
            lo, hi = probe_offset - pad_tol, probe_offset
            gate.check(
                f"{lay.link_name}: probes on the pad face",
                bool(np.all(along >= lo - 1e-5) and np.all(along <= hi + 1e-5)),
                f"{along.min() * 1e3:.2f}..{along.max() * 1e3:.2f} mm from the reference plane "
                f"(want {lo * 1e3:.2f}..{hi * 1e3:.2f} = standoff minus pad-slab scatter)",
            )
            # The grid must not spill off the pad onto the recessed finger body.
            axis = int(np.argmax(np.abs(lay.inner_dir)))
            tang = [a for a in range(3) if a != axis]
            inside = all(
                bool(np.all(np.abs(offset[:, a]) <= 0.5 * lay.pad_box_size[a] + 1e-6)) for a in tang
            )
            gate.check(
                f"{lay.link_name}: probes within pad extents",
                inside,
                f"pad {np.round(lay.pad_box_size[tang] * 1e3, 1).tolist()} mm | "
                f"probe span "
                f"{np.round((lay.probe_local_pos[:, tang].max(0) - lay.probe_local_pos[:, tang].min(0)) * 1e3, 1).tolist()} mm",
            )
    inner = [lay.inner_dir for lay in bundle.layouts]
    if len(inner) == 2:
        gate.check(
            "fingers face each other",
            float(np.dot(inner[0], inner[1])) < -0.99,
            f"dot={float(np.dot(inner[0], inner[1])):+.3f}",
        )


def check_sensor_reads(gate: Gate, bundle) -> None:
    from tempvla.sensors import read_tactile, read_temperature

    gate.section("4. live sensor readouts")
    tactile = read_tactile(bundle.tactile, temporal_reduction=bundle.cfg.tactile.temporal_reduction)
    for h in bundle.tactile:
        arr = _np(tactile[h.name])
        gate.ok(
            f"{h.name}",
            f"shape {arr.shape} dtype {arr.dtype} range [{arr.min():.3e}, {arr.max():.3e}]",
        )
    links = read_tactile(bundle.link_contacts, temporal_reduction="last")
    for h in bundle.link_contacts:
        arr = _np(links[h.name])
        gate.ok(f"{h.name}", f"shape {arr.shape} value {arr.reshape(-1)[:4]}")

    temps = read_temperature(bundle.temperatures, ground_truth=bundle.cfg.temperature.record_ground_truth)
    for h in bundle.temperatures:
        arr = _np(temps[h.name])
        gate.check(f"{h.name}: shape", arr.shape == (bundle.n_envs, *h.grid_size), f"{arr.shape}")
        gate.ok(f"{h.name}", f"mean {arr.mean():.3f} C | range [{arr.min():.3f}, {arr.max():.3f}] C")

    from tempvla.sensors import read_link_temperatures

    lt = read_link_temperatures(bundle.temperatures[-1])
    if lt is None:
        gate.warn("link_temperatures", "empty (simulate_all_link_temperatures off?)")
    else:
        arr = _np(lt)
        gate.ok("link_temperatures", f"shape {arr.shape} | {fmt_floats(arr.reshape(-1)[:6], 2)} C ...")


def check_rotations(gate: Gate) -> None:
    """Our pure-numpy quaternion helpers vs genesis.utils.geom (wxyz, Hamilton)."""
    from tempvla.rotations import quat_mul, quat_rotate, quat_to_matrix, quat_to_rotvec, rotvec_to_quat

    gate.section("5. quaternion conventions vs genesis.utils.geom")
    try:
        import genesis.utils.geom as gu
    except Exception as exc:  # noqa: BLE001
        gate.fail("import genesis.utils.geom", f"{exc}")
        return

    rng = np.random.default_rng(0)
    quats = rng.normal(size=(16, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    vecs = rng.normal(size=(16, 3))

    # vector rotation: gu.transform_by_quat expects (N,3) and (N,4)
    ref = _np(gu.transform_by_quat(vecs.copy(), quats.copy()))
    ours = np.stack([quat_rotate(q, v) for q, v in zip(quats, vecs)])
    gate.check("quat_rotate == transform_by_quat", bool(np.allclose(ref, ours, atol=1e-6)),
               f"max err {np.abs(ref - ours).max():.2e}")

    # Matrix form (obs.py rotates the whole marker block with it; readers invert
    # a world-frame file with it): v_world = R @ v_link, so v_link = R.T @ v_world.
    mats = np.stack([quat_to_matrix(q) for q in quats])
    via_matrix = np.einsum("nij,nj->ni", mats, vecs)
    gate.check("quat_to_matrix == quat_rotate", bool(np.allclose(ref, via_matrix, atol=1e-6)),
               f"max err {np.abs(ref - via_matrix).max():.2e}")
    gate.check("quat_to_matrix is orthonormal", bool(np.allclose(mats @ mats.transpose(0, 2, 1), np.eye(3), atol=1e-9)),
               f"max |RR^T - I| {np.abs(mats @ mats.transpose(0, 2, 1) - np.eye(3)).max():.2e}")

    # composition order: transform_quat_by_quat(v, u) == u (x) v in our convention
    u, v = quats[0], quats[1]
    ref_c = _np(gu.transform_quat_by_quat(v.copy(), u.copy())).reshape(4)
    uv = quat_mul(u, v)
    vu = quat_mul(v, u)
    sign = float(np.sign(np.dot(ref_c, uv)))
    gate.check("transform_quat_by_quat(v,u) == u (x) v", bool(np.allclose(ref_c, sign * uv, atol=1e-6)),
               f"u(x)v err {np.abs(ref_c - sign * uv).max():.2e} | v(x)u err {np.abs(ref_c - sign * vu).max():.2e}")

    # axis-angle round trips
    ref_rv = _np(gu.quat_to_rotvec(quats.copy()))
    ours_rv = np.stack([quat_to_rotvec(q) for q in quats])
    gate.check("quat_to_rotvec matches", bool(np.allclose(ref_rv, ours_rv, atol=1e-6)),
               f"max err {np.abs(ref_rv - ours_rv).max():.2e}")
    ref_q = _np(gu.rotvec_to_quat(ours_rv.copy()))
    # q and -q are the same rotation: align the sign per row before comparing.
    sign = np.sign((ref_q * quats).sum(axis=1, keepdims=True))
    sign[sign == 0.0] = 1.0
    err_q = float(np.abs(sign * ref_q - quats).max())
    gate.check("rotvec_to_quat matches", bool(err_q < 1e-6), f"max err {err_q:.2e} (sign-aligned)")


def calibrate_tcp(gate: Gate, bundle) -> np.ndarray:
    """Grip-center in the hand link frame, from the live pad geometry."""
    from tempvla.rotations import quat_conj, quat_rotate

    gate.section("6. TCP (grip center) calibration")
    hand_pos, hand_quat = bundle.hand_pose()
    centers = []
    for lay in bundle.layouts:
        link = bundle.robot.get_link(lay.link_name)
        lpos = _np(link.get_pos()).reshape(-1)[:3]
        lquat = _np(link.get_quat()).reshape(-1)[:4]
        # Probe plane center = pad-face center + inner_dir * (half extent*0 +
        # probe offset); on a mesh pad the face lives at the pad plane itself.
        half = 0.5 * float((lay.pad_box_size * np.abs(lay.inner_dir)).sum())
        face_local = lay.pad_box_origin + lay.inner_dir * (half + bundle.cfg.tactile.probe_offset_m)
        centers.append(lpos + quat_rotate(lquat, face_local))
    grip_world = 0.5 * (centers[0] + centers[1])
    grip_hand = quat_rotate(quat_conj(hand_quat), grip_world - hand_pos)
    spacing = float(np.linalg.norm(centers[0] - centers[1]))
    gate.ok("pad face centers (world)", f"{fmt_floats(centers[0])} / {fmt_floats(centers[1])}")
    gate.ok("closed-gap pad spacing (probe planes)", f"{spacing:.4f} m")
    gate.ok("grip center (world)", fmt_floats(grip_world))
    gate.ok("grip center (hand frame)", fmt_floats(grip_hand))
    cfg_offset = np.asarray(bundle.cfg.robot.tcp_offset, dtype=np.float64)
    err = float(np.linalg.norm(grip_hand - cfg_offset))
    if err < 0.005:
        gate.ok("tcp_offset", f"configured {fmt_floats(cfg_offset)} matches ({err * 1000:.1f} mm)")
    else:
        gate.warn(
            "tcp_offset",
            f"configured {fmt_floats(cfg_offset)} differs by {err * 1000:.1f} mm from measured "
            f"{fmt_floats(grip_hand)} — update robot.tcp_offset",
        )
    return grip_hand


# ------------------------------------------------------------------- press test


def _temps(bundle) -> tuple[dict[str, float], float, dict[int, float]]:
    """(finger means, targeted-object mean, {object_index: mean})."""
    from tempvla.sensors import read_temperature

    raw = read_temperature(bundle.temperatures, ground_truth=bundle.cfg.temperature.record_ground_truth)
    fingers = {
        h.link_name: float(_np(raw[h.name]).mean()) for h in bundle.temperatures if h.kind == "finger"
    }
    objects = {
        int(h.object_index): float(_np(raw[h.name]).mean())
        for h in bundle.temperatures
        if h.kind == "object" and h.object_index is not None
    }
    obj = objects.get(int(bundle.target_index), float("nan"))
    return fingers, obj, objects


def _tactile_summary(bundle) -> tuple[float, float, bool, float]:
    """(max depth m, max force magnitude N, any link contact, max |marker disp| m).

    The last value is the elastomer channel's peak displacement norm — the
    signal the height-map channel is synthesized from. It is 0.0 for a config
    that does not enable the marker modality, which is why the assertion built
    on it is conditioned on the modality being present.
    """
    from tempvla.sensors import read_tactile

    depth = 0.0
    force = 0.0
    marker = 0.0
    for h in bundle.tactile:
        arr = _np(read_tactile([h], temporal_reduction="last")[h.name])
        if h.modality == "depth":
            depth = max(depth, float(arr.max()))
        elif h.modality in ("force", "force_torque"):
            force = max(force, float(np.linalg.norm(arr.reshape(-1, 3), axis=-1).max()))
        elif h.modality == "elastomer":
            marker = max(marker, float(np.linalg.norm(arr.reshape(-1, 3), axis=-1).max()))
    contact = any(
        bool(_np(read_tactile([h], temporal_reduction="last")[h.name]).any()) for h in bundle.link_contacts
    )
    return depth, force, contact, marker


def _thermal_diag(bundle) -> tuple[str, float]:
    """Contact-heat scale report; returns (text, worst amplification factor).

    The engine's per-voxel contact update is explicit —
    ``dT = dt * k_eff * (T_other - T_cell) * A^2 / (V^2 * rho_cp)`` — so the
    amplification ``c`` printed here must stay well below 1. ``A`` is
    ``max(contact_polygon_area, pi * contact_depth_weight * penetration)``:
    at 4.6 mm penetration the fallback alone is 14 cm^2, which drives c >> 1
    and pins cells at the engine's +-1000 C clamp (docs §9-R16).

    ``A`` is taken per sensor (only contacts that involve that sensor's link):
    the scene-wide max also covers pairs like object-vs-table, whose ~14 cm^2
    base contact is a real, cold heat sink but says nothing about the pads.

    ``V`` must be the volume the kernel actually divides by, which is
    ``voxel_volume[i_s] + gs.EPS`` (``gs.EPS`` = float32 eps = 1.192e-7, see
    ``genesis.init``). That term is not cosmetic: a pad's ``voxel_volume`` is
    1.4e-9 m^3 (its AABB is reconstructed from two corners, so the y extent
    collapses), i.e. *below* ``gs.EPS``, so the pad's effective cell volume is
    the epsilon — 85x its own voxel and, by coincidence, within 2% of the
    geometric 1.17e-7 m^3 cell. Using the raw field would overstate ``c`` by
    7160x and fail every configuration (docs §9-R16).
    """
    import genesis as gs

    from tempvla.sensors import read_temperature

    handles = list(bundle.temperatures)
    if not handles:
        return "", 0.0
    md = handles[0].sensor._shared_metadata
    dt = float(md.solver._sim.dt)
    eps = float(gs.EPS)
    # links_idx holds scene-global link indices (bundle.link_names is robot-local).
    names = {
        int(link.idx): link.name for entity in md.solver._entities for link in entity.links
    }
    area_col = _np(md.contact_area_buffer)[:, 0]
    # Per-sensor contact area: the buffer is indexed by sorted contact slot, so
    # map each slot back to its link pair (same indexing as the heat kernel).
    cs = md.solver.collider.collider_state
    n_c = int(_np(cs.n_contacts).reshape(-1)[0])
    order = _np(cs.contact_sort_idx)[:n_c, 0].astype(np.int64)
    la = _np(cs.contact_data.link_a)[order, 0].astype(np.int64)
    lb = _np(cs.contact_data.link_b)[order, 0].astype(np.int64)
    a_col = area_col[:n_c]
    a_all = float(a_col.max()) if a_col.size else 0.0
    props = _np(md.link_material_properties)  # rows: base T, k, emissivity, rho*cp
    if props.ndim == 2 and props.shape[0] != 4:
        props = props.T  # engine allocates (n_props, n_mats), default field is (n_mats, n_props)
    to_mat = _np(md.link_to_material_idx).astype(np.int64)
    links_idx = _np(md.links_idx).astype(np.int64)
    vols = _np(md.voxel_volume)
    raw = read_temperature(handles, ground_truth=bundle.cfg.temperature.record_ground_truth)
    link_temps = _np(md.link_temps)[0] if md.link_temps.numel() else None

    c_max = 0.0
    parts = [f"A_all {a_all * 1e4:.3f} cm^2"]
    for i, h in enumerate(handles):
        li = int(links_idx[i]) if i < len(links_idx) else -1
        name = names.get(li, h.link_name)
        arr = _np(raw[h.name])
        span = f"{arr.min():+.1f}..{arr.max():+.1f} C"
        if link_temps is not None and 0 <= li < link_temps.size:
            span += f" Tlink {link_temps[li]:+.1f}"
        mat = int(to_mat[li]) if 0 <= li < to_mat.size else -1
        if mat < 0 or i >= len(vols):
            parts.append(f"{name}: no material | {span}")
            continue
        k_s, rcp, v = float(props[1, mat]), float(props[3, mat]), float(vols[i]) + eps
        # Mirror the kernel exactly: k_eff from the *partner's* material, and
        # contacts whose partner has no material are skipped by the engine too
        # (an all-materials max would rate the object against itself, k=205).
        hit = (la == li) | (lb == li) if li >= 0 else np.zeros(n_c, dtype=bool)
        a_s, k_eff = 0.0, 0.0
        for slot in np.flatnonzero(hit):
            other = int(lb[slot]) if int(la[slot]) == li else int(la[slot])
            if not (0 <= other < len(to_mat)) or int(to_mat[other]) < 0:
                continue
            k_o = float(props[1, int(to_mat[other])])
            k_eff = max(k_eff, float(2.0 * k_s * k_o / (k_s + k_o + 1e-12)))
            a_s = max(a_s, float(a_col[slot]))
        c = dt * k_eff * a_s**2 / (v**2 * rcp) if a_s > 0.0 else 0.0
        c_max = max(c_max, c)
        v_eps = abs(float(vols[i])) < eps
        parts.append(
            f"{name}: A {a_s * 1e4:.3f} cm^2 V {v * 1e9:.2f} mm^3"
            f"{' (eps)' if v_eps else ''} k {k_s:g} c {c:.2e} | {span}"
        )
    return " | ".join(parts), c_max


def _max_link_contact(bundle) -> tuple[float, str]:
    """Largest net contact force over the robot's links, and the link's name."""
    force = _np(bundle.robot.get_links_net_contact_force()).reshape(-1, 3)
    if force.size == 0:
        return 0.0, "-"
    mags = np.linalg.norm(force, axis=-1)
    i = int(np.argmax(mags))
    name = bundle.link_names[i] if i < len(bundle.link_names) else f"link{i}"
    return float(mags[i]), name


def _drive(bundle, ctrl, target_pos, target_quat, grip: float, *, max_steps: int, tol: float = 0.004,
           trace: bool = False):
    """Closed-loop normalized-action drive toward a TCP target.

    With ``trace``, one row per control step: measured TCP, target error,
    setpoint-vs-measured distance (a runaway setpoint shows up here), the IK
    residual, the per-step joint motion (a branch flip shows as a huge jump),
    and the largest link contact force (a jam shows up here).
    """
    from tempvla.actions import normalized_action_to, tcp_pose

    cc = bundle.cfg.controller
    err = float("inf")
    q_prev = None
    for step in range(max_steps):
        pos, quat = tcp_pose(bundle)
        err = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - pos))
        if trace:
            print(
                f"    t{step:>3} | tcp {fmt_floats(pos)} | err {err * 1e3:>7.2f} mm | "
                f"setpoint {np.linalg.norm(ctrl.target_pos - pos) * 1e3:>6.2f} mm",
                end="",
            )
        if err < tol:
            if trace:
                print("  <- converged", flush=True)
            break
        action = normalized_action_to(
            pos, quat, target_pos, target_quat, grip,
            pos_scale_m=cc.pos_scale_m, rot_scale_rad=cc.rot_scale_rad,
        )
        info = ctrl.step(action)
        q_now = bundle.get_qpos()[bundle.arm_dofs]
        if trace:
            dq = 0.0 if q_prev is None else float(np.abs(q_now - q_prev).max())
            f_max, f_name = _max_link_contact(bundle)
            print(
                f" | ik {info.ik_pos_err_m * 1e3:>6.2f} mm {np.degrees(info.ik_rot_err_rad):>5.1f} deg"
                f" | dq {np.degrees(dq):>6.2f} deg | F {f_max:>6.2f} N {f_name}",
                flush=True,
            )
        q_prev = q_now
        for _ in range(cc.decimation):
            bundle.step()
    return err, step + 1


def press_test(gate: Gate, bundle, hot_c: float, *, close_steps: int, trace: bool = False) -> None:
    from tempvla.actions import DeltaPoseController
    from tempvla.expert import TOP_DOWN_QUAT

    gate.section("7. press test — contact, tactile, heat flow")
    cfg = bundle.cfg
    from tempvla.scene import scripted_episode

    bundle.reset_episode(scripted_episode(cfg, xy_offset=np.zeros(2), yaw=0.0, temp_c=hot_c))
    bundle.step()
    fingers0, obj0, objs0 = _temps(bundle)
    gate.ok(
        "object temperature grids (one per object)",
        ", ".join(
            f"object_{i}={v:.2f} C{' (target)' if i == bundle.target_index else ''}"
            for i, v in sorted(objs0.items())
        ),
    )
    gate.check(
        "every object temperature grid reads back",
        len(objs0) == len(bundle.objects) and all(np.isfinite(v) for v in objs0.values()),
        f"{len(objs0)} of {len(bundle.objects)} grids",
    )
    gate.ok("seeded object temperature", f"{obj0:.2f} C (requested {hot_c:.1f} C)")
    gate.check(
        "temperature seed reaches the read cache",
        abs(obj0 - hot_c) < 2.0,
        f"delta {obj0 - hot_c:+.3f} C",
    )
    gate.ok("finger temperatures (start)", ", ".join(f"{k}={v:.2f}" for k, v in fingers0.items()))

    ctrl = DeltaPoseController(bundle)
    ctrl.reset()
    obj_pos, _ = bundle.object_pose()

    err, steps = _drive(
        bundle, ctrl, obj_pos + np.array([0.0, 0.0, 0.08]), TOP_DOWN_QUAT, -1.0, max_steps=40, trace=trace
    )
    gate.check("approach TCP above object", err < 0.01, f"err {err * 1000:.1f} mm in {steps} steps")

    err, steps = _drive(bundle, ctrl, obj_pos, TOP_DOWN_QUAT, -1.0, max_steps=40, trace=trace)
    gate.check("descend TCP to object center", err < 0.01, f"err {err * 1000:.1f} mm in {steps} steps")
    depth_free, force_free, contact_free, marker_free = _tactile_summary(bundle)
    gate.ok(
        "pre-close readings",
        f"depth {depth_free:.2e} m | force {force_free:.3f} N | contact {contact_free} "
        f"| marker {marker_free * 1e6:.2f} um",
    )

    # Close the gripper and hold, sampling contact + temperature as it happens.
    trajectory = []
    for step in range(close_steps):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # hold pose, close gripper
        ctrl.step(action)
        for _ in range(cfg.controller.decimation):
            bundle.step()
        if step % 5 == 0 or step == close_steps - 1:
            fingers, obj, _ = _temps(bundle)
            depth, force, contact, marker = _tactile_summary(bundle)
            trajectory.append((step, depth, force, contact, marker, obj, fingers))
            if trace:
                text, c_max = _thermal_diag(bundle)
                print(f"    thermal | {text} | c_max {c_max:.2e}", flush=True)
    last = trajectory[-1]
    gate.check("contact registered after closing", bool(last[3]), f"link contact={last[3]}")
    # depth/force probe assertions are conditional: the pair config runs an
    # elastomer-only stack (docs §5.2), where those channels are simply not
    # recorded and their absence is not a failure.
    modalities = tuple(cfg.tactile.modalities)
    if "depth" in modalities:
        gate.check("tactile depth > 0", last[1] > 0.0, f"max depth {last[1] * 1e3:.3f} mm")
    else:
        gate.ok("tactile depth", "modality disabled")
    if "force" in modalities or "force_torque" in modalities:
        gate.check("tactile force > 0", last[2] > 0.0, f"max force {last[2]:.3f} N")
    else:
        gate.ok("tactile force", "modality disabled")
    if "elastomer" in modalities:
        gate.check(
            "elastomer marker field responds",
            last[4] > 0.0,
            f"peak |disp| {last[4] * 1e6:.2f} um",
        )
    else:
        gate.ok("elastomer marker field", "modality disabled")

    print("  step | depth(mm) | force(N) | contact | marker(um) | obj(C) | " + " | ".join(f"{k}(C)" for k in fingers0))
    for step, depth, force, contact, marker, obj, fingers in trajectory:
        row = " | ".join(f"{v:.2f}" for v in fingers.values())
        print(f"  {step:>4} | {depth * 1e3:>9.4f} | {force:>8.3f} | {str(contact):>7} | "
              f"{marker * 1e6:>10.2f} | {obj:>6.2f} | {row}")

    fingers1, obj1 = last[6], last[5]
    heating = {k: fingers1[k] - fingers0[k] for k in fingers0}
    gate.ok("finger heating", ", ".join(f"{k}={v:+.3f} C" for k, v in heating.items()))
    # Threshold is deliberately low: the commanded k sets the pad's rate, and a
    # legitimate patch can be much smaller than the ~9 cm^2 this press happens
    # to make (docs §9-R16).
    gate.check(
        "heat flows from hot object into the pads",
        max(heating.values()) > 1e-3,
        f"max rise {max(heating.values()):+.3f} C over {close_steps} steps",
    )
    _, c_max = _thermal_diag(bundle)
    gate.check(
        "contact-heat coupling stable",
        c_max < 0.5,
        f"worst amplification c={c_max:.2e} per substep (want <0.5; docs §9-R16)",
    )
    # Independent divergence signature: an unstable gain shows up as voxels
    # slamming between the engine's +-1000 C clamps even when the c estimate
    # (which trusts the area buffer) misses the cause.
    from tempvla.sensors import read_temperature

    raw_all = read_temperature(
        list(bundle.temperatures), ground_truth=bundle.cfg.temperature.record_ground_truth
    )
    lo = min(float(_np(v).min()) for v in raw_all.values())
    hi = max(float(_np(v).max()) for v in raw_all.values())
    gate.check(
        "temperature field unclamped",
        hi < 900.0 and lo > -900.0,
        f"range {lo:+.1f}..{hi:+.1f} C (clamp is +-1000)",
    )

    pos_now, _ = bundle.object_pose()
    gate.ok("object displacement", f"{np.linalg.norm(pos_now - obj_pos) * 1e3:.2f} mm during grasp")


# ------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="task YAML (defaults to dataclass defaults)")
    parser.add_argument("--material", default=None, help="object material key (default: first in config)")
    parser.add_argument("--hot", type=float, default=None, help="object temperature for the press test (C)")
    parser.add_argument("--close-steps", type=int, default=40, help="policy steps to hold the closed gripper")
    parser.add_argument("--viewer", action="store_true", help="open the interactive viewer")
    parser.add_argument("--skip-press", action="store_true", help="skip the press test")
    parser.add_argument("--trace", action="store_true", help="print a per-step drive trace in the press test")
    args = parser.parse_args()

    gate = Gate("Gate 1 — sensors / conventions / TCP / contact")
    cfg = load_config(args.config)
    material = args.material or (cfg.materials[0] if cfg.materials else "aluminum")
    hot = args.hot if args.hot is not None else float(np.mean(cfg.object_list()[0].temp_c))

    bundle = check_build(gate, cfg, material, args.viewer)
    if bundle is None:
        return gate.summary()
    try:
        check_robot(gate, bundle)
        check_layouts(gate, bundle)
        check_sensor_reads(gate, bundle)
        check_rotations(gate)
        calibrate_tcp(gate, bundle)
        if not args.skip_press:
            press_test(gate, bundle, hot, close_steps=args.close_steps, trace=args.trace)
    except Exception as exc:  # noqa: BLE001
        gate.fail("gate1 body", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    return gate.summary("Gate 1 passed — proceed to: python scripts/gate2_scene.py")


if __name__ == "__main__":
    raise SystemExit(main())
