#!/usr/bin/env python
"""Gate 4 — RGB channels in the recorded dataset (the v2 increment).

Gate 3 proved the renderer produces correct frames. This proves the frames
reach the HDF5 intact **and** that turning cameras on does not perturb the
simulation.

    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \\
        python scripts/gate4_camera_obs.py --episodes 2

Three rollouts of the same seeds are compared:

    armed      cameras on, rendering every policy step  (recorded to HDF5)
    disarmed   same scene, ``bundle.cameras = {}``      (in-memory only)
    control    the same disarmed rollout again
    off        a second build with ``cameras.enabled=false``

* **armed vs disarmed** is the check that proves rendering only observes: one
  scene, one build, the only difference is whether ``render_cameras()`` runs.
* **control** measures the baseline: how much two *identical, render-free*
  rollouts already differ. Without it, a camera effect cannot be told apart
  from the engine's own noise.
* **off** spans two builds, and checks that switching cameras off changes
  nothing a rebuild would not.

Trajectory channels (proprioception, actions, contact, temperature) must be
bit-identical in every comparison; they are the observable consequence of the
simulation and they are perfectly deterministic here.

``tactile/*`` is reported against the control rather than asserted at zero,
because of a pre-existing, camera-independent effect (measured 2026-10-03,
documented as R19 in docs/architecture.md §9): the per-taxel force readout at
the first-touch substeps is not reproducible run to run. Two identical
render-free rollouts of seed 1 on one scene differed by 4.9 N on a single taxel
at step 11 while ``ee_pos`` and ``actions`` matched to the last bit. The cause
is float32 reduction order in the GPU contact solve — the taxel depth itself
moves by ~1e-6 m and the force amplifies it. It is confined to 3-4 contact-onset
steps, does not propagate, and does not depend on the renderer.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import Gate, Timer, load_config  # noqa: E402

CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")
# Deterministic to the last bit: proprioception, actions, contact, temperature, and
# the finger pose (a pure FK readback, which is what makes the world-frame marker
# field invertible — so it must survive the armed/disarmed camera switch bit for bit).
TRAJECTORY_CHANNELS = (
    "ee_pos",
    "ee_ori",
    "gripper_states",
    "joint_states",
    "link_pose",
    "object_pose",
    "contact/link_contact",
    "contact/link_force",
    "temperature/fingertip_grid",
    "temperature/object_grid",
    "temperature/objects_grid",
    "temperature/object",
    "actions",
)
# Subject to the cross-build contact-onset transient (R19); reported, not asserted.
# The marker field is here for the same reason as force/depth — it is a physics
# readout, so the same bounded-transient rule applies.
TACTILE_CHANNELS = ("tactile/force", "tactile/depth", "tactile/elastomer")
ALL_CHANNELS = TRAJECTORY_CHANNELS + TACTILE_CHANNELS


def flatten(prefix: str, value, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten(f"{prefix}/{key}" if prefix else key, sub, out)
    else:
        out[prefix] = np.asarray(value)


def save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    img = np.asarray(arr)
    if img.ndim == 4:
        img = img[0]
    Image.fromarray(img[..., :3].astype(np.uint8)).save(path)


def diff_channels(a: dict, b: dict, channels) -> tuple[list[tuple[str, float]], list[str]]:
    """[(channel, max|diff|)] for channels that moved; plus structural errors."""
    diffs: list[tuple[str, float]] = []
    errors: list[str] = []
    for ch in channels:
        x, y = a.get(ch), b.get(ch)
        if x is None or y is None:
            errors.append(f"{ch}: present={x is not None}/{y is not None}")
        elif x.shape != y.shape:
            errors.append(f"{ch}: shape {x.shape} != {y.shape}")
        elif x.size:
            d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            if d > 0.0:
                diffs.append((ch, d))
    return diffs, errors


def fmt(diffs: list[tuple[str, float]], errors: list[str]) -> str:
    if errors:
        return "structural: " + "; ".join(errors)
    if not diffs:
        return "all channels bit-identical"
    return f"{len(diffs)} channel(s) differ: " + ", ".join(f"{c} {d:.3g}" for c, d in sorted(diffs, key=lambda kv: -kv[1]))


def run_batch(bundle, seeds: list[int], writer=None):
    """Roll out ``seeds``; returns [(flat obs, success, steps)]."""
    from tempvla.recording import EpisodeRunner

    runner = EpisodeRunner(bundle)
    runner.writer = writer
    out = []
    for seed in seeds:
        res = runner.run_episode(episode_seed=seed)
        flat: dict[str, np.ndarray] = {}
        flatten("", res.data.obs, flat)
        flat["actions"] = np.asarray(res.data.actions)
        out.append((flat, res.success, res.steps))
    return out


def main() -> int:  # noqa: C901 - a gate script is a linear checklist
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=2, help="episodes per rollout (the gate builds 2 scenes)")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--material", default=None)
    ap.add_argument("--out", default="/tmp/tempvla_gate4", help="output directory (recreated)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--unsafe-gl", action="store_true")
    args = ap.parse_args()

    gate = Gate("Gate 4 — RGB observation channels")
    cfg = load_config(args.config)
    material = args.material or (cfg.materials[0] if cfg.materials else "aluminum")
    seeds = [cfg.seed + i for i in range(args.episodes)]

    out_dir = Path(args.out)
    if out_dir.exists() and not args.keep:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"{cfg.task_id}_{material}_camera.hdf5"

    # ------------------------------------------------------------ environment
    gate.section("environment")
    if str(cfg.cameras.renderer).lower() in ("raster", "rasterizer") and not args.unsafe_gl:
        if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
            gate.fail(
                "software GL",
                "renderer=raster without LIBGL_ALWAYS_SOFTWARE=1 renders a degenerate image with no error "
                "(docs §8); re-run with LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe, or pass --unsafe-gl",
            )
            return gate.summary()
        gate.ok("software GL", f"GALLIUM_DRIVER={os.environ.get('GALLIUM_DRIVER', '(unset)')}")
    gate.info(f"config: cameras.renderer={cfg.cameras.renderer} res={tuple(cfg.cameras.resolution)} material={material}")

    from tempvla.recording import HDF5Writer, build_env_args, build_problem_info
    from tempvla.scene import build_scene

    # ------------------------------- build 1: cameras on, armed vs disarmed
    gate.section(f"build 1 — cameras enabled, {args.episodes} episodes x2 rollouts")
    cfg.cameras.enabled = True
    cfg.cameras.resolution = (args.res, args.res)
    try:
        with Timer() as t_build1:
            bundle = build_scene(cfg, material=material, seed=cfg.seed)
    except Exception as exc:  # noqa: BLE001
        gate.fail("scene build", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return gate.summary()

    gate.check("cameras registered", set(bundle.cameras) == {"agentview", "eye_in_hand"}, f"{sorted(bundle.cameras)}")
    from tempvla.recording import EpisodeRunner

    writer = HDF5Writer(
        h5_path,
        task_id=cfg.task_id,
        problem_info=build_problem_info(cfg, material),
        env_args=build_env_args(cfg, bundle=bundle),
        layouts=EpisodeRunner(bundle).file_layouts(),
        failed_path=None,
        compression=cfg.recording.compression,
        compression_opts=cfg.recording.compression_opts,
    )
    armed_cams = dict(bundle.cameras)
    try:
        with Timer() as t_armed:
            armed = run_batch(bundle, seeds, writer=writer)
        # Same scene, same build: only the render call goes away. Two disarmed
        # rollouts, so the second one gives the noise floor to compare against.
        bundle.cameras = {}
        with Timer() as t_disarmed:
            disarmed = run_batch(bundle, seeds)
            control = run_batch(bundle, seeds)
        bundle.cameras = armed_cams
    except Exception as exc:  # noqa: BLE001
        gate.fail("rollout", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return gate.summary()
    finally:
        writer.close()

    for ch in CAMERAS:
        present = all(ch in obs for obs, _, _ in armed)
        gate.check(f"obs has {ch}", present, "" if present else "missing from ObsBuilder.build()")

    # ------------------------------------- rendering only observes (the key check)
    gate.section("rendering does not perturb the simulation (one build, armed vs disarmed)")
    for i, ((a, _, n_a), (b, _, n_b), (c, _, n_c)) in enumerate(zip(armed, disarmed, control, strict=True)):
        if not (n_a == n_b == n_c):
            gate.fail(f"ep {seeds[i]} length", f"{n_a} armed / {n_b} disarmed / {n_c} control steps")
            continue
        diffs, errors = diff_channels(a, b, TRAJECTORY_CHANNELS)
        gate.check(
            f"ep {seeds[i]} trajectory identical with rendering armed/disarmed",
            not diffs and not errors,
            f"{n_a} steps | {len(TRAJECTORY_CHANNELS)} channels | {fmt(diffs, errors)}",
        )
        # Noise floor: what two render-free rollouts of the same seed already do.
        # Reported, never asserted on — a single draw from a heavy-tailed
        # distribution cannot decide anything (measured: the *render-free* pair
        # differed by 4.9 N while the armed pair differed by 0.005 N).
        c_diffs, _ = diff_channels(b, c, ALL_CHANNELS)
        a_diffs, _ = diff_channels(a, b, TACTILE_CHANNELS)
        gate.info(
            f"  ep {seeds[i]} tactile armed-vs-disarmed: {fmt(a_diffs, []) if a_diffs else 'bit-identical'} "
            f"| render-free control floor: {fmt(c_diffs, [])} (R19, not an assertion)"
        )
        # What *is* assertable: the tactile difference is a bounded transient at
        # contact onset, not a divergence. A renderer that perturbed the contact
        # solve would show a difference that persists through close/squeeze/lift.
        for ch in TACTILE_CHANNELS:
            x, y = a.get(ch), b.get(ch)
            if x is None or y is None or x.shape != y.shape:
                continue
            per_step = np.abs(x.astype(np.float64) - y.astype(np.float64)).reshape(len(x), -1).max(axis=1)
            n_steps = int((per_step > 0).sum())
            gate.check(
                f"ep {seeds[i]} {ch} differs only as a contact-onset transient",
                n_steps <= max(4, len(x) // 4) and per_step[-1] == 0.0,
                f"{n_steps}/{len(x)} steps differ (max {per_step.max():.3g}), final step identical",
            )

    # --------------------------------------- second build with cameras disabled
    gate.section(f"build 2 — same seeds, cameras.enabled=false")
    cfg.cameras.enabled = False
    try:
        with Timer() as t_build2:
            bundle_off = build_scene(cfg, material=material, seed=cfg.seed)
        with Timer() as t_off:
            off = run_batch(bundle_off, seeds)
    except Exception as exc:  # noqa: BLE001
        gate.fail("no-camera build", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return gate.summary()
    del bundle, bundle_off
    gate.ok("builds", f"cameras on {t_build1}, cameras off {t_build2}")

    for i, ((a, _, n_a), (b, _, n_b)) in enumerate(zip(armed, off, strict=True)):
        if n_a != n_b:
            gate.fail(f"ep {seeds[i]} length", f"{n_a} steps with cameras vs {n_b} without")
            continue
        t_diffs, t_errors = diff_channels(a, b, TRAJECTORY_CHANNELS)
        gate.check(
            f"ep {seeds[i]} trajectory matches a no-camera build",
            not t_diffs and not t_errors,
            f"{n_a} steps | {len(TRAJECTORY_CHANNELS)} channels | {fmt(t_diffs, t_errors)}",
        )
        # Expected to be non-empty at the contact-onset step (R19) — and expected
        # to be exactly the same whether or not cameras exist, which the two
        # builds cannot distinguish, so it is only reported here.
        k_diffs, k_errors = diff_channels(a, b, TACTILE_CHANNELS)
        gate.info(f"  ep {seeds[i]} tactile across builds: {fmt(k_diffs, k_errors)}")

    # ------------------------------------------------------ file round trip
    gate.section("HDF5 round trip")
    import h5py

    from tempvla.tools.validate import validate_file

    if not h5_path.exists():
        gate.fail("file written", f"{h5_path} does not exist")
        return gate.summary()
    gate.ok("file written", f"{h5_path.stat().st_size / 1024**2:.2f} MiB")

    with h5py.File(h5_path, "r") as f:
        demos = sorted((k for k in f["data"] if k.startswith("demo_")), key=lambda s: int(s.split("_")[1]))
        gate.check("demo count", len(demos) == args.episodes, f"{demos}")
        for name in demos:
            g = f["data"][name]
            T = int(g.attrs["num_samples"])
            for ch in CAMERAS:
                if ch not in g["obs"]:
                    gate.fail(f"{name}/obs/{ch}", "missing from the file")
                    continue
                d = g["obs"][ch]
                gate.check(
                    f"{name}/obs/{ch}",
                    tuple(d.shape) == (T, args.res, args.res, 3) and d.dtype == np.uint8,
                    f"{tuple(d.shape)} {d.dtype} (want ({T},{args.res},{args.res},3) uint8, T = num_samples)",
                )
                frames = np.asarray(d)
                nonblack = [float((frames[j].max(axis=-1) > 0).mean()) for j in range(len(frames))]
                changed = float((np.abs(frames[-1].astype(np.int16) - frames[0]) > 8).mean())
                gate.ok(
                    f"  {ch} pixels",
                    f"non-black min {min(nonblack) * 100:.1f}% | mean {frames.mean():.1f} | "
                    f"changed {changed * 100:.1f}% vs frame 0",
                )
                gate.check(f"{name}/{ch} frames are not blank", min(nonblack) > 0.05, f"darkest {min(nonblack) * 100:.1f}% non-black")
                gate.check(f"{name}/{ch} is not a frozen buffer", not np.array_equal(frames[0], frames[-1]), "first != last frame")
                for j in (0, len(frames) - 1):
                    save_png(out_dir / f"{name}_{ch}_{j:03d}.png", frames[j])

    report = validate_file(h5_path, strict_channels=True)
    gate.check("validate_file", report.ok, f"{len(report.issues)} issues" + (f": {report.issues[:3]}" if report.issues else ""))

    # --------------------------------------------------------------- the cost
    gate.section("cost")
    cam_steps = sum(s for _, _, s in armed)
    # t_disarmed covers two rollouts (disarmed + control); halve it for a like-for-like rate.
    disarmed_s = t_disarmed.s / 2.0
    gate.ok(
        "render overhead",
        f"{t_armed.s:.1f} s for {cam_steps} policy steps armed vs {disarmed_s:.1f} s disarmed (same build) "
        f"-> {(t_armed.s - disarmed_s) / max(cam_steps, 1) * 1e3:.0f} ms/step, "
        f"{(t_armed.s - disarmed_s) / max(disarmed_s, 1e-9) * 100:+.1f}%",
    )
    cam_mb = h5_path.stat().st_size / 1024**2
    gate.ok("file size", f"{cam_mb:.2f} MiB for {args.episodes} eps ({cam_mb / max(args.episodes, 1):.2f} MiB/ep)")
    gate.info(f"  PNGs (dumped from the file, not from memory): {out_dir}")
    return gate.summary("Gate 4 passed — RGB reaches obs/ intact and rendering only observes.")


if __name__ == "__main__":
    raise SystemExit(main())
