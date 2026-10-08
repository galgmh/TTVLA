#!/usr/bin/env python
"""Generate TemperatureVLA datasets: one HDF5 file per material.

    python scripts/generate.py --episodes 50 --out data/
    python scripts/generate.py --material aluminum --episodes 5 --out /tmp/x   # smoke run

One scene is built per material (material thermal properties, object size and
mass are frozen at build time — see docs/architecture.md §9), then N episodes
are rolled out with per-episode pose/yaw/temperature randomization. Successful
episodes go to <task>_<material>.hdf5, failures to <task>_<material>_failed.hdf5.

Memory stays flat: one episode lives in memory at a time and is flushed to disk
as soon as it finishes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import Timer, load_config, meminfo  # noqa: E402


def gl_guard(cfg, *, allow_unsafe: bool) -> int | None:
    """Refuse ``renderer=raster`` without software GL — it corrupts frames silently.

    Measured (docs §8): WSLg's default EGL → D3D12 (Intel) path renders a
    degenerate image (content collapsed into the last two pixel rows, depth
    pinned at 0.1) with no exception and no warning — the run would succeed and
    fill 100 MB with unusable frames. ``LIBGL_ALWAYS_SOFTWARE=1`` uses llvmpipe
    and is correct. Non-raster renderers are not affected by this trap.
    """
    if not cfg.cameras.enabled or str(cfg.cameras.renderer).lower() not in ("raster", "rasterizer"):
        return None
    if allow_unsafe or os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1":
        return None
    print("ERROR: cameras.enabled with renderer=raster needs software GL on this box.")
    print("  Without it the EGL -> WSLg -> D3D12 path renders a degenerate image")
    print("  (content in the last two pixel rows) and reports no error at all.")
    print("  Re-run as:")
    print("    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe python scripts/generate.py ...")
    print("  or pass --unsafe-gl to record whatever the current GL path produces.")
    return 2


def run_material(cfg, material: str, args, index: int, total: int) -> dict:
    """Build one scene and roll out ``args.episodes`` demos for one material."""
    from tempvla.recording import (
        EpisodeRunner,
        HDF5Writer,
        build_env_args,
        build_problem_info,
        summarize_episode,
    )
    from tempvla.scene import build_scene

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg.task_id}_{material}.hdf5"
    failed_file = out_dir / f"{cfg.task_id}_{material}_failed.hdf5" if cfg.recording.keep_failed else None

    print(f"\n=== [{index}/{total}] material={material} -> {out_file} ===")
    scene_seed = cfg.seed + 1000 * (index + 1)
    with Timer() as t_build:
        bundle = build_scene(cfg, material=material, seed=scene_seed, show_viewer=args.viewer)
    print(f"  scene built in {t_build}: " + " | ".join(
        f"object_{i} {s.shape} r={s.radius_m:.4f} m h={s.height_m:.4f} m "
        f"m={s.mass_kg:.4f} kg rho={s.mass_kg / s.volume_m3:.1f} kg/m3"
        for i, s in enumerate(bundle.object_specs)
    ))

    runner = EpisodeRunner(bundle)
    writer = HDF5Writer(
        out_file,
        task_id=cfg.task_id,
        problem_info=build_problem_info(cfg, material),
        env_args=build_env_args(cfg, bundle=bundle),
        layouts=runner.file_layouts(),
        failed_path=failed_file,
        compression=cfg.recording.compression,
        compression_opts=cfg.recording.compression_opts,
    )
    runner.writer = writer
    runner.verbose = not args.quiet

    stats = {
        "material": material,
        "file": str(out_file),
        "episodes": 0,
        "success": 0,
        "steps": [],
        "events": [],
        "windows": [],
        "heat": [],
        "ik": [],
        "errors": 0,
    }
    t_start = time.perf_counter()
    try:
        for i in range(args.episodes):
            episode_seed = cfg.seed + i
            try:
                with Timer() as t_ep:
                    res = runner.run_episode(episode_seed=episode_seed)
            except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
                stats["errors"] += 1
                print(f"  [ep {i:>4}] ERROR {type(exc).__name__}: {exc}")
                if args.traceback:
                    traceback.print_exc()
                continue

            stats["episodes"] += 1
            stats["success"] += int(res.success)
            metrics = summarize_episode(res)
            stats["steps"].append(metrics["steps"])
            stats["events"].append(metrics["events"])
            stats["windows"].append(metrics["windows"])
            stats["ik"].append(metrics["ik_err_max"])
            stats["heat"].append(metrics["heating_c"])

            done = i + 1
            rate = done / max(time.perf_counter() - t_start, 1e-6)
            eta = (args.episodes - done) / max(rate, 1e-9)
            print(
                f"  [ep {i:>4}] {'ok  ' if res.success else 'fail'} steps={res.steps:>4} "
                f"events={res.stats['events']:>2} windows={res.stats['windows']:>2} "
                f"heat={stats['heat'][-1] if stats['heat'] else 0.0:+.3f}C "
                f"ik={res.stats['ik_err_max'] * 1000:>5.2f}mm "
                f"({rate:.2f} ep/s, {res.steps / max(t_ep.s, 1e-9):.1f} steps/s, eta {eta / 60:.1f} min)"
            )
            del res  # free the episode before the next rollout
    except KeyboardInterrupt:
        print("  interrupted — closing the file with the episodes written so far")
    finally:
        writer.close()

    stats["wall_s"] = time.perf_counter() - t_start
    stats["file_mb"] = out_file.stat().st_size / 1024**2 if out_file.exists() else 0.0
    print(
        f"  done: {stats['success']}/{stats['episodes']} success"
        + (f", {stats['errors']} errors" if stats["errors"] else "")
        + f" | {stats['file_mb']:.2f} MiB | {stats['wall_s'] / 60:.1f} min"
    )
    if stats["heat"]:
        print(
            f"  mean per episode: steps {np.mean(stats['steps']):.0f} | "
            f"events {np.mean(stats['events']):.1f} | windows {np.mean(stats['windows']):.1f} | "
            f"max finger heating {np.mean(stats['heat']):+.3f} C"
        )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="task YAML (defaults to dataclass defaults)")
    parser.add_argument("--material", action="append", default=None, help="repeatable; default all in config")
    parser.add_argument("--episodes", type=int, default=50, help="episodes per material file")
    parser.add_argument("--seed", type=int, default=None, help="override cfg.seed (episode i uses seed+i)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory (default: recording.output_dir from the config)")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="no per-episode randomization line")
    parser.add_argument("--traceback", action="store_true", help="print full tracebacks for failed episodes")
    parser.add_argument("--no-validate", action="store_true", help="skip the post-run HDF5 validation")
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="record obs/agentview_rgb + obs/eye_in_hand_rgb (needs software GL, see docs §8)",
    )
    parser.add_argument("--no-cameras", action="store_true", help="force cameras.enabled=false from the config")
    parser.add_argument(
        "--unsafe-gl",
        action="store_true",
        help="allow camera rendering on a GL path known to produce degenerate frames (see docs §8)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cameras:
        cfg.cameras.enabled = True
    if args.no_cameras:
        cfg.cameras.enabled = False
    if args.out is None and cfg.cameras.enabled and cfg.recording.output_dir == "data":
        # Don't quietly overwrite the camera-less v1 dataset with a camera one.
        args.out = "data_cam"
    guard = gl_guard(cfg, allow_unsafe=args.unsafe_gl)
    if guard is not None:
        return guard
    if args.out is None:
        args.out = cfg.recording.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_steps:
        cfg.max_episode_steps = args.max_steps
    materials = args.material or list(cfg.materials)
    unknown = [m for m in materials if m not in cfg.material_properties]
    if unknown:
        print(f"ERROR: materials missing from material_properties: {unknown}")
        print(f"known: {sorted(cfg.material_properties)}")
        return 2
    # scene_seed = cfg.seed + 1000 * (index + 1) — the *position in this list*, not the
    # material. Object size and mass are drawn from that seed at build time, so a subset
    # run silently reuses the seeds of a full run's first materials: `--material copper`
    # alone gives copper the object aluminium had (measured 2026-10-03 — the two files
    # came out with identical geometry, trajectories and RGB, differing only in the
    # thermal channels). Keep one dataset to one invocation.
    subset = [m for m in materials if m not in cfg.materials]
    scene_seeds = {
        m: cfg.seed + 1000 * (i + 1) for i, m in enumerate(materials)
    }
    if len(materials) != len(cfg.materials):
        print("WARNING: this is a subset run — scene seeds are assigned by position, so they")
        print(f"  will differ from a full run's: {', '.join(f'{m}={s}' for m, s in scene_seeds.items())}")
        print(f"  a full run of {list(cfg.materials)} would give "
              + ", ".join(f"{m}={cfg.seed + 1000 * (i + 1)}" for i, m in enumerate(cfg.materials)))
        if subset:
            print(f"  (not in the config's materials list at all: {subset})")

    print("TemperatureVLA generation")
    print(f"  task        : {cfg.task_id}")
    print(f"  materials   : {', '.join(materials)}")
    print(f"  scene seeds : {', '.join(f'{m}={s}' for m, s in scene_seeds.items())}")
    print(f"  episodes    : {args.episodes} per material ({args.episodes * len(materials)} total)")
    print(f"  control     : {cfg.controller.control_hz:.0f} Hz policy / {1 / cfg.controller.sim_dt:.0f} Hz sim "
          f"(decimation {cfg.controller.decimation}) | max {cfg.max_episode_steps} steps")
    objects = cfg.object_list()
    print(f"  objects     : {len(objects)} ({', '.join(o.shape for o in objects)})"
          + ("  | target sampled per episode" if len(objects) > 1 else ""))
    for i, o in enumerate(objects):
        print(f"    object_{i}   : xy +-{o.xy_jitter_m * 1000:.0f} mm | yaw +-{np.degrees(o.yaw_jitter_rad):.0f} deg "
              f"| temp {o.temp_c[0]:.0f}-{o.temp_c[1]:.0f} C | place {tuple(o.place_xy)}")
    if cfg.cameras.enabled:
        w, h = cfg.cameras.resolution
        cams = ["agentview"] + (["eye_in_hand"] if cfg.cameras.eye_in_hand else [])
        print(f"  cameras     : {', '.join(cams)} @ {w}x{h} uint8 via {cfg.cameras.renderer} "
              f"(rendered at policy rate, ~{cfg.controller.control_hz:.0f} Hz)")
    else:
        print("  cameras     : off (set cameras.enabled=true to record RGB)")
    mem = meminfo()
    if mem:
        print(f"  memory      : {mem.get('MemAvailable', 0):.1f} GiB available of {mem.get('MemTotal', 0):.1f} GiB")
    print(f"  output      : {Path(args.out).resolve()}")

    all_stats = []
    for i, material in enumerate(materials):
        all_stats.append(run_material(cfg, material, args, i, len(materials)))

    print("\n=== summary ===")
    total_ep = sum(s["episodes"] for s in all_stats)
    total_ok = sum(s["success"] for s in all_stats)
    print(f"episodes: {total_ok}/{total_ep} success ({100 * total_ok / max(total_ep, 1):.1f}%)")
    print(f"{'material':<12} {'eps':>5} {'ok':>5} {'err':>4} {'steps':>6} {'events':>7} {'windows':>8} {'heat C':>7} {'MiB':>7}")
    for s in all_stats:
        print(
            f"{s['material']:<12} {s['episodes']:>5} {s['success']:>5} {s['errors']:>4} "
            f"{np.mean(s['steps']) if s['steps'] else 0:>6.0f} "
            f"{np.mean(s['events']) if s['events'] else 0:>7.1f} "
            f"{np.mean(s['windows']) if s['windows'] else 0:>8.1f} "
            f"{np.mean(s['heat']) if s['heat'] else 0:>+7.3f} {s['file_mb']:>7.2f}"
        )

    if not args.no_validate:
        from tempvla.tools.validate import validate_file

        print("\n=== validation ===")
        ok = True
        for s in all_stats:
            path = Path(s["file"])
            if not path.exists():
                print(f"{path}: MISSING")
                ok = False
                continue
            report = validate_file(path, strict_channels=True)
            issues = len(report.issues)
            print(f"{path.name}: {len(report.demos)} demos, {issues} issues")
            for issue in report.issues[:10]:
                print(f"    - {issue}")
            ok &= report.ok
        if not ok:
            print("validation found issues (see above)")
            return 1
    n_failed = sum(s["errors"] for s in all_stats)
    return 1 if (n_failed or total_ok == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
