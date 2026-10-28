#!/usr/bin/env python
"""Generate ThermTacVLA datasets: one HDF5 file per job.

    python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml --episodes 50
    python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml \
        --episodes 1 --no-cameras --out /tmp/x   # smoke run

One file per invocation: ``material_list`` gives one material per body. One
scene is built per job (material thermal properties, object size and mass are
frozen at build time — see docs/architecture.md §9 in git HEAD), then N episodes are rolled
out with per-episode pose/yaw/temperature randomization. Successful episodes go
to <task>_<tag>.hdf5, failures to <task>_<tag>_failed.hdf5.

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
from sim.expert.pipeline import PIPELINE, target_kind  # noqa: E402


def gl_guard(cfg, *, allow_unsafe: bool) -> int | None:
    """Refuse ``renderer=raster`` without software GL — it corrupts frames silently.

    Measured (docs §8): WSLg's default EGL → D3D12 (Intel) path renders a
    degenerate image (content collapsed into the last two pixel rows, depth
    pinned at 0.1) with no exception and no warning — the run would succeed and
    fill 100 MB with unusable frames. ``LIBGL_ALWAYS_SOFTWARE=1`` uses llvmpipe
    and is correct. Non-raster renderers are not affected by this trap.
    """
    from sim.obs import RENDERER

    if not cfg["cameras"]["enabled"] or str(RENDERER).lower() not in ("raster", "rasterizer"):
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


def run_job(cfg, args) -> dict:
    """Build one scene and roll out ``args.episodes`` demos for one job.

    The scene is ``material_list`` -- one entry per object: a file can hold
    bodies of different materials and is named after all of them.
    """
    from sim.recording import (
        EpisodeRunner,
        HDF5Writer,
        build_env_args,
        build_problem_info,
        summarize_episode,
    )
    from sim.scene import build_scene

    materials = list(cfg["material_list"])
    tag = "_".join(materials)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg['task_id']}_{tag}.hdf5"
    failed_file = (
        out_dir / f"{cfg['task_id']}_{tag}_failed.hdf5"
        if cfg["recording"]["keep_failed"]
        else None
    )

    print(f"\n=== job={tag} -> {out_file} ===")
    scene_seed = cfg["seed"] + 1000
    with Timer() as t_build:
        bundle = build_scene(
            cfg,
            seed=scene_seed,
            show_viewer=args.viewer,
        )
    print(f"  scene built in {t_build}: " + " | ".join(
        f"object_{i} {s.shape} r={s.radius_m:.4f} m h={s.height_m:.4f} m "
        f"m={s.mass_kg:.4f} kg rho={s.mass_kg / s.volume_m3:.1f} kg/m3"
        for i, s in enumerate(bundle.object_specs)
    ))

    runner = EpisodeRunner(bundle)
    writer = HDF5Writer(
        out_file,
        task_id=cfg["task_id"],
        problem_info=build_problem_info(cfg, materials, pipeline=PIPELINE),
        env_args=build_env_args(cfg, bundle=bundle),
        layouts=runner.file_layouts(),
        failed_path=failed_file,
        compression=cfg["recording"]["compression"],
        compression_opts=cfg["recording"]["compression_opts"],
    )
    runner.writer = writer
    runner.verbose = not args.quiet

    stats = {
        "material": tag,
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
            episode_seed = cfg["seed"] + i
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
    parser.add_argument("--config", default=None, help="task YAML (default: sim/configs/thermal_recognition/pick_cold_can.yaml)")
    parser.add_argument("--episodes", type=int, default=50, help="episodes in the output file")
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
        cfg["cameras"]["enabled"] = True
    if args.no_cameras:
        cfg["cameras"]["enabled"] = False
    guard = gl_guard(cfg, allow_unsafe=args.unsafe_gl)
    if guard is not None:
        return guard
    if args.out is None:
        args.out = cfg["recording"]["output_dir"]
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_steps:
        cfg["max_episode_steps"] = args.max_steps

    materials = list(cfg["material_list"])
    unknown = [m for m in materials if m not in cfg["material_properties"]]
    if unknown:
        print(f"ERROR: materials missing from material_properties: {unknown}")
        print(f"known: {sorted(cfg['material_properties'])}")
        return 2
    tag = "_".join(materials)
    # HDF5Writer opens with mode="w": an existing dataset is truncated, not merged.
    # The output dir is version-scoped with no camera-suffix convention any more,
    # so replacing files is allowed — but never silent.
    out_dir = Path(args.out)
    out_file = out_dir / f"{cfg['task_id']}_{tag}.hdf5"
    if out_file.exists():
        print(f"NOTE: existing file under {out_dir} will be REPLACED "
              f"(truncate-and-write): {out_file.name}")

    scene_seed = cfg["seed"] + 1000
    print("ThermTacVLA generation")
    print(f"  task        : {cfg['task_id']}")
    print(f"  file        : {tag}")
    print(f"  scene seed  : {scene_seed}")
    print(f"  episodes    : {args.episodes}")
    print(f"  control     : {cfg['controller']['control_hz']:.0f} Hz policy / "
          f"{1 / cfg['controller']['sim_dt']:.0f} Hz sim "
          f"(decimation {cfg['controller']['decimation']}) | max {cfg['max_episode_steps']} steps")
    objects = cfg["objects"]
    print(f"  objects     : {len(objects)} ({', '.join(str(o['shape']) for o in objects)})"
          f"  | target profile anchor object {cfg['target_index']}")
    for i, o in enumerate(objects):
        own = f"{o['temp_c'][0]:.0f}-{o['temp_c'][1]:.0f} C"
        print(f"    object_{i}   : xy +-{o['xy_jitter_m'] * 1000:.0f} mm | "
              f"yaw +-{np.degrees(o['yaw_jitter_rad']):.0f} deg "
              f"| temp profile {own} | place {tuple(o['place_xy'])}")
    if cfg["temperature_groups"]:
        print(
            f"  temp groups : {cfg['temperature_groups']} -> profiles permuted per "
            f"episode; target is the body that receives object {cfg['target_index']}'s profile"
        )
    print("  materials   : " + ", ".join(materials))
    expert = cfg["expert"]
    if expert["temperature_judge_enabled"] and expert["approach"] == "place_in_tray":
        lo, hi = expert["temperature_judge_threshold_c"]
        lo_txt = "-inf" if lo is None else f"{lo:g}"
        hi_txt = "+inf" if hi is None else f"{hi:g}"
        kind = target_kind(expert["temperature_judge_threshold_c"])
        print(
            f"  expert      : PutInTrayExpert | temperature judge over bodies "
            f"{list(expert['temperature_candidate'])} (random order) | {kind} target: tip in "
            f"[{lo_txt}, {hi_txt}] C -> carry now; no tip heading there -> release -> next | "
            f"all candidates judged, every match carried"
        )
    elif expert["approach"] == "place_in_tray":
        print("  expert      : PutInTrayExpert | no judge (pinned target) -> carry to tray")
    else:
        print(f"  expert      : approach {expert['approach']!r}")
    if PIPELINE.transit_clearance_m > 0:
        print(f"  transit     : APPROACH rises to table + tallest body + "
              f"{PIPELINE.transit_clearance_m:.2f} m before any lateral motion")
    tray = cfg["tray"]
    if tray["enabled"]:
        print(f"  tray        : inner {tray['inner_x_m'] * 1000:.0f}x{tray['inner_y_m'] * 1000:.0f} mm, "
              f"rim {tray['rim_m'] * 1000:.0f} mm, "
              f"at table-relative {tuple(tray['place_xy'])} | success by {cfg['success_criterion']}")
    cameras = cfg["cameras"]
    if cameras["enabled"]:
        from sim.obs import RENDERER

        w, h = cameras["resolution"]
        cams = ["agentview"] + (["eye_in_hand"] if cameras["eye_in_hand"] else [])
        print(f"  cameras     : {', '.join(cams)} @ {w}x{h} uint8 via {RENDERER} "
              f"(rendered at policy rate, ~{cfg['controller']['control_hz']:.0f} Hz)")
    else:
        print("  cameras     : off (set cameras.enabled=true to record RGB)")
    mem = meminfo()
    if mem:
        print(f"  memory      : {mem.get('MemAvailable', 0):.1f} GiB available of {mem.get('MemTotal', 0):.1f} GiB")
    print(f"  output      : {out_dir.resolve()}")

    stats = run_job(cfg, args)

    print("\n=== summary ===")
    total_ep = stats["episodes"]
    total_ok = stats["success"]
    print(f"episodes: {total_ok}/{total_ep} success ({100 * total_ok / max(total_ep, 1):.1f}%)")
    print(f"{'material':<16} {'eps':>5} {'ok':>5} {'err':>4} {'steps':>6} {'events':>7} {'windows':>8} {'heat C':>7} {'MiB':>7}")
    print(
        f"{stats['material']:<16} {stats['episodes']:>5} {stats['success']:>5} {stats['errors']:>4} "
        f"{np.mean(stats['steps']) if stats['steps'] else 0:>6.0f} "
        f"{np.mean(stats['events']) if stats['events'] else 0:>7.1f} "
        f"{np.mean(stats['windows']) if stats['windows'] else 0:>8.1f} "
        f"{np.mean(stats['heat']) if stats['heat'] else 0:>+7.3f} {stats['file_mb']:>7.2f}"
    )

    if not args.no_validate:
        from sim.tools.validate import validate_file

        print("\n=== validation ===")
        path = Path(stats["file"])
        if not path.exists():
            print(f"{path}: MISSING")
            return 1
        report = validate_file(path, strict_channels=True)
        print(f"{path.name}: {len(report.demos)} demos, {len(report.issues)} issues")
        for issue in report.issues[:10]:
            print(f"    - {issue}")
        if not report.ok:
            print("validation found issues (see above)")
            return 1
    errors = stats["errors"]
    return 1 if (errors or total_ok == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
