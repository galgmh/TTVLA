#!/usr/bin/env python
"""Gate 2 — end-to-end scripted episodes + dataset validation.

Runs the full stack the generator will use (scene -> controller -> expert ->
runner -> HDF5) for a couple of episodes, prints what each episode actually
did (phases, contact events, burst windows, temperature rise, IK error), then
validates the written file with tempvla.tools.validate.

    python scripts/gate2_scene.py [--episodes 2] [--material aluminum] [--out /tmp/tempvla_gate2]

A pass here means generate.py can be run for real.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import Gate, Timer, fmt_floats, load_config, meminfo  # noqa: E402


def describe_episode(gate: Gate, res, index: int) -> None:
    data = res.data
    obs = data.obs
    steps = res.steps
    events = data.events or {}
    n_events = len(events.get("contact_events", {}).get("event_id", []))
    n_windows = int(data.attrs.get("num_windows", 0))

    depth = obs.get("tactile", {}).get("depth")
    force = obs.get("tactile", {}).get("force")
    temp = obs.get("temperature", {})
    contact = obs.get("contact", {}).get("link_contact")

    gate.ok(
        f"episode {index}",
        f"steps {steps} | success {res.success} | events {n_events} | burst windows {n_windows} | "
        f"ik_err_max {res.stats['ik_err_max'] * 1000:.2f} mm",
    )
    gate.ok(f"episode {index} actions", f"range [{data.actions.min():+.3f}, {data.actions.max():+.3f}]")
    if contact is not None and contact.size:
        per_link = [int(np.asarray(contact)[:, i].sum()) for i in range(np.asarray(contact).shape[1])]
        gate.ok(f"episode {index} contact steps", f"per link {per_link} of {steps}")
    if depth is not None:
        gate.ok(f"episode {index} tactile depth", f"max {float(np.asarray(depth).max()) * 1e3:.3f} mm")
    if force is not None:
        mag = np.linalg.norm(np.asarray(force).reshape(len(force), -1, 3), axis=-1)
        gate.ok(f"episode {index} tactile force", f"max {float(mag.max()):.3f} N")
    if "fingertip_grid" in temp:
        fg = np.asarray(temp["fingertip_grid"]).reshape(len(temp["fingertip_grid"]), temp["fingertip_grid"].shape[1], -1)
        rise = fg[-1].mean(axis=-1) - fg[0].mean(axis=-1)
        gate.ok(f"episode {index} finger heating", f"{fmt_floats(rise, 3)} C per finger")
    if "object" in temp:
        obj = np.asarray(temp["object"]).reshape(-1)
        gate.ok(
            f"episode {index} object temperature",
            f"start {obj[0]:.2f} C -> end {obj[-1]:.2f} C",
        )
    if obs.get("ee_pos") is not None:
        ee = np.asarray(obs["ee_pos"])
        gate.ok(f"episode {index} tcp z", f"{ee[0, 2]:.3f} -> {ee[-1, 2]:.3f} m | xy {fmt_floats(ee[-1, :2], 3)}")

    for i, win in enumerate((events.get("burst") or {}).values()):
        if i == 0:
            shapes = {k: tuple(v.shape) for k, v in win.items() if k.startswith("fingertip") or k in ("tcp_pos", "object_temp")}
            gate.ok(
                f"episode {index} burst window 0",
                f"trigger frame {int(np.asarray(win['trigger_frame']).reshape(-1)[0])} | {shapes}",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--material", default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default="/tmp/tempvla_gate2", help="output directory (recreated)")
    parser.add_argument("--keep", action="store_true", help="keep an existing output directory")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    gate = Gate("Gate 2 — end-to-end episodes + validation")
    cfg = load_config(args.config)
    if args.max_steps:
        cfg.max_episode_steps = args.max_steps
    material = args.material or (cfg.materials[0] if cfg.materials else "aluminum")

    # The shipped configs enable cameras, and gate2 has no other GL guard: without
    # software GL the rasterizer returns the same degenerate frame for every step
    # (no error, image not blank) and the only symptom is the validator's
    # "first and last frame are identical" — 20 minutes of scene build later.
    import os

    if cfg.cameras.enabled and str(cfg.cameras.renderer).lower() in ("raster", "rasterizer"):
        if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
            gate.fail(
                "software GL",
                "cameras.enabled with renderer=raster without LIBGL_ALWAYS_SOFTWARE=1 records a frozen "
                "degenerate image with no error (docs §8); re-run with "
                "LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe",
            )
            return gate.summary()

    out_dir = Path(args.out)
    if out_dir.exists() and not args.keep:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg.task_id}_{material}.hdf5"

    from tempvla.recording import (
        EpisodeRunner,
        HDF5Writer,
        build_env_args,
        build_problem_info,
    )
    from tempvla.scene import build_scene

    mem0 = meminfo().get("MemAvailable", 0.0)
    try:
        with Timer() as t_build:
            bundle = build_scene(cfg, material=material, seed=cfg.seed, show_viewer=args.viewer)
        gate.ok("scene build", f"{t_build}")
    except Exception as exc:  # noqa: BLE001
        gate.fail("scene build", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return gate.summary()

    writer = None
    try:
        runner = EpisodeRunner(bundle)  # no writer yet: layouts come from the built scene
        writer = HDF5Writer(
            out_file,
            task_id=cfg.task_id,
            problem_info=build_problem_info(cfg, material),
            env_args=build_env_args(cfg, bundle=bundle),
            layouts=runner.file_layouts(),
            failed_path=out_dir / f"{cfg.task_id}_{material}_failed.hdf5" if cfg.recording.keep_failed else None,
            compression=cfg.recording.compression,
            compression_opts=cfg.recording.compression_opts,
        )
        runner.writer = writer

        results = []
        for i in range(args.episodes):
            with Timer() as t_ep:
                res = runner.run_episode(episode_seed=cfg.seed + i)
            results.append(res)
            describe_episode(gate, res, i)
            gate.ok(f"episode {i} wall clock", f"{t_ep} ({res.steps} steps -> {res.steps / t_ep.s:.1f} steps/s)")

        mem1 = meminfo().get("MemAvailable", 0.0)
        gate.ok("writer totals", f"{writer.num_episodes} success / {writer.num_failed} failed")
        gate.ok("RAM", f"-{max(mem0 - mem1, 0.0):.2f} GiB since start")
        gate.check("at least one success", any(r.success for r in results))
        gate.check(
            "every demo saw contact",
            all(r.data.attrs.get("num_contact_events", 0) > 0 for r in results),
        )
    except Exception as exc:  # noqa: BLE001
        gate.fail("episode run", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()

    # ------------------------------------------------------------------ validate
    gate.section("dataset validation")
    if not out_file.exists():
        gate.fail("output file", f"{out_file} was not written")
        return gate.summary()
    size_mb = out_file.stat().st_size / 1024**2
    try:
        from tempvla.tools.validate import validate_file

        report = validate_file(out_file, strict_channels=True)
        print(report.summary())
        print()
        gate.check("validator", report.ok, f"{len(report.issues)} issues")
    except Exception as exc:  # noqa: BLE001
        gate.fail("validator", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    gate.ok("file size", f"{out_file} ({size_mb:.2f} MiB)")
    return gate.summary(f"Gate 2 passed — generate with: python scripts/generate.py --material {material}")


if __name__ == "__main__":
    raise SystemExit(main())
