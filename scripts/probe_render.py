"""Minimal renderer probe: does BatchRenderer work on this box (sm_120 / WSL2)?

Deliberately trivial on purpose (plane + box + one 128x128 camera): if this
fails, the failure is the renderer/GPU, not the task scene. Kept because it is
the fastest way to re-check after a driver, torch or genesis change.

    python scripts/probe_render.py [--renderer batch|raster|raytrace] [--res 128]

Prints, per renderer: build wall time, output dtype/shape, pixel stats
(min/max/mean, fraction of non-black pixels) and render latency over 5 frames.
An all-black image with a correct shape is the failure mode that matters: the
camera can be "working" while the lighting or the GPU path is silently dead.

Reference (RTX 5060 Laptop, WSL2, 2026-10-02): see docs/architecture.md Gate 3.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default="batch", choices=("batch", "raster", "raytrace"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    import genesis as gs

    print(f"genesis {gs.__version__}")
    gs.init(backend=gs.gpu, logging_level="warning")
    print(f"backend={gs.backend} device={gs.device}")

    renderers = {
        "batch": gs.renderers.BatchRenderer,
        "raster": gs.renderers.Rasterizer,
        "raytrace": gs.renderers.RayTracer,
    }
    opts = renderers[args.renderer]()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        renderer=opts,
        show_viewer=False,
        vis_options=gs.options.VisOptions(shadow=True),
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.4, 0.0, 0.05), fixed=False),
        material=gs.materials.Rigid(rho=1000.0),
    )
    cam = scene.add_camera(
        res=(args.res, args.res), pos=(0.9, -0.6, 0.7), lookat=(0.4, 0.0, 0.05), fov=45.0, GUI=False
    )
    print(f"camera class={type(cam).__name__}")

    t0 = time.perf_counter()
    scene.build(n_envs=1)
    t_build = time.perf_counter() - t0
    print(f"build: {t_build:.2f} s")
    cam.build()

    # Let the box drop a few steps so the frames are not all identical.
    for _ in range(10):
        scene.step()

    lat = []
    for i in range(args.frames):
        t0 = time.perf_counter()
        rgb, depth, seg, normal = cam.render(rgb=True, depth=True, segmentation=False, normal=False)
        lat.append(time.perf_counter() - t0)
        arr = np.asarray(rgb)
        d = np.asarray(depth) if depth is not None else None
        nz = float((arr[..., :3].max(axis=-1) > 0).mean())
        print(
            f"frame {i}: rgb {arr.shape} {arr.dtype} min {arr.min()} max {arr.max()} "
            f"mean {arr.mean():.2f} | non-black {nz * 100:.1f}%"
            + (f" | depth {d.shape} {float(d.min()):.3f}..{float(d.max()):.3f}" if d is not None else "")
        )
    print(f"render latency: mean {np.mean(lat) * 1e3:.1f} ms over {args.frames} frames")
    print(f"verdict: {'PASS' if nz > 0.05 else 'FAIL (image is all black)'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
