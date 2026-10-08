#!/usr/bin/env python
"""Gate 3 — cameras: agentview + eye-in-hand rendering on this box.

Builds the real task scene with ``cameras.enabled=true``, runs a scripted
expert episode, and renders both cameras along the way. What it decides:

* does the chosen renderer produce a correct, non-black 128² frame at all;
* is the eye-in-hand camera actually attached (its world pose must follow the
  hand, and its image must change as the arm moves);
* what one render pass costs on this machine, and what that projects to for a
  full 200-episode dataset;
* how much VRAM/RAM the cameras add.

    python scripts/gate3_render.py [--renderer raster|batch] [--res 128]
                                   [--steps 8] [--out /tmp/tempvla_gate3]

Renderer notes (measured 2026-10-02, RTX 5060 Laptop / WSL2):

* ``batch``  — BatchRenderer needs a Vulkan loader **and** an ICD. This WSL2
  image has neither (no ``libvulkan.so.1`` anywhere, no ``/usr/share/vulkan``),
  so madrona aborts the process with "Couldn't find vulkan loader". Installable
  with ``sudo apt install libvulkan1`` (pulls ``mesa-vulkan-drivers``).
* ``raster`` — the Rasterizer goes through EGL; under WSLg that lands on Mesa's
  D3D12 (Intel) driver, which collapses the image into the last two pixel rows.
  ``LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`` renders correctly and is
  what the recorded numbers use. The gate warns when it is not set.
"""

from __future__ import annotations

import argparse
import ctypes.util
import gzip
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import Gate, Timer, fmt_floats, load_config, meminfo  # noqa: E402

CAMERAS = ("agentview", "eye_in_hand")
_ICD_DIRS = (Path("/usr/share/vulkan/icd.d"), Path("/etc/vulkan/icd.d"))


def vulkan_status() -> tuple[bool, str]:
    """(usable, description) for madrona's Vulkan requirement."""
    loader = ctypes.util.find_library("vulkan")
    if loader is None:
        for cand in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/wsl/lib/libvulkan.so.1"):
            if Path(cand).exists():
                loader = cand
                break
    icds = [p.name for d in _ICD_DIRS if d.is_dir() for p in sorted(d.glob("*.json"))]
    ok = loader is not None and bool(icds)
    return ok, f"loader={loader or 'MISSING'} icds={icds or 'NONE'}"


def gl_renderer_string(bundle) -> str:
    """GL_RENDERER the Rasterizer actually draws through (best effort)."""
    try:
        from OpenGL import GL

        renderer = bundle.scene.visualizer.rasterizer._renderer  # pyrender OffscreenRenderer
        renderer.make_current()
        try:
            name = GL.glGetString(GL.GL_RENDERER)
        finally:
            renderer.make_uncurrent()
        return name.decode() if isinstance(name, bytes) else f"({name})"
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return f"(unavailable: {type(exc).__name__}: {exc})"


def image_stats(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    nonblack = float((a[..., :3].max(axis=-1) > 0).mean())
    return (
        f"{a.shape} {a.dtype} min {a.min()} max {a.max()} mean {a.mean():.1f} "
        f"non-black {nonblack * 100:.1f}%"
    )


def save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    img = np.asarray(arr)
    if img.ndim == 4:
        img = img[0]
    Image.fromarray(img[..., :3].astype(np.uint8)).save(path)


def gzip_size(arr: np.ndarray, level: int = 4) -> int:
    return len(gzip.compress(np.ascontiguousarray(arr).tobytes(), compresslevel=level))


def main() -> int:  # noqa: C901 - a gate script is a linear checklist
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--renderer", default="raster", choices=("raster", "batch", "raytrace"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--material", default=None)
    ap.add_argument("--steps", type=int, default=8, help="policy steps to run the expert for")
    ap.add_argument("--render-every", type=int, default=2, help="render a frame every N policy steps")
    ap.add_argument("--out", default="/tmp/tempvla_gate3", help="PNG output directory (recreated)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    gate = Gate("Gate 3 — camera rendering")
    cfg = load_config(args.config)
    cfg.cameras.enabled = True
    cfg.cameras.renderer = args.renderer
    cfg.cameras.resolution = (args.res, args.res)
    material = args.material or (cfg.materials[0] if cfg.materials else "aluminum")

    out_dir = Path(args.out)
    if out_dir.exists() and not args.keep:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ environment
    gate.section("environment")
    if args.renderer == "batch":
        # Pre-flight: madrona aborts the whole process (no exception to catch)
        # when the loader is missing, so stop here instead of core-dumping.
        ok, desc = vulkan_status()
        gate.info(f"vulkan: {desc}")
        if not ok:
            gate.fail(
                "vulkan loader/ICD",
                "madrona would abort the process at build; install with: sudo apt install libvulkan1 "
                "(pulls mesa-vulkan-drivers) — or re-run with --renderer raster",
            )
            return gate.summary()
        gate.ok("vulkan loader/ICD", desc)
    if args.renderer == "raster":
        soft = os.environ.get("LIBGL_ALWAYS_SOFTWARE")
        if soft != "1":
            gate.warn(
                "LIBGL_ALWAYS_SOFTWARE",
                "not set — under WSLg the D3D12 (Intel) path collapses the image to the last 2 rows; "
                "re-run with LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe",
            )
        else:
            gate.ok("software GL", f"GALLIUM_DRIVER={os.environ.get('GALLIUM_DRIVER', '(unset)')}")

    from tempvla.scene import build_scene

    mem0 = meminfo().get("MemAvailable", 0.0)

    # ------------------------------------------------------------------ build
    gate.section("scene build")
    try:
        with Timer() as t_build:
            bundle = build_scene(cfg, material=material, seed=cfg.seed)
        gate.ok("scene build", f"{t_build}")
    except Exception as exc:  # noqa: BLE001
        gate.fail("scene build", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return gate.summary()

    gpu_free0 = None
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu_free0 = free / 1024**3
            gate.info(f"VRAM after build: free {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
    except Exception:  # noqa: BLE001
        pass

    gate.ok("gl renderer", gl_renderer_string(bundle))
    gate.check("cameras registered", set(bundle.cameras) == set(CAMERAS), f"{sorted(bundle.cameras)}")
    if not bundle.cameras:
        return gate.summary()
    for name, cam in bundle.cameras.items():
        attached = getattr(cam, "_attached_link", None)
        gate.info(
            f"  {name}: res {tuple(cam.res)} fov {cam.fov} near {cam.near} far {cam.far} "
            f"attached={getattr(attached, 'name', None) if attached is not None else None} pos {fmt_floats(cam.pos, 3)}"
        )
    res_ok = len({tuple(c.res) for c in bundle.cameras.values()}) == 1
    gate.check("cameras share one resolution", res_ok, "BatchRenderer requires this")

    # ----------------------------------------------------------- still frame
    gate.section("still frame")
    try:
        with Timer() as t_first:
            frames = bundle.render_cameras()
        gate.ok("first render pass", f"{t_first}")
    except Exception as exc:  # noqa: BLE001
        gate.fail("first render", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return gate.summary()

    for name in CAMERAS:
        arr = frames.get(name)
        if arr is None:
            gate.fail(f"{name} frame", "missing from render_cameras()")
            continue
        gate.ok(f"{name} frame", image_stats(arr))
        save_png(out_dir / f"{name}_step000.png", arr)
        nonblack = float((np.asarray(arr)[..., :3].max(axis=-1) > 0).mean())
        gate.check(f"{name} is not all black", nonblack > 0.05, f"non-black {nonblack * 100:.1f}%")

    # How large is the object on screen? A grasp policy needs more than a pixel.
    # The mask holds compact seg ids, not link indices: map each back to its
    # (entity_idx, link_idx) key to find the object's pixels.
    try:
        seg = np.asarray(bundle.cameras["agentview"].render(rgb=False, depth=False, segmentation=True)[2])
        seg = seg.reshape(-1, args.res, args.res)[0]
        key_to_id = {k: v for v, k in bundle.scene.visualizer.context.seg_color_map.idxc_map.items()}
        target = (int(bundle.obj.idx), int(bundle.obj.links[0].idx))
        obj_val = key_to_id.get(target)
        px = int((seg == obj_val).sum()) if obj_val is not None else 0
        gate.ok(
            "object footprint (agentview)",
            f"{px} px of {args.res * args.res} ({px / args.res**2 * 100:.2f}%, ~{np.sqrt(px):.0f}x{np.sqrt(px):.0f}) "
            f"| seg level {bundle.scene.options.vis.segmentation_level} | object key {target} -> seg id {obj_val}",
        )
        gate.check("object is resolvable in agentview", px >= 100, f"{px} px (want >= 100, i.e. ~10x10)")
    except Exception as exc:  # noqa: BLE001
        gate.warn("object footprint", f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------------------- the episode
    gate.section(f"expert episode with rendering ({args.steps} policy steps)")
    from tempvla.recording import EpisodeRunner
    from tempvla.scene import sample_episode

    runner = EpisodeRunner(bundle)
    rng = np.random.default_rng(cfg.seed)
    ep = sample_episode(cfg, rng)
    bundle.reset_episode(ep)
    bundle.step()
    runner.controller.reset()
    runner.expert.reset(bundle, rng)

    camera_pos: dict[str, list[np.ndarray]] = {n: [] for n in CAMERAS}
    shots: dict[str, list[np.ndarray]] = {n: [] for n in CAMERAS}
    render_lat: list[float] = []
    compressed: list[tuple[int, int]] = []
    phases: list[str] = []
    import time as _time

    try:
        for step in range(args.steps):
            obs = runner.obs_builder.build()
            action, info = runner.expert.act(obs)
            phases.append(str(info.get("phase")))
            runner.controller.step(action)
            for _ in range(cfg.controller.decimation):
                bundle.step()
            if step % args.render_every and step != args.steps - 1:
                continue
            t0 = _time.perf_counter()
            frames = bundle.render_cameras()
            render_lat.append(_time.perf_counter() - t0)
            for name in CAMERAS:
                if name not in frames:
                    continue
                shots[name].append(np.asarray(frames[name]))
                camera_pos[name].append(np.asarray(bundle.cameras[name].pos, dtype=np.float64).reshape(-1)[:3])
                save_png(out_dir / f"{name}_step{step:03d}.png", frames[name])
        gate.ok("episode ran", f"phases {phases}")
    except Exception as exc:  # noqa: BLE001
        gate.fail("episode render", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return gate.summary()

    # --------------------------------------------------- attachment / motion
    gate.section("attachment and motion")
    for name in CAMERAS:
        pos = camera_pos[name]
        moved = float(np.linalg.norm(pos[-1] - pos[0])) if len(pos) > 1 else 0.0
        if name == "eye_in_hand":
            gate.check("eye_in_hand follows the hand", moved > 0.01, f"camera moved {moved * 1e3:.1f} mm over the episode")
        else:
            gate.info(f"  {name} camera moved {moved * 1e3:.1f} mm (should be a fixed camera)")
            gate.check("agentview is static", moved < 1e-6, f"moved {moved:.2e} m")

    for name in CAMERAS:
        imgs = shots[name]
        if len(imgs) < 2:
            continue
        first, last = imgs[0].astype(np.int16), imgs[-1].astype(np.int16)
        diff = np.abs(last - first).mean()
        changed = float((np.abs(last - first).max(axis=-1) > 8).mean())
        gate.check(
            f"{name} image changes as the arm moves",
            changed > 0.02,
            f"mean|d| {diff:.2f} | changed pixels {changed * 100:.1f}%",
        )
    for name in CAMERAS:
        for i, img in enumerate(shots[name][:1] + shots[name][-1:]):
            kind = "first" if i == 0 else "last"
            raw = gzip_size(img)
            compressed.append((img[0].size * 3, raw))
            gate.info(f"  {name} {kind} frame: {raw / 1024:.1f} KiB gzip")

    # ------------------------------------------------------------- throughput
    gate.section("throughput and projection")
    if render_lat:
        lat = np.asarray(render_lat)
        gate.ok(
            "render pass (both cameras)",
            f"mean {lat.mean() * 1e3:.0f} ms  median {np.median(lat) * 1e3:.0f} ms  "
            f"min {lat.min() * 1e3:.0f}  max {lat.max() * 1e3:.0f}  n {len(lat)}",
        )
        per_episode_s = lat.mean() * 34  # the expert averages 34 policy steps
        gate.ok(
            "projected cost",
            f"{per_episode_s:.1f} s/episode -> {per_episode_s * 200 / 60:.1f} min for 200 episodes "
            f"(rendering only; v1 took 944 s total)",
        )
    if compressed:
        raw_mean = np.mean([r for r, _ in compressed])
        gz_mean = np.mean([g for _, g in compressed])
        ep_mb = gz_mean * 34 / 1024**2
        gate.ok(
            "projected size",
            f"{gz_mean / 1024:.1f} KiB/frame gzip ({gz_mean / raw_mean * 100:.0f}% of raw) -> "
            f"{ep_mb:.2f} MiB/episode for 34 steps x 1 camera -> {ep_mb * 2 * 200:.0f} MiB for 200 episodes x 2 cameras",
        )

    mem1 = meminfo().get("MemAvailable", 0.0)
    gate.ok("RAM", f"-{max(mem0 - mem1, 0.0):.2f} GiB since start")
    try:
        import torch

        if torch.cuda.is_available() and gpu_free0 is not None:
            free, total = torch.cuda.mem_get_info()
            gate.ok(
                "VRAM",
                f"free {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB "
                f"(peak allocated {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB)",
            )
    except Exception:  # noqa: BLE001
        pass

    gate.ok("PNGs", str(out_dir))
    return gate.summary("Gate 3 passed — cameras render correctly; see docs/architecture.md §8 for the cost numbers.")


if __name__ == "__main__":
    raise SystemExit(main())
