#!/usr/bin/env python
"""Contact sheet for eye-in-hand camera poses, rendered on the real task scene.

Choosing a wrist-camera mount is geometric guesswork until you look at the
frames: ``panda_hand``'s own mesh spans z -0.026..0.066 m in the hand frame
(the fingers run z 0.058..0.112), so a camera parked behind the flange stares
straight into the hand. This script builds the scene once and renders the same
grasp moment from a list of candidate hand-frame poses, so the choice is made
by looking rather than by arithmetic.

    python scripts/probe_camera_pose.py [--steps 11] [--out /tmp/tempvla_camposes]

Candidate poses come from ``--poses`` (a JSON file) or the built-in defaults.
Writes ``<index>_<label>.png`` plus a printed summary of what each pose sees
(non-black fraction, pixel count of the object via the segmentation mask).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import load_config  # noqa: E402

# (label, pos, lookat, up) in the panda_hand frame, metres.
#
# up=(0,0,-1) puts image-up along the hand's -z. The hand's +z is the approach
# axis, so with the gripper pointing down at the table the -z direction is world
# up: the object and workspace land below the gripper in the frame, the way a
# wrist camera sees a real grasp. With up=(0,0,1) the image is rolled 180 deg
# (table at the top), which no real wrist camera looks like.
UP = (0.0, 0.0, -1.0)
DEFAULT_POSES = [
    ("side_mid", (0.10, 0.0, -0.02), (0.0, 0.0, 0.10), UP),
    ("side_mid_out", (0.12, 0.0, -0.02), (0.0, 0.0, 0.10), UP),
    ("side_fwd", (0.10, 0.0, 0.02), (0.0, 0.0, 0.10), UP),
    ("side_out_fwd", (0.12, 0.0, 0.01), (0.0, 0.0, 0.10), UP),
    ("side_far", (0.14, 0.0, -0.03), (0.0, 0.0, 0.09), UP),
    ("side_far_out", (0.16, 0.0, -0.02), (0.0, 0.0, 0.10), UP),
    ("side_y", (0.12, -0.03, 0.0), (0.0, 0.0, 0.10), UP),
    ("robosuite_like", (0.05, 0.0, 0.05), (0.0, 0.0, 0.13), UP),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--material", default="aluminum")
    ap.add_argument("--steps", type=int, default=11, help="expert steps before the sweep (11 lands on 'close')")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--poses", default=None, help="JSON file: [[label, pos, lookat, up], ...]")
    ap.add_argument("--agentview", action="store_true", help="also render the agentview at the same moment")
    ap.add_argument("--out", default="/tmp/tempvla_camposes")
    args = ap.parse_args()

    from PIL import Image

    import genesis as gs
    from tempvla.recording import EpisodeRunner
    from tempvla.scene import build_scene, sample_episode

    cfg = load_config(args.config)
    cfg.cameras.enabled = True
    cfg.cameras.renderer = "raster"
    cfg.cameras.eye_in_hand = False  # the sweep adds its own probe camera below
    cfg.cameras.resolution = (args.res, args.res)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = DEFAULT_POSES
    if args.poses:
        poses = [tuple(p) for p in json.loads(Path(args.poses).read_text())]

    bundle = build_scene(cfg, material=args.material, seed=cfg.seed)
    hand = bundle.robot.get_link(bundle.hand_link_name)

    # Drive the expert to the grasp phase so every candidate sees the same moment.
    runner = EpisodeRunner(bundle)
    rng = np.random.default_rng(cfg.seed)
    ep = sample_episode(cfg, rng)
    bundle.reset_episode(ep)
    bundle.step()
    runner.controller.reset()
    runner.expert.reset(bundle, rng)
    for _ in range(args.steps):
        obs = runner.obs_builder.build()
        action, _info = runner.expert.act(obs)
        runner.controller.step(action)
        for _ in range(cfg.controller.decimation):
            bundle.step()

    if args.agentview:
        rgb = np.asarray(bundle.cameras["agentview"].render(rgb=True)[0]).reshape(args.res, args.res, 3)
        Image.fromarray(rgb).save(out_dir / "agentview.png")

    # The sweep needs a camera it can teleport: reuse the agentview one, which is
    # not attached to anything, and set its world pose per candidate.
    probe_cam = bundle.cameras["agentview"]
    key_to_id = {k: v for v, k in bundle.scene.visualizer.context.seg_color_map.idxc_map.items()}
    obj_seg_id = key_to_id.get((int(bundle.obj.idx), int(bundle.obj.links[0].idx)))
    # How much of the frame the robot's own hand eats: a wrist camera that mostly
    # shows the gripper is a wrist camera that mostly shows nothing.
    self_seg_ids = [
        v
        for key, v in key_to_id.items()
        if isinstance(key, tuple) and key[0] == int(bundle.robot.idx) and key[1] in {l.idx for l in bundle.robot.links}
    ]
    n_px = args.res * args.res

    print(f"hand link {bundle.hand_link_name} pos {np.round(np.asarray(hand.get_pos().cpu()).reshape(-1)[:3], 3)}")
    results = []
    for i, (label, pos, lookat, up) in enumerate(poses):
        try:
            offset_T = gs.utils.geom.pos_lookat_up_to_T(
                np.asarray(pos, dtype=np.float64), np.asarray(lookat, dtype=np.float64), np.asarray(up, dtype=np.float64)
            )
            link_pos = np.asarray(hand.get_pos(relative=False).detach().cpu()).reshape(-1)[:3]
            link_quat = np.asarray(hand.get_quat(relative=False).detach().cpu()).reshape(-1)[:4]
            link_T = np.asarray(gs.utils.geom.trans_quat_to_T(link_pos, link_quat), dtype=np.float64)
            world_T = link_T @ offset_T
            probe_cam.set_pose(transform=world_T)
            rgb, _, seg, _ = probe_cam.render(rgb=True, segmentation=True)
            rgb = np.asarray(rgb).reshape(args.res, args.res, 3)
            seg = np.asarray(seg).reshape(args.res, args.res)
            px = int((seg == obj_seg_id).sum()) if obj_seg_id is not None else -1
            robot_px = int(np.isin(seg, self_seg_ids).sum()) if self_seg_ids else -1
            Image.fromarray(rgb).save(out_dir / f"{i}_{label}.png")
            results.append((label, px, robot_px))
            print(
                f"  [{i}] {label:16s} pos {np.round(pos, 3)} | object {px:5d} px ({px / n_px * 100:4.1f}%) "
                f"| robot {robot_px:5d} px ({robot_px / n_px * 100:4.1f}%)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {label:16s} FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if results:
        best = max(results, key=lambda r: r[1])
        print(f"\nmost object pixels: {best[0]} ({best[1]} px) — PNGs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
