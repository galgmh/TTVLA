#!/usr/bin/env python
"""Closed-loop evaluation of a policy in the ThermTacVLA Genesis scene.

The loop is the generator's, minus the scripted expert: build the scene once,
sample one episode per seed, reset, run the policy at policy rate through the
same ``DeltaPoseController``, and score from the scene. The success rule is the
runner's ``place_in_tray`` check without its expert-phase condition (a learned
policy has no expert): the targeted body at rest inside the cavity, on the
cavity floor, pads open, held for ``stable_steps``.

    # harness smoke tests (no policy checkpoint needed)
    python experiment/eval_closed_loop.py --policy zero --episodes 1
    python experiment/eval_closed_loop.py --policy expert --episodes 4

    # pi0.5 policy server (openpi) on the network
    python experiment/eval_closed_loop.py --policy remote --episodes 20 \
        --remote-host 192.168.1.20 --remote-port 8000 --state-mode proprio16

Two metrics are reported, and they answer different questions:

* **task success**   -- the *target* body is in the tray (the cold can).
* **manipulation**   -- *any* body is in the tray (did the policy pick and
  place something at all?).

For the pure-vision baseline the second should be high and the first should be
near the chance rate the scene allows (three visually identical cans). That gap
is the baseline's whole point.

``--permute-slots`` is a diagnostic, not the training distribution: it moves
the bodies to a fresh permutation of the home slots at evaluation time. Use it
to measure how much of a fixed-slot policy's score came from the target's
position. See experiment/README.md for why the real fix is a generation-side
change.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiment.common import assemble_state_from_obs  # noqa: E402
from experiment.policy_interface import (  # noqa: E402
    ExpertPolicy,
    PolicyObservation,
    RandomPolicy,
    RemotePolicy,
    ZeroPolicy,
)
from sim.config import language_templates, load_task_config  # noqa: E402
from sim.expert.actions import DeltaPoseController  # noqa: E402
from sim.expert.pipeline import PIPELINE  # noqa: E402
from sim.obs import ObsBuilder  # noqa: E402
from sim.recording.meta import language_instruction  # noqa: E402
from sim.scene import build_scene, sample_episode  # noqa: E402
from sim.sensors.temperature import SENSOR_TIME_CONSTANT  # noqa: E402


# ------------------------------------------------------------------- scoring


def _bodies_in_tray(bundle: Any, tol: float) -> list[int]:
    """Object indices whose world AABB bottom rests on the cavity floor inside it.

    Same geometry the runner uses (``_placed_ok``): the engine's world AABB, so
    a body that settled tilted still counts. The gripper state is handled by
    the caller -- a body held at exactly this height also rests on the floor.
    """
    tray = bundle.tray
    if tray is None:
        return []
    out: list[int] = []
    for i, entity in enumerate(bundle.objects):
        aabb = np.asarray(entity.get_AABB().detach().cpu(), dtype=np.float64).reshape(-1, 3)
        lo, hi = aabb[0], aabb[1]
        if abs(float(lo[2]) - float(tray.floor_z)) > float(tol):
            continue
        if tray.contains_xy(0.5 * (float(lo[0]) + float(hi[0])), 0.5 * (float(lo[1]) + float(hi[1]))):
            out.append(i)
    return out


def _placed(
    bundle: Any,
    grip: float,
    *,
    tol: float,
    open_tol: float,
    target_only: bool,
) -> bool:
    if grip > float(open_tol):
        return False
    bodies = _bodies_in_tray(bundle, tol)
    if target_only:
        return int(bundle.target_index) in bodies
    return bool(bodies)


# -------------------------------------------------------------------- episode


def _permute_slots(ep: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Give every body a fresh home slot (diagnostic only).

    The slot *set* is unchanged, so the pairwise separation the generator
    checked still holds; only who stands where changes. Ground-truth object
    poses and the temperature sensor geometry are build-time, so this is a
    vision-only diagnostic -- do not read the thermal channels' numbers as
    physical under permutation.
    """
    out = dict(ep)
    slots = np.asarray(ep["slots"], dtype=np.float64)
    out["slots"] = slots[rng.permutation(len(slots))]
    return out


def _policy_action(
    policy: Any,
    obs: dict[str, Any],
    *,
    state_mode: str,
    instruction: str,
    step: int,
) -> np.ndarray:
    if isinstance(policy, ExpertPolicy):
        return policy.act_raw(obs)
    view = PolicyObservation(
        agentview=np.asarray(obs["agentview_rgb"], dtype=np.uint8),
        wrist=(
            None
            if "eye_in_hand_rgb" not in obs
            else np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
        ),
        state=assemble_state_from_obs(obs, state_mode),
        instruction=instruction,
        step=step,
    )
    action = np.asarray(policy.act(view), dtype=np.float64).reshape(-1)
    if action.size < 7:
        raise ValueError(f"policy returned {action.size} action dims, need 7")
    return action[:7]


def _make_policy(args: argparse.Namespace, cfg: dict[str, Any], bundle: Any) -> Any:
    if args.policy == "zero":
        return ZeroPolicy()
    if args.policy == "random":
        return RandomPolicy(seed=args.policy_seed)
    if args.policy == "expert":
        return ExpertPolicy(cfg, bundle)
    if args.policy == "remote":
        return RemotePolicy(
            host=args.remote_host,
            port=args.remote_port,
            requery=args.remote_requery,
        )
    raise ValueError(f"unknown policy {args.policy!r}")


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default="sim/configs/thermal_recognition/pick_cold_can.yaml",
        help="task YAML (default: the shipped pick_cold_can config)",
    )
    ap.add_argument(
        "--policy",
        choices=("zero", "random", "expert", "remote"),
        default="zero",
        help="zero/random = harness smoke; expert = generator validation; "
        "remote = openpi policy server",
    )
    ap.add_argument("--episodes", type=int, default=10, help="number of episodes (default 10)")
    ap.add_argument("--seed0", type=int, default=1000, help="first episode seed (default 1000)")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="policy steps per episode (default: YAML max_episode_steps)",
    )
    ap.add_argument(
        "--state-mode",
        choices=("proprio16", "touch28", "full30"),
        default="proprio16",
        help="state assembled for the policy (default proprio16)",
    )
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--viewer", action="store_true", help="open the Genesis viewer")
    ap.add_argument(
        "--permute-slots",
        action="store_true",
        help="diagnostic: permute the home slots per episode (vision-only)",
    )
    ap.add_argument("--open-tol", type=float, default=float(PIPELINE.release_open_tol))
    ap.add_argument("--policy-seed", type=int, default=0, help="RandomPolicy seed")
    ap.add_argument("--remote-host", default="127.0.0.1")
    ap.add_argument("--remote-port", type=int, default=8000)
    ap.add_argument(
        "--remote-requery",
        type=int,
        default=1,
        help="ask the server every N steps; 0 = execute the whole returned chunk",
    )
    ap.add_argument("--trace", action="store_true", help="write per-step trace.jsonl")
    args = ap.parse_args()

    cfg = load_task_config(args.config)
    if args.policy != "expert" and not cfg["cameras"]["enabled"]:
        print("ERROR: this eval feeds RGB to the policy; enable cameras in the config")
        return 2
    if cfg["success_criterion"] != "place_in_tray" or not cfg["tray"]["enabled"]:
        print("ERROR: this scorer implements place_in_tray only")
        return 2

    max_steps = int(args.max_steps) if args.max_steps else int(cfg["max_episode_steps"])
    decimation = int(cfg["controller"]["decimation"])
    sim_dt = float(cfg["controller"]["sim_dt"])
    tol = float(cfg["tray_rest_tol_m"])
    stable_needed = int(cfg["stable_steps"])
    scene_seed = int(cfg["seed"]) + 1000  # exactly what scripts/generate.py used

    if args.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = REPO / "experiment" / "runs" / f"{args.policy}_{args.state_mode}_{stamp}"
    else:
        out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"config: {args.config}")
    print(f"policy: {args.policy}  state: {args.state_mode}  episodes: {args.episodes}")
    print(f"scene seed: {scene_seed}  episode seeds: {args.seed0}..{args.seed0 + args.episodes - 1}")
    print(f"output: {out_dir}")
    if not args.permute_slots:
        print(
            "note: slots are fixed per episode -- a vision-only policy can learn the "
            "target's position; --permute-slots measures that shortcut"
        )

    t_start = time.perf_counter()
    bundle = build_scene(cfg, seed=scene_seed, show_viewer=bool(args.viewer))
    builder = ObsBuilder(bundle)
    controller = DeltaPoseController(bundle)
    policy = _make_policy(args, cfg, bundle)

    records: list[dict[str, Any]] = []
    trace_fh = (out_dir / "trace.jsonl").open("w", encoding="utf-8") if args.trace else None

    try:
        for k in range(int(args.episodes)):
            seed = int(args.seed0) + k
            rng = np.random.default_rng(seed)
            ep = sample_episode(cfg, rng)
            if args.permute_slots:
                ep = _permute_slots(ep, rng)

            bundle.reset_episode(ep)
            bundle.step()
            tau = float(SENSOR_TIME_CONSTANT)
            if tau > 0.0 and bundle.temperatures:
                for _ in range(int(np.ceil(5.0 * tau / sim_dt))):
                    bundle.step()
            controller.reset()

            target_index = int(ep["target_index"])
            spec = bundle.object_specs[target_index]
            instruction = language_instruction(
                cfg,
                spec.material,
                spec.shape,
                template_index=seed % len(language_templates(cfg)),
            )
            policy.reset(instruction, rng=rng)

            stable = 0
            any_stable = 0
            success = False
            manipulation = False
            min_tip = float("inf")
            last_grip = float("nan")
            steps = 0

            for step in range(max_steps):
                obs = builder.build()
                tip_block = obs.get("temperature") or {}
                tip = (
                    np.asarray(tip_block["fingertip_grid"], dtype=np.float64).reshape(-1)
                    if "fingertip_grid" in tip_block
                    else np.empty(0, dtype=np.float64)
                )
                if tip.size:
                    min_tip = min(min_tip, float(tip.min()))

                action = _policy_action(
                    policy,
                    obs,
                    state_mode=args.state_mode,
                    instruction=instruction,
                    step=step,
                )
                controller.step(action)
                for _ in range(decimation):
                    bundle.step()
                steps = step + 1

                last_grip = float(
                    np.asarray(obs["gripper_states"], dtype=np.float64).reshape(-1)[0]
                )
                ok_target = _placed(
                    bundle,
                    last_grip,
                    tol=tol,
                    open_tol=float(args.open_tol),
                    target_only=True,
                )
                ok_any = _placed(
                    bundle,
                    last_grip,
                    tol=tol,
                    open_tol=float(args.open_tol),
                    target_only=False,
                )
                stable = stable + 1 if ok_target else 0
                any_stable = any_stable + 1 if ok_any else 0

                if trace_fh is not None:
                    trace_fh.write(
                        json.dumps(
                            {
                                "episode": k,
                                "seed": seed,
                                "step": step,
                                "gripper": last_grip,
                                "fingertip_c": [float(v) for v in tip],
                                "ee_pos": [
                                    float(v)
                                    for v in np.asarray(obs["ee_pos"], dtype=np.float64).reshape(-1)[:3]
                                ],
                                "target_in_tray": bool(ok_target),
                                "any_in_tray": bool(ok_any),
                            }
                        )
                        + "\n"
                    )

                if stable >= stable_needed:
                    success = True
                    break

            manipulation = any_stable >= stable_needed
            bodies = _bodies_in_tray(bundle, tol)
            record = {
                "episode": k,
                "seed": seed,
                "success": bool(success),
                "manipulation_success": bool(manipulation),
                "steps": int(steps),
                "instruction": instruction,
                "target_index": target_index,
                "target_material": str(spec.material),
                "target_shape": str(spec.shape),
                # Ground truth for the *log only*: never a policy input.
                "target_temp_c": float(ep["temps_c"][target_index]),
                "min_fingertip_c": None if not np.isfinite(min_tip) else float(min_tip),
                "final_gripper": last_grip,
                "bodies_in_tray": [int(i) for i in bodies],
                "target_in_tray": bool(target_index in bodies),
                "expert_phase": policy.phase if isinstance(policy, ExpertPolicy) else None,
            }
            records.append(record)
            tip_text = f"{min_tip:.1f} C" if np.isfinite(min_tip) else "n/a"
            print(
                f"[{k + 1}/{args.episodes}] seed {seed}  "
                f"{'TASK OK ' if success else 'task fail'}  "
                f"{'placed ' + str(bodies) if bodies else 'tray empty'}  "
                f"steps {steps}  tip min {tip_text}"
            )
    finally:
        if trace_fh is not None:
            trace_fh.close()

    n = len(records)
    task_rate = float(np.mean([r["success"] for r in records])) if n else 0.0
    manip_rate = float(np.mean([r["manipulation_success"] for r in records])) if n else 0.0
    steps_mean = float(np.mean([r["steps"] for r in records])) if n else 0.0
    wall = time.perf_counter() - t_start

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    summary = {
        "config": str(args.config),
        "policy": policy.name,
        "state_mode": args.state_mode,
        "episodes": n,
        "task_success": int(sum(r["success"] for r in records)),
        "task_success_rate": task_rate,
        "manipulation_success": int(sum(r["manipulation_success"] for r in records)),
        "manipulation_success_rate": manip_rate,
        "mean_steps": steps_mean,
        "seed0": int(args.seed0),
        "scene_seed": scene_seed,
        "permute_slots": bool(args.permute_slots),
        "wall_time_s": wall,
        "definition": {
            "task_success": "the targeted body is at rest in the tray, pads open",
            "manipulation_success": "any body is at rest in the tray, pads open",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"task success          {summary['task_success']}/{n} ({100.0 * task_rate:.1f}%)")
    print(f"manipulation success  {summary['manipulation_success']}/{n} ({100.0 * manip_rate:.1f}%)")
    print(f"mean steps            {steps_mean:.1f}")
    print(f"wall time             {wall:.1f}s")
    print(f"wrote                 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
