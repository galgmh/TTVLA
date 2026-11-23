"""Closed-loop evaluation on the raw_v4 grasp episodes.

`sim.run_pi_simpler` samples `ThermalContactSampler` datasets (the four-region
*probing* task), so it cannot evaluate a policy trained on the raw_v4 grasp
data. This evaluator rebuilds episodes from the policy-style dataset config
used to generate that data (default: `sim/configs/datasets/pi05_v1.yaml`) and
scores the policy on the same task distribution it was trained on.

Usage
-----
Start the policy server first (models env), then run this in the Genesis env::

    PYOPENGL_PLATFORM=osmesa python -m sim.run_pi_thermal \
        --host 127.0.0.1 --port 8000 \
        --episodes-per-policy 2 \
        --video-dir datasets/eval_videos_v4 \
        --output datasets/eval_thermal_v4.json

The success criterion defaults to `cup-lift` with a 0.06 m threshold, matching
`contact.lift_success_height_m` in the dataset config. For a held-out check,
pass a different `--seed` (episodes are rebuilt with that seed) and/or
`--attempt` (0 reproduces the first-attempt instance of every slot).

Headless machines need the software OpenGL fallback that Genesis supports:
`export PYOPENGL_PLATFORM=osmesa`.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
from typing import Any

from sim.config import load_dataset_config, load_task_config
from sim.generate_pi05_v1 import _make_episode
from sim.model_env import GenesisModelEnv
from sim.run_pi_simpler import (
    _EpisodeVideoWriter,
    _WebsocketClientPolicy,
    _action_chunk,
    _cup_position,
    _is_success,
    _to_model_observation,
    _video_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop evaluation on raw_v4 grasp episodes"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--task-config", default="sim/configs/tasks/thermal_contact_probe.yaml"
    )
    parser.add_argument(
        "--dataset-config", default="sim/configs/datasets/pi05_v1.yaml"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="Only evaluate these policy ids (default: all in the dataset config)",
    )
    parser.add_argument(
        "--episodes-per-policy",
        type=int,
        default=None,
        help="Slots evaluated per policy (default: policy['episodes'], i.e. 20)",
    )
    parser.add_argument("--start-slot", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the dataset seed (use for held-out evaluation)",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=0,
        help="Attempt index passed to the episode builder (0 = first attempt)",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--success-kind",
        choices=("none", "contact", "cup-move", "cup-lift"),
        default="cup-lift",
    )
    parser.add_argument("--success-threshold", type=float, default=0.06)
    parser.add_argument(
        "--video-dir",
        default=None,
        help="If set, write one MP4 per episode (base | wrist cameras side by side)",
    )
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--output", default=None, help="Optional result JSON path")
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be positive")
    if args.episodes_per_policy is not None and args.episodes_per_policy < 1:
        raise ValueError("--episodes-per-policy must be positive")
    if args.start_slot < 0:
        raise ValueError("--start-slot must be >= 0")

    task = load_task_config(args.task_config)
    dataset = load_dataset_config(args.dataset_config)
    if args.seed is not None:
        dataset = {**dataset, "seed": int(args.seed)}
    policies = list(dataset["policies"])
    if args.policies:
        wanted = set(args.policies)
        policies = [p for p in policies if str(p["id"]) in wanted]
        missing = wanted - {str(p["id"]) for p in policies}
        if missing:
            raise SystemExit(f"unknown policies: {sorted(missing)}")

    client = _WebsocketClientPolicy(args.host, args.port)
    env = GenesisModelEnv(
        args.task_config,
        None,  # episodes come from `_make_episode`, not the ThermalContact sampler
        max_steps=args.max_steps,
        show_viewer=args.show_viewer,
    )

    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    video_fps = max(1, round(1.0 / float(env.control_dt)))

    episodes: list[dict[str, Any]] = []
    attempt_counter = 0
    try:
        for policy in policies:
            policy_id = str(policy["id"])
            total_slots = int(policy["episodes"])
            start = min(args.start_slot, total_slots)
            end = (
                total_slots
                if args.episodes_per_policy is None
                else min(total_slots, start + args.episodes_per_policy)
            )
            for slot in range(start, end):
                spec, _, _ = _make_episode(
                    task, dataset, policy, slot, args.attempt, attempt_counter
                )
                attempt_counter += 1

                video_path = (
                    video_dir / f"{policy_id}_{slot:03d}.mp4"
                    if video_dir is not None
                    else None
                )
                video = (
                    _EpisodeVideoWriter(video_path, video_fps)
                    if video_path is not None
                    else None
                )

                obs = env.reset(spec=spec)
                if video is not None:
                    video.add(_video_frame(obs))
                initial_cup_position = _cup_position(env)

                action_plan: deque[Any] = deque()
                inferred_chunks = 0
                success = False
                while not env.done:
                    if not action_plan:
                        model_obs = _to_model_observation(
                            obs, task_index=args.task_index
                        )
                        chunk = _action_chunk(
                            client.infer(model_obs), action_dim=args.action_dim
                        )
                        if chunk.shape[0] < args.replan_steps:
                            raise ValueError(
                                f"policy horizon {chunk.shape[0]} "
                                f"< --replan-steps {args.replan_steps}"
                            )
                        action_plan.extend(chunk[: args.replan_steps])
                        inferred_chunks += 1

                    obs, done, _ = env.step(action_plan.popleft())
                    if video is not None:
                        video.add(_video_frame(obs))
                    success = _is_success(
                        args.success_kind,
                        env,
                        initial_cup_position=initial_cup_position,
                        threshold=args.success_threshold,
                    )
                    if success or done:
                        break

                if video is not None:
                    video.close()

                row = {
                    "policy": policy_id,
                    "slot": slot,
                    "success": bool(success),
                    "policy_steps": int(env.policy_step),
                    "sim_time_s": float(env.sim_time),
                    "inferred_chunks": inferred_chunks,
                    "initial_cup_position": initial_cup_position.tolist(),
                    "final_cup_position": _cup_position(env).tolist(),
                }
                if video_path is not None:
                    row["video"] = str(video_path)
                episodes.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        client.close()

    per_policy: dict[str, dict[str, Any]] = {}
    for row in episodes:
        entry = per_policy.setdefault(
            row["policy"], {"episodes": 0, "successes": 0, "success_rate": 0.0}
        )
        entry["episodes"] += 1
        entry["successes"] += int(row["success"])
    for entry in per_policy.values():
        entry["success_rate"] = entry["successes"] / max(entry["episodes"], 1)

    successes = sum(int(row["success"]) for row in episodes)
    summary = {
        "host": args.host,
        "port": args.port,
        "task_config": args.task_config,
        "dataset_config": args.dataset_config,
        "seed": args.seed,
        "attempt": args.attempt,
        "success_kind": args.success_kind,
        "success_threshold": args.success_threshold,
        "episodes": episodes,
        "total_episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / max(len(episodes), 1),
        "per_policy": per_policy,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate(args)
    print(
        f"success {summary['successes']}/{summary['total_episodes']} "
        f"({summary['success_rate']:.1%}) kind={summary['success_kind']} "
        f"threshold={summary['success_threshold']}"
    )
    for policy_id, entry in summary["per_policy"].items():
        print(
            f"  {policy_id}: {entry['successes']}/{entry['episodes']} "
            f"({entry['success_rate']:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
