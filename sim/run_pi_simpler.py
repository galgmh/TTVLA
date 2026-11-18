"""Closed-loop evaluation for a models inference server.

The model server is expected to return already denormalized physical actions::

    [dx, dy, dz, drot_x, drot_y, drot_z, gripper_width]

Genesis execution is handled by :class:`sim.model_env.GenesisModelEnv`.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.model_env import GenesisModelEnv
from sim.obs import get_policy_observation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run π0.5 closed-loop evaluation in Genesis")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-config", default="sim/configs/tasks/thermal_contact_probe.yaml")
    parser.add_argument("--dataset-config", default="sim/configs/datasets/thermal_contact_v1.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-sample-steps", type=int, default=10)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--success-kind",
        choices=("none", "contact", "cup-move", "cup-lift"),
        default="none",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.06,
        help="Distance for cup-move or height gain for cup-lift, in metres",
    )
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--output", default=None, help="Optional result JSON path")
    return parser


def _to_model_observation(obs: dict[str, Any], *, task_index: int) -> dict[str, Any]:
    """Convert GenesisModelEnv output to models's raw repack keys."""
    base = np.asarray(obs["image"]["base_0_rgb"], dtype=np.uint8)
    wrist = np.asarray(obs["image"]["left_wrist_0_rgb"], dtype=np.uint8)
    return {
        "observation.images.up": np.ascontiguousarray(base.transpose(2, 0, 1)),
        "observation.images.side": np.ascontiguousarray(wrist.transpose(2, 0, 1)),
        "observation.state": np.asarray(obs["state"], dtype=np.float32),
        "task_index": np.int64(task_index),
        "prompt": str(obs["prompt"]),
    }


def _action_chunk(result: dict[str, Any], *, action_dim: int) -> np.ndarray:
    """Extract a [H,A] physical action chunk from a policy-server response."""
    if "actions" not in result:
        raise KeyError(f"policy response has no 'actions'; got {sorted(result)}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError(f"policy actions must be [H,{action_dim}], got {actions.shape}")
    if actions.shape[-1] != action_dim:
        raise ValueError(
            f"policy action_dim={actions.shape[-1]} does not match env action_dim={action_dim}"
        )
    if actions.shape[0] < 1 or not np.all(np.isfinite(actions)):
        raise ValueError("policy returned an empty or non-finite action chunk")
    return actions


def _cup_position(env: GenesisModelEnv) -> np.ndarray:
    return np.asarray(env.bundle.cup_pose()[0], dtype=np.float64).reshape(3)


def _is_success(
    kind: str,
    env: GenesisModelEnv,
    *,
    initial_cup_position: np.ndarray,
    threshold: float,
) -> bool:
    if kind == "none":
        return False
    if kind == "contact":
        return bool(get_policy_observation(env.bundle)["contact_mask"].any())

    displacement = _cup_position(env) - initial_cup_position
    if kind == "cup-move":
        return bool(np.linalg.norm(displacement) >= threshold)
    if kind == "cup-lift":
        return bool(displacement[2] >= threshold)
    raise ValueError(f"unknown success kind: {kind}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be positive")

    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "openpi_client is required. Install the openpi client package in the Genesis environment."
        ) from exc

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = GenesisModelEnv(
        args.task_config,
        args.dataset_config,
        split=args.split,
        max_steps=args.max_steps,
        show_viewer=args.show_viewer,
    )

    episodes: list[dict[str, Any]] = []
    for episode_offset in range(args.episodes):
        index = args.start + episode_offset
        obs = env.reset(index=index)
        initial_cup_position = _cup_position(env)
        action_plan: deque[np.ndarray] = deque()
        inferred_chunks = 0
        success = False

        while not env.done:
            if not action_plan:
                model_obs = _to_model_observation(obs, task_index=args.task_index)
                result = client.infer(model_obs)
                chunk = _action_chunk(result, action_dim=args.action_dim)
                if chunk.shape[0] < args.replan_steps:
                    raise ValueError(
                        f"policy horizon {chunk.shape[0]} < --replan-steps {args.replan_steps}"
                    )
                action_plan.extend(chunk[: args.replan_steps])
                inferred_chunks += 1

            obs, done, info = env.step(action_plan.popleft())
            success = _is_success(
                args.success_kind,
                env,
                initial_cup_position=initial_cup_position,
                threshold=args.success_threshold,
            )
            if success or done:
                break

        row = {
            "index": index,
            "success": bool(success),
            "policy_steps": int(env.policy_step),
            "sim_time_s": float(env.sim_time),
            "inferred_chunks": inferred_chunks,
            "initial_cup_position": initial_cup_position.tolist(),
            "final_cup_position": _cup_position(env).tolist(),
        }
        episodes.append(row)
        print(json.dumps(row, ensure_ascii=False))

    successes = sum(int(episode["success"]) for episode in episodes)
    summary = {
        "host": args.host,
        "port": args.port,
        "split": args.split,
        "start": args.start,
        "episodes": episodes,
        "total_episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "success_kind": args.success_kind,
        "success_threshold": args.success_threshold,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate(args)
    print(
        f"success {summary['successes']}/{summary['total_episodes']} "
        f"({summary['success_rate']:.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
