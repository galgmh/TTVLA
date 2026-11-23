"""Closed-loop evaluation for a models inference server.

The model server is expected to return already denormalized physical actions::

    [dx, dy, dz, drot_x, drot_y, drot_z, gripper_width]

Genesis execution is handled by :class:`sim.model_env.GenesisModelEnv`.
"""

from __future__ import annotations

import argparse
from collections import deque
import functools
import inspect
import json
import logging
from pathlib import Path
import time
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client

from sim.model_env import GenesisModelEnv
from sim.obs import get_policy_observation


# ---------------------------------------------------------------------------
# Minimal websocket policy client (wire-compatible with models/server_policy.py).
# Kept local so this evaluator does not need the `openpi_client` package; the
# numpy codec mirrors server_policy._pack_array/_unpack_array.
# ---------------------------------------------------------------------------
def _encode_numpy(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _decode_numpy(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_pack = functools.partial(msgpack.packb, default=_encode_numpy)
_unpack = functools.partial(msgpack.unpackb, object_hook=_decode_numpy)


class _WebsocketClientPolicy:
    """Sync client for models/server_policy.py (no openpi_client dependency)."""

    def __init__(self, host: str, port: int, *, retry_s: float = 5.0) -> None:
        uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        connect_kwargs: dict[str, Any] = {"compression": None, "max_size": None}
        params = inspect.signature(websockets.sync.client.connect).parameters
        # Newer websockets warn unless the connection is kept open via a context
        # manager; this client holds it for many infer() calls.
        if "legacy" in params:
            connect_kwargs["legacy"] = True
        # websockets >= 15 picks HTTP(S)_PROXY from the environment and urllib's
        # no_proxy matcher doesn't understand patterns like "127.*", so loopback
        # connections would otherwise be routed through a local proxy.
        if "proxy" in params:
            if host in ("localhost", "::1") or host.startswith("127."):
                connect_kwargs["proxy"] = None
        while True:
            try:
                self._ws = websockets.sync.client.connect(uri, **connect_kwargs)
                break
            except ConnectionRefusedError:
                logging.info(f"Waiting for policy server at {uri} ...")
                time.sleep(retry_s)
        self.server_metadata = _unpack(self._ws.recv())

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(_pack(obs))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"Error in inference server:\n{reply}")
        return _unpack(reply)

    def close(self) -> None:
        self._ws.close()


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
    parser.add_argument(
        "--video-dir",
        default=None,
        help="If set, write one MP4 per episode (base | wrist cameras side by side)",
    )
    return parser


def _to_model_observation(obs: dict[str, Any], *, task_index: int) -> dict[str, Any]:
    """Convert GenesisModelEnv output to models's raw repack keys."""
    base = np.asarray(obs["image"]["base_0_rgb"], dtype=np.uint8)
    wrist = np.asarray(obs["image"]["left_wrist_0_rgb"], dtype=np.uint8)
    return {
        "observation.images.up": np.ascontiguousarray(base.transpose(2, 0, 1)),
        "observation.images.side": np.ascontiguousarray(wrist.transpose(2, 0, 1)),
        "observation.state": np.asarray(obs["state"], dtype=np.float32),
        # The server's RepackTransform maps "actions" <- "action" and requires
        # the key even at inference. The model never reads it (no ground-truth
        # action exists at inference time), so a zero placeholder is enough.
        "action": np.zeros(7, dtype=np.float32),
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


def _video_frame(obs: dict[str, Any]) -> np.ndarray:
    """Base camera and wrist camera side by side (RGB uint8)."""
    images = obs["image"]
    base = np.asarray(images["base_0_rgb"], dtype=np.uint8)
    wrist = np.asarray(images["left_wrist_0_rgb"], dtype=np.uint8)
    height = min(base.shape[0], wrist.shape[0])
    return np.concatenate([base[:height], wrist[:height]], axis=1)


class _EpisodeVideoWriter:
    """Write policy-step frames to an MP4 (opencv `mp4v` codec)."""

    def __init__(self, path: Path, fps: int) -> None:
        import cv2

        self._cv2 = cv2
        self._path = path
        self._fps = max(1, int(fps))
        self._writer = None

    def add(self, frame: np.ndarray) -> None:
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = self._cv2.VideoWriter(
                str(self._path),
                self._cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                (width, height),
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"failed to open video writer for {self._path}")
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be positive")

    client = _WebsocketClientPolicy(args.host, args.port)
    env = GenesisModelEnv(
        args.task_config,
        args.dataset_config,
        split=args.split,
        max_steps=args.max_steps,
        show_viewer=args.show_viewer,
    )

    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    video_fps = max(1, round(1.0 / float(env.control_dt)))

    episodes: list[dict[str, Any]] = []
    for episode_offset in range(args.episodes):
        index = args.start + episode_offset
        video_path = video_dir / f"episode_{index:04d}.mp4" if video_dir is not None else None
        video = _EpisodeVideoWriter(video_path, video_fps) if video_path is not None else None
        obs = env.reset(index=index)
        if video is not None:
            video.add(_video_frame(obs))
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
            "index": index,
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
        print(json.dumps(row, ensure_ascii=False))

    client.close()

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
