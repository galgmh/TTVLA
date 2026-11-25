"""Closed-loop LIBERO evaluation for the TTVLA policy server.

This is adapted from Physical Intelligence's openpi LIBERO evaluator. The
LIBERO environment and task definitions remain the official implementation;
only the observation payload is adapted to the data keys used by TTVLA.

Example:

    Terminal 1:
        cd TTVLA/models
        python server_policy.py \
            --config_name libero_object_lora \
            --checkpoint_dir /path/to/checkpoints/15000 \
            --host 0.0.0.0 --port 8000

    Terminal 2, using an environment with LIBERO, msgpack, and websockets:
        MUJOCO_GL=egl python test_libero.py \
            --host 127.0.0.1 --port 8000 \
            --task-suite-name libero_object \
            --num-trials-per-task 50 \
            --output data/libero_object_results.json

For the first smoke test, use ``--num-trials-per-task 1 --max-steps 50``.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
import json
import logging
import math
import pathlib
import time
from typing import Any

import cv2
import imageio
import msgpack
import numpy as np
import websockets.sync.client
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _encode_numpy(obj: Any) -> Any:
    """Encode numpy values using the codec expected by server_policy.py."""
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
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _decode_numpy(obj: dict[Any, Any]) -> Any:
    """Decode the msgpack numpy codec used by server_policy.py."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_pack = functools.partial(msgpack.packb, default=_encode_numpy)
_unpack = functools.partial(msgpack.unpackb, object_hook=_decode_numpy)


class WebsocketClientPolicy:
    """Minimal sync client compatible with models/server_policy.py."""

    def __init__(self, host: str, port: int, *, retry_s: float = 5.0) -> None:
        uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        connect_kwargs: dict[str, Any] = {
            "compression": None,
            "max_size": None,
        }
        parameters = inspect.signature(websockets.sync.client.connect).parameters
        if "legacy" in parameters:
            connect_kwargs["legacy"] = True
        if "proxy" in parameters and (
            host in ("localhost", "::1") or host.startswith("127.")
        ):
            connect_kwargs["proxy"] = None

        while True:
            try:
                self._websocket = websockets.sync.client.connect(
                    uri,
                    **connect_kwargs,
                )
                break
            except ConnectionRefusedError:
                logging.info("Waiting for policy server at %s ...", uri)
                time.sleep(retry_s)

        self.server_metadata = _unpack(self._websocket.recv())
        logging.info("Policy server metadata: %s", self.server_metadata)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._websocket.send(_pack(observation))
        response = self._websocket.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in policy server:\n{response}")
        return _unpack(response)

    def close(self) -> None:
        self._websocket.close()


@dataclasses.dataclass
class Args:
    # TTVLA policy server.
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Official LIBERO evaluation protocol.
    task_suite_name: str = "libero_object"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7
    max_steps: int | None = None
    stop_on_error: bool = True

    # Result and video outputs.
    video_out_path: str = "data/libero/videos"
    save_videos: bool = False
    output: str | None = None


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to an axis-angle vector.

    Matches robosuite and the official openpi LIBERO evaluator.
    """
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    axis_angle = (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator
    return np.asarray(axis_angle, dtype=np.float32)


def _convert_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(image * 255.0)
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _resize_with_pad(
    image: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize an HWC image while preserving aspect ratio and add black padding."""
    image = _convert_to_uint8(image)
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / width, current_height / height)
    resized_height = max(1, int(current_height / ratio))
    resized_width = max(1, int(current_width / ratio))
    interpolation = (
        cv2.INTER_AREA
        if resized_height < current_height or resized_width < current_width
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pad_top = (height - resized_height) // 2
    pad_left = (width - resized_width) // 2
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return canvas


def _make_policy_observation(
    obs: dict[str, Any],
    *,
    task_id: int,
    task_description: str,
    resize_size: int,
) -> dict[str, Any]:
    """Build the observation keys required by TTVLA's LIBERO RepackTransform.

    The observation is flipped 180 degrees and resized in the same way as the
    official openpi evaluator. ``actions`` and ``task_index`` are required by
    the current training-oriented RepackTransform, but neither is used by
    inference. Their values are therefore harmless placeholders.
    """
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    image = _resize_with_pad(image, resize_size, resize_size)
    wrist_image = _resize_with_pad(wrist_image, resize_size, resize_size)

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    if state.shape != (8,):
        raise ValueError(f"LIBERO state must have shape (8,), got {state.shape}")

    return {
        # TTVLA's LeRobot pipeline stores images as CHW. The server converts
        # CHW to the NHWC layout expected by SigLIP internally.
        "image": np.ascontiguousarray(image.transpose(2, 0, 1)),
        "wrist_image": np.ascontiguousarray(wrist_image.transpose(2, 0, 1)),
        "state": state,
        "actions": np.zeros(7, dtype=np.float32),
        "task_index": np.int64(task_id),
        "prompt": str(task_description),
    }


def _extract_action_chunk(
    result: dict[str, Any],
    *,
    replan_steps: int,
) -> np.ndarray:
    if "actions" not in result:
        raise KeyError(f"Policy response has no 'actions'; got {sorted(result)}")

    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(
            f"Policy actions must have shape [H, 7], got {actions.shape}"
        )
    if actions.shape[0] < replan_steps:
        raise ValueError(
            f"Policy horizon {actions.shape[0]} is smaller than "
            f"replan_steps={replan_steps}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned non-finite actions")
    return actions


def _get_libero_env(task: Any, resolution: int, seed: int) -> tuple[Any, str]:
    task_description = str(task.language)
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    # The seed also affects object placement even when using a fixed init state.
    env.seed(seed)
    return env, task_description


def _write_summary(
    output: str | None,
    *,
    task_suite_name: str,
    task_results: list[dict[str, Any]],
    total_episodes: int,
    total_successes: int,
) -> dict[str, Any]:
    summary = {
        "task_suite_name": task_suite_name,
        "total_tasks": len(task_results),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": (
            total_successes / total_episodes if total_episodes > 0 else 0.0
        ),
        "tasks": task_results,
    }
    if output is not None:
        path = pathlib.Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("Saved evaluation summary to %s", path)
    return summary


def evaluate(args: Args) -> dict[str, Any]:
    if args.num_trials_per_task < 1:
        raise ValueError("--num-trials-per-task must be positive")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else DEFAULT_MAX_STEPS.get(args.task_suite_name)
    )
    if max_steps is None:
        raise ValueError(
            f"Unknown task suite {args.task_suite_name!r}; "
            f"known suites: {sorted(DEFAULT_MAX_STEPS)}"
        )

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    client = WebsocketClientPolicy(args.host, args.port)

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.info(
        "Connected to policy server; evaluating %s for %d trials/task",
        args.task_suite_name,
        args.num_trials_per_task,
    )

    task_results: list[dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
        )

        task_episodes = 0
        task_successes = 0
        episode_results: list[dict[str, Any]] = []

        try:
            for episode_idx in range(args.num_trials_per_task):
                logging.info(
                    "\nTask %d/%d %r, episode %d/%d",
                    task_id + 1,
                    suite.n_tasks,
                    task_description,
                    episode_idx + 1,
                    args.num_trials_per_task,
                )

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                action_plan: collections.deque[np.ndarray] = collections.deque()
                replay_images: list[np.ndarray] = []
                success = False
                error: str | None = None
                step = 0

                try:
                    for step in range(max_steps + args.num_steps_wait):
                        if step < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            if done:
                                success = True
                                break
                            continue

                        policy_obs = _make_policy_observation(
                            obs,
                            task_id=task_id,
                            task_description=task_description,
                            resize_size=args.resize_size,
                        )
                        replay_images.append(
                            np.ascontiguousarray(
                                policy_obs["image"].transpose(1, 2, 0)
                            )
                        )

                        if not action_plan:
                            result = client.infer(policy_obs)
                            action_chunk = _extract_action_chunk(
                                result,
                                replan_steps=args.replan_steps,
                            )
                            action_plan.extend(
                                action_chunk[: args.replan_steps]
                            )

                        action = action_plan.popleft()
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            success = True
                            break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logging.exception(
                        "Episode failed during inference or simulation"
                    )
                    if args.stop_on_error:
                        client.close()
                        raise

                task_episodes += 1
                total_episodes += 1
                task_successes += int(success)
                total_successes += int(success)

                if args.save_videos and replay_images:
                    suffix = "success" if success else "failure"
                    video_name = (
                        f"task_{task_id:02d}_episode_{episode_idx:03d}_"
                        f"{suffix}.mp4"
                    )
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / video_name,
                        replay_images,
                        fps=20,
                    )

                row = {
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "success": bool(success),
                    "steps": int(step),
                    "error": error,
                }
                episode_results.append(row)
                logging.info(
                    "Result: success=%s; task %d/%d; total %d/%d (%.1f%%)",
                    success,
                    task_successes,
                    task_episodes,
                    total_successes,
                    total_episodes,
                    (
                        100.0 * total_successes / total_episodes
                        if total_episodes
                        else 0.0
                    ),
                )
        finally:
            if hasattr(env, "close"):
                env.close()

        task_result = {
            "task_id": task_id,
            "task_description": task_description,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": (
                task_successes / task_episodes if task_episodes else 0.0
            ),
            "results": episode_results,
        }
        task_results.append(task_result)
        logging.info(
            "Task success rate: %.1f%% (%d/%d)",
            100.0 * task_result["success_rate"],
            task_successes,
            task_episodes,
        )

        # Persist partial results so long evaluations remain inspectable.
        _write_summary(
            args.output,
            task_suite_name=args.task_suite_name,
            task_results=task_results,
            total_episodes=total_episodes,
            total_successes=total_successes,
        )

    summary = _write_summary(
        args.output,
        task_suite_name=args.task_suite_name,
        task_results=task_results,
        total_episodes=total_episodes,
        total_successes=total_successes,
    )
    client.close()
    logging.info(
        "Final success rate: %d/%d (%.1f%%)",
        total_successes,
        total_episodes,
        100.0 * summary["success_rate"],
    )
    return summary


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    evaluate(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
