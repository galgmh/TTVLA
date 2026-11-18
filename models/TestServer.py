"""Websocket smoke test for an already-running `server_policy.py` instance.

The synthetic obs matches the repack transform in `configs.configs.test_run`:

  {
      "observation.images.up":   np.uint8,  (H, W, 3)  in [0, 255]
      "observation.images.side": np.uint8,  (H, W, 3)
      "observation.state":       np.float32 (6,)        in [-1, 1]
      "action":                  np.float32 (6,)        (consumed by repack
                                                         but ignored by the model)
      "task_index":              np.int64   ()          (consumed by repack)
      "prompt":                  str                    (REQUIRED — server only
                                                         carries InjectDefaultPrompt
                                                         as a fallback, not the
                                                         LeRobot task lookup)
  }

Usage
-----
  python TestServer.py --host 127.0.0.1 --port 8000

Runtime deps: `pip install msgpack websockets`.
"""
from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import tyro

from server_policy import _msgpack_packer, _msgpack_unpackb, init_logging


# ---------------------------------------------------------------------------
# Synthesize one raw observation.
# ---------------------------------------------------------------------------
def make_synthetic_obs(
    *,
    img_height: int = 240,
    img_width: int = 320,
    raw_action_dim: int = 6,
    prompt: str = "pick up the red cube and place it in the bowl",
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    logging.warning("ONLY ACCEPT CHW IMAGES, HWC IS NOT SUPPORTED!")
    img_shape = (3, img_height, img_width)
    return {
        "observation.images.up":   rng.integers(0, 255, size=img_shape, dtype=np.uint8),
        "observation.images.side": rng.integers(0, 255, size=img_shape, dtype=np.uint8),
        "observation.state":       rng.uniform(-5.0, 5.0, size=raw_action_dim).astype(np.float32),
        # `action` and `task_index` are pulled by the repack transform but the
        # model never reads them at inference time — values are irrelevant.
        "action":                  np.zeros(raw_action_dim, dtype=np.float32),
        "task_index":              np.int64(0),
        "prompt":                  prompt,
    }


def _describe_obs(obs: dict) -> None:
    print("Sending synthetic obs:")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k!s:35s} np.ndarray shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k!s:35s} {type(v).__name__} = {v!r}")


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
def _validate_action_frame(reply: dict) -> tuple[int, int]:
    assert isinstance(reply, dict), f"reply must be dict, got {type(reply)}: {reply!r}"
    assert "actions" in reply, f"reply missing 'actions' key. Got: {sorted(reply)}"

    actions = reply["actions"]
    assert isinstance(actions, np.ndarray), f"actions must be np.ndarray, got {type(actions)}"
    assert actions.ndim == 2, f"actions must be (horizon, dim), got shape {actions.shape}"
    assert actions.dtype in (np.float32, np.float64), f"unexpected action dtype: {actions.dtype}"
    assert np.isfinite(actions).all(), "actions contain non-finite values"

    assert "server_timing" in reply, "reply missing 'server_timing'"
    assert "infer_ms" in reply["server_timing"], "server_timing missing 'infer_ms'"
    return actions.shape


# ---------------------------------------------------------------------------
# Websocket round-trip
# ---------------------------------------------------------------------------
def _round_trip(host: str, port: int, obs: dict, num_iters: int) -> None:
    try:
        from websockets.sync.client import connect as ws_connect
    except ImportError as e:
        raise RuntimeError("Need `pip install websockets msgpack`") from e

    url = f"ws://{host}:{port}"
    print(f"\nConnecting to {url} ...")

    packer = _msgpack_packer()
    with ws_connect(url, compression=None, max_size=None) as ws:
        # First frame from the server is the metadata dict.
        meta = _msgpack_unpackb(ws.recv())
        print(f"  server metadata: {meta}")

        latencies = []
        last_shape: tuple[int, int] | None = None
        for i in range(num_iters):
            t0 = time.monotonic()
            ws.send(packer.pack(obs))
            reply = _msgpack_unpackb(ws.recv())
            rtt_ms = (time.monotonic() - t0) * 1000.0
            latencies.append(rtt_ms)

            shape = _validate_action_frame(reply)
            timing = reply["server_timing"]
            print(
                f"  [iter {i}] reply keys={sorted(reply)} actions shape={shape} "
                f"rtt={rtt_ms:.1f}ms server_infer={timing['infer_ms']:.1f}ms"
            )
            print(f"    actions[0] = {np.array2string(reply['actions'][0], precision=3)}")
            if last_shape is not None and shape != last_shape:
                raise AssertionError(f"action shape changed across iters: {last_shape} -> {shape}")
            last_shape = shape

        arr = np.asarray(latencies)
        print(
            f"\n  round-trip latency over {num_iters} iters: "
            f"mean={arr.mean():.1f}ms p50={np.percentile(arr, 50):.1f}ms "
            f"p90={np.percentile(arr, 90):.1f}ms"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    num_iters: int = 3

    # Synthetic-sample knobs.
    img_height: int = 240
    img_width: int = 320
    raw_action_dim: int = 6
    prompt: str = "pick up the red cube and place it in the bowl"
    seed: int = 0


def main(args: Args) -> None:
    init_logging()
    logging.getLogger("websockets").setLevel(logging.WARNING)

    obs = make_synthetic_obs(
        img_height=args.img_height,
        img_width=args.img_width,
        raw_action_dim=args.raw_action_dim,
        prompt=args.prompt,
        seed=args.seed,
    )
    _describe_obs(obs)

    _round_trip(args.host, args.port, obs, args.num_iters)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
