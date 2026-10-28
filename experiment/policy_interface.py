"""The policy interface used by ``experiment/eval_closed_loop.py``.

A learned policy sees exactly a :class:`PolicyObservation`:

* the two RGB frames,
* ``state`` assembled by :mod:`experiment.common` (proprio16 / touch28 /
  full30),
* the episode's language instruction.

No ground-truth object pose and no ground-truth object temperature is exposed;
a policy that needs them is outside the experiment. :class:`ExpertPolicy` is
the one deliberate exception: it wraps the scripted expert so the evaluation
loop itself can be validated against the generator, and the expert's API takes
the raw observation dict. That bypass is never used for a learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from experiment.common import assemble_state_from_obs


@dataclass(frozen=True)
class PolicyObservation:
    """One policy-rate observation, in the units the dataset stores."""

    agentview: np.ndarray  # (H, W, 3) uint8, RGB
    wrist: np.ndarray | None  # (H, W, 3) uint8, RGB, when the camera exists
    state: np.ndarray  # (D,) float32, layout from experiment.common
    instruction: str
    step: int = 0


class Policy(Protocol):
    """Minimal interface a baseline must implement."""

    name: str

    def reset(self, instruction: str, *, rng: np.random.Generator | None = None) -> None:
        """Start a new episode. ``rng`` is the episode's generator (for experts)."""

    def act(self, observation: PolicyObservation) -> np.ndarray:
        """Return a 7-dim normalized delta action."""


class ZeroPolicy:
    """No-op arm, open gripper -- the floor a real policy must beat."""

    name = "zero"

    def reset(self, instruction: str, *, rng: np.random.Generator | None = None) -> None:
        self._instruction = instruction

    def act(self, observation: PolicyObservation) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class RandomPolicy:
    """Uniform random actions -- a harness smoke test, not a baseline."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        self._episode = 0
        self._rng = np.random.default_rng(self._seed)

    def reset(self, instruction: str, *, rng: np.random.Generator | None = None) -> None:
        # Do not draw from `rng`: that stream belongs to the episode/ expert.
        self._rng = np.random.default_rng(self._seed + self._episode)
        self._episode += 1
        self._instruction = instruction

    def act(self, observation: PolicyObservation) -> np.ndarray:
        action = self._rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
        return action


class ExpertPolicy:
    """Wraps the scripted expert; used only to validate the harness.

    The expert reads the raw observation (tactile + measured fingertip
    temperature), so this class deliberately does *not* implement ``act`` on a
    :class:`PolicyObservation`. ``eval_closed_loop.py`` calls ``act_raw``.
    """

    name = "expert"

    def __init__(self, cfg: dict[str, Any], bundle: Any, *, pipeline: Any | None = None):
        from sim.expert.expert import build_expert

        self._bundle = bundle
        self.expert = build_expert(
            cfg["expert"],
            pipeline=pipeline,
            pos_scale_m=float(cfg["controller"]["pos_scale_m"]),
            rot_scale_rad=float(cfg["controller"]["rot_scale_rad"]),
        )

    def reset(self, instruction: str, *, rng: np.random.Generator | None = None) -> None:
        self._instruction = instruction
        self.expert.reset(
            self._bundle,
            rng if rng is not None else np.random.default_rng(),
        )

    def act_raw(self, obs: dict[str, Any]) -> np.ndarray:
        action, _info = self.expert.act(obs)
        return np.asarray(action, dtype=np.float32).reshape(-1)[:7]

    def act(self, observation: PolicyObservation) -> np.ndarray:
        raise RuntimeError(
            "ExpertPolicy reads the raw observation; eval_closed_loop.py calls act_raw()"
        )

    @property
    def phase(self) -> str:
        return str(getattr(self.expert.state, "phase", ""))


class RemotePolicy:
    """Client for an openpi policy server (pi0.5, or any websocket policy).

    The observation dict uses openpi's LIBERO keys by default:

        observation/image, observation/wrist_image, observation/state, prompt

    If the server's input transform expects other keys, pass them here (or edit
    ``_infer``); the values are exactly the dataset's channels plus the
    instruction.

    ``requery`` controls action chunking:

    * ``1`` (default): ask the server every control step. Required for the
      thermal rungs, where the decision happens after contact and a long chunk
      would lock the action before the fingertip has answered.
    * ``N > 1``: ask every N steps and execute the returned chunk in between.
    * ``0``: ask once and execute the whole returned chunk.
    """

    name = "remote"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        image_key: str = "observation/image",
        wrist_key: str = "observation/wrist_image",
        state_key: str = "observation/state",
        prompt_key: str = "prompt",
        action_dim: int = 7,
        requery: int = 1,
    ):
        self.host = str(host)
        self.port = int(port)
        self.image_key = image_key
        self.wrist_key = wrist_key
        self.state_key = state_key
        self.prompt_key = prompt_key
        self.action_dim = int(action_dim)
        self.requery = int(requery)
        self._client = None
        self._instruction = ""
        self._queue: list[np.ndarray] = []
        self._steps = 0

    # ------------------------------------------------------------------ client

    def _connect(self) -> None:
        if self._client is not None:
            return
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "ERROR: RemotePolicy needs openpi's client. Install it on the "
                "evaluation machine: pip install openpi-client (from the openpi "
                "repo, packages/openpi-client), or run the eval where openpi is "
                "installed."
            ) from exc
        try:
            self._client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)
        except Exception as exc:  # noqa: BLE001 - connection errors are transport specific
            raise SystemExit(
                f"ERROR: cannot connect to the policy server at "
                f"{self.host}:{self.port} ({exc}); start "
                f"scripts/serve_policy.py on the GPU machine first"
            ) from exc
        print(f"remote policy: connected to {self.host}:{self.port}")

    # ------------------------------------------------------------------- policy

    def reset(self, instruction: str, *, rng: np.random.Generator | None = None) -> None:
        self._connect()
        self._instruction = instruction
        self._queue.clear()
        self._steps = 0
        # A server-side reset is optional across openpi versions.
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 - optional protocol detail
                pass

    def act(self, observation: PolicyObservation) -> np.ndarray:
        self._connect()
        refill = not self._queue or (self.requery > 0 and self._steps % self.requery == 0)
        if refill:
            actions = self._infer(observation)
            self._queue = [row for row in actions]
        if not self._queue:
            raise RuntimeError("policy server returned no actions")
        action = np.asarray(self._queue.pop(0), dtype=np.float32).reshape(-1)
        self._steps += 1
        if action.size < self.action_dim:
            raise ValueError(
                f"policy returned {action.size} action dims, need {self.action_dim}"
            )
        return action[: self.action_dim]

    def _infer(self, observation: PolicyObservation) -> np.ndarray:
        payload = {
            self.image_key: np.asarray(observation.agentview, dtype=np.uint8),
            self.state_key: np.asarray(observation.state, dtype=np.float32),
            self.prompt_key: observation.instruction,
        }
        if observation.wrist is not None:
            payload[self.wrist_key] = np.asarray(observation.wrist, dtype=np.uint8)

        result = self._client.infer(payload)
        if isinstance(result, dict):
            result = result.get("actions", result.get("action"))
        actions = np.asarray(result, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"unexpected action shape from server: {actions.shape}")
        return actions


__all__ = [
    "ExpertPolicy",
    "Policy",
    "PolicyObservation",
    "RandomPolicy",
    "RemotePolicy",
    "ZeroPolicy",
]
