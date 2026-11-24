"""Pi0.5 PyTorch inference server.

Loads a fine-tuned `PI0Pytorch` checkpoint and serves it via the openpi
websocket protocol so that openpi-client based runtimes (DROID, ALOHA, sim
evaluators, etc.) can talk to it unchanged.

Usage
-----
  python server_policy.py \
      --config_name test_run \
      --checkpoint_dir ./runs/test_run/checkpoints/30000 \
      --port 8000

The config name is looked up in `configs.configs.ALLCONFIGS` so it shares the
exact same DataConfig / Pi0Config that the matching training run used. Dense
runs load `model.safetensors`; LoRA runs load the configured base model plus
`adapter_model.safetensors`.
"""
# Patches MUST be installed before any code touches `transformers`.
from pi05_model._patches import install as install_transformers_patches
install_transformers_patches()

import asyncio
import dataclasses
import functools
import http
import logging
import pathlib
import socket
import time
import traceback
from typing import Any

import msgpack
import numpy as np
import safetensors.torch
import torch
import tyro
import websockets.asyncio.server as _ws_server
import websockets.frames

from pi05_model.config import Pi0Config
from pi05_model.model import PI0Pytorch
from pi05_model.lora import inject_lora, load_lora_adapter
from pi05_model.observation import Observation
from pi05_model.preprocessing import IMAGE_RESOLUTION
from data.data_loader import (
    DataConfig,
    InjectDefaultPrompt,
    Normalize,
    NormalizeImages,
    PadStatesAndActions,
    PaligemmaTokenizer,
    ResizeImages,
    TokenizePrompt,
)
from utils.normalize import NormStats, load as load_norm_stats


# ---------------------------------------------------------------------------
# msgpack numpy codec (wire-compatible with openpi_client.msgpack_numpy so
# existing openpi-client runtimes can talk to this server unchanged).
# ---------------------------------------------------------------------------
def _pack_array(obj):
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


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_msgpack_packer = functools.partial(msgpack.Packer, default=_pack_array)
_msgpack_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


# ---------------------------------------------------------------------------
# Inference-side transforms
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Unnormalize:
    """Inverse of `data.data_loader.Normalize` for predicted action arrays.

    Applied after the model emits a padded action tensor; this both undoes the
    normalization and slices the trailing axis back to the dataset's true
    action_dim (taken from the norm_stats shape).
    """

    norm_stats: dict[str, NormStats] | None
    use_quantiles: bool = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.norm_stats is None:
            return data
        out = dict(data)
        for key, stats in self.norm_stats.items():
            if key not in out:
                continue
            x = np.asarray(out[key], dtype=np.float32)
            true_dim = stats.mean.shape[-1]
            x = x[..., :true_dim]
            if self.use_quantiles:
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(f"Quantile unnorm requested but stats for {key!r} have no q01/q99")
                out[key] = (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
            else:
                out[key] = x * (stats.std + 1e-6) + stats.mean
        return out


def build_input_transforms(data_config: DataConfig, model_config: Pi0Config):
    """Mirrors the training pipeline in `data.data_loader.create_data_loader`,
    minus the LeRobot-specific `PromptFromLeRobotTask` (the client supplies the
    prompt directly) and `Normalize` over `actions` (no ground-truth actions at
    inference time)."""
    if data_config.tokenizer_path is None:
        raise ValueError("DataConfig.tokenizer_path is required for inference")

    tokenizer = PaligemmaTokenizer(data_config.tokenizer_path, max_len=model_config.max_token_len)
    transforms: list = [*data_config.repack_transforms]
    if data_config.default_prompt is not None:
        transforms.append(InjectDefaultPrompt(data_config.default_prompt))
    transforms.extend(data_config.data_transforms)
    transforms.append(Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.append(NormalizeImages())
    transforms.append(ResizeImages(*IMAGE_RESOLUTION))
    transforms.append(TokenizePrompt(tokenizer, discrete_state_input=model_config.pi05))
    transforms.append(PadStatesAndActions(model_config.action_dim))
    return transforms


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------
class Pi05Policy:
    """Inference-only wrapper around `PI0Pytorch`.

    `infer(obs)` accepts a single (unbatched) dict with raw `images`, `state`,
    and `prompt` (or LeRobot-style nested keys consumed by the repack
    transform), and returns a dict with `actions` of shape
    (action_horizon, true_action_dim) in real-world units.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        input_transforms: list,
        output_transforms: list,
        device: torch.device,
        num_sample_steps: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._input_transforms = list(input_transforms)
        self._output_transforms = list(output_transforms)
        self._device = device
        self._num_sample_steps = int(num_sample_steps)
        self._metadata = metadata or {}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @torch.no_grad()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        sample = dict(obs)
        prompt = sample["prompt"]
        for t in self._input_transforms:
            sample = t(sample)
            if t.__class__.__name__ == "RepackTransform":
                # After tokenization, we have no PromptFromLeRobotTask, so the prompt must be explicitly carried through the repack transform to the model input.
                sample["prompt"] = prompt

        observation = self._sample_to_observation(sample)

        start = time.monotonic()
        actions = self._model.sample_actions(
            self._device, observation, num_steps=self._num_sample_steps
        )
        infer_ms = (time.monotonic() - start) * 1000.0

        actions_np = actions[0].detach().to(torch.float32).cpu().numpy()
        outputs: dict[str, Any] = {
            "state": sample["state"],
            "actions": actions_np,
        }
        for t in self._output_transforms:
            outputs = t(outputs)
        outputs["policy_timing"] = {"infer_ms": infer_ms}
        return outputs

    def _sample_to_observation(self, sample: dict[str, Any]) -> Observation:
        device = self._device
        images = {
            k: torch.from_numpy(np.ascontiguousarray(v)).to(device)[None, ...]
            for k, v in sample["images"].items()
        }
        if "image_masks" in sample:
            # Accept scalar / 1-D / 2-D masks from upstream and reduce them to
            # one value per sample: `embed_prefix` expects shape (B,). The
            # server serves a single observation, so B == 1.
            image_masks = {
                k: torch.from_numpy(np.asarray(v, dtype=bool).reshape(-1)[:1]).to(device)
                for k, v in sample["image_masks"].items()
            }
        else:
            image_masks = {k: torch.ones(1, dtype=torch.bool, device=device) for k in images}

        state = torch.from_numpy(np.ascontiguousarray(sample["state"])).to(torch.float32).to(device)[None, ...]
        tokens = torch.from_numpy(np.ascontiguousarray(sample["tokenized_prompt"])).to(torch.long).to(device)[None, ...]
        tokens_mask = torch.from_numpy(np.ascontiguousarray(sample["tokenized_prompt_mask"])).to(torch.bool).to(device)[None, ...]

        return Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=tokens_mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )


class WebsocketPolicyServer:
    """Wire-compatible with `openpi.serving.websocket_policy_server` so existing
    openpi-client runtimes connect unchanged."""

    def __init__(
        self,
        policy: Pi05Policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self):
        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _ws_server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = _msgpack_packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = _msgpack_unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _ws_server.ServerConnection, request: _ws_server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def load_policy(
    config_name: str,
    checkpoint_dir: str | pathlib.Path,
    default_prompt: str | None = None,
    device: str | None = None,
    num_sample_steps: int = 10,
    pytorch_inference_precision: str = "bfloat16",
) -> Pi05Policy:
    from configs.configs import ALLCONFIGS

    if config_name not in ALLCONFIGS:
        raise ValueError(
            f"Unknown config name {config_name!r}. Available: {sorted(ALLCONFIGS)}"
        )

    bundle = ALLCONFIGS[config_name]
    data_config: DataConfig = bundle["data_config"]
    model_config: Pi0Config = bundle["model_config"]
    train_config = bundle["train_config"]

    # Training-time `create_data_loader` hydrates norm stats from `norm_stats_dir`;
    # the server uses `data_config.norm_stats` directly, so load them here.
    # Without this, Normalize/Unnormalize silently no-op.
    if data_config.norm_stats is None and data_config.norm_stats_dir is not None:
        data_config = dataclasses.replace(
            data_config,
            norm_stats=load_norm_stats(pathlib.Path(data_config.norm_stats_dir)),
        )
        logging.info(f"Loaded norm stats from {data_config.norm_stats_dir}")
    if data_config.norm_stats is None:
        raise ValueError(
            "DataConfig has neither norm_stats nor norm_stats_dir; "
            "serving without normalization would not match training."
        )

    if default_prompt is not None:
        data_config = dataclasses.replace(data_config, default_prompt=default_prompt)
    if data_config.norm_stats is None and data_config.norm_stats_dir is not None:
        data_config = dataclasses.replace(
            data_config,
            norm_stats=load_norm_stats(data_config.norm_stats_dir),
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    object.__setattr__(model_config, "dtype", pytorch_inference_precision)
    logging.info(f"Building PI0Pytorch on {torch_device} (dtype={model_config.dtype})")
    model = PI0Pytorch(model_config).to(torch_device)

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    weights_path = checkpoint_dir / "model.safetensors"
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if adapter_path.is_file():
        if not train_config.lora.enabled:
            raise ValueError(
                f"{adapter_path} is a LoRA checkpoint but config {config_name!r} has LoRA disabled"
            )
        base_path = pathlib.Path(train_config.pytorch_weight_path) / "model.safetensors"
        if not base_path.is_file():
            raise FileNotFoundError(f"LoRA base model not found at {base_path}")
        logging.info(f"Loading LoRA base weights from {base_path}")
        safetensors.torch.load_model(model, str(base_path), device=str(torch_device))
        inject_lora(model, train_config.lora)
        load_lora_adapter(model, checkpoint_dir)
        logging.info(f"Loaded LoRA adapter from {adapter_path}")
    elif weights_path.is_file():
        logging.info(f"Loading weights from {weights_path}")
        safetensors.torch.load_model(model, str(weights_path), device=str(torch_device))
    else:
        raise FileNotFoundError(
            f"No model.safetensors or adapter_model.safetensors found in {checkpoint_dir}"
        )
    model.eval()

    input_transforms = build_input_transforms(data_config, model_config)
    output_transforms = [
        Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
    ]

    metadata = {
        "config_name": config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "action_horizon": model_config.action_horizon,
        "action_dim": model_config.action_dim,
        "pi05": model_config.pi05,
    }
    return Pi05Policy(
        model=model,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
        device=torch_device,
        num_sample_steps=num_sample_steps,
        metadata=metadata,
    )


@dataclasses.dataclass
class Args:
    """Arguments for serving a Pi0.5 PyTorch policy."""

    # Config name in `configs.configs.ALLCONFIGS` (defines data/model config).
    config_name: str

    # Directory holding `model.safetensors` or `adapter_model.safetensors`. Typically
    # `./runs/<exp>/checkpoints/<step>` written by `finetune_new.save_checkpoint`.
    checkpoint_dir: str

    # Port for the websocket server.
    port: int = 8000

    # Host to bind. Leave at 0.0.0.0 for remote clients.
    host: str = "0.0.0.0"

    # Injected when the inbound obs does not already carry a `prompt` key.
    default_prompt: str | None = None

    # Number of Euler steps for the flow-matching denoiser.
    num_sample_steps: int = 10

    # Inference precision for `PaliGemmaWithExpertModel`.
    pytorch_inference_precision: str = "bfloat16"

    # Torch device override (e.g. "cuda:0"). Auto-detects when None.
    device: str | None = None


def main(args: Args) -> None:
    init_logging()

    policy = load_policy(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        device=args.device,
        num_sample_steps=args.num_sample_steps,
        pytorch_inference_precision=args.pytorch_inference_precision,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    logging.info(f"Creating server (host: {hostname}, ip: {local_ip}, port: {args.port})")

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
