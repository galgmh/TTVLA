# models

[中文版 / Chinese version](README_zh.md)

A research-friendly **simplified PyTorch implementation of Pi0 / Pi0.5**. The model behavior is kept faithful to the official release, while the upstream `openpi` package and the JAX runtime are stripped out, leaving a single, flat, readable, hackable PyTorch codebase.

> Training and inference are pure PyTorch. JAX is only optionally needed for a one-shot weight-conversion script.

---

## Why this repo exists

[The official openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence is the canonical implementation of Pi0 / Pi0.5. It is engineered very thoroughly, but for researchers who want to do downstream work — modifying the model, writing new losses, swapping in their own data pipeline — the barrier is not trivial:

1. **Coupled JAX / Flax NNX + PyTorch stacks**
   The official training stack is primarily JAX; the PyTorch side is mirrored from the JAX model. Understanding any module usually means reading both implementations, which is unfriendly to PyTorch-only researchers.
2. **Many internal abstractions inside `openpi`**
   `openpi.models`, `openpi.models_pytorch`, `openpi.shared`, `openpi.training.sharding`, and related modules reference each other, on top of sharding / array-typing / nnx_utils infrastructure. You essentially have to learn the whole framework before you can change a single forward pass.
3. **Heavy dependencies**
   A full run requires JAX + Flax + NNX + openpi's own training framework + a config registry system. The install and environment-debugging cost is high, and it makes the model hard to drop into an existing PyTorch project.
4. **Over-engineered configuration system**
   The official codebase uses a deep dataclass + registry hierarchy. For paper-level experiments, researchers typically prefer dropping hyperparameters straight into a script and changing one line — not registering a new `TrainConfig` first.

These are reasonable design choices for productionization, but they get in the way of a **modify-the-model / run-ablations / plug-in-my-own-dataset** research workflow. The goal of this repo is to turn Pi0 / Pi0.5 into **plain PyTorch code that a researcher can read at a glance and edit on the fly**.

---

## What has been simplified

* **PyTorch only**
  The model, training script ([finetune_new.py](finetune_new.py)) and inference server ([server_policy.py](server_policy.py)) all run on pure PyTorch + `torch.distributed`. JAX / Flax / NNX are no longer in the runtime path.
* **No `openpi` runtime dependency**
  All modules needed for Pi0 / Pi0.5 (PaliGemma + Action Expert, the flow-matching training objective, observation preprocessing, etc.) have been collected into [pi05_model/](pi05_model/):
  * [pi05_model/model.py](pi05_model/model.py) — `PI0Pytorch`, the flow-matching backbone
  * [pi05_model/gemma.py](pi05_model/gemma.py) — PaliGemma + Expert dual-tower
  * [pi05_model/preprocessing.py](pi05_model/preprocessing.py) — image / state preprocessing
  * [pi05_model/config.py](pi05_model/config.py) — single-file `Pi0Config`
  * [pi05_model/observation.py](pi05_model/observation.py) — Observation data structure
* **No JAX dependency (except for weight conversion)**
  The logic that converts the official JAX checkpoints to PyTorch weights lives separately under [convert/](convert/); JAX is only needed when you run `convert/convert_jax_to_torch.py`. Day-to-day training / fine-tuning / inference requires no JAX install at all.
* **Minimal `transformers` monkey-patches**
  The small modifications to HuggingFace `transformers` Gemma / PaliGemma / SigLIP have been consolidated under [pi05_model/_patches/](pi05_model/_patches/) and installed at entry-points via `install()`. From the outside, the standard `transformers` API is preserved.
* **Configs are plain Python**
  All experiment configurations live in the `ALLCONFIGS` dict in [configs/configs.py](configs/configs.py). Adding a new experiment = copy a dataclass block; no extra registration step.
* **Only LeRobot dataset supported**
  The RLDS datasets require dlimp and tensorflow installition, which is heavy and annoying. So we simply removed them. But, if you wish, you could just re-implement them for future use.
* **Flat directory layout**
  ```
  pi05_model/        # the model itself (PyTorch)
  data/              # data loading (LeRobot dataset)
  configs/           # experiment configs
  convert/           # JAX -> PyTorch weight conversion (one-shot)
  utils/             # normalize and other small helpers
  finetune_new.py    # training entry
  server_policy.py   # inference server (openpi-client protocol compatible)
  compute_norm_stats.py
  Test*.py           # sanity-check scripts for data / model / tokenizer / server
  ```
  No deep package nesting, no cross-module sharding/nnx_utils abstraction — finding code is just walking the directory tree.

---

## Relationship to the official repo

* **Behaviorally aligned**: the forward pass, attention masks, flow-matching sampling and action-expert architecture in `pi05_model` match `openpi.models_pytorch.pi0_pytorch`. Official JAX checkpoints can be converted into the `model.safetensors` this repo loads via `convert/convert_jax_to_torch.py`.
* **Protocol-compatible inference**: [server_policy.py](server_policy.py) implements the same websocket inference protocol as openpi, so existing **openpi-client** code (DROID, ALOHA, sim evaluators, etc.) talks to it unchanged.
* **Not aiming for feature parity**: the official training framework, sharding strategies, registry system, multi-task training, and other "complete engineering" features are intentionally not replicated. The goal stays "hackable for researchers", not "turnkey product".

---

## Quick start

End-to-end walkthrough using the simplest example, **`svla_so101_pickplace`** (the `test_run` config in [configs/configs.py](configs/configs.py)). All paths below are placeholders — replace them with your own.

### Step 0. Download assets

You will need three things on disk:

1. **The LeRobot dataset** — e.g. `lerobot/svla_so101_pickplace`. Download it via the `lerobot` CLI / `huggingface_hub`, then point `DataConfig.local_root` at the directory.
2. **The Pi0.5 JAX checkpoint** — e.g. `pi05_base` from the official openpi release. Used as the starting point for fine-tuning.
3. **The PaliGemma tokenizer** — `paligemma_tokenizer.model` (a SentencePiece file). Path goes into `DataConfig.tokenizer_path`.

### Step 1. Install the environment

See [Requirements](Requirements) for the full list. Key constraints:

* `python=3.12`, with `ffmpeg` from conda-forge
* `lerobot` installed from source (`pip install -e .`), plus `lerobot[dataset]`
* **`transformers==4.53.2`** — this is required, the `pi05_model._patches` monkey-patches are written against this exact version
* `jax==0.9.2` — **only** needed for Step 2 (weight conversion); not used at training / inference time
* `wandb`, `tyro`, `sentencepiece`, `numpydantic`

### Step 2. Convert the JAX checkpoint to PyTorch

```bash
python convert/convert_jax_to_torch.py \
    --checkpoint_dir /path/to/pi05_base \
    --output_path    /path/to/pi05_base_torch \
    --pi05
```

The resulting `pi05_base_torch` directory is what `TrainConfig.pytorch_weight_path` should point to.

### Step 3. Compute dataset normalization statistics

```bash
python compute_norm_stats.py \
    --args.repo_id        lerobot/svla_so101_pickplace \
    --args.local_root     /path/to/svla_so101_pickplace \
    --args.asset_id       svla_so101_pickplace_norm_stats+action_horizon_10 \
    --args.output_dir     /path/to/Assets \
    --args.action-horizon 10
```

Writes `norm_stats.json` to `<output_dir>/<asset_id>/`. Point `DataConfig.norm_stats_dir` at that directory.

### Step 4. Sanity-check scripts

Once the dataset, weights, tokenizer and norm stats are in place, run the self-checks:

```bash
python TestData.py        # dataset & repack transform — also writes TestData_batch_preview.png
python TestTokenizer.py   # paligemma tokenizer loads & encodes correctly
python Testpi05Model.py   # PI0Pytorch builds + forward / sample passes
```

All three should run cleanly before proceeding.

### Step 5. Fine-tune

Edit / add an entry in [configs/configs.py](configs/configs.py) (the `test_run` entry is the template), then:

```bash
# Single-GPU
python finetune_new.py --config_name test_run

# Single-node multi-GPU
torchrun --standalone --nnodes=1 --nproc_per_node=3 finetune_new.py --config_name test_run
```

Training metrics are pushed to **wandb** (toggle with `TrainConfig.wandb_enabled`). Checkpoints land under `<run_dir>/<exp_name>/checkpoints/<step>/`.

### Step 6. Serve and verify

Launch the inference server against a trained checkpoint:

```bash
python server_policy.py \
    --config_name     test_run \
    --checkpoint_dir  ./runs/test_run/checkpoints/30000 \
    --port            8000
```

Verify the server is healthy with the websocket smoke test:

```bash
python TestServer.py --host 127.0.0.1 --port 8000
```

After this passes, any openpi-client based runtime (DROID / ALOHA / sim evaluators) can connect to the same port.

---

## Who this is for

* Researchers who want to modify Pi0 / Pi0.5 — change the model, the loss, or the data;
* Anyone who wants to use Pi0 / Pi0.5 as **a plain PyTorch module** inside an existing PyTorch project;
* People who are unfamiliar with JAX, or do not want to learn a whole training framework just to read a model.

If your goal is to "reproduce the official training pipeline in production", please keep using [the official openpi](https://github.com/Physical-Intelligence/openpi).

---

## Citation

If this repository helps your work, please consider citing:

```bibtex
@inproceedings{sun2026mirth,
  title={MIRTH: Mutual-Information Reasoning with Temporal Hubs for Vision-Language-Action Agents},
  author={Sun, Hao and Song, Yu and Teng, Shiyu and Niu, Ziwei and Chen, Yen-Wei},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  address={San Diego, California, USA},
  pages={1--}
}
```

---

## Acknowledgements

The model implementation and weight format come from the [openpi](https://github.com/Physical-Intelligence/openpi) project by Physical Intelligence; this repo is only a secondary PyTorch-ification and code-simplification pass on top of it.
