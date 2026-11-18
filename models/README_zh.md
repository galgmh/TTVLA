# models

[English version](README.md)

一个面向研究者的 **Pi0 / Pi0.5 PyTorch 简化实现**。在保留官方模型行为一致性的前提下，剥离了上游 `openpi` 包与 JAX 运行时，重写为单一、扁平、可读、可改的 PyTorch 代码库。

> 训练 / 推理过程完全是纯 PyTorch；JAX 仅在一次性权重转换脚本中可选使用。

---

## 为什么会有这个仓库？

[官方 openpi](https://github.com/Physical-Intelligence/openpi) 是 Physical Intelligence 团队发布的 Pi0 / Pi0.5 实现，工程上非常完整，但对很多想做下游研究、修改模型结构、写新 loss、换数据 pipeline 的研究者来说，门槛偏高，主要体现在：

1. **JAX / Flax NNX 双栈耦合**
   官方训练栈以 JAX 为主，PyTorch 端是从 JAX 模型镜像出来的。要读懂一个模块往往得同时跟两套实现，对只熟悉 PyTorch 的同学不够友好。
2. **`openpi` 包内部抽象层较多**
   `openpi.models`、`openpi.models_pytorch`、`openpi.shared`、`openpi.training.sharding` 等模块互相引用，加上 sharding / array typing / nnx_utils 等基础设施，需要先理解整个框架才能改一个简单的前向。
3. **依赖较重**
   完整跑通需要 JAX + Flax + NNX + openpi 自身的训练框架 + 一套配置注册系统，安装与环境调试成本不低，也不利于在已有的 PyTorch 项目里直接复用模型。
4. **配置系统过于"工程化"**
   官方使用了较深的 dataclass + registry 体系。对论文级实验来说，往往希望把超参直接写在脚本里、改一行就能跑，而不是先去注册一个新的 `TrainConfig`。

这些问题对工程落地是合理的设计，但对 **改模型 / 做消融 / 接自己的数据集** 的研究流程不够顺手。本仓库的目标就是：**把 Pi0 / Pi0.5 拆成一份普通 PyTorch 研究者一眼能看懂、随手能改的代码**。

---

## 这个仓库做了什么简化

* **只保留 PyTorch**
  模型、训练 ([finetune_new.py](finetune_new.py))、推理服务 ([server_policy.py](server_policy.py)) 全部是纯 PyTorch + `torch.distributed`，不再依赖 JAX / Flax / NNX 运行。
* **去掉 `openpi` 运行时依赖**
  Pi0 / Pi0.5 所需的模块（PaliGemma + Action Expert、flow-matching 训练目标、observation 预处理等）被整理到 [pi05_model/](pi05_model/)：
  * [pi05_model/model.py](pi05_model/model.py) — `PI0Pytorch`，flow-matching 主体
  * [pi05_model/gemma.py](pi05_model/gemma.py) — PaliGemma + Expert 双塔
  * [pi05_model/preprocessing.py](pi05_model/preprocessing.py) — 图像 / 状态预处理
  * [pi05_model/config.py](pi05_model/config.py) — `Pi0Config` 单文件配置
  * [pi05_model/observation.py](pi05_model/observation.py) — Observation 数据结构
* **去掉 JAX 依赖（除权重转换外）**
  原版从官方 JAX checkpoint 到 PyTorch 权重的转换逻辑独立放在 [convert/](convert/) 下，只有运行 `convert/convert_jax_to_torch.py` 时才需要 JAX。日常训练 / 微调 / 推理完全不需要装 JAX。
* **`transformers` 的最小化 monkey-patch**
  对 HuggingFace `transformers` 中 Gemma / PaliGemma / SigLIP 的几处修改被整理为 [pi05_model/_patches/](pi05_model/_patches/)，由 `install()` 在入口处统一打上，对外仍然是标准 `transformers` API。
* **配置直接是 Python 文件**
  所有实验配置集中在 [configs/configs.py](configs/configs.py) 的 `ALLCONFIGS` 字典，新增一个实验 = 复制一段 dataclass，没有额外的注册流程。
* **仅支持LeRobot数据集**
  RLDS 数据格式需要dlimp和tensorflow安装，这非常繁琐。所以我们直接去除了。如果有需要，你也可以实现并加入。
* **扁平的目录结构**
  ```
  pi05_model/        # 模型本体（PyTorch）
  data/              # 数据加载（LeRobot dataset）
  configs/           # 实验配置
  convert/           # JAX → PyTorch 权重转换（一次性）
  utils/             # normalize 等小工具
  finetune_new.py    # 训练入口
  server_policy.py   # 推理服务（openpi-client 协议兼容）
  compute_norm_stats.py
  Test*.py           # 数据 / 模型 / tokenizer / server 的自检脚本
  ```
  没有多层 package、没有跨模块的 sharding/nnx_utils 抽象，找代码靠目录树就够了。

---

## 与官方仓库的关系

* **模型行为对齐**：`pi05_model` 中的前向、注意力 mask、flow-matching 采样、action expert 结构与 `openpi.models_pytorch.pi0_pytorch` 保持一致；可用 `convert/convert_jax_to_torch.py` 把官方 JAX checkpoint 转成本仓库可加载的 `model.safetensors`。
* **推理协议兼容**：[server_policy.py](server_policy.py) 实现了与 openpi 一致的 websocket 推理协议，因此 DROID / ALOHA / 各种 sim evaluator 这类 **openpi-client** 端代码无需修改即可对接。
* **不追求功能完备**：官方仓库中的训练框架、sharding 策略、注册系统、多任务训练等"完整工程"功能本仓库不复刻；目标始终是"够研究者改"，而不是"开箱即用的产品"。

---

## 快速上手

以最简单的 **`svla_so101_pickplace`**（即 [configs/configs.py](configs/configs.py) 中的 `test_run`）为例完整走一遍流程。下面所有路径都是占位，请替换成你自己的本地路径。

### Step 0. 下载所需资产

需要把以下三样东西准备好：

1. **LeRobot 数据集** — 例如 `lerobot/svla_so101_pickplace`。通过 `lerobot` CLI / `huggingface_hub` 下载到本地，路径填到 `DataConfig.local_root`。
2. **Pi0.5 JAX 权重** — 例如官方 openpi release 中的 `pi05_base`，用作微调起点。
3. **PaliGemma tokenizer** — `paligemma_tokenizer.model`（SentencePiece 文件），路径填到 `DataConfig.tokenizer_path`。

### Step 1. 安装环境

完整依赖见 [Requirements](Requirements)，关键约束：

* `python=3.12`，并通过 conda-forge 装 `ffmpeg`
* `lerobot` 源码安装（`pip install -e .`），并安装 `lerobot[dataset]`
* **`transformers==4.53.2`** — 必须是这个版本，`pi05_model._patches` 的 monkey-patch 是针对它写的
* `jax==0.9.2` — **仅** Step 2 权重转换时需要，训练 / 推理过程都不会用到
* `wandb`、`tyro`、`sentencepiece`、`numpydantic`

### Step 2. JAX 权重转 PyTorch

```bash
python convert/convert_jax_to_torch.py \
    --checkpoint_dir /path/to/pi05_base \
    --output_path    /path/to/pi05_base_torch \
    --pi05
```

转换后的 `pi05_base_torch` 目录就是 `TrainConfig.pytorch_weight_path` 要指向的位置。

### Step 3. 计算数据集的 normalization 统计量

```bash
python compute_norm_stats.py \
    --args.repo_id        lerobot/svla_so101_pickplace \
    --args.local_root     /path/to/svla_so101_pickplace \
    --args.asset_id       svla_so101_pickplace_norm_stats+action_horizon_10 \
    --args.output_dir     /path/to/Assets \
    --args.action-horizon 10
```

会在 `<output_dir>/<asset_id>/` 下生成 `norm_stats.json`。把该目录路径填到 `DataConfig.norm_stats_dir`。

### Step 4. 自检脚本

数据集、权重、tokenizer、norm stats 都到位后，先跑三个自检脚本：

```bash
python TestData.py        # 数据集 & repack transform — 同时会生成 TestData_batch_preview.png
python TestTokenizer.py   # paligemma tokenizer 加载与编码是否正常
python Testpi05Model.py   # PI0Pytorch 构建 + 前向 / 采样是否能跑通
```

三个都通过后再继续。

### Step 5. 微调训练

在 [configs/configs.py](configs/configs.py) 中编辑或新增一个配置（`test_run` 是模板），然后：

```bash
# 单卡
python finetune_new.py --config_name test_run

# 单机多卡
torchrun --standalone --nnodes=1 --nproc_per_node=3 finetune_new.py --config_name test_run
```

训练指标会推送到 **wandb**（开关在 `TrainConfig.wandb_enabled`）。Checkpoint 保存在 `<run_dir>/<exp_name>/checkpoints/<step>/`。

### Step 6. 部署 & 验证

用训练好的 checkpoint 启动推理服务：

```bash
python server_policy.py \
    --config_name     test_run \
    --checkpoint_dir  ./runs/test_run/checkpoints/30000 \
    --port            8000
```

用 websocket smoke test 验证服务是否正常：

```bash
python TestServer.py --host 127.0.0.1 --port 8000
```

走通这一步后，所有基于 openpi-client 的运行环境（DROID / ALOHA / sim evaluator 等）都可以直接连这个端口。

---

## 适合谁用

* 想在 Pi0 / Pi0.5 基础上做改模型 / 改 loss / 改数据的研究者；
* 希望把 Pi0 / Pi0.5 当作一个**普通 PyTorch 模块**嵌入到已有 PyTorch 项目里的人；
* 对 JAX 不熟悉、或不想为了读懂模型先学一套训练框架的人。

如果你的需求是"在生产环境复现官方训练 pipeline"，请仍然使用 [官方 openpi](https://github.com/Physical-Intelligence/openpi)。

---

## 引用

如果这个仓库对你的工作有帮助，请引用：

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

## 致谢

模型实现与权重格式来自 Physical Intelligence 团队的 [openpi](https://github.com/Physical-Intelligence/openpi)；本仓库仅做了 PyTorch 化与代码精简的二次整理。
