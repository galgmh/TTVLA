# experiment/ — 纯视觉 baseline 的导出、评测与训练接口

这个目录把仿真器与训练栈隔开：`sim/` 只通过 HDF5 产出数据，
`experiment/` 把它转成 LeRobot 数据集、定义策略 I/O 契约、跑闭环评测。
`sim` 不 import 这里的任何东西。

主 baseline：**π0.5 + 16 维本体状态（V+P）**。
受控消融：16D（V+P）→ 28D（V+P+Touch）→ 30D（V+P+Touch+T）。
外部对照：OpenVLA-OFT（后续，走 RLDS 导出，不在本目录当前范围内）。

---

## 1. 文件

| 文件 | 作用 | 依赖 |
|---|---|---|
| `common.py` | state / action 的唯一契约；导出与评测共用 | numpy |
| `export_lerobot.py` | format-1.6 HDF5 → LeRobot v2.1 | h5py, numpy, pyarrow, Pillow, imageio |
| `check_lerobot.py` | 校验导出目录（元数据 / parquet / 视频 / stats / splits） | pyarrow；`--decode` 加 imageio；`--with-lerobot` 加 lerobot |
| `policy_interface.py` | `Policy` 协议、`PolicyObservation`、`Zero/Random/Expert/Remote` | numpy；`RemotePolicy` 加 openpi-client |
| `eval_closed_loop.py` | Genesis 闭环评测（任务成功率 / 操作成功率） | Genesis + 整个 `sim` |

---

## 2. 数据契约

### 2.1 state（`observation.state`，float32）

| 模式 | 维度 | 布局 |
|---|---|---|
| `proprio16` | 16 | `ee_pos(3)` + `ee_ori(3)` + `gripper(1)` + `joint_states(9)` |
| `touch28` | 28 | proprio16 + `tactile/force(12)`（2 指 × `[Fn(3), Ft(3)]`，finger-major） |
| `full30` | 30 | proprio16 + 实测 `fingertip_grid(2)` + `tactile/force(12)` |

`ee_ori` 是世界系旋转向量（rad）；`gripper` 是 -1 张开 … +1 闭合；
9 个关节 = 7 臂 + 2 指。名字表在 `common.STATE_NAMES`，同时写进
`meta/info.json → features["observation.state"]["names"]`。

**只喂策略可见的通道。** 真值温度（`temperature/object`、`object_grid`、
`objects_grid`、`link_temps`）和固定行序的物体位姿都不在 state 里：
它们会泄露答案，看了它们就不是 baseline。

### 2.2 action（`action`，7 维）

与 LIBERO 一致的世界系 delta：

```
a[0:3] * controller.pos_scale_m   -> TCP 平移增量
a[3:6] * controller.rot_scale_rad -> TCP 旋转向量增量
a[6]                              -> 夹爪（-1 张开 … +1 闭合）
```

### 2.3 图像与语言

| LeRobot key | HDF5 通道 | 说明 |
|---|---|---|
| `observation.images.image` | `agentview_rgb` | 固定第三人称 |
| `observation.images.wrist_image` | `eye_in_hand_rgb` | 腕部相机 |
| `meta/tasks.jsonl` | 每条 demo 的 `language_instruction` attr | 去重后按首次出现编号 |

键名是标准 LeRobot v2.1 写法。openpi 的 `LeRobotLiberoDataConfig` **不能直接用**：
它面向 openpi 自己转换的 LIBERO 数据集，repack 映射期望原始列名 `image` /
`wrist_image` / `state` / `actions`，且 `action_sequence_keys=("actions",)`。
我们的导出是 `observation.images.*` / `observation.state` / `action`，所以要像
openpi 类注释建议的那样复制该类，只改 repack 映射和 action key（见 §4 的
`PickColdCanDataConfig`）。

---

## 3. 导出 + 校验（数据机器，不需要 Genesis）

```bash
cd ~/ThermTacVLA
conda activate tempvla
pip install pyarrow imageio imageio-ffmpeg Pillow
python experiment/export_lerobot.py datasets/hdf5/pick_cold_can/pick_cold_can_aluminum_aluminum_lemon_paperboard_aluminum_balsa_iron.hdf5 --out datasets/lerobot/local/pick_cold_can_p16
python experiment/check_lerobot.py datasets/lerobot/local/pick_cold_can_p16 --decode
```

`--out` 故意写成 `.../lerobot/local/<name>`：LeRobot 把 `repo_id="local/<name>"`
解析到 `$HF_LEROBOT_HOME/local/<name>`，训练时只需导出环境变量（见 §4）。

导出结果（50 条成功 demo）：

- 50 个 episode，44 train / 4 val / 2 test（连续切分，写在 `meta/info.json → splits`）
- `data/chunk-000/episode_XXXXXX.parquet` 每集一个
- `videos/chunk-000/<camera_key>/episode_XXXXXX.mp4` 每集每相机一个
- `meta/{info,tasks,episodes,episodes_stats,stats}.json[l]`

常用参数：`--state-mode touch28|full30`（消融）、`--image-size 0`（保留 512 原图）、
`--include-failed`、`--limit N`、`--overwrite`（只清 `data/ videos/ meta/`，
不会动父目录）。

安装 `lerobot` 后再跑一次加载器检查（最接近 openpi 的真实读取路径）：

```bash
pip install lerobot
python experiment/check_lerobot.py datasets/lerobot/local/pick_cold_can_p16 --with-lerobot --state-mode proprio16
```

注意版本：本导出是 **v2.1** 布局。openpi 用的 lerobot 固定版本能直接读；
最新的 LeRobot v3 需要先做格式转换，装之前先看 `~/openpi/pyproject.toml`
固定的版本。如果 `--with-lerobot` 报 `codebase_version` 不匹配，换成 openpi
环境里的那个 lerobot 再跑。

`check_lerobot.py` 逐项核对：episode 数与帧数、parquet 列名/行数/全局 `index`、
`task_index` 与语言指令一致、`timestamp == frame_index / fps`、
视频存在（`--decode` 时还核对帧数/尺寸/fps）、`stats.json` 与逐集 stats、
splits 不重不漏。最后一行 `OK` 才算通过。

---

## 4. π0.5 微调（官方 LoRA 路线，云上 GPU）

采用 openpi 的低显存 LoRA 机制（仓库里的 `pi0_libero_low_mem_finetune`
是 π0 的官方配置）：`paligemma_variant="gemma_2b_lora"` +
`action_expert_variant="gemma_300m_lora"`，`freeze_filter` 由
`Pi0Config.get_freeze_filter()` 生成。效果：

- **冻结**：PaliGemma 2B 与 action expert 300M 的全部原始权重；
- **可训**：两边的 LoRA 适配器、SigLIP 视觉编码器、action 投影与时间 MLP。

上游没有发布 π0.5 + LoRA 的现成配置（π0.5 的官方配置都是全量微调），
这里是同一套机制第一次套到 π0.5 上：先跑 1k 步确认 loss 和显存，再放全量。
openpi README 标称 LoRA 微调需要 >22.5 GB 显存，3090（24 GB）在范围内。
`get_freeze_filter()` 没有 vision 模式，所以 SigLIP 仍参与训练；
要连它也冻住需要手写 filter，不是官方组合，先不用。

训练走 JAX（`scripts/train.py`）。openpi 的 PyTorch 路径
（`scripts/train_pytorch.py`，torch 2.7.1）明确不支持 LoRA、FSDP、混合精度，
全量微调需要 >70 GB 显存——3090 上不可行。PyTorch 只在环境里作为依赖安装
（数据管线等），不用于训练。

在 `src/openpi/training/config.py` 里新增下面两个东西（`pi0_config`、`weight_loaders`、
`DataConfig`、`DataConfigFactory`、`ModelTransformFactory`、`_transforms`、`_model`、
`libero_policy`、`pathlib`、`override` 均已在该文件导入）。

先加数据工厂：复制 `LeRobotLiberoDataConfig`，把 repack 的来源键换成本数据集的标准
键名，并把 action 序列键改成 `("action",)`（否则 loader 会去找不存在的 `actions` 列）：

```python
@dataclasses.dataclass(frozen=True)
class PickColdCanDataConfig(DataConfigFactory):
    """LIBERO-style transforms for the ThermTacVLA LeRobot export."""

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
```

然后是训练配置：

```python
TrainConfig(
    name="pick_cold_can_pi05_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=32,        # state/action 都被补齐到 32 维
        action_horizon=16,    # 接触后要尽快重新决策，默认 50 太长
        discrete_state_input=False,  # 16 维连续状态，同 pi05_libero
    ),
    data=PickColdCanDataConfig(
        repo_id="local/pick_cold_can_p16",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,  # 我们的动作已经是 delta
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    num_train_steps=20_000,
    batch_size=32,          # 全局 batch，多卡均分
    fsdp_devices=1,         # 纯数据并行
    ema_decay=None,         # 省一份权重副本
),
```

如果该版本用 `_CONFIGS` 列表注册训练配置，把这个条目也加进去。

三个字段不能省：`base_config=DataConfig(prompt_from_task=True)` 从 LeRobot
的 `task` 字段取指令；`extra_delta_transform=False` 防止 delta 被叠加两次；
`discrete_state_input=False` 对应 16 维连续状态。

运行（2 卡示例；4 卡把 `CUDA_VISIBLE_DEVICES` 写全）：

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git ~/openpi
cd ~/openpi
uv sync
mkdir -p ~/datasets/lerobot/local
export HF_LEROBOT_HOME=$HOME/datasets/lerobot
export OPENPI_DATA_HOME=$HOME/.cache/openpi   # 权重缓存；云上应指到大盘
uv run scripts/compute_norm_stats.py --config-name pick_cold_can_pi05_lora
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py pick_cold_can_pi05_lora --exp-name=trial1 --overwrite
```

数据集整个 `local/pick_cold_can_p16` 目录一起拷贝，`HF_LEROBOT_HOME` 指向它的
上一级 `.../datasets/lerobot`。没有 wandb 就加 `--no-wandb-enabled`。

- 20k 步 × batch 32 ≈ 43 个 epoch（14710 帧）。50 条 demo 下过拟合风险不低，
  用闭环成功率选 checkpoint（`keep_period=5000`，5k/10k/15k 和最后的目录都会留），
  不要只看最后一步或 loss。
- checkpoint 写到 `checkpoint_base_dir`（云上指到大盘，如
  `/root/autodl-fs/openpi-checkpoints`；默认是 `~/openpi/checkpoints`）。
  确认 `save_interval`/`keep_period` 的实际取值，并用 `du -sh` 看单个
  checkpoint 的大小，避免写满系统盘。
- OOM 时先降 `batch_size`（每卡 8 起步），再考虑 `fsdp_devices=2`。
- 本路线 action expert 是 LoRA 而非全量；要改成全量就把
  `action_expert_variant` 换成 `gemma_300m`（filter 会自动保留它可训）。

**注意**：本轮用的是全量 50 集，`meta/info.json` 里的 val/test 6 集也在训练里。
held-out 只有评测时的新种子（默认 1000+），报告里必须写明。

---

## 5. 闭环评测

评测在 8 GB 的 4060 机器上跑 Genesis，策略可以在本机，也可以在网络另一端的
openpi 服务器上。

先做两个不依赖任何 checkpoint 的冒烟（验证评测循环本身）：

```bash
cd ~/ThermTacVLA
conda activate tempvla
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
python experiment/eval_closed_loop.py --policy zero --episodes 1
python experiment/eval_closed_loop.py --policy expert --episodes 4
```

- `zero`：动作全 0、夹爪张开，应该 0 任务成功率，用来看循环能跑完。
- `expert`：包住脚本专家，用于**验证评测循环**（步数、成功判定、场景 seed
  都与生成器一致）。它读原始触觉/温度，不是学习 baseline，只做校验。

### 5.1 接 openpi 服务器

服务器机（训练机，有 GPU）：

```bash
cd ~/openpi
ls checkpoints/pick_cold_can_pi05_lora/trial1
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pick_cold_can_pi05_lora --policy.dir=checkpoints/pick_cold_can_pi05_lora/trial1/19999
```

先 `ls` 看实际留下的 step 目录（20k 步时最后一个是 `19999`，中间档是
`4999/9999/14999`），换 checkpoint 就换最后一段路径。训练时若设了
`checkpoint_base_dir`（云上为 `/root/autodl-fs/openpi-checkpoints`），根目录就不是
`~/openpi/checkpoints`，以 `ls` 到的实际路径为准。

评测机（只需客户端，不需要 openpi 的训练依赖）：

```bash
pip install -e ~/openpi/packages/openpi-client
python experiment/eval_closed_loop.py --policy remote --episodes 20 --remote-host 192.168.1.20 --remote-port 8000 --state-mode proprio16
```

`RemotePolicy` 每步都向服务器要一次动作（`requery=1`）。这是温度任务的
默认：长动作块会锁死接触后的反应。只做纯视觉 baseline 时可以放宽，
例如 `--remote-requery 8`，但为了各档可比，建议统一用 1。

### 5.2 输出与指标

每次运行写一个目录（默认 `experiment/runs/<policy>_<state>_<时间戳>/`）：

- `records.jsonl`：逐 episode 的 `success`、`manipulation_success`、步数、
  指令、目标真值温度（**只进日志，不进策略**）、`bodies_in_tray` 等
- `summary.json`：两个成功率、平均步数、seed 范围、是否 `--permute-slots`
- `trace.jsonl`：加 `--trace` 时写逐步的指尖温度 / 夹爪 / TCP

两个成功率的定义：

- **task success**：目标物体（冷铝罐）在托盘里、贴底、夹爪张开，连续
  `stable_steps` 步。与生成器的判定一致，只是去掉了"专家 phase == DONE"。
- **manipulation success**：**任意**物体满足同样的静置条件。

纯视觉 baseline 的典型形态是 **manipulation 高、task 低**：它会抓一个罐子
放进托盘，但三种罐子外观完全相同，它无法知道哪个是冷的。两者的差距就是
这个 baseline 要报告的东西。

### 5.3 位置捷径与 `--permute-slots`

当前数据里每个 episode 的物体**槽位固定**（只有 ±1 cm、±0.15 rad 的抖动）。
视觉策略完全可能背下"目标在 (x, y)"，从而在训练分布上刷出很高的任务成功率。
正式报告前必须能量化这一点：

```bash
python experiment/eval_closed_loop.py --policy remote --episodes 20 --permute-slots --remote-host 192.168.1.20 --remote-port 8000
```

`--permute-slots` 每集把 7 个物体的槽位随机重排（槽位集合不变，最小间距
约束仍成立）。它**只是诊断**，不是训练分布：物体温度网格是在建场景时按
**世界系 AABB** 切出来的（见 `docs/architecture.md` §9-R21），大范围移动物体
会让物体系温度网格失效，所以这个模式只用于纯视觉诊断，不要用它解读温度通道。

要真正消除捷径，正确做法是**重新生成一份每集槽位随机排列的数据集**
（`sample_episode` 里加一次 permutation），那会改变训练分布，且温度档必须
一起重算。这一步需要单独确认后再做（M9 全量生成需要明确同意）。

在重排分布下，三种外观相同的罐子意味着**纯视觉任务成功率的上限约为 1/3**；
操作成功率仍应很高。这就是"视觉不够、需要触觉/温度"的核心证据。

### 5.4 checkpoint 选择与最终报告

没有 held-out 数据 split：训练用全部 50 集（episode seed 0..49），测试靠闭环新种子
（`--seed0 1000` 起），物体位姿与温度都是训练时没见过的配置。

中间 checkpoint 用一组固定种子评（如 `--seed0 1000 --episodes 20`），按 task
success 选；最终报告换一组**没参与选择**的种子（如 `--seed0 2000 --episodes 20`），
避免选择偏差。报告写清 seed 段、episode 数、是否 `--permute-slots`，以及训练用了
全量 50 集。

---

## 6. 预期结果与解读

| policy | task success | manipulation | 说明 |
|---|---|---|---|
| `zero` | ~0 | ~0 | 循环健康检查 |
| `expert` | 高（以实际运行为准） | 高 | 评测循环与生成器一致性 |
| π0.5 V+P（固定槽位） | 可能被位置捷径抬高 | 高 | 不能直接当 baseline 数字 |
| π0.5 V+P（重排槽位） | ≈1/3 上限 | 高 | 纯视觉的真实水平 |
| +Touch / +Touch+T | 应显著高于 1/3 | 高 | 消融的增益 |

报告时至少给出：task success、manipulation success、平均步数、episode 数、
seed 范围，以及是否重排槽位。

---

## 7. 故障排查

- **`ERROR: this eval feeds RGB ... cameras`**：评测配置必须开相机；
  只有 `--policy expert` 允许无相机配置。
- **`state mode 'full30' needs ...`**：确认 YAML 里温度传感器开启；
  `touch28` 需要 `tactile.modalities` 含 `force`。
- **`--image-size must be even`**：yuv420p 要求偶数边长，换 256/512。
- **`openpi-client` 找不到**：`pip install -e ~/openpi/packages/openpi-client`，
  或在装了 openpi 的环境里跑评测。
- **服务器返回维度不足 7**：检查 `policy.config` 是否用了 `action_dim=32`
  且输出按 7 维动作训练；客户端只取前 7 维。
- **`check_lerobot.py` 报 `task_index` 不一致**：说明 `tasks.jsonl` 被改过；
  重新导出，不要手改 meta。
- **`Number of dimensions must be at least 1, and is 0 instead.`** /
  **`Shape of 'mean' must be (3,1,1), but is (3,) instead.`**：旧版
  `export_lerobot.py` 的 stats 格式与 LeRobot 不符——`count` 写成了裸数字，
  图像 stats 写成了扁平 `(3,)`。LeRobot 要求每个 stat 都是数组（`ndim >= 1`），
  图像特征为 `(3,1,1)`（`[[[v]], [[v]], [[v]]]`），`count` 为 `[n]`。现已修正；
  已导出的数据集不必重编码视频，按这两条修 `meta/stats.json` 与
  `meta/episodes_stats.jsonl` 即可。`check_lerobot.py` 现在会先报出这两项。
- **`--overwrite` 的安全性**：只删除目标目录下的 `data/ videos/ meta/`，
  并且拒绝把仓库根目录当 `--out`。

---

## 8. 之后

- `--state-mode touch28` / `full30` 导出同一份数据，做三档消融；
- OpenVLA-OFT 对照需要另一条 RLDS 导出路径（本目录未包含）；
- 若决定消除位置捷径：改 `sample_episode` 的槽位分配 → 重新生成 →
  重新导出 → 重跑本 README 的全部检查。
