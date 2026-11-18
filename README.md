# ThermTacVLA

在 Genesis 中构建热触觉抓取场景，生成 task episode，并把数据导出为 LeRobot
格式供 models 微调。代码边界为：

- `sim/`：Genesis 场景、触觉/热觉传感器、接触控制和 task 数据生成。
- `models/`：纯 PyTorch π0.5、LeRobot 微调和推理服务。
- `models/thermal_model/`：已归档的热觉模块，暂不接入 π0.5。

外部 `models/` 已删除。

## sim 结构

```text
sim/
  config.py                 简单 YAML 加载与路径解析
  scene.py                  Genesis 场景构建和 episode reset
  obs.py                    策略观测和特权观测
  episode.py                EpisodeSpec
  sampler.py                确定性 episode 采样
  dataset.py                HDF5 字段契约和 metadata
  episode_recorder.py       逐帧 episode 记录
  recording.py              HDF5 读写
  validate.py               数据和物理一致性校验
  model_env.py              π0.5 7 维动作接入 Genesis 的环境包装
  run_pi_simpler.py         models WebSocket 闭环评测

  generate.py               生成原始 task episode
  generate_grasp.py         生成抓取基线
  generate_pi05_v1.py       生成 thermal-pi05-v1 的 80 集原始数据
  export_pi05_lerobot.py    原始 HDF5 -> LeRobot 微调数据集
  select_episodes.py        筛选完整四接触 episode

  control/                  接近、接触、抓取和 IK 控制
  sensors/                  指尖触觉、温度和指垫布局
  thermal/                  link 温度和热响应
  compat/                   Genesis 兼容补丁
  assets/                   cup/robot 资产和构建校验工具
  configs/                  仿真任务配置
  tools/                    episode 检查、图表和视频工具
```

`sim/config.py` 只负责加载 YAML、解析相对路径和展开物体 manifest，不再维护
一套逐字段的配置校验框架。具体字段由使用它的场景或生成流程检查。

## models 结构

```text
models/
  pi05_model/                纯 PyTorch π0.5 本体
  data/                      LeRobot 数据加载和 transforms
  finetune_new.py            微调入口
  server_policy.py           WebSocket 推理服务
  configs/configs.py         数据集、模型和训练配置
  thermal_model/
    config.py                热觉模型配置
    observation.py           最终在线模型输出容器
    preprocessing.py         12 通道热觉窗口
    model.py                 最终接入 π0.5 的在线热觉模型
    contact_encoder.py       因果接触编码器
    memory_bank.py           接触事件长期记忆
    spatial_memory.py        热空间记忆
    prefix_adapter.py        热觉 prefix 适配
    data/                    预训练数据读取
    encoder_pretraining/     直接使用 ThermalContactEncoder 的预训练工具
    spatial_pretraining/     空间表示、热场 decoder 和辅助头训练
    pi05_training/           热觉训练适配（暂未接入）
    configs/                 热觉模型、预训练和 π0.5 配置
    ARCHITECTURE.md          热觉集成设计
```

`thermal_model` 目前只是独立归档，不影响 models 的普通 π0.5
训练、推理或闭环测试。

## 数据流

```text
sim/configs + sim/assets
  -> EpisodeSpec
  -> Genesis scene.reset
  -> 受控接近、接触、力闭环、退回
  -> 100 Hz 热触觉 + 20 Hz 控制/RGB
  -> sim.dataset / sim.recording
  -> sim.export_pi05_lerobot
  -> models data loader
  -> pi05_model
```

模型侧的单步动作约定为：:

```text
[dx, dy, dz, drot_x, drot_y, drot_z, gripper_width]
```

[model_env.py](/home/ca/TTVLA/sim/model_env.py:1) 把动作作用到当前 TCP 位姿，
通过 IK 控制机械臂，并把 7 个关节角和夹爪宽度组成 8 维 state。它同时维护
模型所需的 25 帧、12 通道热触觉窗口：

```python
env = GenesisModelEnv(
    "sim/configs/tasks/thermal_contact_probe.yaml",
    "sim/configs/datasets/thermal_contact_v1.yaml",
    split="train",
)

observations = env.reset(index=0)
next_observations, done, info = env.step(action_7d)
```

models 闭环评测：

```bash
# 终端 1：在 models 环境中启动模型服务
cd models
python server_policy.py \
  --config_name your_config \
  --checkpoint_dir /path/to/checkpoint \
  --port 8000

# 终端 2：在 Genesis 环境中执行闭环
cd ..
python -m sim.run_pi_simpler \
  --host 127.0.0.1 --port 8000 \
  --split train --start 0 --episodes 10 \
  --replan-steps 5 \
  --success-kind cup-lift \
  --success-threshold 0.06 \
  --output datasets/pi_simpler_eval.json
```

## 环境

```bash
conda activate tempvla
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

π0.5 训练使用：

```bash
conda activate tempvla_pi05
```

项目使用 Python 3.11，不使用 `.venv`。

## task 数据生成

生成单个或部分 episode：

```bash
python -m sim.generate \
  --split train \
  --start 0 \
  --count 10 \
  --out datasets/thermal_contact_multi
```

校验：

```bash
python - <<'PY'
from sim.validate import validate_directory

issues = validate_directory("datasets/thermal_contact_multi/train")
print(issues or "0 issues")
PY
```

检查单个 episode：

```bash
python -m sim.tools.inspect_episode \
  datasets/thermal_contact_multi/train/<episode>.hdf5
```

筛选完整四接触 episode：

```bash
python -m sim.select_episodes \
  --source datasets/thermal_contact_multi/train \
  --out datasets/thermal_contact_1/train \
  --expect 90
```

抓取基线：

```bash
python -m sim.generate_grasp \
  --policy sim/configs/policies/direct_lift.yaml \
  --split train \
  --count 100
```

第一版 Pi0.5 数据：

```bash
# 80 集原始 HDF5
python -m sim.generate_pi05_v1

# 只导出成功 episode 到 LeRobot
python -m sim.export_pi05_lerobot \
  --source datasets/thermal_pi05_v1/raw \
  --root datasets/lerobot/thermal_pi05_v1 \
  --repo-id thermtacvla/thermal_pi05_v1 \
  --fps 20
```

`generate_pi05_v1.py` 按每种专家 20 个成功 episode 计数。失败 episode 不写盘，
会用新的确定性 attempt 自动重试；运行前还会清理旧输出目录中未成功的 HDF5。
单个 episode 默认最多尝试 20 次，可通过 `--max-attempts-per-episode` 调整。

## 热编码器

生成 12 通道派生数据：

```bash
cd /home/ca/TTVLA

PYTHONPATH=models python -m thermal_model.encoder_pretraining.prepare_dataset \
  --source datasets/thermal_contact_1_copy/train \
  --out datasets/thermal_encoder \
  --overwrite
```

90 个 episode 会生成 360 个接触组和 360 个非接触组。每组 51 帧
（50 Hz 下 1 s），从组内切出 25 个长度为 26 的窗口，共 18000 个窗口。
所有窗口均为真实采样，`mask` 全为 True；非接触组只训练接触有效性。

训练：

```bash
PYTHONPATH=models python -m thermal_model.encoder_pretraining.train \
  --config thermal_model/configs/encoder_pretraining.yaml \
  --data-root datasets/thermal_encoder \
  --out datasets/thermal_encoder/thermal_encoder.pt
```

评测：

```bash
PYTHONPATH=models python -m thermal_model.encoder_pretraining.evaluate \
  datasets/thermal_encoder/thermal_encoder.pt \
  --data-root datasets/thermal_encoder \
  --splits validation test
```

空间热觉预训练：

```bash
PYTHONPATH=models python -m thermal_model.spatial_pretraining.prepare_dataset \
  --source datasets/thermal_encoder \
  --raw-source datasets/thermal_contact_1_copy/train \
  --out datasets/thermal_spatial \
  --overwrite

PYTHONPATH=models python -m thermal_model.spatial_pretraining.train \
  --config thermal_model/configs/spatial_pretraining.yaml \
  --data-root datasets/thermal_spatial \
  --out datasets/thermal_spatial/runs/thermal_spatial.pt \
  --memory-mode full

PYTHONPATH=models python -m thermal_model.spatial_pretraining.evaluate \
  datasets/thermal_spatial/runs/thermal_spatial.pt \
  --data-root datasets/thermal_spatial \
  --memory-mode full
```

空间预训练加载 `thermal_encoder.pt`，默认冻结接触编码器，只训练
`ContactTokenBuilder`、`ThermalSpatialMemory` 和辅助 `SpatialFieldDecoder`。
辅助头实现位于 `thermal_model/spatial_pretraining/`，最终在线模型不持有它们。
训练按 `episode_offsets` 逐条运行完整 episode，按时间顺序更新 MemoryBank；
不同长度 episode 在同一 batch 内 padding，并用 mask 排除。
训练或评测传 `--memory-mode current` 可关闭历史 MemoryBank，只使用当前
contact token，用于 full/current 消融。
`thermal-spatial-v2` 额外保存每个窗口末端的 `contact_valid` 和
`contact_region`。默认只对当前接触区域的查询点计算 MSE，并把 current token
显式加入 SpatialMemory 输入。
评测会额外输出 `current`、`seen`、`unseen` 三类区域的样本数、MAE 和
RMSE，用于比较 full-memory 与 current-only。

先用 current-only 训练基础空间映射，再加载该 checkpoint 训练 full-memory：

```bash
PYTHONPATH=models python -m thermal_model.spatial_pretraining.train \
  --config thermal_model/configs/spatial_pretraining.yaml \
  --data-root datasets/thermal_spatial \
  --out datasets/thermal_spatial/runs/with_memory_v2.pt \
  --init-checkpoint datasets/thermal_spatial/runs/current_only_v2.pt \
  --memory-mode full \
  --field-loss mse \
  --epochs 20 \
  --batch-size 2 \
  --lr 0.0001
```

温度 `logvar` 使用有界 sigmoid 参数化：`[-4, 3]`。对应温度标准差约为
`0.135～4.48°C`，loss 中不再使用会把梯度截断的固定 `clamp`。

## models 微调

```bash
cd /home/ca/TTVLA/models
python finetune_new.py --config_name your_config
```

热觉模块暂不接入；当前训练使用普通 π0.5、LeRobot 数据和可选零右腕 mask。
