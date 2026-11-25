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
  lift_test.py              lift 任务 WebSocket 闭环评测

  generate.py               所有仿真专家数据的唯一生成入口
  generation/               各 task 的专家管线和随机化预设

  control/                  接近、接触、抓取和 IK 控制
  sensors/                  指尖触觉、温度和指垫布局
  thermal/                  link 温度和热响应
  compat/                   Genesis 兼容补丁
  assets/                   cup/robot 资产和构建校验工具
  configs/                  仿真任务配置
  tools/                    episode 检查、图表和视频工具
```

配置按 scene / object / sensor / task 分层。task YAML 只允许包含任务名、
instructions 列表和成功标准；专家接触位姿与轨迹位于 `sim/control/`，不属于任务定义。

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
  -> 受控接近、零宽度夹爪闭合、接触保持、退回/抬升
  -> 100 Hz 热触觉 + 20 Hz 控制/RGB
  -> sim.dataset / sim.recording
  -> datasets/convert_hdf5_to_lerobot.py
  -> models data loader
  -> pi05_model
```

模型侧的单步动作约定为：:

```text
[dx, dy, dz, drot_x, drot_y, drot_z, gripper_width]
```

[model_env.py](/home/ca/TTVLA/sim/model_env.py:1) 把动作作用到当前 TCP 位姿，
通过 IK 控制机械臂，并把 7 个关节角和夹爪宽度组成 8 维 state。它同时维护
模型所需的 26 帧、12 通道热触觉窗口：

```python
env = GenesisModelEnv(
    "sim/configs/tasks/lift_with_cup_wall.yaml",
    "sim/generation/presets/lift.yaml",
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
python -m sim.lift_test \
  --host 127.0.0.1 --port 8000 \
  --split train --start 0 --episodes 10 \
  --replan-steps 5 \
  --success-kind cup-lift \
  --success-threshold 0.06 \
  --output datasets/lift_test_eval.json
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

HDF5 转 LeRobot v3 的数据转换使用：

```bash
conda activate tempvla
python -m pip install "lerobot==0.4.3"
```

项目使用 Python 3.11，不使用 `.venv`。

## task 数据生成

所有 task 使用同一个生成入口：

```bash
python -m sim.generate \
  --task sim/configs/tasks/lift_with_cup_wall.yaml \
  --split train \
  --start 0 \
  --num_per_ins 10 \
  --out datasets/
```

校验：

```bash
python - <<'PY'
from sim.validate import validate_directory

issues = validate_directory("datasets/lift_with_cup_wall/train")
print(issues or "0 issues")
PY
```

检查单个 episode：

```bash
python -m sim.tools.inspect_episode \
  datasets/lift_with_cup_wall/train/lift_with_cup_wall_ins00_000000.hdf5
```

检查关节限位余量：

```bash
python -m sim.tools.check_joint_limits \
  datasets/lift_with_cup_wall/train \
  --arm-margin-rad 0.10 \
  --finger-margin-m 0.002 \
  --json datasets/lift_with_cup_wall/joint_limits.json
```

杯壁抓取基线：

```bash
python -m sim.generate \
  --task sim/configs/tasks/lift_with_cup_wall.yaml \
  --split train \
  --start 0 --num_per_ins 100 \
  --out datasets/
```

生成三种抓取任务数据：

```bash
python -m sim.generate \
  --task sim/configs/tasks/lift_with_cup_rim.yaml \
  --split train --start 0 --num_per_ins 40 \
  --out datasets/

# 将多个已生成任务的 HDF5 合并转换为一个 LeRobot 数据集
python datasets/convert_hdf5_to_lerobot.py \
  --source datasets/lift_with_cup_rim/train datasets/lift_with_cup_wall/train \
  --root datasets/lerobot/lift \
  --repo-id thermtacvla/lift \
  --fps 20 \
  --overwrite
```

生成热安全探索+抓取数据：

```bash
for task in \
  sim/configs/tasks/thermal_lift_cool_cup.yaml \
  sim/configs/tasks/thermal_lift_hot_cup.yaml \
  sim/configs/tasks/thermal_lift_hot_handle_cup.yaml
do
  python -m sim.generate \
    --task "$task" \
    --split train \
    --start 0 \
    --num_per_ins 20 \
    --out datasets/
done
```

每个 task 生成 80 集，三个场景共 240 集。每集随机选择首个探测
区域，随后按空间最近原则完成其余区域；首个区域不是 handle 时，
handle 固定最后探测。每次接触保持 0.5 秒并读取当前真实温度：
不安全则原地张开并前往下一区域，安全则直接在接触位置抓取抬升。
相邻 wall 区域之间通常直接从当前接触位移动到下一接触位。
`middle ↔ upper` 例外：先沿杯外退开 3cm，再抬到杯沿上方 2cm，
完成腕部转向后下降到 upper pre_pos。wall 与 handle 之间仍使用
杯口上方 6cm 的全局安全高度。upper 每集在 horizontal 和 vertical
两种接近方向中等概率选择；vertical 使用更高的杯沿握点。middle 与
upper 之间只有 vertical 使用 rim 过渡，horizontal 直接移动。

`--num_per_ins` 表示每条 instruction 的 episode 数，总数为
`instruction 数量 × num_per_ins`。

## 热编码器

生成 12 通道派生数据：

```bash
cd /home/ca/TTVLA

PYTHONPATH=models python -m thermal_model.encoder_pretraining.prepare_dataset \
  --source /path/to/source_hdf5 \
  --dataset /path/to/dataset_config.yaml \
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
  --raw-source /path/to/source_hdf5 \
  --task /path/to/task.yaml \
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
