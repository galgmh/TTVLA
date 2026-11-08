# ThermTacVLA — ThermalContact-v1

在 Genesis 刚性仿真里生成**单区域受控热接触**数据：Franka Panda 用简单的
触碰策略（不判断温度）依次接触四段式杯子的一个热区，逐帧记录指尖温度、
接触力、夹爪开口与指令，用来训练一个只吃传感器历史的热编码器。

- **数据**：每集一次单区域触碰，100 Hz 原始记录；四段杯子（lower / middle /
  upper / handle）各是一个热节点；派生量（ΔT、Ṫ、前缀窗口、归一化）一律
  在加载器里现算，不落盘。
- **编码器**：逐帧 14 维输入 → 因果膨胀卷积 → z ∈ R¹²⁸，两个头：温度回归、
  接触有效性。区域 ID、link 名、真实 link 温度、材料参数、目标力都不进模型输入。

## 目录

```
assets/cup_4link/      四段杯子的 URDF、网格与清单（位置/区域边界的唯一真源）
configs/               物体、传感器、任务、数据集、模型五类 YAML
sim/                   仿真层：场景、区域映射、控制器、采样、观测、温度播种
data/                  数据层：契约、写出、加载窗口、校验
models/                热编码器与损失
training/              训练 / 评估脚本
scripts/               资产生成与校验、清单生成、数据生成、检查与画图
tests/                 单元测试（stdlib unittest）
datasets/              数据输出（不入库）
```

## 环境

```bash
conda activate tempvla                      # genesis 1.4.1 + torch cu128
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH   # WSL2 上必须
```

## 常用命令

```bash
# 生成划分清单（冻结每集的完整参数组合）
python scripts/generate_contact_manifests.py --splits pilot

# 生成一集试生产数据
python scripts/generate_thermal_contact.py --split pilot --count 10

# 并行全量生成（分块 + 断点续跑；中断后重跑同一条命令）
python scripts/generate_all.py --workers 4

# 校验
python - <<'PY'
from data.validation.validate_thermal_contact import validate_directory
print(validate_directory("datasets/thermal_contact_v1/pilot") or "0 issues")
PY

# 看一集、画响应曲线
python scripts/inspect_episode.py datasets/thermal_contact_v1/pilot/<id>.hdf5
python scripts/plot_contact_response.py datasets/thermal_contact_v1/pilot/<id>.hdf5

# 训练 / 评估
python training/train_thermal_encoder.py --epochs 5
python training/evaluate_thermal_encoder.py datasets/thermal_contact_v1/runs/thermal_encoder.pt

# 测试
python -m unittest discover -s tests -t .
```

## 数据流

```
资产清单 -> 场景构建 / 区域解析 -> EpisodeSpec 采样 -> 初始化四段 link 温度与指尖温度
   -> 单区域受控触碰（RESET→APPROACH→SEEK_CONTACT→HOLD→RETRACT）
   -> 逐帧记录 -> HDF5 thermalcontact-v1.8 -> 校验 -> 窗口加载 -> 训练
```

场景与采样都不写随机数：同一 `(seed, split, index)` 必然复现同一集。
