# ThermTacVLA — 主动空间热触觉探索 VLA

研究目标：让 VLA 策略在抓取前**主动**进行空间热触觉探索——用指尖接触感知物体
温度场，判断冷/热（以及材料类别），再决定继续探测、抓取或放弃。项目分三层：

- **仿真/数据层 `sim/` + `data/`**：Genesis 刚性仿真生成受控热接触 episode
  （当前为 4-link 杯子的单区域触碰数据集 `thermalcontact-v1.8`），HDF5 落盘、
  清单冻结、契约校验。
- **模型层 `models/`**：
  - `models/pi05/` —— OpenPI PyTorch π0.5 本体（pinned commit `215abfb`）+
    Thermal-π0.5 热觉分支（`models/pi05/thermal/`）。这是项目中 π0.5 的唯一
    实现，不是 third_party 副本；官方 JAX/Flax 实现已删除，训练/微调/评测
    全部走 PyTorch。
  - `models/thermal_encoder.py` —— 独立热编码器（因果 TCN，基线）。
- **训练层 `training/`**：
  - `training/pi05/` —— 项目自有的 Thermal-π0.5 训练/微调/评测（参考官方
    `scripts/train_pytorch.py` 重写）：YAML 配置、`thermalvla-1.0` 数据集、
    分阶段冻结、三组学习率、检查点续训、离线损失评测。
  - `training/train_thermal_encoder.py` —— 基线编码器训练。

## 目录

```
assets/cup_4link/      四段杯子的 URDF、网格与清单（位置/区域边界的唯一真源）
configs/               物体、传感器、任务、数据集、模型、训练六类 YAML
sim/                   仿真层：场景、区域映射、控制器、采样、观测、温度播种
data/                  数据层：契约、写出、加载窗口、校验
models/                热编码器（thermal_encoder.py）与 π0.5 模型层（pi05/）
training/              训练/评估：基线编码器（顶层脚本）与 π0.5（pi05/ 包）
scripts/               资产生成与校验、清单生成、数据生成、检查与画图
tests/                 单元测试（stdlib unittest）
datasets/              数据输出（不入库）
checkpoints/           训练检查点（不入库）
.venv                  π0.5 运行环境（torch 2.7.1 / transformers 4.53.2，无需 JAX）
```

## 环境

```bash
# 仿真/数据（Genesis 1.4.1 + torch cu128）
conda activate tempvla
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH   # WSL2 上必须

# π0.5 模型与训练（项目根 .venv；已装 transformers_replace）
PYTHONPATH=. .venv/bin/python -m pytest models/pi05/thermal -q
```

注意：`.venv` 的 torch cu126 轮子不支持 RTX 5060（sm_120），本机 GPU 训练需
换 cu128 构建；正式训练与闭环评估在云端 GPU 机器上执行。

## 常用命令

```bash
# ---- 仿真数据 ----
python scripts/generate_contact_manifests.py --splits pilot   # 冻结划分
python scripts/generate_thermal_contact.py --split pilot --count 10
python scripts/generate_all.py --workers 4                    # 并行全量

python - <<'PY'
from data.validation.validate_thermal_contact import validate_directory
print(validate_directory("datasets/thermal_contact_v1/pilot") or "0 issues")
PY

# ---- 基线热编码器 ----
python training/train_thermal_encoder.py --epochs 5
python training/evaluate_thermal_encoder.py datasets/thermal_contact_v1/runs/thermal_encoder.pt

# ---- Thermal-π0.5 ----
# 冒烟：dummy 主干 + 随机数据，CPU 数秒跑完
PYTHONPATH=. .venv/bin/python -m training.pi05.train \
    --config configs/training/pi05_thermal_smoke.yaml --overwrite

# 正式训练（先由仿真导出器产出 data.root 指向的 thermalvla-1.0 数据）
PYTHONPATH=. .venv/bin/python -m training.pi05.train \
    --config configs/training/pi05_thermal_stage1.yaml

# 断点续训
PYTHONPATH=. .venv/bin/python -m training.pi05.train \
    --config configs/training/pi05_thermal_stage1.yaml --resume

# 离线损失评测（加载最新检查点；可用 --data-root 指定评估集）
PYTHONPATH=. .venv/bin/python -m training.pi05.eval \
    --config configs/training/pi05_thermal_stage1.yaml --num-batches 20

# ---- 测试 ----
python -m unittest discover -s tests -t .                     # 仿真/数据/编码器
PYTHONPATH=. .venv/bin/python -m pytest models/pi05/thermal -q      # 39
PYTHONPATH=. .venv/bin/python -m pytest tests/test_pi05_training.py -q  # 8
```

## 数据流

```
资产清单 -> 场景构建/区域解析 -> EpisodeSpec 采样 -> 初始化 link/指尖温度
  -> 受控触碰 -> 逐帧记录 -> HDF5 thermalcontact-v1.8 -> 校验 -> 窗口加载
  -> [导出器，待实现] thermalvla-1.0（13 热觉字段 + 7 监督字段）
  -> training/pi05 -> Thermal-π0.5（Stage 1–4）
```

场景与采样都不写随机数：同一 `(seed, split, index)` 必然复现同一集。

## 文档

- `models/pi05/INTEGRATED_PI05_ARCHITECTURE.md`：Thermal-π0.5 完整 13 节规格。
- `models/pi05/THERMAL_PI05.md`：热觉模块、训练阶段、`training/pi05/` 用法、
  数据格式与验证记录。
- `models/pi05/README.md`：模型层文件对应关系与运行环境。
