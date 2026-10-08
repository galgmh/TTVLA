# TemperatureVLA

面向 VLA 的**温度-触觉**机器人操作数据集生成项目。基于 [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world)
物理引擎（唯一引擎依赖），生成与 **LIBERO 同构**（robomimic HDF5 + 7 维动作 + 语言标注）的任务数据集，
并额外提供**触觉**与**温度**感知通道。

**本体**：Franka Panda + 平行夹爪，两指指尖各 12 个触觉探针 + 温度网格。
**任务**：`pick_hot_cylinder` —— 抓起桌上的热圆柱（材质/初始温度随机化）。

## 状态

v1 代码已实现（2026-10-02），等待引擎环境验证（Gate 0/1/2 + 生成冒烟）。
完整设计见 [docs/architecture.md](docs/architecture.md)；未实现项（相机、replay、
固定初始状态池）在各节标注为 v2。

## 与上游仓库的关系

本项目**只在引擎层依赖 Genesis**（pip `genesis-world==1.4.1`，wheel 自带 `urdf/panda_bullet/` 资产）。
`~/tactile-genesis/Genesis` 仅作**只读参考**，唯一的静态拷贝是：

| 拷贝内容 | 来源 | 去向 |
|---|---|---|
| Genesis 传感器补丁（修 indexed random-walk reset bug） | `dexterous-hands/src/sensor_compat.py` | `tempvla/compat/sensor_compat.py`（逐字保留） |

触觉探针布局**不再拷贝布局文件**：`tempvla/sensors/finger_layout.py` 直接解析夹爪 URDF
的碰撞盒，程序化生成两指内侧探针网格。**不使用**：Eden、rsl_rl、dexterous-hands 的任何 Python 模块。

## 目录

```
TemperatureVLA/
├── docs/architecture.md      # 架构设计（数据契约 §4、模块 §5、Gate §8）
├── configs/                  # 任务规格（每任务一个 YAML，与 config.py 逐字段对应）
├── tempvla/                  # Python 包（config/scene/actions/expert/obs/rotations + sensors/recording/compat/tools）
├── scripts/                  # gate0_engine · gate1_sensors · gate2_scene · autotune · generate · validate_dataset
└── data/                     # 生成输出（不入版本库）
```

## 环境（WSL2 / RTX 5060 Laptop）

```bash
conda create -n tempvla python=3.11 -y && conda activate tempvla
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install genesis-world==1.4.1 h5py pyyaml

# WSL2 必需（官方文档处方，详见架构文档 §2.1）
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_INDIRECT=0
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

## 运行顺序

```bash
cd ~/TemperatureVLA
python scripts/gate0_engine.py          # 引擎 + 资产 + 温度传感器契约
python scripts/gate1_sensors.py         # 机器人 + 探针布局 + 传感器读数 + 压入测试
python scripts/gate2_scene.py --episodes 2   # 完整场景 + 专家 + 落盘 + 校验
python scripts/autotune.py --trials 24 --episodes 5   # 专家参数自动调优 -> *.tuned.yaml
python scripts/generate.py --episodes 2 --out data_smoke   # 冒烟
python scripts/generate.py --config configs/pick_hot_cylinder.tuned.yaml --episodes 50 --out data  # 全量
python scripts/validate_dataset.py data/pick_hot_cylinder_aluminum.hdf5
```

## 数据格式速览

与 LIBERO 相同的 robomimic HDF5 结构（`data/demo_i/obs/...` + 文件级 `problem_info` 语言标注），
**每个材质一个文件**（`pick_hot_cylinder_<material>.hdf5`，另有 `*_failed.hdf5`），新增子树：

- `obs/tactile/force|depth` —— 逐探针触觉读数（两指各 12 探针，前 12 个为左指）
- `obs/temperature/{fingertip_grid, object_grid, link_temps, object}` —— 指尖与物体**双侧**温度场
- `obs/contact/{link_contact, link_force}` + `events/` —— 接触状态/力、接触事件表（L3）、
  接触瞬态的高频（200 Hz）爆发窗口（L4）

完整数据契约见 [docs/architecture.md](docs/architecture.md) §4（接触与热传导细节见 §4.7）。
