# ThermTacVLA

面向 VLA 的**温度-触觉**机器人操作数据集生成项目。基于 [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world)
物理引擎（唯一引擎依赖），生成与 **LIBERO 同构**（robomimic HDF5 + 7 维动作 + 语言标注）的任务数据集，
并额外提供**触觉**与**温度**感知通道。

**本体**：Franka Panda + 平行夹爪。触觉只保留两个与 AT-VLA 对齐的信号（format 1.6）：
**6D 合力**（每指一个 `[Fn(3), Ft(3)]` = 3D 法向 + 3D 切向，12 探针求和，link 系）与
**2D marker**（16×16 = 256 点/指在指垫切平面内的位移 `(u,v)`）。其余触觉参数全部在
`sim/sensors/tactile.py`，YAML 只写 `modalities: [force, elastomer]`。
指尖温度自 format 1.5 起是**官方最粗栅格的单格实测通道**（`grid_size=(1,1,1)`，每指一个集中
温度 + 官方传感器非理想性：一阶 RC / 白噪 / 量化，只录 measured 通道（`record_ground_truth =
false`，与其余温度参数一起定义在 `sim/sensors/temperature.py`），§9-R36）。
**任务**：`pick_cold_can`（当前唯一任务配置）——桌上有两个外观相同的罐子（一冷一温）、
两盒牛奶、一个柠檬和木/金属方块；专家先触碰候选罐、读实测指尖温度判冷，判不冷就原地
释放换下一个，判冷后搬运到托盘（§9-R42/R43）。历史任务配置（`pick_hot_cylinder` /
`pick_by_temperature` / `put_in_tray` 等）已在 2026-10-19 的单配置清理中删除，
设计记录保留在 §9-R38/R39/R42。

## 状态（2026-10-20；2026-10-11/12 重组与旧语料删除、2026-10-13 指尖通道定案、2026-10-20 触觉格式 1.6 后同步）

- 代码已到 format `tempvla-1.6`（触觉 = 6D 合力 + 2D marker，§9-R44；合成热像
  1.7 已按要求全部回退，§9-R45）；Gate 0–4 的验收已于 2026-10-02/03 全部通过
  （实测见架构文档 §8；Gate 脚本在 2026-10-20 的 scripts 清理中删除，§9-R46）；
  旧栈（format 1.0）全量数据已于 2026-10-12 删除（§9-R31）——
  **M9 重生成前仓库内没有当前格式的完整数据集**；布局、侧车约定与数据卡见
  [datasets/README.md](datasets/README.md)。
- **现状 / 已知缺陷 / 目标与差距**：[docs/status.md](docs/status.md)；
  **设计依据与实测记录（单一事实来源）**：[docs/architecture.md](docs/architecture.md)。
- 尚未完成：全量数据用 1.6 栈重生成（M9，落点仍为 `datasets/v1.5/hdf5/`——目录名沿用，
  未随格式号重命名）、M9 后的正式数据卡、replay/analyze、融合基线、真机迁移（M10–M13）。

## 与上游仓库的关系

本项目**只在引擎层依赖 Genesis**（pip `genesis-world==1.4.1`，wheel 自带 `urdf/panda_bullet/` 资产）。
`~/tactile-genesis/Genesis` 仅作**只读参考**，唯一的静态拷贝是：

| 拷贝内容 | 来源 | 去向 |
|---|---|---|
| Genesis 传感器补丁（修 indexed random-walk reset bug） | `dexterous-hands/src/sensor_compat.py` | `sim/compat/sensor_compat.py`（逐字保留） |

触觉探针布局**不再拷贝布局文件**：`sim/sensors/finger_layout.py` 直接解析夹爪 URDF
的碰撞盒，程序化生成两指内侧探针网格；另有同一几何的**独立简化版** `sim/gripper.py`
（只几何、写死 Panda、纯 numpy，管线不 import 它；等价性曾由已删除的
`scripts/probe_gripper_model.py` 断言，§9-R28）。**不使用**：Eden、rsl_rl、
dexterous-hands 的任何 Python 模块。

## 目录

```
ThermTacVLA/
├── docs/                     # architecture.md（设计单一事实来源）· status.md（现状快照）· figures/（报告图 + 演示视频）
├── sim/                      # Python 包（config/scene/expert/obs/rotations/gripper + sensors/recording/compat/tools）
│   └── configs/              # 当前唯一任务配置：thermal_recognition/pick_cold_can.yaml
├── scripts/                  # generate · gate_common · validate_dataset · build_manifest · render_video · scene_check · render_scene · autotune · probe_render · probe_camera_pose
└── datasets/                 # 数据集根（§9-R30）：v1.5/hdf5/ 为 M9 落点；只有 README.md（数据卡）入库
```

**`tempvla/` 包名预留给未来的 VLA 模型代码**（2026-10-11 重组：仿真/数据层全部移入 `sim/`，见 [docs/architecture.md](docs/architecture.md) §9-R29）。

## 环境（WSL2 / RTX 5060 Laptop）

```bash
conda create -n tempvla python=3.11 -y && conda activate tempvla
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install genesis-world==1.4.1 h5py pyyaml scipy pillow

# WSL2 必需：CUDA 运行库路径（不设会让 Quadrants 静默回退 CPU，架构文档 §2.1）
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
# 相机/渲染路径必须软件 GL：WSLg 的 D3D12 路径会静默产出退化图像（内容塌缩到最下 2 行像素）
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
```

## 运行顺序

```bash
cd ~/ThermTacVLA
conda activate tempvla
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe   # 相机路径（WSL2）

# 场景自检：不接专家，查静置/接触/温度播种/随机化/相机 manifest
python scripts/scene_check.py --config sim/configs/thermal_recognition/pick_cold_can.yaml

# 场景初始位姿快照（双相机；不跑 episode、不写 HDF5）
python scripts/render_scene.py --config sim/configs/thermal_recognition/pick_cold_can.yaml --res 512

# 相机位姿候选接触表 / 渲染器最小探针
python scripts/probe_camera_pose.py --config sim/configs/thermal_recognition/pick_cold_can.yaml
python scripts/probe_render.py --renderer raster --res 128

# 生成：单集冒烟（无相机，不碰正式数据）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml \
    --episodes 1 --no-cameras --out /tmp/pick_cold_can_smoke

# 生成：正式（落点 = 配置的 recording.output_dir，现为 datasets/v1.5/hdf5；--out 可覆盖。
# 同名文件截断重写；cameras.enabled=true 时脚本会检查软件 GL，云端可加 --unsafe-gl）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml --episodes 50

# 数据侧：校验 / 侧车 / 回放视频
python scripts/validate_dataset.py datasets/v1.5/hdf5/pick_cold_can_*.hdf5
python scripts/build_manifest.py datasets/v1.5/hdf5
python scripts/render_video.py datasets/v1.5/hdf5/pick_cold_can_aluminum_aluminum_lemon_paperboard_paperboard_balsa_iron.hdf5 --demo 0

# 可选：专家参数搜索（输出 YAML 的 tuning:；常量真源在 sim/expert/pipeline.py）
python scripts/autotune.py --config sim/configs/thermal_recognition/pick_cold_can.yaml --trials 24 --episodes 5
```

> 2026-10-20 scripts 清理（§9-R46）：Gate 0–4、历史取证探针、热像/接触演示、
> `report_figures.py` 与离线盲判工具都已删除；架构文档 §8 里的对应命令保留为历史记录，
> 不再可执行。`docs/figures/` 中的图与视频是归档副本。

## 数据格式速览

与 LIBERO 相同的 robomimic HDF5 结构（`data/demo_i/obs/...` + 文件级 `problem_info` 语言标注），
文件级属性记录格式版本（`format_version`，当前 `tempvla-1.6`）；**每个 job 一个文件**
（`<task>_<tag>.hdf5`，tag 由 `materials` / `material_list` / `group_files` 决定；当前
`pick_cold_can` 是单 job、七材质，文件名列出全部材质），另有 `*_failed.hdf5`。新增子树：

- `obs/tactile/force` `(T,2,6)` —— 每指一个 **6D 合力** `[Fn(3), Ft(3)]`（AT-VLA 口径：
  3D 法向 + 3D 切向合力，不是力+力矩），12 探针求和、link 系、压缩为正
- `obs/tactile/elastomer` `(T,512,2)` —— 16×16 marker/指在**指垫切平面**内的位移 `(u,v)`，
  finger-major（AT-VLA 的 N×2 格式）；1.6 不存法向分量，因此没有 `height_map`
  （format ≤1.5 的 3D 文件仍可用旧工具合成）
- `obs/link_pose` / `obs/object_pose` —— 指尖连杆 / 物体位姿（1.1/1.2；`link_pose` 是
  link 系触觉信号的世界参考：`p_world = R(q)·p_link + pos`）
- `obs/temperature/{fingertip_grid, object_grid, objects_grid, link_temps, object}` —— 指尖与物体**双侧**温度场
  （1.5 起 `fingertip_grid` 是官方最粗栅格的**单格实测**通道：每指一个集中温度，附官方 RC/噪声/量化
  非理想性，§9-R36；1.0–1.4 的历史记录见 R26/R36）
- `obs/contact/{link_contact, link_force}` + `events/` —— 接触状态/力、接触事件表（L3）、
  接触瞬态的高频（200 Hz）爆发窗口（L4）

完整数据契约见 [docs/architecture.md](docs/architecture.md) §4（接触与热传导细节见 §4.7）。
