# ThermTacVLA

面向 VLA 的**温度-触觉**机器人操作数据集生成项目。基于
[Genesis](https://github.com/Genesis-Embodied-AI/genesis-world)（唯一引擎依赖）生成与 LIBERO 同构的
robomimic HDF5（7 维动作 + 语言标注），并额外提供触觉与温度感知通道。
`sim/` 只产出 HDF5；`experiment/` 负责把它转成 LeRobot、定义策略 I/O 契约并跑闭环评测。

**本体**：Franka Panda + 平行夹爪。

**触觉**（format 1.6）：每指一个 **6D 合力** `[Fn(3), Ft(3)]`（12 探针求和，link 系）+
**2D marker**（16×16 = 256 点/指在指垫切平面内的位移 `(u,v)`）。其余触觉参数在
`sim/sensors/tactile.py`，YAML 只写 `modalities: [force, elastomer]`。

**指尖温度**：官方最粗栅格的单格实测通道（`grid_size=(1,1,1)`，每指一个集中温度 +
官方传感器非理想性：一阶 RC / 白噪 / 量化，只录 measured 通道）。

**格式**：`tempvla-1.6`。**任务**：`pick_cold_can` 与 `pick_lemon` 两个配置。
`pick_cold_can` 的桌上有三个外观相同的罐子（冷/温/室温三个温度档）、牛奶、柠檬和木/金属
方块；三个温度档每集随机分配给这三个罐子，`target_index` 跟随冷档，所以冷罐的位置不是
固定槽位。专家先触碰候选罐、读实测指尖温度判冷，判不冷就原地释放换下一个，判冷后搬运
到托盘。`pick_lemon` 是**纯抓取测试**：一个柠檬（固定 target）、两个罐子、两盒牛奶；
不启用温度判定，专家直接抓取柠檬并放入托盘，用来验证抓取/搬运链路。

## 目录

```
ThermTacVLA/
├── sim/                  # 仿真包（config / scene / expert / sensors / recording / tools / compat）
│   ├── configs/
│   │   └── thermal_recognition/
│   │       ├── pick_cold_can.yaml
│   │       └── pick_lemon.yaml
│   └── assets/           # Panda URDF + robosuite 物体网格/贴图（随仓库分发）
├── scripts/              # 数据生成侧入口
│   ├── generate.py       # 生成 HDF5
│   ├── validate_dataset.py
│   ├── build_manifest.py # index / splits / stats 侧车
│   ├── render_video.py   # 回放 MP4
│   └── gate_common.py    # generate.py 的共享辅助
├── experiment/           # HDF5 → LeRobot、策略 I/O、闭环评测；细节见 experiment/README.md
└── datasets/             # 数据根（除 README.md 外均被 .gitignore 排除）
    ├── hdf5/pick_cold_can/               # 旧 50 集（固定温度角色）+ 侧车三件套
    └── lerobot/local/pick_cold_can_p16/  # LeRobot v2.1 导出
```

`docs/` 当前不在工作树中；历史设计记录（数据契约、R# 风险表、Gate 记录）仍可从 git HEAD 取回：
`git show HEAD:docs/architecture.md`。

## 环境（WSL2 / RTX 5060 Laptop）

```bash
conda create -n tempvla python=3.11 -y && conda activate tempvla
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install genesis-world==1.4.1 h5py pyyaml scipy pillow
pip install pyarrow imageio imageio-ffmpeg   # experiment/ 的 LeRobot 导出与校验
```

WSL2 上运行生成/评测前：

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe   # 相机路径必须软件 GL
```

## 现有数据（2026-10-28）

- `datasets/hdf5/pick_cold_can/pick_cold_can_aluminum_aluminum_lemon_paperboard_aluminum_balsa_iron.hdf5`
  —— 1.8 GB，**50 集 / 50 成功**，274–334 步/集，format `tempvla-1.6`，validator 0 issues。
  这份语料早于温度档置换：冷罐固定在物体 0；新分布需要重新生成。
- 侧车：`index.jsonl`（50 集）、`splits.json`（44 train / 4 val / 2 test，按 seed hash）、
  `stats.json`（只在 train split 的成功集上统计）。
- `datasets/lerobot/local/pick_cold_can_p16/` 是 2026-10-22 的 LeRobot v2.1 导出
  （50 集 / 14710 帧 / 100 个 256² 视频）。它**早于 2026-10-27 的 stats 修复**，
  当前 `check_lerobot.py` 会报 `count` 为标量、图像 stats 形状等 59 个 issues；
  使用前需按下面的命令重新导出（或按 `experiment/README.md` §7 修 meta）。
- 注意：YAML 的 `recording.output_dir` 目前仍指向
  `datasets/thermal_recognition/pick_cold_can/hdf5`，与上面这份语料的位置不同。
  在配置对齐前，重生成请显式给 `--out datasets/hdf5/pick_cold_can`。

## 运行顺序

```bash
cd ~/ThermTacVLA
conda activate tempvla
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe

# 生成：单集冒烟（无相机，不碰正式数据）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml \
    --episodes 1 --no-cameras --out /tmp/pick_cold_can_smoke

# 生成：pick_lemon 冒烟（同一条管线，只换任务 YAML）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_lemon.yaml \
    --episodes 4 --no-cameras --out /tmp/pick_lemon_smoke

# 生成：正式 50 集（新温度档置换分布；写新目录，避免覆盖旧语料）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml \
    --episodes 50 --out datasets/hdf5/pick_cold_can_temp_perm

# 数据侧：校验 / 侧车 / 回放视频
python scripts/validate_dataset.py datasets/hdf5/pick_cold_can/pick_cold_can_*.hdf5
python scripts/build_manifest.py datasets/hdf5/pick_cold_can
python scripts/render_video.py \
    datasets/hdf5/pick_cold_can/pick_cold_can_aluminum_aluminum_lemon_paperboard_aluminum_balsa_iron.hdf5 \
    --demo 0

# LeRobot 导出 + 校验（导出侧不需要 Genesis）
python experiment/export_lerobot.py \
    datasets/hdf5/pick_cold_can/pick_cold_can_aluminum_aluminum_lemon_paperboard_aluminum_balsa_iron.hdf5 \
    --out datasets/lerobot/local/pick_cold_can_p16 --overwrite
python experiment/check_lerobot.py datasets/lerobot/local/pick_cold_can_p16 --decode

# 闭环评测（需要 Genesis；policy 可选 zero / expert / remote）
python experiment/eval_closed_loop.py --policy zero --episodes 1
python experiment/eval_closed_loop.py --policy expert --episodes 4
```

π0.5 微调、`--state-mode touch28|full30` 消融、openpi 服务器与 `--permute-slots`
诊断见 [experiment/README.md](experiment/README.md)。

## 数据格式速览

与 LIBERO 相同的 robomimic HDF5 结构（`data/demo_i/obs/...` + 文件级 `problem_info` 语言标注），
文件级属性记录格式版本（`format_version`，当前 `tempvla-1.6`）；**每个 job 一个文件**
（`<task>_<tag>.hdf5`，tag 由 `materials` / `material_list` / `group_files` 决定）。新增子树：

- `obs/tactile/force` `(T,2,6)` —— 每指一个 **6D 合力** `[Fn(3), Ft(3)]`（AT-VLA 口径：
  3D 法向 + 3D 切向合力，不是力+力矩），12 探针求和、link 系、压缩为正
- `obs/tactile/elastomer` `(T,512,2)` —— 16×16 marker/指在**指垫切平面**内的位移 `(u,v)`，
  finger-major；1.6 不存法向分量，因此没有 `height_map`
- `obs/link_pose` / `obs/object_pose` —— 指尖连杆 / 物体位姿
  （`link_pose` 是 link 系触觉信号的世界参考：`p_world = R(q)·p_link + pos`）
- `obs/temperature/{fingertip_grid, object_grid, objects_grid, link_temps, object}` —— 指尖与物体双侧温度场
  （`fingertip_grid` 是官方最粗栅格的单格实测通道，每指一个集中温度）
- `obs/contact/{link_contact, link_force}` + `events/` —— 接触状态/力、接触事件表（L3）、
  接触瞬态的高频（200 Hz）爆发窗口（L4）

完整契约的实现见 `sim/recording/hdf5_writer.py` 的模块 docstring 与 `sim/tools/validate.py`；
LeRobot 侧的 state/action 契约见 `experiment/common.py`。

## 与上游仓库的关系

本项目**只在引擎层依赖 Genesis**（pip `genesis-world==1.4.1`，wheel 自带 `urdf/panda_bullet/` 资产）。
`sim/assets/` 是运行期自包含的资产副本；`sim/compat/` 是钉版引擎的幂等补丁。
