# TemperatureVLA 架构设计

| | |
|---|---|
| 版本 | v1.0（v1 代码已实现，Gate 0/1/2 实测通过） |
| 日期 | 2026-10-02 |
| 状态 | Gate 0/1/2 全部通过（各 0 failures / 0 warnings，GPU 后端）；**全量数据集已生成：4 材质 × 50 = 200/200 成功、0 失败，`validate_dataset.py` 全部 "OK: no issues found"** |
| 目标平台 | WSL2 / RTX 5060 Laptop (sm_120, 8GB VRAM) / 7.4GB RAM |

---

## 1. 目标与非目标

### 1.1 目标

生成**温-触觉 VLA 任务数据集**：与 LIBERO 同构的 robomimic HDF5 格式，额外包含
**触觉（tactile）**与**温度（temperature）**感知通道，可用于：

1. 语言条件 VLA 策略的训练与评测（动作接口与 LIBERO 一致，7 维）；
2. 触觉 / 温度感知的表征学习与消融实验；
3. 热物拾取（pick hot object）类任务的成功率评测（初始状态由 episode seed 复现）。

### 1.2 非目标（明确不做）

| 不做 | 原因 |
|---|---|
| 训练 VLA / RL 策略 | 数据集生成项目，训练在外部进行 |
| 依赖 Eden 的 Manager/Recorder 框架 | `recorder_manager.py` 全仓库零调用者，从未被执行过；自写流式 HDF5 writer |
| 依赖 rsl_rl、论文复现实验矩阵 | 与数据生成无关 |
| 人类遥操作采集 | WSL 环境无遥操作接口；使用**脚本专家 + 域随机化** |
| 光学触觉渲染（GelSight 图像） | 触觉以 taxel 数组记录（可选提供 tactile-as-image 可视化工具） |

---

## 2. 设计约束（已核实事实）

本节所有条目均已在上游源码或官方文档中核实，是架构决策的依据。

### 2.1 平台与渲染（WSL2）

Genesis 有 3 条渲染路径，全部可在 WSL2 下工作（两条不依赖 OpenGL）：

| 路径 | 机制 | WSL2 前提 | 用途 |
|---|---|---|---|
| A. BatchRenderer | Madrona（CUDA 内核） | `gs-madrona` 需 CUDA 12.8+ / torch cu128+；支持 Linux x86-64；**运行时 NVRTC 编译，架构无关（应支持 sm_120，Gate 3 实测确认）** | **首选**：批量 ML 数据生成 |
| B. Rasterizer + EGL | pyrender → Mesa D3D12 翻译层 | `LD_LIBRARY_PATH=/usr/lib/wsl/lib`、`GALLIUM_DRIVER=d3d12`、`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` | 单环境调试 / 可视化 |
| C. Rasterizer + OSMesa | CPU 软件光栅化 | 安装 OSMesa 即可；Genesis 在 `vis/rasterizer.py:18-20` 自动回退 | 保底路径（慢） |

关键约束（官方文档核实）：

- BatchRenderer 要求**所有批量相机分辨率一致**；相机不能绑定单个 `env_idx`；
  `scene.render_all_cameras()` 返回 `(n_cameras, n_envs, H, W, 3)`。
- **Quadrants 在 WSL 下若不设 `LD_LIBRARY_PATH=/usr/lib/wsl/lib` 会静默回退 CPU**——
  运行前必须导出（Gate 脚本会打印当前 backend/device 供确认）。
- torch 与 gs-madrona 的 CUDA 版本错配会在 import 时段错误（segfault）→ 锁定 cu128。

### 2.2 温度仿真（Genesis `TemperatureGridSensor`）

- 按环境批量（cache 带 B 维）；`grid_size` 可配（默认 (1,1,1) = 集中温度）；
  体素尺寸 = 连杆世界 AABB / grid_size。
- **一个传感器 = 一个连杆**；可挂在静态物体连杆上（记录物体温度）。
- 每个环境每传感器状态 = `prod(grid_size)` 个浮点数；物理模型 = FFT 扩散 +
  接触热传导 + 辐射/对流；`MAX_TEMP = 1000°C`。
- ⚠️ `ambient_temperature`、`convection_coefficient`、`properties_dict`、
  `simulate_all_link_temperatures` 是**类级共享的（"last one set wins"）**——
  同一进程内多场景构建会互相覆盖。→ 约束：**环境温度等全局热参数按"场景构建"固定**；
  逐 episode 随机化走物体初始温度——已核实机制：无公开 setter，但可在
  `scene.reset()` 之后直接写 manager 的 ground-truth 缓冲（`_ground_truth_intermediate_cache`，
  转置布局 `(cols, B)`），即 `seed_temperature()`（§5.2）；写入需一步 `scene.step()` 后才可见。
- ⚠️ 体素体积仅对**轴对齐连杆**精确；被抓物体旋转时其网格为近似。→ 物体温度以
  **集中温度通道为主**，网格为辅。

### 2.3 触觉仿真（Genesis 解析式 taxel 传感器）

- 解析计算（无需渲染），10 种模态注册表（源自 `dexterous-hands/src/tactile_sensors.py`）：

| 模态 | features/probe | 类型 | 说明 |
|---|---|---|---|
| `link_bool` / `link_force` | 1 / 3 | 连杆级 | 接触布尔 / 合力 |
| `bool` / `depth` | 1 / 1 | 探针级 | 接触布尔 / 压入深度 |
| `agg_bool` / `agg_force` | 1 / 3 | 探针聚合到连杆 | 每连杆聚合 |
| `force` / `force_torque` | 3 / 6 | 探针级向量 | KinematicTaxel |
| `proximity` | 3 | 探针级向量 | 接近场（跟踪 link 的点云 BVH，**不接触门控**，见 R23）；引擎返回 force+torque，本项目只记录 force |
| `elastomer` | 3 | 探针级向量 | 弹性体位移场（最接近光学触觉）；**正式 schema 的 marker 场**（Q5 已决），16×16/指 |

- 探针布局：**自 Franka 夹爪指节 URDF 的碰撞几何程序化推导**（`finger_layout.py`）。
  Panda 的手指碰撞体是 **mesh**（`meshes/collision/finger.obj`，腕部才是 box），
  且整指 AABB 不能当抓取面用：真正的平面指垫只在**指尖附近**凸起
  （link 系 z ≈ 0.036..0.054），指根/腰部凹陷 2.5..15 mm —— 若按 AABB 面布探针，
  前几排会悬在指面之外、提前触发接触并虚增压入深度。因此流程为：
  取 mesh 顶点 → 以 `v·inner` 的极大值 + `pad_tol_m`（默认 1.5 mm）选出**指垫平板**，
  其切向范围即指垫（非整指）→ 每根探针再吸附到自身切向邻域内的局部表面高度。
  闭合轴由两指**碰撞体中心**在父系（hand）中的连线定出 —— **不用关节原点**
  （Panda 的 mimic 双指关节共享同一原点 `xyz="0 0 0.0584"`），
  **更不能用指垫面心**：夹爪闭合时两指垫近乎共面（左 y=-0.0001 / 右 y=+0.0001），
  其差只有 0.2 mm、符号指向"哪块垫面恰好更高"，会把闭合轴指反
  （实测会把探针布到手指**外**侧）。仅在无碰撞几何时回退到关节原点差。
  最后在指向内侧的平面上生成探针网格 + 外法线；**网格密度逐模态**（2026-10-04 起）：
  `tactile.probe_grid` 可以是 `[4, 3]`（所有模态共用）或
  `{default: [4, 3], elastomer: [16, 16]}`（逐模态覆盖，唯一的取值口是
  `TactileConfig.grid_for(modality)`）。场景按**去重后的网格**调用 `derive_finger_layout`
  （指垫几何与网格无关，一份 `layout_info` 复用），组装出 `layouts_by_modality`；
  `SceneBundle.layouts` 仍是**主模态**网格那一份（§4.2 的根别名）。
  法线仅用于记录/可视化，**不可**传入 ContactProbe/ContactDepthProbe/KinematicTaxel
  （这些类无 `probe_local_normal` 字段）。仅支持**轴对齐**指垫法线（否则报错，
  可用 `box_override`/`inner_dir_override` 逃生口）。
- 传感器不完美度：配置项 `tactile.imperfection`（clean / mild / strong，映射上游
  `tactile_params.yaml` 的噪声/偏置/随机游走档）；v1 默认 `clean`（不注入额外构造参数）。
- ⚠️ 必须携带 `src/sensor_compat.py`（`install_sensor_reset_fix()`）——
  修补 Genesis 索引化随机游走 reset bug。
- 时序：传感器可配 `history_length=DECIMATION`，在每个控制步内对子步读数做
  **median 降噪**后再落盘（本项目沿用）。

### 2.4 控制与资产

- 控制率 = `decimation × sim_dt`；本项目取 **sim_dt = 1/200、decimation = 10 → 20 Hz**
  （与 LIBERO 一致）。
- 机器人：**Franka Panda + 平行夹爪**（Genesis 自带 `urdf/panda_bullet/panda.urdf`，
  7 臂关节 + 2 指关节；动作 7 维语义与 LIBERO 完全一致，底层由 IK 驱动手臂）。
  ⚠️ 实体必须以**单位姿态 build**（温度传感器的 AABB/体素体积在 build 时冻结），
  需要朝向时通过 root DOF 设置后再 build 传感器。
- 脚本专家模板：`scripts/material_id/collect_slides.py`（纯 Genesis、多环境、
  温度+接触采集、npz 输出）。

### 2.5 存储与算力预算

- 8GB VRAM（本机 RTX 5060 Laptop）、7.4GB RAM + 2GB swap（**禁止在内存中缓冲整个数据集**，
  必须流式写盘）。
- 单 episode 估算（**v1/format 1.0 口径**：T=200、无相机、Franka 双指各 12 探针、
  手指网格 4×4×1、物体网格 4×4×4；marker 场的增量见本节的下一张表）：

| 通道 | 字节/步 | 200 步合计 |
|---|---|---|
| 触觉 force (24×3 f32) + depth (24 f32) | 384 B | 77 KB |
| 温度：双指网格 2×16 + 物体 4³ + link_temps + object | ~440 B | 88 KB |
| 接触 + 本体感知 | ~90 B | 18 KB |
| L4 爆发窗口（10 窗口 × 61 帧 × ~0.44 KB） | — | ~270 KB |
| **合计** | | **≈ 0.45 MB / episode** |

- 加相机（Gate 3/4 后）：双 **256×256×3** uint8 = 393 KB/步（原始）→ T=200 口径 +78.6 MB/episode
  上界；gzip 后 **7.2 MiB/episode**（T=200 口径，§8）。128² 时为 98 KB/步 → 19.7 MB 上界。
  这正是"先无相机出数据"的原因。**实测**：34 步/episode、gzip 后 **1.22 MiB/episode**
  （256²，§8 Gate 4；128² 时为 0.49 MiB）。
- 规模参考：v1 交付 1 任务 × 4 材质 × 50 演示 = 200 episodes ≈ **90 MB**（无相机，gzip 后更小）。
  全 LIBERO 规模（130×50）**不在 v1 目标内**。
  **实测修正（2026-10-02）**：专家平均 **34 步**（32–36，≈1.7 s @20 Hz）即完成抓取+抬升并提前
  结束，远低于 T=400 上限 → 实际单 episode 0.063–0.065 MiB、200 条合计 **12.9 MiB**（预算的 1/7）。
  若 v2 需要更长轨迹（如带相机的接近段），上表的 T=200 口径才生效。

**marker 场转正后的预算（2026-10-04，T≈44 实测口径）**：

| 项 | 增量 | 备注 |
|---|---|---|
| `obs/tactile/elastomer`（L1，16×16×2 指 ×3 f32） | 6144 B/step 原始 → **2.1–3.1 KiB/step** gzip-4 | 接触占比决定压缩率：稀疏接触 42 步实测 89.7 KiB（2.1 KiB/step），持续接触 44 步实测 136 KiB |
| `obs/link_pose`（L1，2 指 × 7 f32） | 56 B/step 原始（不可压） | 可忽略（42 步 2.3 KiB） |
| L4 burst 的 `fingertip_elastomer` | **110–218 KiB/窗口** → 10 窗口 ≈ **1.1–2.2 MiB/episode** | 单集最大单项；`recording.burst_modalities` 可收口 |
| 时间 | 无相机 84.4 → **127.7 ms/step（+51%）**，几乎与探针数无关 | 带相机路径因 llvmpipe 占大头，相对增幅 ~+25%（R8） |

即：`data/` 12.9 MiB → **~450 MiB**、`data_cam/` 97 MiB → **~540 MiB**、`data_lift/` 同量级，
三套合计 **~1.4 GiB**（不含旧的 format 1.0 副本）；VRAM 不变（触觉传感器在 CPU 侧）。
`probe_radius_m` 仍全局共用（16×16 下探针球重叠，见 R8 末段）。

---

## 3. 总体架构

### 3.1 分层

```
┌──────────────────────────────────────────────────────────────────┐
│ scripts/      gate0_engine · gate1_sensors · gate2_scene ·        │ ← 验证与生成入口
│               generate（+ tools/validate，校验与消费）             │
├──────────────────────────────────────────────────────────────────┤
│ recording/    EpisodeRunner · HDF5Writer · events · meta          │ ← 数据出口（流式）
├──────────────────────────────────────────────────────────────────┤
│ expert.py     脚本专家 PickExpert（阶段状态机 + 域随机化 + 触觉反馈）│
├──────────────────────────────────────────────────────────────────┤
│ actions.py    7 维动作控制器（6 delta pose + 1 协同抓取 → IK + PD）│
├──────────────────────────────────────────────────────────────────┤
│ obs.py        观测组装（proprio / tactile / temperature / contact）│
├──────────────────────────────────────────────────────────────────┤
│ config.py     任务规格（TaskConfig + configs/*.yaml）· 成功判定参数 │
├──────────────────────────────────────────────────────────────────┤
│ scene.py      场景与重置（地面/桌/物体/机器人/热参数/逐 episode 随机化）│
├──────────────────────────────────────────────────────────────────┤
│ sensors/      传感器装配（URDF 探针布局 → 触觉 + 温度 + 接触）      │
├──────────────────────────────────────────────────────────────────┤
│ compat/       引擎补丁（上游触觉 reset 修复，monkey-patch）         │
├──────────────────────────────────────────────────────────────────┤
│ genesis-world v1.4.1（pip wheel：Quadrants + Madrona + 传感器内核）  │ ← 唯一引擎依赖
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 一次 episode 的生命周期（数据流）

```
configs/*.yaml ──► TaskConfig ──► build_scene(cfg, material) ──► SceneBundle
                                      │  (机器人/物体/桌/传感器/热参数，单位姿态)
                                      ▼
                    ┌──────── EpisodeRunner.run_episode(seed) ────────┐
                    │  sample_episode(rng) → xy/yaw/temp_c             │
                    │  bundle.reset_episode(...) → 物体摆位 + 温度注入   │
                    │  bundle.step()  ← warm step（缓存 post-reset 状态）│
                    │  for t in 0..max_steps:                          │
                    │    obs = ObsBuilder.build()      ← policy 率 20Hz │
                    │    a = expert.act(obs) → 控制器 → IK → 关节 PD     │
                    │    for _ in range(decimation):  ← 200Hz 子步       │
                    │        bundle.step()                             │
                    │        BurstRecorder.push / 接触沿触发（L4）        │
                    │    接触事件累积（L3）· 抬升判定 → reward/done       │
                    └───────────────┬──────────────────────────────────┘
                                    ▼
                    HDF5Writer.add_episode（demo_N 立即落盘；
                    失败 → *_failed.hdf5）
```

### 3.3 依赖策略

**只用 Genesis（pip `genesis-world==1.4.1`），其余全部自研/静态拷贝：**

| 类别 | 处置 |
|---|---|
| 引擎 | `pip install genesis-world==1.4.1`（wheel 自带 `assets/urdf/panda_bullet/`）；vendored 树 `~/tactile-genesis/Genesis` 仅作只读参考 |
| 触觉补丁 | 拷贝 `src/sensor_compat.py` → `tempvla/compat/sensor_compat.py`（逐字保留，import 时 `install_sensor_reset_fix()`） |
| 探针布局 | **程序化推导**（不再拷贝布局文件）：`finger_layout.py` 解析夹爪 URDF 碰撞几何（mesh 顶点 → 指垫平板）生成两指内侧网格 |
| 机器人 URDF | 用 pip wheel 自带的 `urdf/panda_bullet/panda.urdf`（不拷贝资产目录） |
| 传感器参数 | 不需要外部 YAML：触觉/温度参数直接写在任务配置 `configs/*.yaml` 的 `tactile:`/`temperature:` 段 |
| 专家模板 | 参考 `collect_slides.py` 重写（不 import） |
| 明确不用 | Eden / rsl_rl / dexterous-hands 的任何 Python 模块 |

---

## 4. 数据契约（核心）

### 4.1 文件组织（对齐 LIBERO）

```
data/
├── pick_hot_cylinder_aluminum.hdf5          # 每 (任务, 材质) 一个文件，含 50 条成功演示
├── pick_hot_cylinder_aluminum_failed.hdf5   # 失败 episode 单独存放；**0 失败时不落盘**（不产生空文件）
├── pick_hot_cylinder_copper.hdf5
├── pick_hot_cylinder_copper_failed.hdf5
├── ...
└── （规划中）dataset_stats.json             # 各文件成功率 / 参数分布统计
```

**为什么按材质分文件**：热物性（k/ρ/c/ε）是 `TemperatureGridSensor` 的**类级共享参数**
（§2.2，同一 scene 内 "last one wins"），且物体尺寸/质量在场景构建期冻结。
因此材质是**文件级常量**，文件内只随机化位姿/偏航/初始温度。这也让"材质分类"成为
文件级的天然标签，下游可按材质做条件生成。

### 4.2 HDF5 树形结构

```
<task_id>_<material>.hdf5
├── attrs
│   ├── format_version : "tempvla-1.3"   # 1.0 = 单网格/无 link_pose（2026-10-04 前）；1.1 = marker 场转正；1.2 = 多物体加法扩展；1.3 = 可选稠密高度图
│   ├── generator      : JSON str   # 项目名 + 版本 + 生成时间
│   └── task_id        : str
│
├── data/                                   # ── robomimic 兼容层（LIBERO 同构）──
│   ├── attrs
│   │   ├── env_args     : JSON str  # 机器人/控制器/相机/传感器/率参数（见 §4.5）
│   │   ├── problem_info : JSON str  # 语言指令 + 材质 + 热参数 + 随机化范围（§4.5）
│   │   └── total        : int       # demo 数
│   ├── demo_0/
│   │   ├── attrs
│   │   │   ├── num_samples          : int
│   │   │   ├── success              : bool
│   │   │   ├── episode_seed         : int   # 复现本 episode 的位姿/温度
│   │   │   ├── episode_steps        : int
│   │   │   ├── num_contact_events   : int   # 见 §4.7-L3
│   │   │   ├── num_windows          : int   # 见 §4.7-L4
│   │   │   ├── ik_err_max_m         : float # 专家执行质量（IK 位置误差峰值）
│   │   │   ├── expert_final_phase   : str
│   │   │   ├── grasp_confirmed      : bool
│   │   │   ├── object_material      : str
│   │   │   ├── object_radius_m      : float
│   │   │   ├── object_height_m      : float
│   │   │   ├── object_mass_kg       : float
│   │   │   ├── object_temp_c        : float # 本 episode 初始温度（真值）
│   │   │   ├── object_xy            : (2,) f32  # 相对桌面中心的偏移
│   │   │   ├── object_yaw           : float
│   │   │   ├── object_final_pos     : (3,) f32
│   │   │   ├── object_link_idx      : int   # 物体在 link 表中的下标
│   │   │   │   # ↓↓ format 1.2：上面所有 `object_*` 标量 = **本集目标**的值（单物体文件语义不变）
│   │   │   ├── language_instruction : str   # 本集目标的语言串（如 "lift the aluminum box"）
│   │   │   ├── target_index         : int   # 本集目标是 objects_* 里的第几个（单物体 = 0）
│   │   │   ├── objects_shape        : (N,) str  # 全部物体，行序 = obs/object_pose
│   │   │   ├── objects_xy           : (N,2) f32 # 相对桌面中心
│   │   │   ├── objects_yaw          : (N,)  f32
│   │   │   ├── objects_temp_c       : (N,)  f32 # 各物体初始温度（真值）
│   │   │   ├── objects_final_pos    : (N,3) f32
│   │   │   ├── objects_link_idx     : (N,)  int
│   │   │   └── objects_radius_m / _height_m / _mass_kg : (N,) f32
│   │   ├── obs/
│   │   │   │   # ── LIBERO 原生通道 ──
│   │   │   ├── ee_pos           : (T,3)   float32   # TCP（夹爪中心，世界系）
│   │   │   ├── ee_ori           : (T,3)   float32   # 轴角 rotvec
│   │   │   ├── gripper_states   : (T,1)   float32   # -1 张开 / +1 闭合
│   │   │   ├── joint_states     : (T,9)   float32   # 7 臂 + 2 指
│   │   │   ├── link_pose        : (T,L,7) float32   # 每指 pos_world(3)+quat_wxyz(4)，纯 FK 读回
│   │   │   ├── object_pose      : (T,N,7) float32   # 每物体 pos_world(3)+quat_wxyz(4)，编码同 link_pose（1.2）
│   │   │   │   # ── 扩展通道 ① 触觉（每模态用自己的网格，行数 = L × prod(grid_m)）──
│   │   │   ├── tactile/
│   │   │   │   ├── force     : (T,24,3)  float32    # 4×3/指：P = 两指各 12 探针拼接（世界系）
│   │   │   │   ├── depth     : (T,24)    float32    # 4×3/指：压深（米）
│   │   │   │   ├── elastomer : (T,2P,3)  float32    # marker 场 = 每标记的 3D 位移（世界系）；P 由逐模态网格定（1.1）——正式 config 16×16=256/指，pair config 9×7=63/指
│   │   │   │   ├── height_map: (T,L,H,W) float32    # 1.3：marker 场重采样出的**稠密高度图**（米；指垫法向、link 系；0 = 未接触）。`tactile.height_map.enabled` 时才有，见 §4.2 末与 §9-R25
│   │   │   │   └── (proximity : (T,24,3) float32)   # 可选模态，正式 config 未开（Q5）
│   │   │   │   # 世界系的向量模态用下面的 link_pose 精确转回 link 系（§4.2）
│   │   │   │   # ── 扩展通道 ② 温度 ──
│   │   │   ├── temperature/
│   │   │   │   ├── fingertip_grid : (T,2,4,4,1) f32  # 两指温度场
│   │   │   │   ├── object_grid    : (T,4,4,4) f32    # **目标**物体的温度场（热传导另一侧；单物体时 env 轴被压掉）
│   │   │   │   ├── objects_grid   : (T,N,4,4,4) f32  # 全部物体的温度场，行序同 object_pose（1.2）
│   │   │   │   ├── link_temps     : (T,L_all) float32 # 全连杆集中温度（慢通道）
│   │   │   │   └── object         : (T,1)  float32   # **目标**物体网格均值（集中温度）
│   │   │   │   # ── 扩展通道 ③ 接触 ──
│   │   │   └── contact/
│   │   │       ├── link_contact : (T,2) bool       # 每指接触状态
│   │   │       └── link_force   : (T,2,3) float32  # 每指接触合力
│   │   │   # 相机通道 agentview_rgb / eye_in_hand_rgb：cameras.enabled=true 时写入（Gate 4）
│   │   ├── events/                                 # 接触事件与瞬态窗口（§4.7）
│   │   │   ├── contact_events/
│   │   │   │   ├── event_id / start_step / end_step : (K,) int32
│   │   │   │   ├── start_frame / time_s : (K,) int32 / f32
│   │   │   │   ├── peak_force_n / max_depth_m : (K,) f32
│   │   │   │   └── link_bool : (K,L) bool         # 上升沿涉及哪些指
│   │   │   └── burst/                              # sim 率（200Hz）瞬态窗口
│   │   │       └── window_0/ window_1/ ...         # 每窗口一个组
│   │   │           ├── trigger_frame / start_frame / end_frame : (1,) int32
│   │   │           ├── tcp_pos / finger_joint_pos : (Tw,3) / (Tw,2) f32
│   │   │           ├── contact_bool / contact_force : (Tw,2) bool / (Tw,2,3) f32
│   │   │           ├── fingertip_<modality> : (Tw,L×prod(grid_m)[,3]) f32  # 含 16×16 marker 场
│   │   │           ├── finger_temp  : (Tw,2) f32
│   │   │           └── object_temp  : (Tw,1) f32
│   │   │           #  Tw = pre + 1 + post = 61 帧 @200 Hz（[−0.05 s, +0.25 s] 闭区间）
│   │   │   # states (T,S) 未写入：全量状态回放校验见 §5.8，留待 v2
│   │   ├── actions             : (T,7) float32
│   │   ├── rewards             : (T,)  float32      # 稀疏：抬升稳定达成的那一步 = 1
│   │   └── dones               : (T,)  bool
│   └── demo_1/ ... demo_{N-1}/
│
└── layouts/                                # ── 静态几何（每文件一份，与 episode 无关）──
    ├── attrs: n_probes_total, n_links, probe_grid, modality_grids, world_frame,
    │          link_pose_encoding, finger_link_names, hand_link_name, modalities,
    │          object_link_idx, object_link_indices, tcp_offset, temperature_grids
    │          # object_link_idx = 第一个/目标物体（旧语义）；object_link_indices (N,) = 全部（1.2）
    │          # n_probes_total / probe_grid 都是**主模态**的那一份（老读者语义不变）
    │          # height_map (1.3, JSON str) = 稠密高度图的契约：source/grid/size/interpolation/unit/axes
    ├── probe_local_pos    : (P,3) float32    # ← 主模态网格的别名（见下面的 grids/）
    ├── probe_local_normal : (P,3) float32    # 指向另一指的内法线（仅记录/可视化）
    ├── probe_link_index   : (P,)  int32      # 每个探针属于哪根手指（0/1）
    ├── link_names         : (L,)  |S17 bytes # 定长字节串（读时 .decode()，**不是** vlen str）
    └── grids/<tag>/                          # ── 每模态网格（format 1.1 起）──
        ├── attrs: probe_grid (2,) int32, modalities (M,) |S… bytes  # 同 link_names，读时 .decode()
        ├── probe_local_pos    : (P_m,3) float32   # tag = g<rows>x<cols>，如 g4x3 / g16x16
        ├── probe_local_normal : (P_m,3) float32
        └── probe_link_index   : (P_m,)  int32
```

**`layouts/` 的两层结构（2026-10-04 起，format `tempvla-1.1`）**：

- `grids/<tag>/` 是每模态网格的**权威**：`attrs["modality_grids"]`（JSON，`{模态: tag}`）
  说每个 `obs/tactile/<模态>` 的行数该按哪个网格 reshape。`probe_grid` 在根上仍然写着
  **主模态**（`modalities[0]`，即 force 的 4×3）的网格，根上的
  `probe_local_pos`/`normal`/`link_index` 就是主网格的**别名**——所以**只认旧格式的读取器
  读到的字节与 format 1.0 完全一致**，`modality_grids` 对它们是透明附加。
- 没有 `modality_grids` 的文件（format `tempvla-1.0` 及更早）所有模态共用一个
  `probe_grid`；读取端（`tempvla/tools/tactile_field.py`、`render_video.py`、
  `validate.py`）都按"缺 `modality_grids` → 回退到根 `probe_grid`"处理。
- `world_frame` 记录向量模态（force/proximity/elastomer）落盘时是否旋到了世界系；
  `obs/link_pose`（每步每指 pos_world(3)+quat_wxyz(4)）让这件事**精确可逆**——
  `disp_link = R(q)ᵀ · disp_world`，见 §4.3 与 R8⑤。

**多物体（2026-10-07 起，format `tempvla-1.2`）**：一批场景在**同一张桌子上放 N 个物体**
（如圆柱 + 立方体），脚本专家按 episode 均匀挑一个当目标。约定的核心是**旧键不改变语义**：

- 旧键 = **目标物体**：`obs/temperature/object` / `object_grid`、demo 的 `object_*` 标量组、
  `layouts.object_link_idx`（= 第一个物体）、`env_args.object`。单物体 config（`objects` 缺省）
  的旧键与 format 1.1 **逐字节相同**（多目标抽取在 N=1 时不消耗 rng，整条随机流不变）。
- 新键 = **全部物体**（行序一致，与 `env_args.objects` 对齐）：`obs/object_pose (T,N,7)`、
  `obs/temperature/objects_grid (T,N,*object_grid)`、demo attrs 的 `objects_*` 组、
  `layouts.object_link_indices (N,)`；`target_index` + `language_instruction` attr 说明本集抓谁。
- 只认旧格式的读取器读到的内容与今天一致；`validate.py` 只在 `obs/object_pose` 存在时才做
  N>1 一致性检查（形状、四元数归一、`objects_grid` 尺寸、`target_index` 范围），旧文件自动跳过。

**稠密高度图（2026-10-08 起，format `tempvla-1.3`）**：可选通道
`obs/tactile/height_map (T,L,H,W) float32`，单位米——把 marker 场重采样成一张
GelSight/UniVTAC 风格的稠密图（对应 UniVTAC 的 `press_depth`）。纯加法：不开
`tactile.height_map` 的 config 与 1.2 逐字节相同（只差版本串），旧文件读取端自动跳过。

- **是重采样，不是渲染**（§9-R25）：本项目的 gel 是虚拟传感器（`ElastomerTaxel` =
  SDF 查询 + 网格采样），场景里没有可形变几何体可供相机拍摄；UniVTAC 能渲
  `press_depth` 是因为它仿真 FEM gelpad 并在指内装相机。我们沿用它的**语义**
  （指垫表面法向的离面位移场，米），用插值实现。
- **契约**（写在 `layouts.attrs["height_map"]`，JSON）：`source: elastomer`、
  `grid`、`size`、`interpolation`、`unit: m`。行 = 网格轴 0、列 = 网格轴 1，
  **cell-centred**——图像覆盖网格的 *cells*（最外圈标记外各半格），与渲染器已有的
  span 模型（`gel_render.grid_m_per_px` / `lattice_nodes`）一致，所以 `--markers`
  的标记晶格能**精确**叠在同一张图上；最外半格是**最外圈标记线的平延**——采样坐标
  先 clamp 到标记范围再查样条，边缘像素正好落在最外一排（只有 `mode="nearest"` 不够：
  scipy 的样条预滤波用 mirror 边界，分数坐标落到范围外几个百分点会 ringing，实测
  64 px 边距里前 4 px 从 5.05 µm 漂到 5.17 µm；clamp 后严格常值），不是新造的数据。
  样条过样本点但会过冲，结果 clamp 回样本
  min/max：图里不会出现传感器没读到的位移。
- **插值不增加信息**：9×7 的场给不出它 63 个样本没有的细节，`size` 只是存储分辨率。
  栅格也不一定落在每个样本上（9×7 → 64×64 时孤立尖峰丢 ~1.5% 峰值），所以
  **不要用"峰值相等"来验它**——`validate.py` 的做法是对存在该通道的文件**重新合成一遍
  再逐元素比对**（任何轴序/帧/符号/尺寸错位都会远超插值噪声），旧文件跳过。
- **读取端**（`tempvla/tools/height_map.py`，纯 h5py/numpy + 延迟 import scipy）：
  `read_height_map`（文件里存了就直接用）、`synth_from_markers`（**任何**带
  `obs/tactile/elastomer` + `layouts/` 的文件都能现场合成，包括 1.1 老文件）、
  `height_map_or_synth`（存了用存的，否则按元数据合成），`render_video.py` 走后者。

**为什么这样放**：

- `data/` 子树与 LIBERO/robomimic **逐字段兼容**——现有工具链（robomimic 读取器、
  LIBERO 评测脚本）可直接打开；新增子树 `obs/tactile/`、`obs/temperature/` 对旧读取器
  是透明附加。
- `layouts/` 放在根级（不在 `data/` 内）：robomimic 读取器只读 `data/` 组，多余的根级组
  被忽略；同时避免把布局数据塞进 HDF5 attrs。
- 探针世界坐标**不逐步存储**（可由 `joint_states` + `probe_local_pos` 重算），
  存储开销省一半；可视化工具按需重算。
- 探针布局是**程序化推导**的（`finger_layout.derive_finger_layout` 解析 URDF），
  不是外部 layout 文件，因此 `layouts/` 里没有 layout_id / source_hash 字段。
- **marker 场需要的几何已经全在文件里**：`probe_local_pos` 就是未形变的参考位置，
  配 `obs/tactile/elastomer` 的位移即得标记位置（Tabero 的 `M = [参考帧, t_1..t_H]`
  直接可拼）；`probe_local_normal` 给出"出平面"方向（拍在指垫法线上，不是 link 系的
  某个轴——Panda 指垫是 **y**），`obs/link_pose` 给出世界系↔link 系的精确变换。
- **相机通道**（`agentview_rgb`/`eye_in_hand_rgb`）在 `cameras.enabled=true` 时写入（Gate 4），
  v1 默认关闭；**v1 未写入**：全量 `states`（回放校验用，见 §5.8）、
  `init_state_idx`（固定初始状态池未实现——初始状态由 `seed` + episode 序号复现：
  `np.random.default_rng(episode_seed)` → 位姿 / 偏航 / 温度）。

### 4.3 观测通道定义

| 通道 | 形状 | dtype | 采集方式 | 备注 |
|---|---|---|---|---|
| `agentview_rgb` | (T,256,256,3) | uint8 | 第三人称固定相机，`pos=(0.78,-0.38,0.70)` | ✅ Gate 3/4 已验证（物体 **764 px ~28×28**）；`cameras.enabled=true` 时写入 |
| `eye_in_hand_rgb` | (T,256,256,3) | uint8 | 腕部相机，挂 `panda_hand` +x 侧 | ✅ Gate 3/4 已验证（物体占画面 ~21%）；与 agentview **同分辨率**（BatchRenderer 硬性要求，raster 亦保持） |
| `ee_pos` | (T,3) | f32 | TCP（夹爪中心）世界位置 | Franka + 平行夹爪：`panda_hand` link 位姿 + `tcp_offset`（Gate 1 标定） |
| `ee_ori` | (T,3) | f32 | TCP 世界姿态（轴角 rotvec） | 与 `rotations.py` 的 wxyz ↔ rotvec 约定一致 |
| `gripper_states` | (T,1) | f32 | 夹爪开合标量 s ∈ [-1,1] | -1 = 张开，+1 = 闭合；由两指关节均值映射 |
| `joint_states` | (T,9) | f32 | 7 臂关节 + 2 指关节 | 顺序 = `arm_dofs + finger_dofs` |
| `tactile/<modality>` | (T,P_m[,F]) | f32 | Genesis taxel 传感器 | **子步 median 降噪后**按 policy 率采样；**P_m = n_links × prod(grid_m)**，每个模态用自己网格（`layouts/modality_grids`）——正式 config 的 force/depth/proximity 是 4×3（每指 12）、`elastomer` 是 16×16（每指 256），pair config 是 `elastomer` 9×7（每指 63、无刚性探针，§9-R25）；向量模态旋到世界系，用 `obs/link_pose` 精确转回 |
| `tactile/height_map` | (T,L,H,W) | f32 | 由 `tactile/elastomer` 重采样（1.3） | 稠密高度图（米，指垫法向、link 系），`tactile.height_map.enabled` 时写入；参数在 `layouts.attrs["height_map"]`，读取端 `tools/height_map.py`（§4.2 末、§9-R25）。**不进 burst**：200 Hz 路径不做逐子步重采样 |
| `link_pose` | (T,L,7) | f32 | 两指 link 的 FK 读回 | `pos_world(3) + quat_wxyz(4)`（编码写在 `layouts.attrs["link_pose_encoding"]`）；世界系 ↔ link 系的**精确**逆变换 `v_link = R(q)ᵀ·v_world`，与相机开关逐位一致 |
| `object_pose` | (T,N,7) | f32 | 每物体 root DOF 读回（1.2） | 编码同 `link_pose`；行序 = `env_args.objects`；N=1 时的 `objects[target]` 与旧 `object_*` attrs 同源 |
| `temperature/fingertip_grid` | (T,L,gx,gy,gz) | f32 | TemperatureGridSensor（两指 link） | 默认 (4,4,1)/指（config 可调） |
| `temperature/object_grid` | (T,gx,gy,gz) | f32 | TemperatureGridSensor（**目标**物体 link） | 默认 (4,4,4)，目标物体侧温度场，热传导另一侧（`n_envs=1` 时 env 轴被 `_env0` 压掉，旧文件实测即 (T,4,4,4)） |
| `temperature/objects_grid` | (T,N,gx,gy,gz) | f32 | 同上，每物体一个栅格（1.2） | `objects_grid[:, target_index]` 与 `object_grid` **逐位相同**（同一次读取派生） |
| `temperature/link_temps` | (T,L_all) | f32 | `simulate_all_link_temperatures` | 慢通道，可开关 |
| `temperature/object` | (T,1) | f32 | **目标**物体网格均值 | 目标物体集中温度（`object_grid.mean()`；burst 的 `object_temp` 同源） |
| `contact/link_contact` | (T,L) | bool | `gs.sensors.Contact` | 每指接触状态（L=2） |
| `contact/link_force` | (T,L,3) | f32 | `gs.sensors.ContactForce` | 每指接触合力 |

> 接触热流（§4.7-L2）**不作为记录通道**：Genesis 未暴露该中间量（已在 Gate 2 定级为
> "离线推导"），需要时由 `object_grid`/`fingertip_grid` 的网格差分算出。

### 4.4 动作空间契约（7 维，与 LIBERO 同构）

| 索引 | 含义 | 归一化范围 | 映射 |
|---|---|---|---|
| 0–2 | Δ位置（机器人基座系） | [-1, 1] | × pos_scale（默认 0.05 m/步） |
| 3–5 | Δ姿态（轴角，基座系） | [-1, 1] | × rot_scale（默认 0.5 rad/步） |
| 6 | 夹爪协同 | -1 = 张开，+1 = 闭合 | 在张开/闭合两个关节目标间对 2 指关节插值 |

- **控制率 20 Hz**（sim_dt=1/200 × decimation=10）；T 为 policy 步数，上限 400（20 s）。
- pos/rot scale 在实施时与 LIBERO 仓库的 robosuite controller config 逐项对齐
  （量级：±5 cm、±0.5 rad），并写入 `env_args` 注明。
- 控制器实现（`actions.py`）：每步对 `panda_hand` link 做一次阻尼最小二乘 IK
  （`robot.inverse_kinematics(..., local_point=tcp_offset, init_qpos=当前 qpos)`，
  即 TCP = 夹爪中心），解出的臂关节目标写入 `control_dofs_position`（Genesis 内置 PD，
  kp/kv/force 见 §2.4），指关节目标由协同标量在两关键帧间插值。
  动作语义与 LIBERO 的 7 维 OSC delta 同构，底层换成 IK + PD。
- **本体方案（v1 已定）**：**Franka Panda + 平行夹爪**，单臂固定在基座
  （`fixed=True`），指尖装触觉探针阵列 + 温度网格。动作维度/语义与浮动手方案完全一致，
  区别只在底层：位置增量 → IK → 关节 PD。见 §11 Q1/Q2（已决）。

### 4.5 元数据与语言标注

- `data.attrs['problem_info']`（文件级，LIBERO 同位置）：

```json
{
  "task_id": "pick_hot_cylinder",
  "language_instruction": "pick up the hot copper cylinder",
  "language_instructions": ["pick up the hot copper cylinder"],   # 1.2：全部物体指令去重（单物体 = 1 条）
  "language_template": "pick up the hot {material} cylinder",
  "object_material": "copper",
  "material_properties": {"conductivity": 385.0, "density": 8960.0, "specific_heat": 385.0, "emissivity": 0.0},  # 引擎输入，见 §9-R14
  "all_material_properties": {"aluminum": {...}, "copper": {...}, "abs": {...}, "steel": {...}},
  "object_temp_c_range": [60.0, 200.0],
  "ambient_temp_c": 21.0,
  "object_count": 1,                                               # 1.2
  "objects": [                                                     # 1.2：逐物体描述（行序 = obs/object_pose）
    {"shape": "cylinder", "radius_m": [0.015, 0.025], "height_m": [0.05, 0.08],
     "mass_kg": [0.05, 0.15], "temp_c": [60.0, 200.0], "place_xy": [0.0, 0.0],
     "language_instruction": "pick up the hot copper cylinder"}
  ],
  "target_selection": "single_object",                             # 1.2：| "per_episode_uniform"
  "randomization": {
    "per_file": ["object_material", "object_count", "object_radius_m", "object_height_m", "object_mass_kg"],
    "per_episode": ["object_xy_offset_m", "object_yaw_rad", "object_temperature_c", "target_index"],
    "object_xy_jitter_m": 0.04, "object_yaw_jitter_rad": 3.14159,
    "objects_xy_jitter_m": [0.04], "objects_yaw_jitter_rad": [3.14159]   # 1.2：逐物体
  }
}
```

- 因为**一个文件一个材质**（§4.1），文件级 `language_instruction` 已完全确定，
  不需要 per-demo 覆盖；逐 episode 真值（初始温度、位姿）写在
  **demo attrs**（`object_temp_c` / `object_xy` / `object_yaw`）而不是 problem_info。
  **多物体例外（1.2）**：同一文件里有 N 种形状 → 指令按目标分叉，文件级
  `language_instruction` 取**第一个**物体，每个 demo 的 `language_instruction` attr 记
  它实际抓的那个；`language_instructions` 列出全部去重串。
- `data.attrs['env_args']` 记录完整可复现配置：Genesis 版本与 assets 目录、
  sim_dt/decimation/control_hz、机器人 URDF 与解析出的关节/连杆名、
  动作与观测通道契约（含触觉布局/温度网格/接触通道说明）、物体与桌面参数、
  相机配置、录制（爆发窗口）参数、成功判定与步数上限。**1.2 起** `objects` 数组
  逐物体给 `shape`/范围/`place_xy` 及构建期 `resolved_*`（本文件冻结的实测值），
  旧 `object` 块保留 = 第一个物体。
- `material_properties` / `object_*_m` / `object_mass_kg` 是**文件级真值**（构建期冻结），
  `object_temp_c` 是**逐 episode 真值**——两者都保留，供下游做温度回归/条件生成。

### 4.6 与 LIBERO 的对照

| 维度 | LIBERO | TemperatureVLA | 兼容性 |
|---|---|---|---|
| 文件格式 | robomimic HDF5 | 相同（超集） | ✅ 旧读取器可读 `data/` |
| 动作 | 7 维 OSC delta + gripper | 7 维 delta pose + 协同 | ✅ 同构（底层控制器不同） |
| 控制率 | 20 Hz | 20 Hz | ✅ |
| 相机 | agentview + eye_in_hand 128² | ✅ 同通道、**256²**（Gate 3/4 已验证；`cameras.enabled=true` 即写入） | ✅ 可用（v1 数据未含） |
| 语言 | 文件级 `problem_info` | 相同（一文件一材质，指令完全确定） | ✅ |
| 演示来源 | 人类遥操作 × 50 | **脚本专家 + 域随机化** | ⚠️ 质量分布不同，数据卡必须标注 |
| 本体 | Franka + 平行夹爪 | **Franka + 平行夹爪**（指尖触觉 + 温度） | ✅ 同构（差异只在传感器与记录通道） |
| 评测协议 | 固定 init states + 成功率 | 成功率相同；init states 池未实现（seed 复现，§5.4） | ⚠️ 部分 |
| 触觉/温度/接触 | 无 | force/depth + 双侧温度场 + 接触事件 + 200Hz 爆发窗口 | ➕ 核心增量 |

### 4.7 接触事件、温度传导与高频触觉（扩展通道 ③）

**动机**：policy 率（20 Hz）逐帧采样会漏掉接触瞬态——撞击、滑移起始、热流阶跃都发生在
毫秒级；而"接触时刻的温度传导与触觉"正是本数据集相对 LIBERO 的核心增量。

**四层记录策略**（原始保真 + 事件索引 + 瞬态窗口 + 派生汇总）：

**L1 逐帧原始层**（policy 率，全 episode）：
- `obs/contact/link_contact` / `link_force`：每指接触状态与合力
  （`gs.sensors.Contact` / `gs.sensors.ContactForce`，挂到两根手指 link）；
- `obs/temperature/fingertip_grid` + `object_grid`：指尖与物体**双侧**温度场——
  温度传导过程（接触热、扩散、脱离后余热）全部包含在网格随时间的演化中；
  **实测（v1 全量 200 episode）两信号强度高度不对称**：指尖网格峰值升 **+11…+125 °C**
  （按材质均值 铝 35.4 / 铜 61.9 / abs 33.6 / 钢 45.9），而物体网格均值整段只掉
  **0.02…0.23 °C**。这不是 bug：物体 64 格的热容 ≈ 218 J/K，是指垫 16 格（eps 主导，
  ≈0.19 J/K/格）的 ~70 倍，且仅 2–4 格参与接触，故物侧降温本就该是千分之一量级
  ——物理上正确，但意味着 **`object_grid` 近乎常数、几乎不含信息**，可学习的热信号
  在 `fingertip_grid`（以及 `object` 的初值 = 真值温度）。指尖升温幅度偏大是 R16 的
  数值稳定性取值的代价（k=0.02 比"真实指垫"强 ~7×），**相对**排序与 T0 依赖仍保真；
- `obs/tactile/*`：触觉模态（`force` / `depth` / `proximity` / `elastomer` ...）。每模态
  用**自己的**探针网格（`layouts/modality_grids`，§4.2），所以各通道行数互不相同——
  `force`/`depth` 仍是 4×3（每指 12），`elastomer` 是 16×16（每指 256）；
- `obs/link_pose`：每步每指世界位姿 `pos_world(3) + quat_wxyz(4)`，把
  `world_frame: true` 的向量模态变回**精确可逆**（`v_link = R(q)ᵀ·v_world`），
  是 marker 场能还原到 pad 坐标系的前提。

**L2 接触热流层**（验证门控 → 结论：**不录通道，离线推导**）：Gate 2 检查确认
Genesis 温度模型未通过公开 API 暴露接触热项，且不值得为此打补丁（该量可由 L1 网格
差分重建：Q̇ ≈ ρ·c·V·ΔT/Δt，接触窗口内为主项）。需要显式热流时由
`tools/`（规划：`analyze_contact.py`）从 `object_grid` / `fingertip_grid` 离线推导。

**L3 事件层**（`events/contact_events/`）：**policy 率**检测每指接触 onset/offset
（`ContactMonitor`，状态布尔来自 `contact/link_contact`），每事件记录：
`event_id`、`start_step`/`end_step`（policy 步号）、`start_frame`（对应 sim 子步号）、
`time_s`、`peak_force_n`、`max_depth_m`、`link_bool`（上升沿涉及哪几根手指）。
事件步号与 `actions` 直接对齐，便于关联。

**L4 爆发窗口层**（`events/burst/`，默认开）：以**sim 率（200 Hz）**记录每个接触
上升沿前后的瞬态，窗口 [onset−0.05 s, onset+0.25 s]，即 pre 10 + 触发帧 1 + post 50
= **61 帧**（闭区间 0.30 s），每 episode 上限 10 个窗口。每帧通道：`tcp_pos`、
`finger_joint_pos`、`contact_bool`/`contact_force`、`fingertip_<modality>`（触觉原始模态，
**含 `elastomer`**：一个窗口 61 帧 × 512 标记 × 3 f32 = 366 KiB 原始，gzip 后 110–218 KiB
（窗口内接触占比决定），比整条 L1 marker 通道还大）、`finger_temp`、`object_temp`；
附 `trigger_frame`/`start_frame`/`end_frame`。
窗口数与帧数**不因模态集变化**，`recording.burst_modalities`（默认 `None` = 全部模态）
是唯一的收口阀，出数据集前若觉得 burst 体积过大可在这里裁。
实现：`BurstRecorder` 在 sim 步循环内维护预触发环形缓冲，触发锚定在**真实接触子步**
（用 sim 率 payload 的接触沿检测，而非 policy 步边界）。

**触觉空间定位是免费的**：探针世界坐标 = `layouts/probe_local_pos` + `joint_states` 重算
（或直接用 `obs/link_pose` 的逐步位姿），接触区域 = active 探针集合（`depth>0` 探针的
重心即接触中心）——无需额外接触点几何 API。

**marker 场也是免费的**：`obs/tactile/elastomer` 的位移 + `layouts/grids/<tag>/probe_local_pos`
的未形变参考 = 每个标记的完整三维轨迹（`tools/tactile_field.py` 的 `read_markers`
就是这两项的拼接，Tabero 的 `M = [参考帧, t_1..t_H]` 可直接由它组装）；再配
`tactile_field.pad_normal_field` 投影到指垫法线，即得 GelSight 式高度场。

**热物性元数据**：热传导材质参数（k / c / ρ / ε）写在**文件级**
`problem_info.material_properties`（一文件一材质）。⚠️ 受 §2.2 类级共享约束，
**材质按文件分组重建场景**（文件内只随机化温度/位姿），详见 R1。

**存储增量**（单 episode 估算，4×3 探针 / 4³ 物体网格 / T≈200 / 44 步实测外推）：
L1 全通道 ≈ 0.2 MB；L4 默认配置 ≈ 10 窗口 × 61 帧 × ~0.44 KB ≈ **0.27 MB**。
合计 < 0.5 MB/episode（4 材质 × 50 demo ≈ 90 MB），gzip 后更小。

**开启 marker 场（`elastomer`）后这一项要重算**（实测，2026-10-03，aluminum）：
L1 的 `tactile/elastomer` = 6144 B/step 原始 / **2.1–3.1 KiB/step** gzip-4（接触越多越难压）
→ 42–44 步 ≈ 90–136 KiB；L4 的 `fingertip_elastomer` 才是大头：**110–218 KiB/窗口** × 10 窗口
≈ **1.1–2.2 MiB/episode**，比整条 L1 marker 通道还大。即 marker 场把单集从 ~0.5 MB 推到
~1.6–2.9 MB（相机另计），
而**五分之四在 burst 里**——这就是 `recording.burst_modalities` 存在的理由。

---

## 5. 模块设计

### 5.1 `tempvla/compat/` — 引擎补丁层

- `sensor_compat.py`：从 dexterous-hands 拷贝（**逐字保留**），import 时执行
  `install_sensor_reset_fix()`。
- 职责边界：只做 monkey-patch，不含业务逻辑；升级 Genesis 时首先回归此文件。

### 5.2 `tempvla/sensors/` — 传感器装配层（已实现）

| 接口 | 说明 |
|---|---|
| `finger_layout.derive_finger_layout(urdf, (link_l, link_r), probe_grid, probe_offset_m, pad_tol_m) -> list[FingerLayout]` | 解析 URDF 的指节 collision（mesh 顶点，box 兜底）→ 指垫平板 → 两指**内侧抓取面**探针网格（法线 = 指向另一指的单位向量）；闭合轴取两指垫中心连线（关节原点仅作回退）；输出探针位姿（link 系）、pad 平面（`pad_box_origin/size`，法向厚度为 0）、`num_probes`。**pad 几何与网格无关**，所以逐模态网格只需按**去重后的网格**各调一次，`layout_info` 复用。`layouts_to_hdf5()`（单网格包装）与 `layouts_to_hdf5_multi(groups, *, default_grid, modality_grids, attrs)`（§4.2 的两层 `layouts/`，`groups` 的键是 `(rows, cols)`）打包写盘 + `summarize()` 打印 |
| `tactile.build_finger_sensors(scene, *, entity, layouts_by_modality, cfg, decimation, extra_params, track_link_idx)` | 模态注册表 → `gs.sensors.{KinematicTaxel, ContactDepthProbe, ContactProbe, ElastomerTaxel, ProximityTaxel}`；**modality-major 迭代**，每个模态用 `layouts_by_modality[模态]` 那份布局（§4.2 的逐模态网格）取 `probe_local_pos`（+`probe_radius`）与 `num_probes`/`grid_shape`；`history_length=decimation`，`as_grid` 时按该模态自己的 `(ny,nx,3)` 网格传入以启用串扰/FFT 路径。**`probe_local_normal` 只对 `elastomer`/`proximity` 传**（力轴 = 指垫法线，不传则用引擎默认 (0,0,1)，方向错；这两个类有该字段）；`force`/`depth` 的 KinematicTaxel/ContactDepthProbe **不能**传（options `extra="forbid"`，仅 raycast 后端才读该字段）。**多物体（1.2）**：`track_link_idx` 传**全部**物体 link（引擎原文档："indices whose collision geometry is queried by SDF"，复数，天生多压头），`n_sample_points` 相应传**逐 link 列表**——int 会被引擎按 `divmod` 均分（两个物体 = 每个密度减半），列表则每 link 各按自己的预算采。守卫：同一 modality 不得出现两次、句柄名必须唯一（`read_all` 用 name 做 dict key，重名会**静默覆盖**） |
| `tactile.build_link_contact_sensors(scene, entity, link_names)` | `Contact` / `ContactForce` 挂两指 link → `contact/link_contact`、`contact/link_force` |
| `tactile.read_all(handles, temporal_reduction)` | 子步 history 轴（axis 1）median/last 降噪；`rotate_to_world()` 用 `gu.transform_by_quat` 旋到世界系 |
| `temperature.build_finger_temperature_sensors` / `build_object_temperature_sensor` | 指尖 + 物体**双侧**网格；一个 scene 内**所有** TemperatureGrid 共享同一份 `properties_dict` 与共享元数据（ambient/convection/simulate_all 一次设定，见 §2.2）。**多物体（1.2）**：每物体一个物体栅格，句柄名 `temperature_object_<i>`（`name=`/`object_index=` 参数；单物体缺省名仍为 `temperature_object`），`properties_dict` 逐传感器带各自的 link idx |
| `temperature.seed_temperature(sensor, T_c)` | 逐 episode 真值注入：直接写 manager 的 ground-truth 缓冲（无公开 setter），顺序 **reset → seed → step** |
| `temperature.read_temperature(handles, ground_truth=)` / `read_link_temperatures` | 按 policy 率读取；`read_ground_truth()` 取物理量、`read()` 取含滞回/噪声的测量值 |
| `tempvla/obs.py: ObsBuilder` | 组装 §4.3 全部通道（policy 率 `build()` / sim 率 `fast_payload()`），是 L1/L4 的唯一读取入口。`tactile()` 按**模态自己的布局**迭代，`reshape(n_probes, FEATURES_PER_PROBE[模态])`——形状对不上是**显式报错**而不是静默重排；`link_pose()`（同一组位姿既用于旋转也用于记录，所以排在 `tactile()` 之前）输出 `(L,7)`；`object_pose()` 输出 `(N,7)`；`height_map()`（1.3）把 marker 场用**记录下来的那组四元数**（`obs/link_pose`，即读者反演用的同一组）转回 link 系、投影到指垫法向再重采样成 `(L,H,W)`，只在 `tactile.height_map.enabled` 时计算，且**不从 `fast_payload()` 调用**（200 Hz 路径不重采样图像）；`temperature()` 读入全部 `temperature_object_<i>`，`object_grid`/`object` 取**目标**那份、`objects_grid` 堆全部（1.2）；`fast_payload()` 按 `burst_modalities` 过滤 |

> **触觉形态是逐 config 的**（`tactile.modalities` + 逐模态 `probe_grid`，1.1 起）：
> 正式单物体 config（`pick_hot_cylinder` / `lift_object`）是
> `[force, depth, elastomer]` + 16×16 marker 场；多物体 config
> （`lift_object_pair`，2026-10-08 起）对齐 gsmini/UniVTAC 形态，只留
> **`[elastomer]` 9×7 = 63 标记/指**、不采刚性探针，另开 `height_map`。
> 两者都不影响 schema：`force`/`depth` 仍是受支持模态，只是那个文件里没有。
> 去刚性探针后专家照常工作——接触检测回退到 `contact/link_contact`
> （`build_link_contact_sensors` 建的 link 级传感器，**不是探针**），
> [expert.py](../tempvla/expert.py) 的 `depth` 判据在通道缺失时自动失效（§9-R25）。

> 接触热流没有 `read_heat_flux`：已确认 Genesis 不暴露该中间量（§4.7-L2），
> 需要时由 `tools/` 从温度网格离线差分推导。

> **Genesis 1.4.1 名字约定（Gate 0 已实测）**：`gs.sensors.X`（`X` =
> `TemperatureGrid`/`ContactProbe`/…）是 **Options 类**，只描述构造参数；
> `scene.add_sensor(opts)` 返回的运行时对象是 **`XSensor`**
> （`genesis.engine.sensors.*`），`read` / `read_ground_truth` /
> `link_temperatures` 都在它上面。另外 GPU 后端下引擎 getter
> （`get_pos` / `get_quat` / `get_qpos` / `read()`）返回 **CUDA torch 张量**，
> 转 numpy 必须走 `.detach().cpu()`（`obs._np`、`gate1._np` 已封装）。

### 5.3 `tempvla/scene.py` — 场景构建（已实现）

- `build_scene(cfg, material, seed, show_viewer=False) -> SceneBundle`：一次构建
  地面 + 桌面 + 物体（`cfg.object_list()` 逐个：`shape: cylinder` → `gs.morphs.Cylinder`，
  `box` → `gs.morphs.Box(size=(2r,2r,h))`；位置 = 桌面中心 + `place_xy`，材质物性写入
  TemperatureGrid 的 `properties_dict`）+ 机器人（`merge_fixed_links=False`）+ 手指探针
  传感器 + 温度网格 + 接触传感器；返回 `SceneBundle`（scene / 传感器句柄 / 布局 /
  名称表 / 逐物体规格 / `target_index`）。`cfg.objects` 为空时退回单 `object` 字段，
  行为与 1.1 完全一致。
  `bundle.layouts` 是**主模态**（`modalities[0]`，即 force 的 4×3）那份布局，
  `bundle.tactile_layouts` 才是 `{模态: layouts}` 全表——前者仍按"按指分组"的语义
  被 `obs.py`、指尖温度网格、gate1 标定依赖，逐模态的传感器装配读后者。
  **多物体（1.2）**：`bundle.objects` / `bundle.object_specs` 是全部 N 个；
  `bundle.obj` / `bundle.object_spec` 是**属性**，返回 `[target_index]` 那一份——
  专家、成功判据、gate/probe 的既有调用点因此零改动地跟随目标。
- `bundle.reset_episode(ep)`：**不重建场景**——`ep` 是 `sample_episode` /
  `scripted_episode` 返回的一集字典（逐物体 xy 偏移 / 偏航 / 温度 + `target_index`），
  **所有**物体经 root DOF 摆到桌面上的目标位姿（位姿在构建期以单位位姿冻结，
  温度传感器 link 不变），逐物体 `temperature.seed_temperature` 注入初始温度，
  机器人回 `init_qpos`；随后必须 `step()` 一次（warm step）才可读。
- `bundle.step()` / `object_pose()` / `object_center_height()` / `joint_state()`：
  仿真步进与状态查询；`object_pose()` 返回 `(N,7)`；`object_center_height()` 等
  标量查询仍指**目标**物体。`sample_episode(cfg, rng)` 采样全部物体的
  xy 偏移 / 偏航 / 温度 + `target_index`（episode rng 均匀抽取；**N=1 时不抽**，
  随机流与单物体时代逐位相同）。`scripted_episode(cfg, *, xy_offset, yaw, temp_c,
  target_index=0)` 给 gate/probe 用：固定位姿、不起 episode rng。
  采样后过**最小间距守卫**：任两物体表面距 < `min_object_gap_m`（默认 0.02 m）
  时重抽 jitter（≤5 次，兜底回基准位）。
- 相机（Gate 3 已实现，见 §8）：`build_cameras()` 在 `scene.build()` **之前**注册
  `cameras: {agentview, eye_in_hand}`（均 256²，默认 `renderer: raster`）；
  `_renderer_for(cfg)` 在 `enabled=false` 时返回默认 `Rasterizer()`，与 v1 无相机
  路径逐字节一致。eye-in_hand 经 `cam.attach(hand_link, offset_T)` 挂到手，
  `offset_T` 由 `cfg.cameras.eye_in_hand_{pos,lookat,up}` 经
  `pos_lookat_up_to_T` 得到（手坐标系）；attach 只记录，真正的位姿同步必须发生在
  `scene.build()` 之后（`_is_batched` 只在 `Camera.build()` 里设置，而
  `Visualizer.update()` 不同步挂载相机）——`SceneBundle.sync_cameras()` 负责这件事，
  `render_cameras()` 返回 `{name: (n_envs,H,W,3) uint8}`（Rasterizer 的
  `(H,W,3)` 会被补一维）。
- 相机通道（Gate 4 已实现，见 §8）：`ObsBuilder.cameras()` 在 `cameras.enabled` 时把
  上一步的帧降成 `(H,W,3)` 追加到 `obs/agentview_rgb` / `obs/eye_in_hand_rgb`（平铺，
  LIBERO 布局）；关闭时返回 `{}`，通道表逐字段不变。**只在 `build()` 里渲染，不在
  `fast_payload()` 里**——burst 以 200 Hz 采样，渲染会吃光预算。

### 5.4 任务规格与成功判定

- **任务规格 = `tempvla/config.py` 的 dataclass + `configs/*.yaml`**（无独立
  `tempvla/tasks/` 包）：`TaskConfig` 含 robot / controller / tactile / temperature /
  object / objects / table / recording / cameras 子配置与 `task_id`、`language_template`、
  `materials`、`material_properties`、`max_episode_steps`、`lift_success_m`、
  `stable_steps`、`seed`；`load_task_config(path)` 从 YAML 覆盖式加载。
  物体有两种写法：单物体 `object:`（旧）或 `objects:` **列表**（1.2，每项一个
  `ObjectConfig` + `place_xy` 桌面槽位）；`cfg.object_list()` 统一取用（`objects`
  为空则 `[object]`）。`min_object_gap_m` 给多物体的最小表面间距守卫定阈值。
- **成功判定**（在 `runner.EpisodeRunner` 内实现）：物体相对初始高度抬升
  `> lift_success_m`（默认 5 cm）并**连续保持** `stable_steps` 个 policy 步
  （默认 10 步 = 0.5 s）；达成的那一步 reward=1（稀疏），最后一步 done=True。
- `InitStatePool`（固定初始状态池）**未实现**：初始状态由 episode seed 复现
  （`np.random.default_rng(seed)` → xy 偏移 / 偏航 / 温度），demo attrs 记录真值。
  评测若要 LIBERO 式固定初始状态，可在 v2 加入。
- **v1 任务清单**：`pick_hot_cylinder`（已实现）、`lift_object`（已实现，最小任务）、
  `lift_object_pair`（1.2：一桌两物体、圆柱或立方体按 episode 二选一，
  `configs/lift_object_pair.yaml`；box 走**面抓**——专家把接近偏航对齐到
  物体的**当前面**：抖动先量化到 90° 整数倍（选哪一面），再加物体自身 yaw
  （见 §5.6）。`place_hot_object` / `push_hot_object` 为规划项（见 §11 Q4）。

### 5.5 `tempvla/actions.py` — 7 维动作控制器（已实现）

- `DeltaPoseController.step(a7)`：裁剪 → `target_pos += a[:3]*pos_scale`、
  `target_quat = quat_apply_delta(target_quat, a[3:6]*rot_scale)` →
  `robot.inverse_kinematics(link=panda_hand, pos, quat, local_point=tcp_offset,
  init_qpos=当前 qpos, damping, return_error=True)` → 臂关节 `control_dofs_position`
  （Genesis PD）→ 指关节按协同标量插值。返回 `ActionInfo`（IK 位置/姿态误差、指目标、是否裁剪）。
- `tcp_pose(bundle)`：`panda_hand` 位姿 + `tcp_offset` 旋转 = 夹爪中心（TCP）。
- `normalized_action_to(...)`：脚本专家用（给定目标位姿反解归一化动作）。
- `tempvla/rotations.py`：自实现的 wxyz/Hamilton 四元数工具（`quat_mul/rotate/rotvec` 等），
  与 `genesis.utils.geom` 在 Gate 1 交叉验证——不依赖引擎的广播约定。
- 约束：控制器**只消费归一化动作**，不读传感器——保证与下游策略部署接口一致。

### 5.6 `tempvla/expert.py` — 脚本专家（已实现）

- 阶段状态机（`PickExpert`）：

```
APPROACH → DESCEND → CLOSE → LIFT → HOLD → (成功)
             ↑触觉 contact/depth   ↑squeeze_steps 维持   ↑抬升高度稳定
```

- 反馈信号优先级：接触判定（`tactile.depth` 压深 > `contact_depth_m`，**或**
  `contact/link_contact` 任一指接触——后者由 `build_link_contact_sensors` 挂的
  link 级传感器给出，**不是探针**，所以 elastomer-only 的 config 照样能判接触，
  见 §9-R25；`depth` 通道不存在时该判据自动失效，`expert.py` 用
  `obs["tactile"].get("depth", [])` 取值）> 本体位置 > 温度（可选：接触后温度
  变化率，用于生成更真实的热感知行为）。
- 域随机化（逐 episode 采样，写入 demo attrs）：物体 xy 偏移 / 偏航 / 初始温度
  （`scene.sample_episode`）+ 抓取点 xy 抖动 / 接近偏航抖动（专家内部）。
  材质、物体尺寸与质量**每文件一次**（构建期冻结，见 §9-R1/R3）。
- **box 面抓**（`shape == "box"` 时，见 §9-R24）：接近偏航
  `approach_yaw = round(yaw_jitter/(π/2))·(π/2) + yaw_of_quat(obj_quat)` ——
  抖动只用来**选抓哪一面**（最近的 90° 整数倍），再叠加物体自身 yaw，使指垫与
  所选面**精确平行**。cylinder 完全不变（`approach_yaw = yaw_jitter`，±π 全向）。
- 专家只消费 policy 率观测字典（不读引擎），输出归一化 7 维动作，
  经与策略相同的 `DeltaPoseController`（**专家与策略同接口**）。
- 全部旋钮集中在 `config.ExpertConfig`（13 个字段，见 §6 `expert:` 段），
  YAML 可覆盖；`scripts/autotune.py` 自动搜索最优值并写回 `*.tuned.yaml`（§8「自动调参」）。
  控制器尺度（`pos_scale_m`/`rot_scale_rad`）**不参与调优**：它们定义数据集动作空间。

### 5.7 `tempvla/recording/` — EpisodeRunner 与 HDF5Writer（已实现）

- `runner.EpisodeRunner`：单环境串行（`n_envs=1`）；每 episode：
  `sample_episode` → `reset_episode`（reset → 机器人初始位姿 → 物体位姿 → 温度播种）
  → **一次 warm step**（把 post-reset 位姿与播种温度推入缓存）→ `controller.reset()`
  → `expert.reset()` → 循环〔`ObsBuilder.build()` → 专家 → 控制器 → decimation 次
  `scene.step()`（内层逐 sim 步记录 L4 爆发窗口）〕→ 抬升判定 → 流式落盘。
- `events.ContactMonitor`：policy 率检测 per-link 接触 onset/offset，累积峰值力/最大压深，
  产出 L3 事件表（§4.7-L3）。
- `events.BurstRecorder`：预触发环形缓冲 + 触发后继续追加，产出 200 Hz 窗口（§4.7-L4）；
  触发锚定在 sim 子步（`fast_payload` 的接触沿），误差 < 一个 policy 步。
- `hdf5_writer.HDF5Writer`：**流式**（内存中只保留当前 episode 的缓冲）：
  - 创建时写 `attrs`、`data/attrs`（`env_args`/`problem_info` JSON）、`layouts/`（`runner.file_layouts()`）；
  - 每个 episode 完成即在其 `data/demo_N` 下建组写数据集，gzip 压缩，逐 episode flush；
  - 成功 → 主文件；失败 → `*_failed.hdf5`（默认），或按配置丢弃；
  - 编号 `demo_0..N-1` 连续无空洞。
- `runner.file_layouts()`：调 `layouts_to_hdf5_multi` 产出 §4.2 的两层 `layouts/`；
  根 `probe_grid` 显式取 `grid_for(modalities[0])`（主模态），
  `modality_grids` 写成 JSON；另写 `link_pose_encoding` 声明 §4.3 的位姿编码。
- `meta.build_env_args/build_problem_info`：文件级元数据（引擎版本、URDF、动作/观测契约、
  热参数、随机化范围、语言指令）；`env_args.observation.tactile` 里带
  `per_modality_probe_grid`（**逐模态展开**成 `{force: [4,3], elastomer: [16,16]}`，
  不能直接 `list(dict)` 写成 `['default','elastomer']` 那种键名列表——那不是契约）。
- `tools/validate.validate_file` + `scripts/validate_dataset.py`：结构/形状/dtype/NaN 校验；
  `scripts/generate.py` 结束时自动跑一遍。

### 5.8 `tempvla/tools/` — 校验与消费

| 工具 | 状态 | 功能 |
|---|---|---|
| `tempvla/tools/validate.py`（`scripts/validate_dataset.py`） | ✅ 已实现 | schema/形状/dtype 校验；必需通道（`obs/ee_pos|ee_ori|gripper_states|joint_states` + 至少一个 `tactile/` 或 `temperature/` + actions/dones）；`num_samples == T`、dones 末位为 True、NaN 扫描；`generate.py` 结束时自动跑一遍 |
| `scripts/render_video.py` | ✅ 已实现 | 把某个 demo 的**落盘 RGB** 渲成 MP4：双相机并排（nearest 放大）+ 叠加步数/夹爪状态/物体与指垫温度/接触时刻标尺。直接读文件里的像素而非重仿真，所以看到的就是数据集消费者看到的东西；默认 20 Hz（policy 率）= 实时。下方**传感面板**同步画触觉与温度场（`--no-sensing` 关）：左右指 4×3 `tactile/force` 幅值图（带数值）、指尖 4×4 温度网格、物体 4×4×4 体素（4 张 z 切片，按 per-episode 紧量程并标出 span——金属物体近乎等温，不标 span 会把色差误读成温差）。文件若含 `tactile/proximity` 会额外多画两样（缺该通道时自动退化为只有 force 一行，见 2026-10-03 的向后兼容检查）：左右指 4×3 proximity 幅值图（独立色标 `RAMP_PROX`，**不与 force 共量程**——两者标定不同，见 R23），以及右下角两行「每指 Σ\|F\| — proximity（青色=首次非零）vs force（红色=接触门打开）」时间曲线，横轴即整条 episode，当前步有白色游标。最右侧还有**凝胶压印面板**（GelSight 风格，2026-10-03 加；照相模型 2026-10-04 起抽到 `tempvla/tools/gel_render.py`，纯 numpy、Pillow 延迟 import，dataloader 可复用）：把每指的接触高度场双三次上采样 + 高斯模糊后求梯度得表面法向，再用三色 LED（`GEL_LIGHTS`，窄镜面瓣 + 弱漫反射）着色——色相=倾斜方向，是真实 GelSight 帧看起来是 RGB 的原因。**每个通道按自己的网格 reshape**（`layouts/modality_grids` → `tactile_field.modality_grid`，缺该 attrs 的老文件回退到根 `probe_grid`），高度源依次取 `tactile/height_map`（1.3 的稠密图，2026-10-08 起）、`tactile/elastomer`（凝胶标记位移）、12 探针的 `tactile/depth`，**面板标题会写明画的是哪一个**；面板的物理尺寸按**网格的 cells** 取而不是数组形状（`gel_cells`）——64×64 的稠密图是覆盖 9×7 格指垫的正方形栅格，按数组长宽画会把压印拉变形，而同一模型也让 `--markers` 的标记晶格精确落在图上（§4.2 末）：`world_frame: true` 且有 `obs/link_pose` 时用 `tactile_field.read_markers(..., frame="link")` 做**精确**逆变换（标题标 `link frame via link_pose`），只有 `world_frame: false` 的文件才走"已记录即 link 系"这条捷径，两者都缺则退化为 `|disp|`（标题如实写 `world frame, no link_pose`）。物理像素尺度由 `layouts/probe_local_pos` 的探针间距推得，所以坡度是真实坡度而不是任意对比度。真实压痕坡度只有 ~1e-3（12 µm 深 / 12 mm 垫），裸渲染看不见，所以**每集算一个统一放大系数**：把峰值帧最陡处拉到 `GEL_TARGET_SLOPE`=0.6，系数 >1.5 时写进面板标题（`slope x210`）——两指共用同一系数，指间的强弱对比是真的。`--markers` 额外叠一层**标记晶格 + 位移矢量**：亚像素位移（峰值 0.056 mm ≈ 0.49 px @112 px 面板）会被自动放大到 `MARKER_TARGET_PX`=3.5 px/标记，倍数写进标题（`vectors x7`），<1 px 的位移只画一个淡点以免糊成一片。这只是**渲染**已记录的场，不新增空间细节：12 探针源看起来是柔和圆顶，密网格 elastomer 才会是真压印 |
| `tempvla/tools/tactile_field.py` | ✅ 已实现（2026-10-04） | **marker 场的读取契约**，纯 h5py/numpy（与 `validate.py` 同层，不 import genesis）：`modality_grid`/`modality_grids`（逐模态网格，缺 `modality_grids` 回退根 `probe_grid`）、`finger_geometry`（`probe_local_pos`/`probe_local_normal`，**按网格取** —— 根节点那份属于主网格）、`read_markers(file, demo, 模态, frame="world"\|"link")`、`marker_positions`（参考 + 位移 = Tabero 的 `M = [参考帧, t_1..t_H]`）、`pad_normal_field`（投影到指垫法线的高度场）、`as_tabero_sequence`、`world_frame`。世界系文件需要 `obs/link_pose`（`quat_to_matrix` 的转置做逆），缺了就报错而不是静默给错值 |
| `tempvla/tools/gel_render.py` | ✅ 已实现（2026-10-04） | GelSight 照相模型（高度场 → 法向 → 三色 LED），从 `render_video.py` 抽出共用：`gelsight_image`、`gel_field`、`gel_slope`、`grid_m_per_px`、`lattice_nodes`、`pad_tangents`、`smooth`。**只渲染已记录的场，不新增空间细节**；密网格输入直接喂同一个函数 |
| `tempvla/tools/height_map.py` | ✅ 已实现（2026-10-08，format 1.3） | 稠密高度图的**合成与读取契约**，纯 h5py/numpy（scipy 延迟 import，dataloader 可直接用）：`pad_normal_planes`（世界系位移 → link 系 → 投影指垫法向）、`upsample_field`/`synthesize`（cell-centred 样条/双线性重采样，采样坐标先 clamp 到标记范围——最外半格严格平延、无 ringing——再把结果 clamp 回样本范围）、`height_map_meta`/`read_height_map`、`synth_from_markers`（任意带 marker 场的老文件现场合成）、`height_map_or_synth`。语义与 extents 模型见 §4.2 末、「是重采样不是渲染」的理由见 §9-R25 |
| `replay.py` | ⏳ v2 | 读全量 `states` + `actions` 重仿真，对比 obs 在容差内一致 → 抓录制 bug 的最强校验；前提是 L1 之外的 `states` 落盘（v1 未写） |
| `tactile_render.py` | ⏳ v2（部分被 `render_video.py` 覆盖） | 由 `obs/tactile/*` + `layouts/` 重算探针世界坐标，渲染触觉场动画。**探针网格图已由 `render_video.py` 的传感面板提供**（指坐标系下的 4×3 幅值图）；本项剩下的独有价值是**世界坐标系**下的三维布点（把 `probe_local_pos/normal` 经正运动学变换后画在场景里） |
| `analyze_contact.py` | ⏳ v2 | 由温度网格差分离线推导接触热流（§4.7-L2），输出事件级汇总 |
| `stats.py` | ⏳ v2 | 数据集统计报表 / 直方图（成功率、温度分布、事件数分布） |

---

## 6. 配置体系

```
configs/
├── pick_hot_cylinder.yaml      # v1 主任务（4 材质；gate0/1/2 与 generate.py 默认加载它）
├── lift_object.yaml            # 最小任务（同一专家，去掉 hot/pick 语义）
├── lift_object_pair.yaml       # 1.2 多物体：一桌圆柱 + 立方体，按 episode 二选一（见下）
└── lift_gel_preview.yaml       # marker 场往返一致性测试的对照件（world_frame: false）
```

`configs/*.yaml` 与 `tempvla/config.py` 的 dataclass **逐字段一一对应**
（`load_task_config` 按 YAML 键覆盖 dataclass 默认值）。实际字段：

```yaml
task_id: pick_hot_cylinder
language_template: "pick up the hot {material} cylinder"
materials: [aluminum, copper, abs, steel]        # 每个材质输出一个文件
material_properties:                             # -> gs.sensors.TemperatureProperties
  copper: {conductivity: 385.0, density: 8960.0, specific_heat: 385.0, emissivity: 0.0}  # 见 §9-R14

robot:                                           # Franka Panda + 平行夹爪
  urdf: urdf/panda_bullet/panda.urdf
  merge_fixed_links: false
  hand_link_name: panda_hand
  finger_link_names: [panda_leftfinger, panda_rightfinger]
  tcp_offset: [0.0, 0.0, 0.105]                  # Gate 1 标定
  open_finger_pos: 0.04
  closed_finger_pos: 0.0
  init_qpos: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
controller:
  pos_scale_m: 0.05
  rot_scale_rad: 0.5
  control_hz: 20.0
  sim_dt: 0.005                                  # 200 Hz sim × decimation 10
  decimation: 10
  ik_damping: 0.1
  ik_max_iters: 25
  ik_max_samples: 4
expert:                                          # 自动调参对象（scripts/autotune.py）
  approach_clearance_m: 0.08
  grasp_z_offset_m: 0.0
  pos_tol_m: 0.01
  rot_tol_rad: 0.2
  descend_tol_m: 0.015
  contact_depth_m: 1.0e-5
  squeeze_steps: 10
  close_timeout_steps: 40
  lift_height_m: 0.10
  lift_tol_m: 0.012
  grasp_xy_jitter_m: 0.004
  approach_yaw_jitter_rad: 0.15
  close_extra: 1.0
tactile:
  modalities: [force, depth, elastomer]          # elastomer = marker 场（Q5 已决）
  probe_grid: {default: [4, 3], elastomer: [16, 16]}   # 每指 12 / 256 探针，逐模态
  height_map: {enabled: false, size: [64, 64], interpolate: cubic}  # 1.3：marker 场的稠密重采样（米），§4.2 末；pair config 开、这里关
  probe_offset_m: 0.0005
  pad_tol_m: 0.0015                              # mesh 指垫平板厚度（§5.2）
  probe_radius_m: 0.004                          # 全局共用；16×16 的 0.56 mm 间距下严重重叠（见 §9-R8）
  as_grid: true
  elastomer_params: {lambda_d: 50000.0, lambda_s: 20000.0}   # 引擎默认 700 会把响应摊到整块 pad
  contact_links: true
  temporal_reduction: median
  world_frame: true                              # obs/link_pose 使之逐步精确可逆
temperature:
  finger_grid: [4, 4, 1]                         # 每指
  object_grid: [4, 4, 4]                         # 物体
  ambient_temp_c: 21.0
  convection_coefficient: 0.0                    # 见 §9-R14（引擎体素散热缺面积因子）
  simulate_all_link_temperatures: true
  sensor_time_constant: 0.0                      # 0 = 无滞回（录 GT）
  contact_depth_weight: 0.0                      # 见 §9-R16（热核内的 π·穿透 回退只会二次计入穿透；面积本身已由接触缓冲给出）
  record_ground_truth: true
  finger_properties: {conductivity: 0.02, density: 1100.0, specific_heat: 1400.0, emissivity: 0.0}  # k=接触耦合增益，见 §9-R16
  environment_properties: {conductivity: 0.15, density: 700.0, specific_heat: 1600.0, emissivity: 0.0}
object:
  shape: cylinder
  radius_m: [0.015, 0.025]                       # 构建期冻结（每文件一次）
  height_m: [0.05, 0.08]
  mass_kg: [0.05, 0.15]
  temp_c: [60.0, 200.0]                          # 逐 episode 采样
  xy_jitter_m: 0.04
  yaw_jitter_rad: 3.14159
table:
  pos: [0.4, 0.0, 0.0]
  size: [0.6, 0.8, 0.04]
  height_m: 0.4
  # 仅渲染用的表面色（RGB 0-1）；null = 引擎默认灰白（data/、data_cam/ 就是这个）
  color: [0.55, 0.38, 0.22]
recording:
  output_dir: data
  keep_failed: true
  compression: gzip
  compression_opts: 4
  contact_burst: true
  burst_pre_s: 0.05
  burst_post_s: 0.25
  burst_max_windows: 10
  burst_modalities: null                         # null = 全部模态（含 16×16 marker 场，1.1–2.2 MiB/集）
cameras: {enabled: true, resolution: [256, 256]}  # 置 true + renderer: raster 即启用双 256² 相机（§8）
max_episode_steps: 400                           # 20 s @ 20 Hz
lift_success_m: 0.05
stable_steps: 10
seed: 0                                          # episode i 用 seed+i
```

- 输出目录默认取 `recording.output_dir`，`scripts/generate.py --out` 可覆盖；
  renderer / 查看器等运行期开关走命令行（`--viewer`）。
- 传感器参数文件（`configs/sensors/*.yaml`）不是必需：探针布局由 URDF 程序化推导，
  触觉参数直接放在任务 YAML 的 `tactile:` 段。

**多物体任务（1.2，`configs/lift_object_pair.yaml`）**：`objects:` 列表取代单
`object:` 块（两者同时存在时 `objects` 生效），语言模板多一个 `{shape}` 占位符：

```yaml
language_template: "lift the {material} {shape}"
objects:
  - shape: cylinder
    place_xy: [0.0, -0.12]         # 桌面槽位（相对桌面中心）；采样 jitter 叠加在它上面
    radius_m: [0.015, 0.025]
    height_m: [0.05, 0.08]
    mass_kg:  [0.05, 0.15]
    temp_c:   [60.0, 200.0]
    xy_jitter_m: 0.04
    yaw_jitter_rad: 3.14159        # 圆柱整圈随机
  - shape: box
    place_xy: [0.0, 0.12]
    radius_m: [0.012, 0.020]       # box 时是半宽：body = 2r × 2r × h
    height_m: [0.048, 0.065]
    mass_kg:  [0.04, 0.12]
    temp_c:   [60.0, 200.0]
    xy_jitter_m: 0.04
    yaw_jitter_rad: 0.30           # box 走面抓：接近角 = 抖动量化选面 + 物体自身 yaw（§5.6）
min_object_gap_m: 0.02             # 采样后两物体表面最小间距守卫（触发则重抽 jitter ≤5 次）
```

- **一文件一材质**不变（热物性类级共享，§4.1）；物体尺寸/质量构建期冻结，
  文件内只随机化位姿/偏航/温度；`target_index` 逐 episode 均匀抽取（N=1 时不抽，
  随机流与旧版逐位相同），记在 demo attrs。
- **触觉段（2026-10-08 起）对齐 UniVTAC 形态**：`modalities: [elastomer]`、
  `probe_grid: {elastomer: [9, 7]}`（63 标记/指，不采刚性探针）、
  `height_map: {enabled: true, size: [64, 64], interpolate: cubic}`；
  `contact_links: true` 保留（接触判定靠它）。理由、实测与代价见 §9-R25。
- 场景/记录的新键见 §4.2；`lift_object_pair.yaml` 的 agentview 站位覆盖
  y ∈ ±0.16 m 的一对槽位。

---

## 7. 目录结构（完整规划）

```
TemperatureVLA/
├── README.md
├── pyproject.toml
├── .gitignore                       # 生成物不入库：data*/、*.hdf5、*.mp4（docs/figures 除外）等
├── docs/
│   └── architecture.md              # 本文档
├── configs/
│   ├── pick_hot_cylinder.yaml       # §6（← configs/task_pick_hot_cylinder.yaml 为早期草稿，已废弃）
│   ├── lift_object.yaml             # 最小任务：抓取 + 抬起（同一专家，未裁剪传感栈）
│   ├── lift_object_pair.yaml        # 1.2 多物体：圆柱 + 立方体，按 episode 选目标（§6）
│   └── lift_gel_preview.yaml        # marker 场往返一致性测试对照件（world_frame: false）
├── tempvla/
│   ├── __init__.py
│   ├── config.py                    # TaskConfig dataclass + load_task_config
│   ├── rotations.py                 # 自实现 wxyz 四元数工具（Gate 1 交叉验证）
│   ├── actions.py                   # DeltaPoseController / tcp_pose / normalized_action_to
│   ├── expert.py                    # PickExpert（APPROACH→DESCEND→CLOSE→LIFT→HOLD）
│   ├── scene.py                     # build_scene / SceneBundle / sample_episode / reset_episode
│   ├── obs.py                       # ObsBuilder（L1 policy 率 + fast_payload sim 率）
│   ├── compat/sensor_compat.py      # 上游触觉 reset 修复（逐字拷贝，monkey-patch）
│   ├── sensors/
│   │   ├── finger_layout.py         # URDF 解析 → 两指探针布局 + layouts_to_hdf5(_multi)
│   │   ├── tactile.py               # 触觉/接触传感器装配与读取
│   │   └── temperature.py           # 温度网格装配 / seed_temperature / 读取
│   ├── recording/
│   │   ├── events.py                # ContactMonitor（L3）+ BurstRecorder（L4）
│   │   ├── hdf5_writer.py           # 流式 HDF5（成功/失败文件分离）
│   │   ├── meta.py                  # env_args / problem_info / 语言指令
│   │   └── runner.py                # EpisodeRunner（一次 episode 全链路）
│   └── tools/
│       ├── validate.py              # HDF5 结构与通道校验
│       ├── tactile_field.py         # marker 场读取（参考位置 + 位移 → 位置/高度场）
│       ├── height_map.py            # 稠密高度图：重采样合成 + 读取（1.3，scipy 延迟 import）
│       └── gel_render.py            # GelSight 照相模型（高度场 → RGB，Pillow 延迟 import）
├── scripts/
│   ├── gate_common.py               # Gate/计时/内存/配置加载 公共件
│   ├── gate0_engine.py              # Gate 0：引擎 + 资产 + 温度传感器契约
│   ├── gate1_sensors.py             # Gate 1：机器人 + 探针布局 + 传感器读数 + 四元数交叉验证 + 压入测试
│   ├── gate2_scene.py               # Gate 2：完整场景 + 专家 episode + 落盘 + 校验
│   ├── gate3_render.py              # Gate 3：双相机装配 + 渲染质量/帧率/显存（需软件 GL）
│   ├── gate4_camera_obs.py          # Gate 4：RGB 落盘 + "渲染不扰动仿真"（需软件 GL）
│   ├── probe_camera_pose.py         # 相机位姿候选接触表（Gate 3 的辅助脚本）
│   ├── generate.py                  # 主生成入口（每材质一个文件）
│   ├── autotune.py                  # 专家参数自动调优（随机搜索 + 局部细化）
│   ├── render_video.py              # HDF5 → MP4（双相机 + 温度/接触叠加），看一眼数据长什么样
│   └── validate_dataset.py          # CLI 校验
├── genesis_ref/（或 pip wheel）      # 引擎：genesis-world==1.4.1
├── tactile-genesis/                 # 上游参考树（只读，资产来源）
└── data/                            # 生成输出
```

---

## 8. 验证关卡（Gate 0–2 + 生成）

按顺序执行；**任一 Gate 失败则暂停并修订设计**。命令均由用户在 `tempvla` conda 环境执行
（`source activate tempvla && cd ~/TemperatureVLA`）。三个 Gate 脚本共用
`--config`（缺省 `configs/pick_hot_cylinder.yaml`）。

### Gate 0 — 引擎 + 资产 + 温度传感器契约（约 10 min）

```bash
python scripts/gate0_engine.py
```

内容：Python/numpy/h5py/torch/CUDA 版本与内存打印；`import genesis` 并断言版本；
`gs.utils.get_assets_dir()` 下 urdf/xml/mesh 资产计数（**缺失时打印
sparse-checkout 恢复命令**，见 R12）；`gs.init(backend=gpu)` 失败自动回退 CPU；
最小 plane+box 场景构建 + 50 步自由落体检查；对
`genesis.options.sensors` 做内省，核对 `ContactProbe`/`ContactDepthProbe`/
`KinematicTaxel`/`TemperatureGrid` 的字段契约（**禁止字段** `probe_local_normal`
必须不存在）与 `TemperatureProperties` 字段；核对运行时实体类（`*Sensor`）的
`read`/`read_ground_truth`/`link_temperatures` 与 GT 注入所依赖的内部量
（`_ground_truth_intermediate_cache`/`_cache_slices_by_type`/`link_temps`…）。

通过标准：全部 `[ ok ]`，无 `[FAIL]`。任何 `[warn]` 记录在案但不阻塞。

### Gate 1 — 机器人 + 探针布局 + 传感器读数（约 30 min）

```bash
python scripts/gate1_sensors.py
```

内容：构建 Franka + 夹爪场景 → 校验 URDF 解析出的关节/连杆名；由 URDF 推导两指
内侧探针布局（校验探针在抓取面上、两指相对、法线相向）；装配触觉/接触/温度传感器
并读一轮（形状/量级/`link_temperatures`）；**四元数交叉验证**（`rotations.py` vs
`genesis.utils.geom`，组合次序、rotvec 往返）；TCP 标定（由 pad 面中心反推
`tcp_offset`，与配置差 > 5 mm 报警）；**压入测试**（重置 + 注入 120 °C、专家位姿
驱动 TCP 压上物体、40 步保持，打印 depth/force/接触/物体温度/指温轨迹）。

通过标准：布局与探针几何校验通过；触觉/温度读数形状正确；压入后接触为真、
depth/force 非零、指温上升 > 0.01 °C；TCP 标定与配置一致（Gate 1 的核心产出）。

### Gate 2 — 场景 + 专家 + 记录 + 落盘（约 1 h）

```bash
python scripts/gate2_scene.py --episodes 2
```

内容：完整场景（桌 + 热圆柱 + 机器人 + 全部传感器）→ `EpisodeRunner` +
`HDF5Writer` 跑 N 个 episode，逐 episode 打印步数/成功/事件数/爆发窗口数/IK 误差、
动作范围、每指接触步数、触觉峰值、指温升温、物体温度始末、burst 窗口 0 的形状；
结束后对落盘文件跑 `validate_file`。

通过标准：episode 能跑完（成功与否都行）；接触事件非空；指温有升温；
HDF5 校验通过。**Gate 2 同时定级接触热流可得性**（结论：不可得 → 离线推导，§4.7-L2）。

### Gate 3 — 相机渲染（**决定性**）：✅ 通过（2026-10-02，`renderer=raster`）

```bash
# 必须带软件 GL：WSLg 的 D3D12(Intel) 路径会把画面压成最后 2 行像素
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe python scripts/gate3_render.py --renderer raster
# 只想看相机位姿候选（一次性渲染多组手系位姿）：
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe python scripts/probe_camera_pose.py --steps 11
```

**结论：双相机可用**（`agentview` + `eye_in_hand`），实测数据：

| 项 | 128² | **256²（现行）** | 512² |
|---|---|---|---|
| 单次渲染（两相机） | 中位 43–50 ms | 同上量级 | 同上量级 |
| 整场景 ms/step（含仿真，2 ep 均值） | 194.0 | **211.6** | 220.8 |
| gzip 帧尺寸（两相机/步） | 13.0 KiB | **33.0 KiB** | 66.0 KiB* |
| 每 episode（34 步） | 0.44 MiB | **1.10 MiB** | 2.2 MiB* |
| 200 ep 投影 | 90 MiB | **219 MiB** | 450 MiB* |
| agentview 中物体像素 | 188 px（1.15%，~14×14） | **764 px（1.17%，~28×28）** | ~3056 px（~55×55）* |
| VRAM | 峰值分配 **0.01 GiB**（纯 GPU 仿真 6.57 GiB 可用），Rasterizer 走 CPU | | |
| eye-in-hand 可见性 | 相机随手移动 **114 mm**，画面随臂运动变化 **21%** 像素；物体占画面 ~21% | | |

\* 512² 的 KiB/帧 与物体像素为按分辨率平方外推，未逐项实测。

**分辨率裁决（2026-10-03：128² → 256²）**——`llvmpipe` 光栅化的成本几乎与像素数无关
（每帧固定开销占主导）：

| res | ms/step | Δ vs 无相机 | 相对 128² 的额外时间 | 磁盘/200ep |
|---|---|---|---|---|
| 无相机 | 148.6 | — | — | — |
| 128² | 194.0 | +45.4 | — | 90 MiB |
| **256²** | **211.6** | **+63.0** | **+17.6 ms/step（+9%）** | **219 MiB** |
| 512² | 220.8 | +72.2 | +26.8 ms/step（+14%） | ~450 MiB |

即"4 倍像素"只买到 +9% 时间，代价全在磁盘（2.4×）。选择 256² 的理由：128² 下物体
仅 188 px，夹爪-物体接触区在时序上几乎无法分辨（§8 末尾"位姿决定"第 1 条），而
256² 的 764 px 已能看清手指轮廓与圆柱面；再往上到 512² 的收益递减而磁盘翻倍。
对照图：`res_compare_resolutions.png`（同一 rollout、同一帧、同一视场，
仅采样像素数不同）。

| 项 | 实测 |
|---|---|
| 全量投影（34 policy steps/ep × 200 ep，256²） | **≈12 min**（对比 v1 全程 944 s） |

**渲染器裁决（与 R3 相关）**：
- `BatchRenderer` **在本机不可用**：madrona 后端在 `scene.build()` 时构造，缺 Vulkan loader
  直接 `abort()`（core dump，`vk/backend.cpp:123`）——**连 `use_rasterizer=True` 也走这条路**
  （实测崩溃），故配置里不提供该开关。装上 `sudo apt install libvulkan1`（连带
  `mesa-vulkan-drivers`）后可再试，但本机无 `/dev/dri`、无 NVIDIA ICD，即使装上也只能落到
  软件 Vulkan（lavapipe），收益未验证 → **v1/v2 默认走 `raster`**。
- `Rasterizer` 在 WSL2 上**必须** `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`：默认
  EGL→WSLg→`D3D12 (Intel(R) Graphics)` 会产出退化图像（内容塌缩到最下面 2 行，深度恒为 0.1）。
  软件路径产出正确图像（`glGetString(GL_RENDERER)` = `llvmpipe (LLVM 15.0.7, 256 bits)`）。

**两个位姿决定（都是看图定的，不是算出来的）**：
1. **agentview 拉近**到 `pos=(0.78,-0.38,0.70)` → 256² 下物体 **~28×28 px**（128² 时
   ~14×14）。原先的 (1.1,0,1.1) 在 128² 下只有 ~5 px，物体几乎不可分辨——这也是后来
   从 128² 提到 256² 的动机（见上表）。
2. **eye-in-hand 装在手的侧面** `pos=(0.10,0,0.02)`、`lookat=(0,0,0.10)`、`up=(0,0,-1)`：
   `panda_hand` 自身网格在手系占 z∈[-0.026,0.066]，**任何装在法兰后方的相机都只能看见手本身**；
   手沿 x 很薄（|x|<0.032），所以相机挂在 +x 侧、与法兰齐平、看向两指之间的 TCP。
   `up=(0,0,-1)` 是为了让图像上方 = 世界上方（夹爪朝下时），否则画面上下颠倒 180°。

本 Gate 只验证引擎能出正确图像与代价，并按 LIBERO 惯例冻结通道名
`agentview_rgb` / `eye_in_hand_rgb`；把它们真正写进 `obs/` 由 **Gate 4**（下）完成。

### Gate 4 — RGB 写进 `obs/`（v2 增量）：✅ 通过（2026-10-03）

```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe python scripts/gate4_camera_obs.py --episodes 2
```

`cameras.enabled=true` 时 `ObsBuilder.build()` 追加 `agentview_rgb` / `eye_in_hand_rgb`
（`(H,W,3) uint8`，**平铺在 `obs/` 下**，LIBERO 布局），writer 按 `(T,H,W,3)` 落盘。
`cameras.enabled=false` 时 `cameras()` 返回 `{}`，v1 通道表逐字段不变。

**核心断言：渲染只观察、不扰动仿真。** 同一场景同一次 build 内，`bundle.cameras`
置空即可关掉渲染，其余全同——这是唯一能把"相机效应"与"引擎自身噪声"分开的对照。
实测（2 episodes × 34 步）：

| 对比 | 轨迹通道（10 个：proprio / contact / temperature / actions） | `tactile/*` |
|---|---|---|
| 同一 build：armed vs disarmed | **逐位相同** | 见 R19，接触起始 3–4 步的噪声 |
| 跨 build：armed vs `cameras.enabled=false` | **逐位相同** | 同上 |

`tactile/force` 的差异**与相机无关**（R19）：同场景、同 seed、**完全关渲染**连跑两次，
`tactile/force` 差 4.92 N，而 `ee_pos`/`actions` 逐位相同。故 Gate 4 不对 `tactile/*`
做强断言，只断言"差异局限于接触起始步、末步相同"这一有界暂态性质，并同时打印
**无渲染对照**的噪声底。实测中 armed-vs-disarmed 的差值往往与无渲染对照**完全相同**
（ep0：1.39 vs 1.39；ep1：0.00534 vs 0.00534）——即同一个噪声抽样。

落盘校验（`validate_file`）：`env_args.cameras.channels` 声明通道表，校验器据 `enabled`
要求两个通道存在，并检查 dtype/形状/`T == num_samples`、**帧非全黑**（非黑像素 < 5% 即失败，
用于抓 WSLg D3D12 那种"无报错的退化图像"）、**首末帧不相同**（抓渲染缓冲被复用的冻结帧）。
本 Gate 另外把帧从**文件里**读出来存成 PNG（不是从内存里存），确认往返无损。

代价（同一 build 内 armed vs disarmed）：**124 ms/步**（两相机，llvmpipe），
即 +91%；落盘 **0.49 MiB/episode**（两相机 × 34 步，gzip）。

**全量 200 episodes 实测（2026-10-03，`data_cam/`）**：**200/200 成功**、4 个文件各 ~24.2 MiB
（合计 **97 MiB**）、`validate_dataset.py` 全部 **0 issues**；总墙钟 **≈53 min**。
干净集稳定在 **8.3 s/episode**（+124 ms/步 × 34 步 = 8.8 s，代价模型成立），
另有约 10% 的 episode 发生**渲染尖峰**（单集 25–350 s）吃掉约一半墙钟——已定位，见 §9-R20。

**软件 GL 是硬性前提**：`scripts/generate.py` 在 `renderer=raster` 且
`LIBGL_ALWAYS_SOFTWARE != 1` 时**拒绝启动**（`--unsafe-gl` 可越过）——那条路径会静默产出
退化图像并把 100 MB 数据写坏，宁可不开工。

### 自动调参 `scripts/autotune.py`

专家参数（§5.6 的 `ExpertConfig`）不必手调：autotune 批量评估候选参数，
按 **成功率（主）+ 接触事件数 + 指温升温 − IK 误差惩罚** 打分
（`score = success_rate + 0.05·min(events,4)/4 + 0.05·min(heating,3°C)/3°C − 0.05·min(ik_mm,20)/20`），
先随机搜索（40%）再围绕最优做局部细化，最优参数写回 `*.tuned.yaml`。

```bash
# 冒烟：3 试次 × 2 episode，验证调参链路
python scripts/autotune.py --trials 3 --episodes 2 --out /tmp/tune_smoke.yaml

# 正式：24 试次 × 5 episode（约 30–60 min，视 steps/s）
python scripts/autotune.py --trials 24 --episodes 5

# 中断后继续（跳过已完成的随机试次）；--budget-min 限制总时长
python scripts/autotune.py --trials 24 --episodes 5 --resume --budget-min 45
```

- 评估 episode 与 `generate.py` **同种子**（`cfg.seed + i`，场景种子
  `cfg.seed + 1000·(材质序号+1)`）→ 调参测得的成功率即正式生成时的预期成功率。
- 调参不写 HDF5（`writer=None`），每 episode 结束即释放，速度远快于生成。
- 每次试次追加写入 `<out>.trials.jsonl`（参数/分数/聚合指标），`--resume` 据此续跑。
- 输出 YAML 的 `expert:` 段为最优参数，另附 `tuning:` 溯源块（材质、种子、
  试次数、top-5、耗时）；`tuning:` 是未知顶层键，加载器会忽略，不影响使用。
- 先跑冒烟确认链路，再跑正式；正式调参在 **Gate 2 通过之后、全量生成之前** 执行。

### 生成（冒烟 → 全量）

```bash
# 冒烟：每材质 2 条
python scripts/generate.py --episodes 2 --out data_smoke

# 全量：4 材质 × 50 条（推荐用调参后的配置；未调参则去掉 --config）
python scripts/generate.py --config configs/pick_hot_cylinder.tuned.yaml --episodes 50 --out data

# 单材质 + 查看器
python scripts/generate.py --material aluminum --episodes 5 --viewer
```

> ⚠️ **一个数据集 = 一次调用。** `scene_seed` 用的是材质在**本次列表中的下标**，
> 物体尺寸/质量由该 seed 在构建期抽出，所以 `--material copper` 单跑会让 copper
> 拿到全量运行里 aluminum 的 seed 与物体（R21）。`generate.py` 现在会在子集运行时
> 打印每个材质的 scene seed 并与全量运行对照。

通过标准：成功率高（目标 > 80%，`autotune.py` 调参后）；校验通过；单 episode 时延
与内存符合预算（§2.5）。

**v1 实测（2026-10-02，GPU 后端）**：**未调参的** base config 冒烟即 **32/32 成功
（8/材质 × 4 材质，0 失败）**，已超过 80% 目标 → v1 跳过 autotune、直接用 base config
全量生成（省 30–60 min）。**全量结果：200/200 成功、0 失败**，命令为
`python scripts/generate.py --config configs/pick_hot_cylinder.yaml --episodes 50 --out data`，
`WALL 944 s`（含 4 次场景构建；首建 10.3 s，其余 4–5.6 s）。
实测单 episode 4.3–4.6 s（≈7.5 control steps/s），
落盘 0.063–0.065 MiB/episode（200 条合计 **12.9 MiB**），生成进程 RSS 峰值 **2.24 GiB**（预算 8 GiB 内）。
**autotune 的两个坑（留档，v2 若要用需先解决）**：(1) 随机相位的候选参数每个都要
roll 到 400 步上限（~4 min/试次），`--budget-min 30` 只够 ~6 试次，永远进不到局部细化；
(2) 打分里的 `heating` 项上限 3 °C，而实测指尖峰值每 episode 升 11–125 °C，该项对所有候选
都是满分、无区分度。建议 v2：把随机相位改为围绕 base config 的扰动 + 只做局部细化，
并把 heating 的归一化改成实测分位数。

**M5 重生成（marker 场，2026-10-04 起；未执行）**：现有 `data/`（12.9 MiB）、
`data_cam/`（97 MiB）、`data_lift/` 全是 **format 1.0、不含 marker 场**。重生成必须先
写进**暂存目录**（`data_gel/`、`data_cam_gel/`），跑 `validate_dataset.py` + 每材质
渲染一集抽查，**确认后才替换**；仓库自 2026-10-08 起由 git 管理，但 `data*/` 不入库
（数据替换仍不可回滚），旧语料**改名保留而不是删除**。前置条件是本节开头那七项验证
（gate1/gate2/generate 冒烟/往返一致性/`render_video.py --markers`/gate4/向后兼容）全过。

---

## 9. 风险与缓解

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 温度传感器类级共享参数（ambient/convection "last one wins"） | 逐 episode 环境温度随机化受限 | 全局热参数按场景固定；物体初始温度走 **GT 缓冲注入**（`temperature.seed_temperature` 直写 manager ground-truth 缓存 + 镜像 `link_temps`，顺序 reset → seed → step，已实现于 `sensors/temperature.py`，Gate 1 压入测试验证）；材质按**文件**分组重建场景（§4.7） |
| R2 | 体素体积仅对轴对齐连杆精确 | 旋转物体的温度网格近似 | 物体温度以集中温度（`object`）为主通道；文档标注网格精度边界 |
| R3 | gs-madrona 在本机不可用：`backend.cpp:553 findPhysicalDevice` 报 **"Cannot find matching vulkan UUID for GPU"** —— WSL2 下没有 NVIDIA Vulkan ICD（`/usr/share/vulkan/icd.d/` 只有 intel/radeon/lvp/virtio；`/usr/lib/wsl/lib/` 无 `libnvidia-vulkan-producer.so`）。装 `libvulkan1` 后仍缺 GPU ICD，只剩 lavapipe。与 sm_120 无关 | 无法用 BatchRenderer 的 GPU 批量渲染 | **已实测并绕过**：`renderer=raster` + llvmpipe 出正确双相机图，43–50 ms/次、全量仅 +12 min（256²，Gate 3）。**且不需要 madrona**：raster 的 512² 也只比 128² 贵 +14% 时间（Gate 3 分辨率表）——批量渲染的收益是省的编译/驱动折腾，不是速度 |
| R4 | 8GB VRAM / 7.4GB RAM | 批量受限、OOM | v1 `n_envs=1` 串行 + HDF5 流式写盘（单 episode 常驻内存）；单 episode < 0.5 MB；生成脚本逐 episode 释放 |
| R5 | Quadrants 在 WSL 静默回退 CPU | 性能塌方且难察觉 | Gate 0 打印 backend/device 并断言 GPU（`gs.init` gpu→cpu 回退时显式警告）；生成脚本打印 steps/s 与 ETA 供察觉 |
| R6 | torch / gs-madrona CUDA 版本错配 → import 段错误 | 环境不可用 | 独立 `tempvla` conda 环境锁 `genesis-world==1.4.1`（py3-none-any wheel 自带依赖声明）；Gate 0 先验证 import + CUDA |
| R7 | 脚本专家质量 ≠ 人类遥操作 | 数据分布偏差 | 数据卡如实标注；域随机化提升多样性；失败 episode 单独文件保留供分析；`scripts/autotune.py` 自动调参（§8）目标成功率 > 80% 后固化参数 |
| R8 | 触觉是 taxel 数组、非光学图像 | 与真实 GelSight 数据域差距 | **工具已实现（2026-10-03）**：`render_video.py` 的 GelSight 风格压印面板（§8）；高度源优先 `elastomer`，退回 12 探针 `depth`，标题写明来源。**`elastomer` 模态实测（2026-10-03，16×16=512 标记，`lambda_d=5e4/lambda_s=2e4`）**：① 成本 depth-only 84.4 → +elastomer 127.7 ms/step（**+51%**），几乎与探针数无关；② 记录为 `(T, 2P, 3)` float32 = 6144 B/step 原始、gzip-4 后 ~3.1 KiB/step（44 步 episode 里 136 KiB / 2.0 MiB ≈ 7%，相机才是大头）；③ **压痕是真的**：峰值法向位移 17.5 µm（L）/ 11.3 µm（R），平滑单峰；但真实坡度 ~1e-3 m/m（12 µm / 12 mm 垫），裸看是平的，渲染必须显式夸大（面板标题标 `slope x210`）；④ **`compressibility<1` 或 `elastomer_thickness>0` 会切到 FFT dilation 路径并崩**（`UNSUPPORTED DTYPE: complex`），只能留默认；⑤ 指垫法线在 link 系是 **y**（不是 z），高度场必须投影到 `layouts/probe_local_normal` 才不是鞍形。**⑤ 已解决（2026-10-04）**：`obs/link_pose`（每步每指 pos+quat）+ `tactile_field.read_markers(frame="link")` 让 `world_frame: true` 的场**精确可逆**，不必再退到模长。**转正为正式 schema（2026-10-04，Q5）**，两处结构缺陷一并修掉：**① 网格不再共享** —— `tactile.probe_grid` 支持逐模态覆盖，`force`/`depth` 保持 4×3（每指 12）逐字节不变，只有 `elastomer` 是 16×16；**② 帧不再丢失** —— 见上。新增代价：`link_pose` 56 B/step（可忽略）；burst 里每窗口 +110–218 KiB（§4.7-L4，`recording.burst_modalities` 可收口）。**往返一致性实测（2026-10-04，同一 seed 跑 shipped 与 preview 两份配置）**：`disp_world @ R(q)ᵀ` 与 link 系录制值最大差 **3.6e-12 m**（42 步全过 1e-6 阈值），投影到指垫法向的高度场差 **9.5e-13 m**；`force` 同法回转后差 **2.4e-7 N**（旋转后的 float32 量化），而 `depth`、`contact/*`、`link_pose`、`ee_pos`、`joint_states`、`actions`、温度网格**逐位相同** —— 即帧转换是精确逆运算，且 `world_frame` 不扰动任何别的东西。**另一条实测提示**：16×16 网格会触发引擎告警 `ElastomerTaxel grid is not strictly regular ... FFT dilation will use averaged spacing and normal as a best-fit approximation`（指垫是曲面，16×16 下间距/法向不再严格均匀）——只是告警，读数照常。**已知未解**：`probe_radius_m`=4 mm 仍是全局共用，16×16 的 0.56 mm 间距下探针球严重重叠——preview 里场是光滑的，若下游读出过平滑再把该参数也改成逐模态 |
| R9 | 上游 Genesis 升级破坏补丁 | 触觉 reset bug 复现 | `sensor_compat.py` 隔离在 `compat/`，升版先跑 Gate 1 |
| R10 | 接触热流（contact heat）未必经公开 API 暴露 | "具体温度传导"的显式记录依赖它 | **已定级：不暴露**（Gate 2 结论）→ 不设通道；由 L1 温度网格离线差分推导（Q̇ ≈ ρcV·ΔT/Δt，`tools/analyze_contact.py`，v2）|
| R11 | 爆发窗口存储膨胀 | 单 episode 增大 | 默认 [−0.05 s, +0.25 s]、上限 10 窗口、全部触觉模态 + 集中温度；无数值通道时 ≈0.25 MB/episode，**开启 16×16 marker 场后 1.1–2.2 MiB/episode（R8）**——这是当前 burst 的主要成本。兜底：`recording.burst_modalities` 逐模态收口、`contact_burst: false` 整体关闭 |
| R12 | 引擎资产随源码树剥离（vendored Genesis 只剩 textures） | Gate 0 场景构建失败 | 用 pip wheel `genesis-world==1.4.1`（自带 `urdf/`）；Gate 0 检测缺失时打印 sparse-checkout 恢复命令（`git sparse-checkout add genesis/assets/urdf`） |
| R13 | 调参默认只在单一材质/场景种子（config 第一个材质）上评估 → 参数可能过拟合该材质 | 其他材质成功率偏低 | 同种子评估保证与生成一致；需要时对每个材质各跑一次 `--material`，或调参后在其余材质上抽验 5–10 条；调参只动专家旋钮，不影响数据 schema |
| R14 | 引擎体素级辐射/对流项**缺面积因子**（`temperature.py: _radiation_convection_delta_T` 用 per-m² 通量除以 `rho*cp*voxel_volume`，未乘面元面积 dx²）→ 散热快 ~1/dx² ≈ 1e4 倍 | 热物体秒级掉到环境温度（实测 ε=0.1 的铝柱 τ≈3 s），温度通道在 episode 内失去信息 | 三类材质（物件 `material_properties` / 指尖 / 环境）全部 `emissivity: 0.0` + 全场景 `convection_coefficient: 0.0`（episode 内绝热；物理上金属件空气中 τ≈1e3 s ≫ 20 s episode，故 0 反而更接近真实）；真实 ε 值留档于 config 注释与数据卡（铝 0.10 / 铜 0.30 / abs 0.90 / 钢 0.30）。唯一热通道 = 接触热传导 + 内部扩散。指尖 `conductivity: 0.02`（见 R16：k 同时是接触耦合增益，按数值稳定性取值，不按材料表值）。逐 episode 仍由 `reset`（全部回 ambient）→ `seed_temperature`（物体采样温度）重置；唯一热通道 = 接触热传导 + 内部扩散 |
| R15 | `panda_hand` 碰撞几何向下延伸到指根平面（手架原点下方 5.84 cm = TCP 上方 4.66 cm） | 俯视下压时**手掌先于手指**撞上物体顶面：Gate 1 trace 显示 TCP 低于物体中心 19 mm 处 244 N 冲击，臂被弹开→setpoint 跑飞 | 下压目标须满足 `grasp_z_offset_m ≥ h/2 + 余量 − 0.0466`；本任务 h ≤ 0.08 m → dz ≥ 0 时手掌仍留 0.7–2.2 cm（实测 dz=0、h=58 mm 时 F=0）。`expert.grasp_z_offset_m` 即该旋钮，autotune 可调；下压收敛判据须留 ≥5 mm 容差，避免"贴着目标不收敛" |
| R16 | 接触热更新是**显式格式且无稳定性约束**（`temperature.py: _kernel_contact_heat`）：单个体素 `dT = dt·k_eff·(T_other−T_cell)·A²/(V²·ρcp)`，放大因子 `c` 必须 ≪1。`A` = 接触面积缓冲（组内 count<3 时该缓冲本身就是 `Σπ·穿透`；本场景每个指尖-物体对恰好 2 接触点 → `A≈π·穿透≈6 cm²`，量级上正好等于整块指垫 45.5×53.7 mm）。**`V = voxel_volume[i_s] + gs.EPS`，`gs.EPS = 1.19e-7 m³`（float32 eps；`genesis.init` 里 `EPS = max(eps, finfo(float32).eps)`）**。指尖的 `voxel_volume` 因 AABB 反变换 bug（只用世界 AABB 的两个角点回推 link 系，y 向 extent 塌成 −9.4e-6 m）只有 `1.435e-9 m³`（1.44 mm³；几何格容 1.17e-7 m³ = 指垫 1.87e-6 m³ ÷ 16 格，差 81×）——**低于 gs.EPS，所以引擎实际除的体积是 eps 本身**（1.214e-7 m³，与几何格容只差 2%） | 若照字段 `voxel_volume` 估算，c 被高估 `(1.214e-7/1.435e-9)² = 7160×`，任何参数看起来都不稳定；实测反演加热曲线时这 7160 恰好表现为恒定系数 `A_eff/A_buf = 1.18e-2 = 1/√7160`（曾误判为"引擎另有隐式缩放"）。**校核（`scripts/probe_kernel_args.py` 包住内核、打印实参并对 scratch 缓冲跑一遍）**：物体-桌面接触预测 −0.0011 K / 实测 −0.0011 K；指尖首个接触步预测 `0.529 K × 2 接触 ÷ 16 格 = 0.066 C` / 实测 `0.0667 C`（1% 内）——即公式无误，差的就是 eps。真发散也确实存在：`k=10` + `π·穿透` 回退面积 → `c≈11` → 体素在 ±1000 °C 夹紧值间逐 substep 翻转（实测右指尖均温 −322.6 °C） | `contact_depth_weight: 0.0` ＋ 指尖 `conductivity: 0.02`（`k_eff≈0.04`）→ **Gate 1 实测 `c = 7.9e-3`**，指垫 2 s 内爬升 +20.9 °C（= 130 °C 温差的 19%，保留速率信息，非阶跃），贴片再大 ~7× 仍稳定。Gate 1 断言 `c_max < 0.5` 并打印 `thermal | A_all … <传感器>: A … V … c …`（`_thermal_diag`：A 按**该传感器自己的连杆对**取，不用场景全局最大值——物体-桌面 ~9 cm² 是真实冷源但会误报指尖；**V 必须含 `gs.EPS`**，否则 c 恒定虚高 7160×）。k 只按数值稳定性选："按材料密度补偿热容"的思路不成立——pad 的有效热容由引擎常量 eps 决定，不由 `density/specific_heat` 决定 |
| R17 | 物体网格 ↔ 指尖是**单向耦合**：物体单元看到 `T_other = link_temps[指尖]`（指尖连杆保持 ambient），指尖单元看到 `T_other = link_temps[物体]`（保持 seed 值）——两侧都不被对方的**网格**温度反哺 | 接触面单元被过冷到 ambient（物体侧）/ 过快贴近物体温度（指尖侧），"接触热传导"的绝对量值偏大 | 录的是网格均温，受影响单元只占少数（物体 4×4×4 中 ~2–4 格，均值漂移几 °C，量级仍在物理范围内）；pad 侧同时受 R16 增益约束；数据卡标注该近似（v2 若要定量热流，改用 `tools/analyze_contact.py` 离线差分） |
| R18 | **刚体密度 ≠ 材质密度**：`ObjectSpec.mass_kg` 从 `object.mass_kg: [0.05,0.15]` 均匀采样、与材质解耦；`scene.py:296` 建刚体用 `rho = mass_kg/volume_m3`，而温度传感器用 `material_properties.density`（真材质值） | 抓取动力学（惯性/重力/摩擦）与"铜/钢"标签不符：**实测** 铝柱 r=20.2mm h=68.1mm m=0.0971kg → ρ_rigid=**1110**（真值 2700，2.4× 偏轻）；铜 r=20.8mm h=69.7mm m=0.0527kg → **559**（真值 8960，16× 偏轻）；abs 1306（真值 1040，反偏重 1.26×）——金属件在数据里像塑料件一样轻。热学侧（ρcp 决定网格热惯量/冷却速率）用真材质值，故同一 episode 内"热的是铜、动的是塑料" | v1 **不改物理**（改动会失效已通过的 Gate 2 与冒烟统计）：数据卡如实标注 `object_mass_kg`（逐 episode demo attr，已录）为"随机化质量"，不等同于材质密度×体积。**注意尺寸/质量是每文件一次（构建期冻结），文件内无变化**：文件内 pad 加热由 T0 主导（铝 r=+0.973 / 铜 +0.987 / abs +0.974 / 钢 +0.965），跨文件的 mass~dT r=−0.497 纯属材质混淆（铜恰好抽到最小质量、加热最高），质量本身只影响抓取动力学、不进热通道。v1 任务实测 32/32 成功，故不动。v2 二选一：(a) `mass_kg` 由材质密度×几何导出（物理自洽，删该旋钮）；(b) 保留随机化但新增真值 `object_rigid_density_kg_m3` 并在数据卡声明"质量与材质解耦（域随机化）"。倾向 (a) |
| R19 | **触觉力读数在"首次接触"substep 上不可逐位复现**（与渲染无关）：GPU 接触求解的 float32 归约顺序不稳定，`tactile/depth` 在个别 taxel 上差 ~1e-6 m，`tactile/force` 按 taxel 刚度放大成尖峰（实测 0.003–5 N），只出现在 3–4 个接触起始 policy step，之后重新收敛 | 重新生成数据集时 `tactile/*` 在接触起始的个别步上不会逐位相同 | **已实测并定级（Gate 4）**：同一场景、同一 seed、**完全关闭渲染**连跑两次，`tactile/force` 差 4.92 N，而同一次对比里 `ee_pos`/`actions` 逐位相同——即噪声源在引擎侧，不在相机。故 Gate 4 只对**轨迹通道**（proprioception/actions/contact/temperature，实测逐位相同）做强断言；`tactile/*` 改判"差异只出现在接触起始、末步相同"这一有界暂态性质，并同时打印**无渲染对照**的噪声底（不是断言）。数据卡须标注：`tactile/force` 在接触起始步不保证跨次生成逐位一致，绝对量级为 taxel 级尖峰 |
| R20 | **`eye_in_hand` 相机在个别帧上渲染尖峰**：同一集内中位 **24 ms**，但个别帧 **11–16 s**（~600×）。实测 ep15（aluminum）35 s 里 **29.4 s 全在 `eye_in_hand`**，集中于 step 4–6（DESCEND 下探、腕相机贴近桌面时）。尖峰**由内容决定且可复现**：同 `(scene_seed, episode_seed)` 在不同批次的集号完全一致（seed 1000 → `[11,15,20,42,44,45]` 两次逐集相同；seed 2000/3000 各复现一次；共 3 组独立复现），且**关掉相机后同一集只要 4.1 s** | 相机开启时约 10% 的 episode 从 8.3 s 涨到 25–350 s（全量 200 集约 53 min 而非 28 min）。**只影响耗时，不影响数值**（全部 0 issues），与 R19 性质不同 | **已定位、未修**：抬高 `near` 或把腕相机挪远都会改相机内参→已生成的 RGB 全部失效，故 v2 不动。将来重做相机标定时优先试 `cameras.near: 0.01 → 0.03` 并复测 ep15 |
| R21 | **材质与物体几何在数据集内完全混淆**：`scene_seed = cfg.seed + 1000*(index+1)` 取的是**材质列表下标**，而物体半径/高度/质量都在构建期由该 seed 抽出 → 每个文件里"材质"与"这一个物体"是 1:1 绑定（aluminum 永远是 seed-1000 那根柱子） | 做 VLA 训练没问题（多样性足够），但**不能用于"固定几何、只变材质"的对比分析**：材质效应无法从本数据集单独辨识。跨文件热通道差异里同时含几何与材质两个因素 | **已加防呆（2026-10-03）**：`generate.py` 在 `--material` 只选子集时打印各材质的 scene seed 并与全量运行对照。此前正是分两次跑（copper 落到下标 0）产出过与 aluminum **逐位相同**的 copper 文件（actions/proprio/contact/tactile/两个 RGB 通道全为 0 差，仅热通道差 ~1e-3 C），已单次全量重跑修正。数据卡须标注此混淆 |
| R22 | **触觉"压力突变"不是动力学，是接触门控 + 探针预压几何**：① `tactile/depth`/`force` 的 SDF 查询只对 rigid solver 报过接触的 geom 运行（`_kernel_build_sensor_geom_idx` 的 geom 列表逐帧取自 `collider_state.contact_data`）→ 接触前无论多近都恒读 0；② 探针浮在指垫表面**外** `probe_offset_m`=0.5 mm（设计原话"contact onset happens slightly before mesh penetration"）→ 门一开的那个 substep，探针中心已在物体表面内 ~0.5 mm，`depth ≈ probe_radius(4.0)+0.5 = 4.5 mm` → `force ≈ 1000 N/m × 4.5 mm × 12 taxel ≈ 45 N`；③ 之后真实的"压实"只有 ~0.25 mm（≈3 N），于是该通道在抓取期近似二值（0 / 45–48 N），动态范围 ~5%。**实测（2026-10-03，aluminum，3 seeds × 2 速度）**：`controller.finger_speed_m_per_s=0.04`（物理闭合放慢 ~10×，episode 34→42–44 步、+25% 时间、成功率不变）下，注册接触的 substep 手指只动 **0.01 mm**、物体位置不动，`depth` 仍 **0 → 4.4688 mm** 一跳；该集全部非零 depth ∈ [4.459, 4.649] mm（基线 [4.316, 5.670]）；两个速度下 policy 级最大单步跳变都是 **~45–47 N**，`median`/`last` 归约都不产生中间值 | `tactile/force` 抓取段近似接触指示器＋~5% 调制；**降低夹爪速度/换归约方式都不能得到平滑受力曲线**（此前把阶跃归因于 20 Hz `median` 硬切换的结论已修正——tie-break 只解释同帧 `force>0 & depth==0` 的 2/133 帧矛盾）。要平滑只能动传感器侧：减 `probe_radius_m`/`probe_offset_m` 压低"触碰基线"（半径 4→1 mm 时台阶 45→~14 N，仍是台阶只是矮），或改用 `elastomer` 模态，或用 **`proximity`** 拿到真正渐变的接近场（实测见 R23）。绝对幅值由 `normal_stiffness` 定标，等比缩放不改变曲线形状 | **已定位（2026-10-03）**：新增 `controller.finger_speed_m_per_s`（默认 0 = 原行为，`lift_object.yaml` 置 0.04）保留物理慢速能力；数据卡须标注 force 通道的准二值性质 |
| R23 | **渐变曲线在 `proximity` 模态里，且实测确实"从无到有"**：`gs.sensors.ProximityTaxel` 走**被跟踪 link 的点云 BVH**（`_kernel_point_cloud_proximity_taxel_bvh`，探针球 vs 点/`sphere_intersects_aabb`，全程不读 `collider_state.contact_data`）→ 无接触门控；读数 = `stiffness × (density_scalar/N_pc) × Σ_i (R − ‖p_i − o‖)`（球内采样点贡献之和），探针球一碰到物体就开始连续增长。**实测（2026-10-03，aluminum，对称抓取，`finger_speed_m_per_s=0.02` → 0.04 m/s 合拢，substep=5 ms）**：proximity 在指垫离物体**每侧尚差 ~3.4 mm**（gap 49.7 mm，物体直径 42.8 mm）时从 0 起步——即 `probe_radius 4 mm + probe_offset 0.5 mm` 的球面触及时刻——随后 **44/44 个 substep 每个值都不同**（连续增长 0 → 0.73，无台阶），一直到 44 步（0.22 s）后指垫真正接触的那一 substep，`depth` 才 0 → 4.42 mm、`force` 0 → 46.1 N（与 R22 一致）。采样密度 500 → 4000 点几乎不改变起点（255 vs 256）与形状（顶端差 ~1.4%）——500 点时曲线已逐 substep 变化，起点的 3.4 mm 是几何（球半径+预压）不是采样量化 | **记录速率（20 Hz + median 归约）下 ramp 占 5 个 policy step（0.25 s）**：实测 HDF5 `obs/tactile/proximity` step 23 起 0.0026 → 0.084 → 0.258 → 0.458 → 0.616（step 28 才 depth 跳变）。**边界**：proximity 是点云近似，幅值依赖采样密度与 `density_scalar`（默认 100），与 KinematicTaxel 的 N 不同标定、**不可直接互比幅值**；只对 `track_link_idx`（=物体）敏感，对指垫/桌面无响应；`probe_local_normal` 必须传指垫法线（否则力轴为默认 (0,0,1)） | **已实现（2026-10-03）**：`tactile.modalities` 可加 `proximity`（本次经 `/tmp/lift_proximity.yaml` 实验副本验证，正式 config 未改，见 Q5）；新增 `TactileConfig.proximity_sample_points`（默认 500 = 引擎默认）；`build_finger_sensors` 对 elastomer/proximity 传 `probe_local_normal`；`obs.VECTOR_MODALITIES` 补 `proximity`；`generate.py` 全流程 + validator 0 issues |
| R24 | **双物体场景（format 1.2）依赖的四条假设**：① 引擎 marker 场传感器真能同时跟踪多个压头（`track_link_idx` 原文档："indices whose collision geometry is queried by SDF"，复数）且 box 的 SDF 查询可用；② `n_sample_points` 传 **int** 会被引擎按 link `divmod` **均分**（两个物体 = 每个密度减半）——必须传逐 link 列表；③ 非目标物体是**热且可动**的刚体，机械臂接近路径可能撞到它（会在数据里留下非目标位移）；④ 物体间热耦合不存在（相距 0.24 m + `convection_coefficient: 0`，本就不该有，只是声明清楚） | ①/box-SDF 不成立 → marker 场对 box 无响应、demo 失败（兜底：按物体各建一套 elastomer 只读目标的，成本 ×2）；②不处理会**静默**把两物体的 marker 密度各减半；③会让"没碰的物体"出现位移 | **已验证（2026-10-07，`configs/lift_object_pair.yaml`）**：①+box SDF — Gate 1 全绿（0 failures/0 warnings；`temperature_object_0/1` 两栅格各自播种/读回：目标 130.0 C、非目标 21.0 C）；场景探针核对引擎实参 `track_link_idx=(15,16)`（Cylinder link 15 + Box link 16）；**8 集生成（seed 0 四集全 box + seed 5 四集含 cylinder/box）8/8 成功、validator 0 issues**，box 集 `tactile/elastomer` 两指 max\|disp\| = **26–80 µm**（非零 → 多压头跟踪 + box SDF 都工作）。② elastomer/proximity 均传 `(500, 500)` 逐 link 列表。③ `min_object_gap_m: 0.02` 守卫（基距 0.24 m + jitter ±0.04 下不触发）+ agentview 帧目测两物体全程在画面内、非目标静置。④ 不做物体间热耦合。**接触形状实测**：box 面抓早期实现把 `object_yaw + jitter` **整体**取整到 90° 整数倍 → 对齐的是**世界轴**，残余角 = 物体自身 yaw（box jitter ±17°）；5 集 box 的 depth 平整度与残余角逐一对应：残余 ≤6° → spread 0.10–0.67 mm；残余 10.7°/11.3° → spread 2.1–2.4 mm、marker 峰值只剩 7–8.7 µm（手指楔在棱上）。**2026-10-08 已修**（`expert.py`：抖动先量化选面、再加物体 yaw）：同 seed 重跑 5 集 box，yaw −10.7°…+11.3° 全覆盖，depth spread 全部 **0.03–0.26 mm**、marker 峰值 35–41 µm，8/8 成功、validator 0 issues；圆柱路径逐位不变。修后形状对照（`/tmp/gelpair3` 同文件相邻两集 demo_1 box vs demo_2 cylinder，共享刻度）：marker 场里 box 压印**铺满整块指垫**（16/16 列，峰值 36 µm，剖面平缓、两端仍有 ~17/10 µm），cylinder 只占 **11/16 列**（8.9 mm，33.5 µm，剖面更尖、两端跌到 ~6/1 µm）；刚性 depth 探针 box **0.09/0.10 mm** spread（平行面接触），cylinder **1.65/1.73 mm**（中列 ~4.5–4.9 vs 外列 ~3.2–3.9 mm，即曲率矢高 + 软垫顺应）。但**marker 场整体相似度确实高**——gel 弹性核把压力分布平滑成宽穹顶，两种形状看图像都像"一团"；形状的强信号在 **depth 通道**与印记**宽度**上（8.9 mm vs 13.0 mm），单看 marker 灰度图的形状差异很弱。对照图 `/tmp/imprint_compare2.png`，视频 `/tmp/pair_box_fixed.mp4` / `/tmp/pair_cyl_fixed.mp4`。**兼容性**：9 个旧文件（404 demos，format 1.0/1.1）validate 0 issues；单物体路径 N=1 不抽 `target_index`，rng 流不变。**2026-10-08 起 `lift_object_pair.yaml` 不再记录刚性 depth**（改为 elastomer-only 9×7 + 稠密高度图，§9-R25），所以该 config 的形状强信号只剩 marker 场压印**宽度**（box 铺满 7×9 格 vs cylinder 4×9 格）——上面那段"depth 通道里的强形状信号"是对**当时**那份 config 的实测记录，用旧 config 重跑仍成立 |
| R25 | **触觉对齐 UniVTAC 形态：63 点 marker 场 + 去刚性探针 + 稠密高度图**（format 1.3）。两个必须先讲清楚的机制：① **弹性核的物理宽度随格距变宽**——HydroShear 的 dilation 在 FFT 格上做，核宽以**格**为单位，所以格距 0.81 mm → 2.04 mm（16×16 → 9×7）时同一个核摊开约 2.5×；② **稠密高度图只能是插值合成**——本项目 gel 是虚拟传感器（SDF + 网格采样），场景里没有可形变几何体给相机拍，UniVTAC 能渲 `press_depth` 是因为它有 FEM gelpad + 指内相机 | ① 同一抓取、同一 seed 下 marker 场峰值 27.4/36.0 µm（16×16）→ **5.8/7.9 µm**（9×7），掉的不是采样密度而是核宽（见左）；② 没有 `force`/`depth` 通道时 gate1 的两条探针断言与依赖探针的专家判据会失效 | ① **已实测并决定：接受，不重标 `lambda_d`/`lambda_s`**（2026-10-08，用户决定）。机制判定：把 16×16 的场在 63 个标记**位置**上重采样，预测峰值 **26.3/36.0 µm** ≈ 原值 —— 即"采样位置"几乎不掉幅度，所以掉幅度的只能是核宽；形状信号仍在（box 压印满 **7×9** 格、cylinder **4×9** 格；16×16 时是 16×16 满 vs 10×16）。代价是 63 点场绝对值小（~6–8 µm），需要绝对幅度时用 16×16 的旧 config。② **专家不依赖探针**：接触检测回退 `contact/link_contact`（link 级传感器），elastomer-only 实测 2/2 成功；**gate1 的两条断言已条件化**（模态没开时打印 `modality disabled` 而不是 FAIL），并新增 marker 场响应检查（`peak |disp| … um`）与轨迹表 marker 列。**高度图**：`tactile.height_map {enabled, size, interpolate}` 把 marker 场重采样成 `(T,L,H,W)` 米（§4.2 末）；`validate.py` **重合成比对**而不是比峰值（栅格不落在每个样本上，孤立尖峰丢 ~1.5% 峰值，比峰值会 flaky）；不进 burst。**实测（2026-10-08，aluminum，seed 5，2 集 × 46 步，`configs/lift_object_pair.yaml`）**：Gate 1 全绿（0 failures/0 warnings；depth/force 显示 `modality disabled`、marker 响应 17.81 µm）、**生成 2/2 成功、validator 0 issues**（含新检查）、Gate 2 回归全绿（1/1 成功，通道表含 `obs/tactile/height_map (50,2,64,64)`）。**独立复核**（自己写 numpy/scipy 重推，不走 `tools/height_map.py`）：`format_version tempvla-1.3`；`layouts.attrs["height_map"]` 契约字段齐全；`(46,2,64,64) f32` 全有限；**从 marker 场重合成与存档逐元素 `max|Δ| = 0.000e+00`**（两集都精确相等）；峰值就等于场峰值（5.7762/7.8944 µm，clamp 上界成立）；接触前 23 步全 0、压合后升至峰值。**边距实测**：只给 `mode="nearest"` 时 scipy 样条预滤波的 mirror 边界会让最外 4 个像素从 5.05 µm 漂到 5.17 µm，**已改为先把采样坐标 clamp 到标记范围**——现在最外 3 行/列逐帧严格等于最外标记线（`max|Δ| = 0`）。**压印形状（峰值帧，9×7 场）**：cylinder 超过峰值 50% 的区域 = **4/7 列 × 9/9 行**（单侧条带），box = **7/7 列 × 9/9 行**（铺满整垫）——形状信号仍在，只是从刚性 depth 挪到了 marker 场的**宽度**上。逐指峰值：cylinder 5.78/5.65 µm、box 7.89/4.92 µm（指垫 patch 14.2×13.8 mm，格距 2.033 mm 列 / 1.536 mm 行）。**体积实测**（同 seed 同 2 集，仅开关 `height_map`）：4.27 MiB vs 3.63 MiB → **+0.32 MiB/集（+17%）**，相机帧仍是主体。**兼容**：`data/`、`data_cam/` 各一个旧文件（format 1.0）validator **0 issues**（新检查自动跳过）；`pick_hot_cylinder.yaml`（force+depth+16×16）Gate 1 **0 failures/0 warnings**，`depth 4.584 mm` / `force 4.584 N` 断言原样通过。**渲染**：`render_video.py` 的 gel 面板优先用 height map，标题源显示 `height_map 64x64 (marker field resampled)`，`--markers` 晶格叠在同一张图上（cell-centred 的收益）。交付图 `/tmp/hm3_pair_compare.png`（两物体 × 场/图，共享 0–8 µm 绝对刻度）、面板对照 `/tmp/hm3_panel_compare.png`（左 9×7 场 vs 右 64×64 图，同 gain）、视频 `/tmp/hm3_demo1.mp4`（带图）vs `/tmp/hm3_nohm_demo1.mp4`（不带图） |

---

## 10. 里程碑（工作量量级）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境搭建 + Gate 0/1 | ✅ `tempvla` 环境（genesis-world 1.4.1 + cu128 torch）；Gate 0/1/2 在 GPU 后端全绿（failures 0 / warnings 0），见 §9-R16 |
| M1 | 单 episode 最小链路（无相机、单 env、HDF5 写出 + 校验） | ✅ 代码完成（`gate2_scene.py`） |
| M2 | 每材质 50 demo + 成功判定 + 失败分离 | ✅ **完成**：`data/` 下 4 文件 × 50 demo = **200/200 成功、0 失败、0 error**（15.7 min，`WALL 944 s`）；逐文件 3.17–3.25 MiB，合计 12.9 MiB；`validate_dataset.py data/*.hdf5` 四个文件全部 "OK: no issues found"；每文件 50 demos、`total` attr = 50 一致。逐材质：步数 32–36、接触事件 1.8–1.9、T0 62–196 °C、指垫峰值升温 铝 35.4 / 铜 61.9 / abs 33.6 / 钢 45.9 °C |
| M3 | 相机接入（Gate 3 + Gate 4 均已通过）+ 双相机 128² | ✅ **代码完成 + 全量数据已生成**：`renderer=raster` + llvmpipe；`cameras.enabled=true` 即写入 `obs/agentview_rgb` + `obs/eye_in_hand_rgb`，渲染不扰动仿真（Gate 4）。代价 +124 ms/步（含首帧着色器编译口径）、+0.49 MiB/episode；`data_cam/` **200/200 成功、97 MiB、53 min**（干净集 8.3 s/ep；尖峰见 R20）。**后续（2026-10-03）**：分辨率提到 256²——`data_cam/` 仍为 128²，`configs/*.yaml` 已改为 256²，未重新出全量 |
| M4 | 任务套件 v1（3 任务）+ replay/analyze + 数据卡 | ⏳ v2（v1 交付 `pick_hot_cylinder` 一个任务、4 个材质文件） |
| M5 | **marker 场转正**（format `tempvla-1.1`：逐模态网格 + `obs/link_pose` + `elastomer` 进正式 schema，Q5） | ✅ **代码 + 七项验证全过（2026-10-04）**：`config.py`/`scene`/`obs`/`sensors`/`recording`/`tools/tactile_field.py`/`gel_render.py`/`render_video.py --markers`/`validate.py`；Gate 1（0 failures/0 warnings，两套网格各按指列出）、Gate 2（0 issues）、`generate.py` 冒烟（`/tmp/gelship`，通道形状逐项核对）、**往返一致性**（R8）、`render_video.py --markers`、Gate 4（0 failures，`link_pose`/`elastomer` 逐位一致）、旧语料向后兼容（12 个 format 1.0 文件 validate 0 issues + 渲染回退正常）。**待做：全量重生成** —— `data/`、`data_cam/`、`data_lift/` 现均为 format 1.0、**不含 marker 场**；先写暂存目录、抽查后再替换旧目录（旧目录改名保留；`data*/` 不入库，git 也回滚不了数据本身） |
| M6 | **双物体任务**（format `tempvla-1.2`：`objects:` 列表 + box morph + 逐物体通道/attrs + box 面抓专家） | ✅ **代码 + 验证（2026-10-07，R24 汇总）**：`configs/lift_object_pair.yaml`（圆柱 + 立方体、按 episode 二选一）；Gate 1 / Gate 2 全绿（0 failures/0 warnings，validator 0 issues）；demo 生成 8 集 8/8 成功、两种目标都覆盖；box 面抓 4/4 成功且 marker 场两指有响应（26–80 µm）；`render_video.py --markers` 两集视频 + 圆柱 vs box 压印对照图；旧语料 9 文件 validate 0 issues。**2026-10-08 修 box 面抓残余角缺陷**（R24）：修正后 box 抓取 depth spread 0.03–0.26 mm（任意物体 yaw），圆柱 vs box 的形状对照在同一文件相邻两集成立。**未做**：双物体全量数据集（与 M5 的全量重生成一起，待定） |
| M7 | **触觉形态对齐 UniVTAC**（format `tempvla-1.3`：63 点 marker 场 + 去刚性探针 + 合成 64×64 height map） | ✅ **代码 + 验证（2026-10-08，R25 汇总）**：`configs/lift_object_pair.yaml` 改 9×7 elastomer-only（间距 2.03/1.54 mm，覆盖 14.2×13.8 mm 指垫）；`tempvla/tools/height_map.py` 新模块（cell-centred 栅格 + 坐标 clamp 防 mirror 边界 ringing），`validate.py` 以重合成比对校验（独立复核 `max|Δ| = 0.000e+00`）；Gate 1（含新 marker 响应检查，depth/force 显示 `modality disabled`）/ Gate 2 全绿，生成 2/2 成功、validator 0 issues（含 `obs/tactile/height_map (50,2,64,64)`）；体积 +0.32 MiB/episode（+17%）；`render_video.py` 优先以 height map 作 gel 面板源；单物体两 config 与全部旧文件不受影响。**未做**：1.3 全量数据集（旧 data/、data_cam/ 仍为 1.0 / 128²） |

---

## 11. 待决问题（已决 = 已实现进 v1）

| # | 问题 | 结论 |
|---|---|---|
| Q1 | 本体：浮动手 vs 手臂+手 | ✅ **已决：Franka Panda 单臂 + 平行夹爪**（`fixed=True`；动作语义与浮动手同构，底层为 IK + 关节 PD） |
| Q2 | 手型号 | ✅ **已决：Franka 平行夹爪**（两指、指尖温度网格）。探针网格**逐模态**（2026-10-04）：`force`/`depth`/`proximity` 每指 12（4×3），`elastomer` 每指 256（16×16）——见 Q5 |
| Q3 | v1 相机 | ✅ **已决：v1 数据无相机**（`cameras.enabled=false`）；**v2 相机链路已完成并验证**（Gate 3 渲染 + Gate 4 落盘/不扰动仿真，§8）：置 `enabled: true` 即写 `obs/agentview_rgb` / `obs/eye_in_hand_rgb`。madrona/BatchRenderer 在本机被缺失的 Vulkan loader 挡住（`sudo apt install libvulkan1` 可解，非 sm_120 问题） |
| Q4 | v1 任务清单 | `pick_hot_cylinder`（✅ 已实现，v1/v2 数据即它）/ `lift_object`（✅ 已实现，最小任务：同一专家去掉"hot/pick"语义，传感栈不裁剪，`configs/lift_object.yaml`）/ `place_hot_object` / `push_hot_object`（v2） |
| Q5 | 触觉模态记录集合 | ✅ **已决（2026-10-04）：`force` + `depth` + `elastomer`** —— `elastomer` 从实验配置**转正为正式 schema**，它是唯一能给出真压痕形状的通道（R8）。转正时把当初列的两条代价都从根上修掉，而不是接受：① **网格共享 → 逐模态网格**（`tactile.probe_grid` 支持 `{default: [4,3], elastomer: [16,16]}`，`force`/`depth` 的 4×3 与语义逐字节不变，`layouts/` 加一层 `grids/<tag>/` 别名层保持老读者可读，§4.2）；② **帧丢失 → `obs/link_pose`**（每步每指 pos+quat，让 `world_frame: true` 精确可逆，`v_link = R(q)ᵀ·v_world`），因此**不必**把 `world_frame` 记成 false —— force 的世界系语义与已有语料保持可比。代价如实记账：时间 +51%（无相机路径 84.4→127.7 ms/step，R8）、L1 ~3.1 KiB/step、**burst 每窗口 +110–218 KiB（约 1.1–2.2 MiB/集，`recording.burst_modalities` 可收口）**；语料规模见 §2.5。`proximity` **仍待定**：实测给出 `force`/`depth` 没有的接触前渐变（R23），代价 +1 个通道与点云采样开销，本次**不随 marker 场一起进**（配置层已支持，需要时单独开）。实验夹具 `configs/lift_gel_preview.yaml`（`world_frame: false`）**保留**：它是往返一致性测试的对照件（用 `tactile_field` 断言两种帧下的高度场一致）。 |
| Q6 | 失败 episode 策略 | ✅ **已决：`*_failed.hdf5` 单独保留**（`keep_failed: true`）；全成功时该文件在 `close()` 里被删除——避免 `data/*.hdf5` 通配符把空文件喂给 `validate_dataset.py` 报 "no demo_N groups found" |
| Q7 | 分辨率 / 步数上限 | ✅ **已决：256×256**（2026-10-03 由 128² 上调；LIBERO 用 128²，但那是 Isaac RTX + 照片级资产，llvmpipe 下 128² 的物体只有 188 px ~14×14，夹爪-接触区难以分辨。256² → 764 px ~28×28，时间只 +9%、磁盘 2.4×；详见 §8 分辨率裁决表与 `res_compare_resolutions.png`）；T ≤ 400（20 s，✅ 已实现） |
| Q8 | 接触记录默认档位 | ✅ **已决：L1+L3 常开；L4 默认开**（[−0.05 s, +0.25 s]、上限 10）。模态随 `tactile.modalities` 全带（含 16×16 marker 场，2026-10-04）；要压体积就设 `recording.burst_modalities: [force, depth]`——**窗口数与帧数不变**，只换模态集 |
