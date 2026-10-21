# ThermTacVLA 架构设计

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

Genesis 有 3 条渲染路径（两条不依赖 OpenGL）；本机实测只有软件光栅路径可用，见下表与 §8/§9-R3：

| 路径 | 机制 | WSL2 前提 | 用途 |
|---|---|---|---|
| A. BatchRenderer | Madrona（CUDA 内核） | `gs-madrona` 需 CUDA 12.8+ / torch cu128+；支持 Linux x86-64；**运行时 NVRTC 编译，架构无关**（与 sm_120 无关，§9-R3） | ❌ **本机不可用**：WSL2 缺 NVIDIA Vulkan ICD，`scene.build()` 直接 abort（§9-R3） |
| B. Rasterizer + EGL | pyrender → Mesa D3D12 翻译层 | `LD_LIBRARY_PATH=/usr/lib/wsl/lib`、`GALLIUM_DRIVER=d3d12`、`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` | ❌ 相机图像经 D3D12 路径塌缩（内容只剩最下 2 行像素，§8 Gate 3） |
| C. Rasterizer + OSMesa | CPU 软件光栅化 | 安装 OSMesa 即可；Genesis 在 `vis/rasterizer.py:18-20` 自动回退 | ✅ **本机采用**：`renderer=raster` + llvmpipe（Gate 3/4 实测，43–50 ms/帧） |

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
- **指尖通道定案（format 1.5）**：指尖 = 官方最粗栅格 `grid_size=(1,1,1)`（即上面的"集中温度"默认，**不新增任何自写物理**）+ 官方传感器非理想性（一阶 RC `sensor_time_constant` **0.05 s**、白噪 σ **0.05 °C**、量化 **0.1 °C**，作用顺序 RC → random_walk → noise → bias → resolution，全在 **measured 缓冲**上），数据集只录 `read()`（`record_ground_truth: false`）；物体传感器非理想性全 0，`read()` ≡ 真值（逐位）。材质对比由指垫 `conductivity = 0.25` 经 `k_eff = 2·k_p·k_o/(k_p+k_o)` 带出。机制推导、已知局限与读数约定见 §9-R33/R35/R36。
- **接触耦合率 λ 与温度无关**：`λ = k_eff·A²/(V²ρc)` 只由材质与几何决定，**绝对温度不含材质信息** —— 这是 `pick_by_temperature` 里"每物体温度各自独立随机"不破坏可判性的唯一理由；实测冷热两卷 λ̂ 逐位相同、`k_eff` 在 `k_pad=0.25` 下饱和（金属 0.4997/0.4994/0.4992 差 <0.5%、ABS 0.2222），故一条阈值覆盖所有金属/非金属对，而**两种金属同场不可分**。数字与推导见 §9-R38。

### 2.3 触觉仿真（Genesis 解析式 taxel 传感器）

- 解析计算（无需渲染），10 种模态注册表（源自 `dexterous-hands/src/tactile_sensors.py`）：

| 模态 | features/probe | 类型 | 说明 |
|---|---|---|---|
| `link_bool` / `link_force` | 1 / 3 | 连杆级 | 接触布尔 / 合力 |
| `bool` / `depth` | 1 / 1 | 探针级 | 接触布尔 / 压入深度 |
| `agg_bool` / `agg_force` | 1 / 3 | 探针聚合到连杆 | 每连杆聚合 |
| `force` / `force_torque` | 3 / 6 | 探针级向量 | KinematicTaxel |
| `proximity` | 3 | 探针级向量 | 接近场（跟踪 link 的点云 BVH，**不接触门控**，见 R23）；引擎返回 force+torque，本项目只记录 force |
| `elastomer` | 3 | 探针级向量 | 弹性体位移场（最接近光学触觉）；**正式 schema 的 marker 场**（Q5 已决），16×16/指；1.6 起落盘只保留切平面 2D 分量（§4.2） |

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
  网格是 `sim/sensors/tactile.py` 的模块常量 `PROBE_GRID`（`{default: (4, 3),
  elastomer: (16, 16)}`，不是 YAML 键；唯一的取值口是 `tactile.grid_for(modality)`）。
  场景按**去重后的网格**调用 `derive_finger_layout`
  （指垫几何与网格无关，同一份 pad 几何被各网格复用），组装出 `layouts_by_modality`；
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

即：旧栈三套（`data/` 12.9 MiB、`data_cam/` 97 MiB、`data_lift/` 2.5 MiB；**2026-10-12 已删除**，
§9-R31）用 1.4 栈重生成后合计 **~1.4 GiB**；VRAM 不变（触觉传感器在 CPU 侧）。
`probe_radius_m` 仍全局共用（16×16 下探针球重叠，见 R8 末段）。

---

## 3. 总体架构

### 3.1 分层

```
┌──────────────────────────────────────────────────────────────────┐
│ scripts/      generate · scene_check · render_scene ·             │ ← 生成与检查入口
│               validate_dataset · build_manifest · render_video   │
├──────────────────────────────────────────────────────────────────┤
│ recording/    EpisodeRunner · HDF5Writer · events · meta          │ ← 数据出口（流式）
├──────────────────────────────────────────────────────────────────┤
│ expert.py     脚本专家 PickExpert（阶段状态机 + 域随机化 + 触觉反馈）│
├──────────────────────────────────────────────────────────────────┤
│ actions.py    7 维动作控制器（6 delta pose + 1 协同抓取 → IK + PD）│
├──────────────────────────────────────────────────────────────────┤
│ obs.py        观测组装（proprio / tactile / temperature / contact）│
├──────────────────────────────────────────────────────────────────┤
│ config.py     任务规格（sim/configs/*.yaml → 校验后的 dict）· 成功判定参数   │
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
sim/configs/*.yaml ──► TaskConfig ──► build_scene(cfg) ──► SceneBundle
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
| 触觉补丁 | 拷贝 `src/sensor_compat.py` → `sim/compat/sensor_compat.py`（逐字保留，import 时 `install_sensor_reset_fix()`） |
| 探针布局 | **程序化推导**（不再拷贝布局文件）：`finger_layout.py` 解析夹爪 URDF 碰撞几何（mesh 顶点 → 指垫平板）生成两指内侧网格 |
| 机器人 URDF | 用 pip wheel 自带的 `urdf/panda_bullet/panda.urdf`（不拷贝资产目录） |
| 传感器参数 | 不需要外部 YAML：触觉/温度参数直接写在任务配置 `sim/configs/*.yaml` 的 `tactile:`/`temperature:` 段 |
| 专家模板 | 参考 `collect_slides.py` 重写（不 import） |
| 明确不用 | Eden / rsl_rl / dexterous-hands 的任何 Python 模块 |

---

## 4. 数据契约（核心）

### 4.1 文件组织（对齐 LIBERO）

```
datasets/v1.5/hdf5/                            # 数据根与落点见 §9-R30（数据卡 datasets/README.md；目录名沿用 1.5，内容格式以 format_version 为准 = 1.6）
├── pick_hot_cylinder_aluminum.hdf5          # 每 (任务, 材质) 一个文件，含 50 条成功演示
├── pick_hot_cylinder_aluminum_failed.hdf5   # 失败 episode 单独存放；**0 失败时不落盘**（不产生空文件）
├── pick_hot_cylinder_copper.hdf5
├── pick_hot_cylinder_copper_failed.hdf5
├── ...
└── index.jsonl / splits.json / stats.json   # 侧车三件套（`scripts/build_manifest.py`，§9-R30）
```

**为什么按材质分文件**：热物性（k/ρ/c/ε）是 `TemperatureGridSensor` 的**类级共享参数**
（§2.2，同一 scene 内 "last one wins"），且物体尺寸/质量在场景构建期冻结。
因此材质是**文件级常量**，文件内只随机化位姿/偏航/初始温度。这也让"材质分类"成为
文件级的天然标签，下游可按材质做条件生成。

### 4.2 HDF5 树形结构

```
<task_id>_<material>.hdf5
├── attrs
│   ├── format_version : "tempvla-1.6"   # 1.0 = 单网格/无 link_pose（2026-10-04 前）；1.1 = marker 场转正；1.2 = 多物体加法扩展；1.3 = 可选稠密高度图；1.5 = 指尖改官方单格 (1,1,1) + 官方非理想性、只录 measured（§9-R36）；1.6 = 触觉只留 6D 合力 + 2D marker（§9-R44）
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
│   │   │   │   # ── 扩展通道 ① 触觉（format 1.6：两个派生信号；每模态用自己的网格）──
│   │   │   ├── tactile/
│   │   │   │   ├── force     : (T,L,6)   float32    # 1.6：每指一个 6D **合力** [Fn(3), Ft(3)]（link 系）
│   │   │   │   ├── elastomer : (T,L×P,2) float32    # 1.6：2D marker 场 = 每标记的切平面位移 (u,v)，finger-major；P=16×16
│   │   │   │   └── (proximity : (T,24,3) float32)   # 可选模态，当前 config 未开（Q5）
│   │   │   │   # 1.6 不再写 depth / height_map；≤1.5 的 3D/世界系口径见下文「触觉口径」小节
│   │   │   │   # ── 扩展通道 ② 温度 ──
│   │   │   ├── temperature/
│   │   │   │   ├── fingertip_grid : (T,2,1,1,1) f32 # 两指**实测**温度：官方最粗单格 (1,1,1)/指（§9-R36），含官方 RC/噪声/量化；`record_ground_truth: true` 的老文件是同一形状的真值。历史：1.3 前 (4,4,1) AABB 切格（§9-R26）
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
│   │   │           ├── fingertip_<modality> : (Tw,·) f32  # 与 L1 同形状：1.6 = force (Tw,L,6) / elastomer (Tw,L×P,2)
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
    │          object_link_idx, object_link_indices, tcp_offset, temperature_grids,
    │          channel_formats
    │          # object_link_idx = 第一个/目标物体（旧语义）；object_link_indices (N,) = 全部（1.2）
    │          # n_probes_total / probe_grid 都是**主模态**的那一份（老读者语义不变）
    │          # channel_formats (1.6, JSON str) = 每个派生通道的 {features, frame, shape}（见下文）
    │          # height_map (1.3–1.5, JSON str) = 旧稠密高度图的契约；1.6 不再写该 attr
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
  `probe_grid`；读取端（`sim/tools/tactile_field.py`、`render_video.py`、
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

**稠密高度图（2026-10-08 起，format `tempvla-1.3`；1.6 起停用）**：可选通道
`obs/tactile/height_map (T,L,H,W) float32`，单位米——把 **3D** marker 场重采样成一张
GelSight/UniVTAC 风格的稠密图（对应 UniVTAC 的 `press_depth`）。纯加法：不开
`tactile.height_map` 的 config 与 1.2 逐字节相同（只差版本串），旧文件读取端自动跳过。
format 1.6 的 marker 只有切平面 2D 分量，该通道不再写入（§9-R44），本节保留为 ≤1.5 文件
的读取契约。

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
- **读取端**（`sim/tools/height_map.py`，纯 h5py/numpy + 延迟 import scipy）：
  `read_height_map`（文件里存了就直接用）、`synth_from_markers`（**任何**带
  `obs/tactile/elastomer` + `layouts/` 的文件都能现场合成，包括 1.1 老文件）、
  `height_map_or_synth`（存了用存的，否则按元数据合成），`render_video.py` 走后者。

**触觉口径简化（2026-10-20 起，format `tempvla-1.6`，§9-R44）**：数据只保留两个与 AT-VLA
对齐的信号；其余触觉参数全部从 YAML 收进 `sim/sensors/tactile.py`（YAML 只剩 `tactile.modalities`）：

- `obs/tactile/force (T,L,6)`：每指一个 **6D 合力**（AT-VLA 口径）= 3D 法向 + 3D 切向合力，
  **不是** 力 + 力矩。来源 = 该指 12 个 `KinematicTaxel` 探针力的和，沿指垫外法线
  `n = -inner_dir` 分解：`Fn = (F·n)·n`、`Ft = F - Fn`，**压缩为正**。
  `obs/contact/link_force (T,L,3)` 仍是求解器级合力，保留作对照。
- `obs/tactile/elastomer (T,L×P,2)`：每 marker 的 **2D 切平面位移**（AT-VLA 的 N×2 格式）。
  基向量由未变形 16×16 网格的首行/首列差分构造并对 `n` 正交化：`u` 沿网格轴 0（垫长）、
  `v` 沿轴 1（垫宽）；**法向分量被丢弃**。两个轴的方向继承该指的网格朝向，`(u,v)` 不是
  世界定向的右手系。
- 两个信号都在 **link / pad 系**，`world_frame` 固定 `false`；`obs/link_pose (T,L,7)` 保留，
  世界系由 `p_world = R(quat)·p_link + pos` 恢复。
- 逐模态网格不变：`g4x3`（12/指）供 force 求和与法向，`g16x16`（256/指）供 marker。
- `layouts.attrs["channel_formats"]`（JSON）声明每个派生通道：
  `force: {features:6, frame:"link", shape:[L,6]}`、
  `elastomer: {features:2, frame:"pad_tangent", shape:[L·P,2]}`；`validate.py` 按它校验形状
  （缺该 attr 的旧文件回退到网格派生的 `(T,L·P[,3])` 规则）。
- `tactile/depth` 与 `tactile/height_map` 不再写入（2D marker 无法合成离面高度场）。
  format ≤1.5 的 3D marker / height map 文件仍可用旧工具读取（`tools/height_map.py`、
  `gel_render.py`、`render_video.py` 只服务旧数据；对 2D 输入显式报错）。

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
| `tactile/force` | (T,L,6) | f32 | 该指 12 个 `KinematicTaxel` 探针力求和 → 沿指垫法向分解（1.6） | AT-VLA 6D 合力 `[Fn(3), Ft(3)]`，link 系、压缩为正；子步 median 降噪后按 policy 率采样；`contact/link_force` 是求解器级合力，二者可对照（§4.2「触觉口径」） |
| `tactile/elastomer` | (T,L·P,2) | f32 | `ElastomerTaxel` 16×16 marker 场（1.6） | 每 marker 的切平面位移 `(u,v)`，finger-major；`u`/`v` = 网格轴 0/1；无法向分量、无 height map（§9-R44）。旧文件是 3D 世界系 `(T,L·P,3)` |
| `tactile/height_map` | (T,L,H,W) | f32 | 由 3D `tactile/elastomer` 重采样（1.3–1.5） | **仅旧格式**：1.6 的 2D marker 不含离面分量，无法合成。参数在 `layouts.attrs["height_map"]`，读取端 `tools/height_map.py`（§4.2 末、§9-R25） |
| `link_pose` | (T,L,7) | f32 | 两指 link 的 FK 读回 | `pos_world(3) + quat_wxyz(4)`（编码写在 `layouts.attrs["link_pose_encoding"]`）；世界系 ↔ link 系的**精确**逆变换 `v_link = R(q)ᵀ·v_world`，与相机开关逐位一致 |
| `object_pose` | (T,N,7) | f32 | 每物体 root DOF 读回（1.2） | 编码同 `link_pose`；行序 = `env_args.objects`；N=1 时的 `objects[target]` 与旧 `object_*` attrs 同源 |
| `temperature/fingertip_grid` | (T,L,1,1,1) | f32 | TemperatureGridSensor（两指 link，**measured**） | 格式 1.5（§9-R36）：官方最粗 `grid_size=(1,1,1)` = 每指**一个**集中温度读数，官方 RC/噪声/量化在内；`record_ground_truth: true` 的老文件是同一形状的真值。历史：1.3 前默认 (4,4,1) = 引擎 AABB 切格（§9-R26）。形状以 `layouts.attrs["temperature_sensor"]`（1.5）/ `["temperature_grids"]`（前三位 = 指）为准 |
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
  "language_templates": ["pick up the hot {material} cylinder"],   # 1.5：全部句式（单句式 = 1 条）
  "language_instruction_variants": ["pick up the hot copper cylinder"],  # 1.5：全部渲染串（句式 × 物体）
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
  **多句式（1.5）**：`language_template` 也可写成 YAML 列表；runner 按
  `episode_seed % n` 逐 episode 轮换（同一 seed 重跑得到同一句）。文件级
  `language_template` / `language_instruction` 仍取第一条，`language_templates` 列全部句式，
  `language_instruction_variants` 列全部去重渲染串；列表内各句的占位符集合必须一致（加载时校验）。
- **材质判据文件（1.5，`pick_by_temperature`，§9-R38）**：`problem_info` 另带
  `objects_material` / `objects_material_group` / `material_judgement`（协议、统计量、阈值/窗长/
  沉降、温区表、槽位置换标志、语言与真值的可达性说明）；`randomization.per_episode` 追加
  `slot_permutation` / `requested_group`；每个 demo 另记 `requested_group`、`objects_material`、
  `objects_group`、`target_material`、`front_object_index`（**专家永远先试的那块 = 0**，与
  `target_index` 分开记，这正是"不能靠找目标逃避摸一下"的可检查点）、`judge`（JSON，每次抓取一条：
  λ̂/逐指 λ̂ 与 τ̂/R²/幅值/onset 步/判定/`group_true`/`correct`/`committed`）、
  `judge_grasps`、`judge_correct`、`judge_committed_correct`。指令用**组词**（`{group}` =
  `material_group`）而不是材质名——两块方块无视觉材质，材质名是不可学的标签，组词才是温度通道
  能揭示的东西。
- `data.attrs['env_args']` 记录完整可复现配置：Genesis 版本与 assets 目录、
  sim_dt/decimation/control_hz、机器人 URDF 与解析出的关节/连杆名、
  动作与观测通道契约（含触觉布局/温度网格/接触通道说明）、物体与桌面参数、
  相机配置、录制（爆发窗口）参数、成功判定与步数上限。**1.2 起** `objects` 数组
  逐物体给 `shape`/范围/`place_xy` 及构建期 `resolved_*`（本文件冻结的实测值），
  旧 `object` 块保留 = 第一个物体。
- `material_properties` / `object_*_m` / `object_mass_kg` 是**文件级真值**（构建期冻结），
  `object_temp_c` 是**逐 episode 真值**——两者都保留，供下游做温度回归/条件生成。

### 4.6 与 LIBERO 的对照

| 维度 | LIBERO | ThermTacVLA | 兼容性 |
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
  **0.02…0.23 °C**。这不是 bug：物体 64 格的热容 ≈ 218 J/K，是指垫侧（eps 主导，
  ≈0.19 J/K/格）的 ~70 倍，且仅 2–4 格参与接触，故物侧降温本就该是千分之一量级
  ——物理上正确，但意味着 **`object_grid` 近乎常数、几乎不含信息**，可学习的热信号
  在 `fingertip_grid`（以及 `object` 的初值 = 真值温度）。指尖升温幅度偏大是 R16 的
  数值稳定性取值的代价（k=0.02 比"真实指垫"强 ~7×），**相对**排序与 T0 依赖仍保真。
  以上实测数字来自 v1 的 AABB 网格（每指 16 格、格心不与 marker 对齐，§9-R26）；
  1.5 起指尖是官方最粗栅格的**单格实测**通道（每指一个集中读数，§9-R33/R36）——
  格数与热流分摊方式都不同，但"物侧近乎等温、热信号在指尖侧"的结论不变；
- `obs/tactile/*`：format 1.6 只有 `force (T,L,6)` 与 `elastomer (T,L×P,2)`（§4.2「触觉口径」）。
  `force` 的行数是**每指一个** 6D 合力（不再等于探针数）；`elastomer` 仍是每指 16×16 = 256 个
  marker，finger-major 展平。逐模态网格仍写在 `layouts/modality_grids`（`force` 4×3 供求和，
  `elastomer` 16×16 供 marker）；
- `obs/link_pose`：每步每指世界位姿 `pos_world(3) + quat_wxyz(4)`。1.6 的两个触觉信号都已在
  link / pad 系，`link_pose` 只作**世界参考**（`p_world = R(q)·p_link + pos`）；
  旧 1.5 文件的 `world_frame: true` 向量才需要 `v_link = R(q)ᵀ·v_world` 反演。

**L2 接触热流层**（验证门控 → 结论：**不录通道，离线推导**）：Gate 2 检查确认
Genesis 温度模型未通过公开 API 暴露接触热项，且不值得为此打补丁（该量可由 L1 网格
差分重建：Q̇ ≈ ρ·c·V·ΔT/Δt，接触窗口内为主项）。需要显式热流时由
`tools/`（规划：`analyze_contact.py`）从 `object_grid` / `fingertip_grid` 离线推导。

**L3 事件层**（`events/contact_events/`）：**policy 率**检测每指接触 onset/offset
（`ContactMonitor`，状态布尔来自 `contact/link_contact`），每事件记录：
`event_id`、`start_step`/`end_step`（policy 步号）、`start_frame`（对应 sim 子步号）、
`time_s`、`peak_force_n`、`max_depth_m`（1.6 无 depth 通道 ⇒ 恒为 0.0，保留键以兼容旧读者）、
`link_bool`（上升沿涉及哪几根手指）。
事件步号与 `actions` 直接对齐，便于关联。

**L4 爆发窗口层**（`events/burst/`，默认开）：以**sim 率（200 Hz）**记录每个接触
上升沿前后的瞬态，窗口 [onset−0.05 s, onset+0.25 s]，即 pre 10 + 触发帧 1 + post 50
= **61 帧**（闭区间 0.30 s），每 episode 上限 10 个窗口。每帧通道：`tcp_pos`、
`finger_joint_pos`、`contact_bool`/`contact_force`、`fingertip_<modality>`（触觉派生模态，
与 L1 同形状；**含 `elastomer`**：1.6 一个窗口 61 帧 × 512 标记 × **2** f32 = 244 KiB 原始
（旧 3D 格式为 366 KiB），gzip 大小取决于窗口内接触占比）、`finger_temp`、`object_temp`；
附 `trigger_frame`/`start_frame`/`end_frame`。
窗口数与帧数**不因模态集变化**，`recording.burst_modalities`（默认 `None` = 全部模态）
是唯一的收口阀，出数据集前若觉得 burst 体积过大可在这里裁。
实现：`BurstRecorder` 在 sim 步循环内维护预触发环形缓冲，触发锚定在**真实接触子步**
（用 sim 率 payload 的接触沿检测，而非 policy 步边界）。

**触觉空间定位是免费的**：探针世界坐标 = `layouts/probe_local_pos` + `joint_states` 重算
（或直接用 `obs/link_pose` 的逐步位姿），接触区域 = active 探针集合（`depth>0` 探针的
重心即接触中心）——无需额外接触点几何 API。

**marker 场也是免费的**：`obs/tactile/elastomer` 的位移 + `layouts/grids/<tag>/probe_local_pos`
的未形变参考 = 每个标记的切平面轨迹（`tools/tactile_field.py` 的 `read_markers` /
`reference_positions_2d` 就是这两项；Tabero 的 `M = [参考帧, t_1..t_H]` 可直接由它们组装）。
1.6 只有 2D，`pad_normal_field` 的 GelSight 式高度场仅对 ≤1.5 的 3D 数据成立。

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

### 5.1 `sim/compat/` — 引擎补丁层

- `sensor_compat.py`：从 dexterous-hands 拷贝（**逐字保留**），import 时执行
  `install_sensor_reset_fix()`。
- 职责边界：只做 monkey-patch，不含业务逻辑；升级 Genesis 时首先回归此文件。

### 5.2 `sim/sensors/` — 传感器装配层（已实现）

| 接口 | 说明 |
|---|---|
| `finger_layout.derive_finger_layout(urdf, (link_l, link_r), probe_grid, probe_offset_m, pad_tol_m) -> list[FingerLayout]` | 解析 URDF 的指节 collision（mesh 顶点，box 兜底）→ 指垫平板 → 两指**内侧抓取面**探针网格（法线 = 指向另一指的单位向量）；闭合轴取两指垫中心连线（关节原点仅作回退）；输出探针位姿（link 系）、pad 平面（`pad_box_origin/size`，法向厚度为 0）、`num_probes`。**pad 几何与网格无关**，所以逐模态网格只需按**去重后的网格**各调一次，同一份 pad 几何被各网格复用。`layouts_to_hdf5()`（单网格包装）与 `layouts_to_hdf5_multi(groups, *, default_grid, modality_grids, attrs)`（§4.2 的两层 `layouts/`，`groups` 的键是 `(rows, cols)`）打包写盘 + `summarize()` 打印 |
| `tactile.build_finger_sensors(scene, *, entity, layouts_by_modality, cfg, decimation, extra_params, track_link_idx)` | 模态注册表 → `gs.sensors.{KinematicTaxel, ContactDepthProbe, ContactProbe, ElastomerTaxel, ProximityTaxel}`；**modality-major 迭代**，每个模态用 `layouts_by_modality[模态]` 那份布局（§4.2 的逐模态网格）取 `probe_local_pos`（+`probe_radius`）与 `num_probes`/`grid_shape`；`history_length=decimation`，`as_grid` 时按该模态自己的 `(ny,nx,3)` 网格传入以启用串扰/FFT 路径。**`probe_local_normal` 只对 `elastomer`/`proximity` 传**（力轴 = 指垫法线，不传则用引擎默认 (0,0,1)，方向错；这两个类有该字段）；`force`/`depth` 的 KinematicTaxel/ContactDepthProbe **不能**传（options `extra="forbid"`，仅 raycast 后端才读该字段）。**多物体（1.2）**：`track_link_idx` 传**全部**物体 link（引擎原文档："indices whose collision geometry is queried by SDF"，复数，天生多压头），`n_sample_points` 相应传**逐 link 列表**——int 会被引擎按 `divmod` 均分（两个物体 = 每个密度减半），列表则每 link 各按自己的预算采。守卫：同一 modality 不得出现两次、句柄名必须唯一（`read_all` 用 name 做 dict key，重名会**静默覆盖**） |
| `tactile.force_6d` / `tactile.marker_2d` / `tactile.channel_format` | 1.6 的派生口径与唯一参数源：`PROBE_GRID` / `PROBE_OFFSET_M` / … 等常量定义在本模块（YAML 只留 `modalities`）；`force_6d((P,3), inner_dir)` 求和并分解为 `[Fn,Ft]`，`marker_2d((P,3), pos, inner_dir, grid)` 投影到切平面 `(P,2)`（基由网格差分 + 法向正交化构造），`channel_format(modality, L, grid)` 给出 `layouts.attrs["channel_formats"]` 的声明。模块级不 import torch（`_extract` 内局部 import，仅 `force_torque` 用） |
| `tactile.build_link_contact_sensors(scene, entity, link_names)` | `Contact` / `ContactForce` 挂两指 link → `contact/link_contact`、`contact/link_force` |
| `tactile.read_all(handles, temporal_reduction)` | 子步 history 轴（axis 1）median/last 降噪；`rotate_to_world()` 用 `gu.transform_by_quat` 旋到世界系 |
| `temperature.build_finger_temperature_sensors` / `build_object_temperature_sensor` | 指尖 + 物体**双侧**网格；一个 scene 内**所有** TemperatureGrid 共享同一份 `properties_dict` 与共享元数据（ambient/convection/simulate_all 一次设定，见 §2.2）。**多物体（1.2）**：每物体一个物体栅格，句柄名 `temperature_object_<i>`（`name=`/`object_index=` 参数；单物体缺省名仍为 `temperature_object`），`properties_dict` 逐传感器带各自的 link idx |
| `temperature.seed_temperature(sensor, T_c)` | 逐 episode 真值注入：直接写 manager 的 ground-truth 缓冲（无公开 setter），顺序 **reset → seed → step** |
| `temperature.read_temperature(handles, ground_truth=)` / `read_link_temperatures` | 按 policy 率读取；`read_ground_truth()` 取物理量、`read()` 取含滞回/噪声的测量值 |
| `sim/obs.py: ObsBuilder` | 组装 §4.3 全部通道（policy 率 `build()` / sim 率 `fast_payload()`），是 L1/L4 的唯一读取入口。`tactile()` 按**模态自己的布局**迭代：`force` 把该指探针力求和后走 `force_6d` → `(L,6)`；`elastomer` 走 `marker_2d` → `(L·P,2)`；其余 legacy 模态仍 `reshape(n_probes, FEATURES_PER_PROBE[模态])`——形状对不上是**显式报错**而不是静默重排；`link_pose()`（位姿记录与旋转参考，排在 `tactile()` 之前）输出 `(L,7)`；`object_pose()` 输出 `(N,7)`；1.6 不再有 `height_map()`；`temperature()` 读入全部 `temperature_object_<i>`，`object_grid`/`object` 取**目标**那份、`objects_grid` 堆全部（1.2）；`cameras()` 渲染 RGB；`fast_payload()` 按 `burst_modalities` 过滤 |

> **触觉形态（format 1.6，2026-10-20）**：唯一的任务 config（`pick_cold_can.yaml`）用
> **`[force, elastomer]`**。force 的 4×3 探针只用于求和成每指一个 6D 合力；
> elastomer 的 16×16 场投影到切平面成 2D。`depth` / `proximity` / `force_torque` 仍是
> 受支持模态，只是当前配置没有；contact 检测始终可用 `contact/link_contact`
> （`build_link_contact_sensors` 建的 link 级传感器，**不是探针**）。
> 1.6 之前的 per-config 形态（`[force, depth, elastomer]` 16×16、pair config 的
> `[elastomer]` 9×7 + height_map）保留在 §9-R25 作为历史记录。

> 接触热流没有 `read_heat_flux`：已确认 Genesis 不暴露该中间量（§4.7-L2），
> 需要时由 `tools/` 从温度网格离线差分推导。

> **Genesis 1.4.1 名字约定（Gate 0 已实测）**：`gs.sensors.X`（`X` =
> `TemperatureGrid`/`ContactProbe`/…）是 **Options 类**，只描述构造参数；
> `scene.add_sensor(opts)` 返回的运行时对象是 **`XSensor`**
> （`genesis.engine.sensors.*`），`read` / `read_ground_truth` /
> `link_temperatures` 都在它上面。另外 GPU 后端下引擎 getter
> （`get_pos` / `get_quat` / `get_qpos` / `read()`）返回 **CUDA torch 张量**，
> 转 numpy 必须走 `.detach().cpu()`（`obs._np`、`gate1._np` 已封装）。

### 5.3 `sim/scene.py` — 场景构建（已实现）

- `build_scene(cfg, seed, show_viewer=False) -> SceneBundle`：一次构建
  地面 + 桌面 + 物体（`cfg["objects"]` 逐个：`shape: cylinder` → `gs.morphs.Cylinder`，
  `box` → `gs.morphs.Box(size=(2r,2r,h))`；位置 = 桌面中心 + `place_xy`，材质取
  `cfg["material_list"]` 中同下标的条目，物性写入
  TemperatureGrid 的 `properties_dict`）+ 机器人（`merge_fixed_links=False`）+ 手指探针
  传感器 + 温度网格 + 接触传感器；返回 `SceneBundle`（scene / 传感器句柄 / 布局 /
  名称表 / 逐物体规格 / `target_index`）。
  `bundle.layouts` 是**主模态**（`modalities[0]`，即 force 的 4×3）那份布局，
  `bundle.tactile_layouts` 才是 `{模态: layouts}` 全表——前者仍按"按指分组"的语义
  被 `obs.py`、指尖温度网格、gate1 标定依赖，逐模态的传感器装配读后者。
  **多物体（1.2）**：`bundle.objects` / `bundle.object_specs` 是全部 N 个；
  `bundle.obj` / `bundle.object_spec` 是**属性**，返回 `[target_index]` 那一份——
  专家、成功判据、gate/probe 的既有调用点因此零改动地跟随目标。
- `bundle.reset_episode(ep)`：**不重建场景**——`ep` 是 `sample_episode` 返回的一集
  字典（逐物体 xy 偏移 / 偏航 / 温度 + `target_index`），
  **所有**物体经 root DOF 摆到桌面上的目标位姿（位姿在构建期以单位位姿冻结，
  温度传感器 link 不变），逐物体 `temperature.seed_temperature` 注入初始温度，
  机器人回 `init_qpos`；随后必须 `step()` 一次（warm step）才可读。
- `bundle.step()` / `object_pose()` / `object_pose_of()` / `object_poses()` / `joint_state()`：
  仿真步进与状态查询；`object_poses()` 返回 `(N,7)`，`object_pose()` 返回目标的
  `(pos, quat)`，`object_pose_of(i)` 按索引返回某一个物体。`sample_episode(cfg, rng)`
  采样全部物体的 xy 偏移 / 偏航 / 温度 + `target_index`（episode rng 均匀抽取；**N=1 时不抽**，
  随机流与单物体时代逐位相同）。
  采样后过**最小间距守卫**：任两物体表面距 < `min_object_gap_m`（默认 0.02 m）
  时重抽 jitter（≤5 次，兜底回基准位）。
- 相机（Gate 3 已实现，见 §8）：`build_cameras()` 在 `scene.build()` **之前**注册
  `cameras: {agentview, eye_in_hand}`；渲染后端不是 YAML 键，而是 `sim/obs.py` 的
  `RENDERER` 模块常量（`raster`；`cfg.cameras.renderer` 是返回它的只读 property）；
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

- **任务规格 = `sim/configs/*.yaml`，只从 YAML 加载**（无独立 `sim/tasks/` 包，
  也没有 dataclass）：`load_task_config(path)` 读 YAML 后做**严格校验**（逐节键集合、
  类型与交叉引用，未知键直接报错），返回普通嵌套 `dict`；所有模块直接
  `cfg["section"]["key"]` 取值。顶层键：`task_id`、`language_template`、
  `material_properties`、`material_list`（每个物体一个材质）、`objects` **列表**
  （每项按 `shape` 校验键集合 + `place_xy` 桌面槽位）、`min_object_gap_m`、
  `target_index`，以及 robot / controller / expert / tactile / temperature / table /
  tray / recording / cameras 与 `max_episode_steps`、`success_criterion`、
  `stable_steps`、`tray_rest_tol_m`、`seed`。`min_object_gap_m` 给多物体的最小表面
  间距守卫定阈值；`target_index` 固定任务目标。
- **成功判定**（在 `runner.EpisodeRunner` 内实现）：`success_criterion: place_in_tray`
  —— 任务目标物体**在托盘腔内**、底部落在腔底 `tray_rest_tol_m` 以内、夹爪已张开、
  专家已 DONE，并**连续保持** `stable_steps` 个 policy 步（默认 10 步 = 0.5 s）；
  达成的那一步 reward=1（稀疏），最后一步 done=True。
- `InitStatePool`（固定初始状态池）**未实现**：初始状态由 episode seed 复现
  （`np.random.default_rng(seed)` → xy 偏移 / 偏航 / 温度），demo attrs 记录真值。
  评测若要 LIBERO 式固定初始状态，可在 v2 加入。
- **v1 任务清单**：`pick_hot_cylinder`（已实现）、`lift_object`（已实现，最小任务）、
  `lift_object_pair`（1.2：一桌两物体、圆柱或立方体按 episode 二选一，
  `sim/configs/lift_object_pair.yaml`；box 走**面抓**——专家把接近偏航对齐到
  物体的**当前面**：抖动先量化到 90° 整数倍（选哪一面），再加物体自身 yaw
  （见 §5.6）。`place_hot_object` / `push_hot_object` 为规划项（见 §11 Q4）。

### 5.5 `sim/expert/actions.py` — 7 维动作控制器（已实现）

- `DeltaPoseController.step(a7)`：裁剪 → `target_pos += a[:3]*pos_scale`、
  `target_quat = quat_apply_delta(target_quat, a[3:6]*rot_scale)` →
  `robot.inverse_kinematics(link=panda_hand, pos, quat, local_point=tcp_offset,
  init_qpos=当前 qpos, damping, return_error=True)` → 臂关节 `control_dofs_position`
  （Genesis PD）→ 指关节按协同标量插值。返回 `ActionInfo`（IK 位置/姿态误差、指目标、是否裁剪）。
- `tcp_pose(bundle)`：`panda_hand` 位姿 + `tcp_offset` 旋转 = 夹爪中心（TCP）。
- `normalized_action_to(...)`：脚本专家用（给定目标位姿反解归一化动作）。
- `sim/rotations.py`：自实现的 wxyz/Hamilton 四元数工具（`quat_mul/rotate/rotvec` 等），
  与 `genesis.utils.geom` 在 Gate 1 交叉验证——不依赖引擎的广播约定。
- 约束：控制器**只消费归一化动作**，不读传感器——保证与下游策略部署接口一致。

### 5.6 `sim/expert/expert.py` — 脚本专家（已实现）

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
- **安全转场（`PipelineConfig.transit_clearance_m`，默认 `0.10`，§9-R43）**：>0 时 APPROACH
  拆成「原地竖直升到 `max(桌面+最高物体, 托盘沿) + clearance` → 在该高度横移 → 竖直降到
  approach 点」三段，任何横移都不在桌面高度发生；释放后换候选、`rejudge` 重摸同样走这条路径
  （`_advance(APPROACH)` 重置子阶段）。默认 0.10 即 pick_cold_can 的取值；0 = 旧直线路径。
- 专家只消费 policy 率观测字典（不读引擎），输出归一化 7 维动作，
  经与策略相同的 `DeltaPoseController`（**专家与策略同接口**）。
- **旋钮分层（2026-10-19 重构）**：任务 YAML 的 `expert:` 只剩**任务契约** 4 个键
  （`approach` + `temperature_judge_enabled` / `temperature_candidate` /
  `temperature_judge_threshold_c`）；其余全部管线常量集中在冻结的
  `sim/expert/pipeline.py::PipelineConfig`（单例 `PIPELINE`：接近/转场高度、抓取抖动、
  CLOSE/SQUEEZE 时序、抬升/放置余量、温度窗口/基线/相对落差、λ 窗口与阈值等）。
  需要时可用 `dataclasses.replace(PIPELINE, ...)` 在进程内覆写并注入
  `EpisodeRunner(pipeline=...)`，不改 YAML、也不改全局单例。
  控制器尺度（`pos_scale_m`/`rot_scale_rad`）属于数据集动作空间，放在 YAML 的
  `controller:`，不属于管线调优旋钮。
- **`PickByTemperatureExpert`（λ 材质判定，§9-R38；该判定路径已在 2026-10-21 的 YAML→dict
  清理中删除——YAML 已无法表达 `requested_groups`，以下阶段机描述保留为历史记录）**：阶段机为
  `APPROACH(0)→DESCEND(0)→CLOSE(0)→JUDGE(0) ─┬ 判对 → LIFT→HOLD→DONE`
  `└ 判错 → RELEASE → APPROACH(1)→…→JUDGE(1)→LIFT`。判据在**抓取姿态保持不动**时读取
  `PIPELINE.judge_settle_s`（默认 0.15 s，跨过引擎接触记账与传感器 RC 的暂态）之后的
  `PIPELINE.judge_window_s`（2.0 s）实测指尖轨迹，两指各拟一次取均值（见 `sim/expert/thermal_judge.py`）。
  ①候选下标 `cand` 是**专家自己的**，`bundle.target_index`（跟随成功判定与 attrs）绝不改动；
  抓取位姿数学从 `PickExpert.reset` 抽成 `_set_goal(bundle, index, rng)`（既有两个任务逐位兼容）；
  ②每回合**永远从 object 0 试起**，而槽位是逐回合置换的（§5.3），所以"先试哪块"与"哪块是
  请求的组"解耦——这正是演示可学的前提；③第二次抓取**同样**走满判据窗（行为差异只有"没放回去"，
  不泄露停留时长），且 50% 的双抓回合白送一对同场景标注；④抓到最后一个候选时**必须抬走**
  （无路可退），判错即 `success=False` 落 `*_failed.hdf5` —— 混淆矩阵是 attrs 的免费产物。
  全程只消费 policy 率 obs（`obs["temperature"]["fingertip_grid"]` = **measured** 通道），
  不读真值、不读材质表。
- **`sim/expert/thermal_judge.py` —— 判据本体（纯 numpy，不 import genesis，任意纯 numpy 工具可直接用；
  当时的出图脚本 `scripts/report_figures.py` 已于 2026-10-20 删除，§9-R46）**：把 `T(t) = T_c + (T_0 − T_c)·e^{−λ(t−t_on)}` 在 `[t_on+settle, t_on+settle+W]`
  上做二参数拟合——`T_c` 是物体侧温度的**播种值**（网格搜索），`T_0` = onset 前 `0.15 s`
  的基线均值，λ 由**跨 15–85% 幅值带**的两点闭式解给出（只在带内取点，避免饱和段把 λ 压小）。
  返回 τ̂/λ̂/R²/样本数，样本不足或 |幅值| < `min_span_k`（0.5 K）时返回 `None`（宁缺勿滥）；
  `judge_fingers` 逐指给 onset（**两指 onset 实测可差 0.25 s**，共用一个 onset 会把慢指 τ
  偏置 ~45%）、池化两指拟合；`decide(λ̂, threshold)` → 组。与
  `render_grasp_thermal_video.py::fit_rise_tau`（该脚本已于 2026-10-20 删除，§9-R46；口径保留作对照）的区别是**有意**的：那个是离线整段拟合
  （极值后全网搜渐近线），本模块是固定窗在线估计（由物体通道播种 T_c、判据只用实测通道），
  两者窗口策略不同，强行合一有改动已发布 R34/R37 τ 数字的风险。
- **冷热判定模式（`expert.temperature_judge_enabled`，§9-R42，2026-10-19；契约重构同日）**：
  不是材质判定——没有 λ 拟合、没有 `material_group`、没有 `requested_group`，问题只是
  "这一块是不是目标温度（冷/热/一个区间）"。`PutInTrayExpert` 的 JUDGE 相因此换成
  **触碰后判断（全部候选都判，命中即搬运）**：`_order_candidates` 取 YAML 的
  `temperature_candidate`（本 config `[0, 1]`），
  **只要有多个候选就每回合用回合 RNG 重新置换**（没有开关），"先摸哪一罐"与"哪一罐是目标"解耦；
  `_on_grasped` 从 CLOSE 进 JUDGE，`_act_judge` 分派到 `_act_temperature_judge`，全程只消费
  policy 率 obs（`obs/temperature/fingertip_grid` = **measured** 通道），不读真值、不读材质表。
  - **目标区间**：`temperature_judge_threshold_c: [low, high]`，`null` = 开区间端。
    `[null, 15]` = 冷目标、`[45, null]` = 热目标、`[10, 20]` = 双侧带。
    区间内、且趋势不会在一个趋势窗内把读数带出区间、且该指的趋势窗已满（有斜率），
    才算命中——没有相对臂，也没有固定窗口。
  - **触碰逻辑（事件驱动，2026-10-21 改；取代 R42 最初的 1.5 s 定窗）**：合爪后停在抓取位姿，
    **每个策略步**重新读一次实测指尖轨迹，按顺序问三个问题：
    ① 任一碰上的指，其 `reading`（末 `PIPELINE.temperature_read_tail_s` = 0.15 s 的均值，
    压住 0.05 °C 噪声与 0.1 °C 量化）落在目标区间内 **且** 趋势不把读数带出该区间
    （`reading + slope × trend_window_s` 仍在区间内）⇒ **立刻** `_on_positive_verdict()`；
    趋势窗未满（无斜率）的指**不算命中**：区间内但方向未知只是"现在在里面"，不是到达；
    ② **没有任何**碰上的指还能到达该区间（每个已有斜率的指都平直/反向，或读数在区间内但
    一个趋势窗后就会离开）⇒ **立刻**原地 `RELEASE`，换下一个候选；单指的"否"不能否决
    另一指的"未定"；
    ③ 否则继续保持、继续读（"等待逻辑"）。
    趋势 = 该指 **onset 之后**最近 `PIPELINE.temperature_trend_window_s`（0.4 s ≈ 8 个策略步）
    样本的最小二乘斜率；样本不足（刚碰上的指）时这一票为"未定"：既不否决别的指，也不算命中。
    "能到达" = 冷目标斜率 ≤ −`PIPELINE.temperature_trend_min_k_per_s`（0.25 K/s）、
    热目标 ≥ +该值、双侧带按 reading 在带的上/下方取降/升——**平直的指垫算"到不了"**，
    所以 35–40 °C 的孪生罐在趋势窗一满（且没有晚碰上的指仍"未定"）就会被否掉，
    不必等超时。
    正证据优先仍然成立，但"进区间"必须**不是穿过**：2026-10-21 首次遍历冒烟抓到反例——
    搬完冷罐的指垫只有 ~2 °C，再摸 55 °C 孪生罐时读数在 +52 K/s 的升温中掠过 9.97 °C，
    旧规则（区间内即命中）把它当成冷目标又搬一块，`all correct: False` 且托盘里两块重叠
    （见 §9-R42 第二次修订）。现在同一读数 +52 K/s 的投影为 30.8 °C > 15 ⇒ 判"离开"并释放。
    baseline 仍逐指取自己 onset 前 `PIPELINE.temperature_judge_baseline_s`（0.3 s）的均值，
    只做记录诊断，不再参与判据。
    绝对区间覆盖"指垫本来就冷"的情况（实测 seed 3 第二摸 reading 1.13 °C；seed 2 一根指
    −11.77 °C、另一根 18.57 °C / Δ −2.42 K——后者按 1.5 s 平均仍有约 −1.6 K/s，高于
    0.25 K/s 下限，不会误否决；单指接触不良时正证据的 OR 仍然成立）。
  - **分支（2026-10-21 第二次修订：全部候选都判、命中即搬运）**：判中 → `_on_positive_verdict()`
    （此任务 = CARRY → LOWER → DROP → RETREAT），托盘**首个落点占中心**、之后的落点按
    `PLACE_SLOTS` 沿 ±x 错开 0.055 m（手指沿世界 y 开合，x 向错开才不会让第二次下爪的指垫
    压到第一块；两枚 25 mm 半径物体需 ≥50 mm 心距，而 17 cm 内腔最多只允许 60 mm——
    0.045 m 曾重叠 5 mm 并让一次 LOWER 卡死，见 §9-R42 第二次修订）；撤退完成后**继续判
    下一个尚未摸过的候选**。判不中 → **不抬升**，从同一触碰位姿 `RELEASE`，回到 APPROACH
    摸下一个候选。**最后一个候选也不再强制提交**：没有下一块时 RELEASE / RETREAT 结束于
    `DONE`，被否掉的块留在桌上。整个保持期没有任何接触 onset ⇒ 无测量：
    `PIPELINE.retry_on_none` 允许时重摸同一块，否则按"不中"走同一候选机器——**绝不进入
    CARRY/LIFT**。一集因此可搬运 0、1 或多块：凡命中（且不是穿过区间）的候选都搬走，未命中的
    都留下。
  - **成功判定不变**：仍只认 `target_index`（0 号冷罐）在托盘内，且要求专家已 `DONE`（= 全部
    候选判完）。判错（把非目标搬进托盘、或目标在最后被否）则该集失败并落 `*_failed.hdf5`，
    成为可查的拒绝/错误样本，而不是被藏掉的错。
  - **记录与材质判据分开**：`temperature_log` 独立成块；`problem_info.temperature_judgement`
    写协议/统计量/候选/顺序/阈值区间/目标类型/趋势窗/baseline/读尾，每个 demo 的
    `temperature_judge`（JSON）逐次触碰记 `cand`/`body`/`verdict_is_target`/`reason`/
    `measured_c`/`baseline_c`/`delta_c`/`slope_c_per_s`/`per_finger`/`threshold_c`/
    `target_kind`/`hold_s`/`trend_window_s`/`trend_min_k_per_s`/`onsets_s`，另有
    `temperature_judge_touches`、`temperature_judge_first_body`、`temperature_judge_correct`。
    注意区分两个键：专家的**实测判定**叫 `verdict_is_target`，runner 落盘前补的**真值标签**
    才叫 `is_target`。

### 5.7 `sim/recording/` — EpisodeRunner 与 HDF5Writer（已实现）

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

### 5.8 `sim/tools/` — 校验与消费

| 工具 | 状态 | 功能 |
|---|---|---|
| `sim/tools/validate.py`（`scripts/validate_dataset.py`） | ✅ 已实现 | schema/形状/dtype 校验；必需通道（`obs/ee_pos\|ee_ori\|gripper_states\|joint_states` + 至少一个 `tactile/` 或 `temperature/` + actions/dones）；`num_samples == T`、dones 末位为 True、NaN 扫描；`generate.py` 结束时自动跑一遍 |
| `sim/tools/tactile_field.py` | ✅ 已实现（2026-10-04） | **marker 场的读取契约**，纯 h5py/numpy（与 `validate.py` 同层，不 import genesis）：`modality_grid(s)`（逐模态网格，缺 `modality_grids` 回退根 `probe_grid`）、`finger_geometry`（`probe_local_pos`/`probe_local_normal`，**按网格取**——根节点那份属于主网格）、`read_markers(file, demo, 模态, frame="world"\|"link")`、`marker_positions`（参考 + 位移 = Tabero 的 `M = [参考帧, t_1..t_H]`）、`pad_normal_field`（投影到指垫法线的高度场）、`as_tabero_sequence`、`world_frame`。世界系文件需要 `obs/link_pose`（`quat_to_matrix` 的转置做逆），缺了就报错而不是静默给错值 |
| `sim/tools/height_map.py` | ✅ 已实现（2026-10-08，format 1.3） | 稠密高度图的**合成与读取契约**，纯 h5py/numpy（scipy 延迟 import，dataloader 可直接用）：`pad_normal_planes`、`upsample_field`/`synthesize`（cell-centred 样条/双线性重采样，采样坐标先 clamp 到标记范围——最外半格严格平延、无 ringing——再把结果 clamp 回样本范围）、`height_map_meta`/`read_height_map`、`synth_from_markers`（任意带 marker 场的老文件现场合成）、`height_map_or_synth` |
| `sim/tools/gel_render.py` | ✅ 已实现（2026-10-04） | GelSight 照相模型（高度场 → 法向 → 三色 LED），从 `render_video.py` 抽出共用：`gelsight_image`、`gel_field`、`gel_slope`、`grid_m_per_px`、`lattice_nodes`、`pad_tangents`、`smooth`。**只渲染已记录的场，不新增空间细节** |
| `scripts/render_video.py` | ✅ 已实现 | 把某个 demo 的**落盘 RGB** 渲成 MP4：双相机并排（nearest 放大）+ 叠加步数/夹爪/物体与指垫温度/接触时刻标尺；直接读文件像素而非重仿真。默认 20 Hz = 实时。传感面板细节见下 |
| `scripts/judge_temperature_material.py` | ⛔ 已删除（2026-10-20，§9-R46；2026-10-14 实现，§9-R38） | **盲判离线复算**：只读落盘 `obs/temperature/fingertip_grid`（measured）+ `obs/contact/link_contact`（接触由这一路自行分段，逐指取各自 `t_on` 上升沿），跑 `sim/expert/thermal_judge.py` 重算每条抓取的 λ̂，再与 attrs 里专家当时记的 `judge` 逐条比对——真值只用于打分、**从不作输入**。打印混淆矩阵、两组 λ̂ 带（含按温区拆的冷/热两列）、分离度与两侧余量、不可判数、指间分歧、文件记录 vs 复算分歧、逐集成功；`--sweep` 给几何阈值扫描（假收/假拒 + 零错窗口）；另带捷径审计 `channel_audit`（接触**前** `objects_grid` 梯度增长率，按 \|T−ambient\| 归一）。用法 `python scripts/judge_temperature_material.py <file.hdf5> [--sweep] [--json out.json]` |
| `replay.py` | ⏳ v2 | 读全量 `states` + `actions` 重仿真，对比 obs 在容差内一致 → 抓录制 bug 的最强校验；前提是 `states` 落盘（v1 未写） |
| `tactile_render.py` | ⏳ v2（部分被 `render_video.py` 覆盖） | 由 `obs/tactile/*` + `layouts/` 重算探针世界坐标，渲染触觉场动画。剩下的独有价值是**世界坐标系**下的三维布点（把 `probe_local_pos/normal` 经正运动学变换后画进场景） |
| `analyze_contact.py` | ⏳ v2 | 由温度网格差分离线推导接触热流（§4.7-L2），输出事件级汇总 |
| `stats.py` | ⏳ v2 | 数据集统计报表 / 直方图（成功率、温度分布、事件数分布） |

**`render_video.py` 传感面板**（`--no-sensing` 关）：左右指 4×3 `tactile/force` 幅值图（带数值）、指尖 4×4 温度网格、物体 4×4×4 体素（4 张 z 切片，按 per-episode 紧量程并标出 span——金属物体近乎等温，不标 span 会把色差误读成温差）。文件含 `tactile/proximity` 时额外多画：左右指 4×3 proximity 幅值图（独立色标 `RAMP_PROX`，**不与 force 共量程**——两者标定不同，§9-R23），以及右下角两行「每指 Σ\|F\| — proximity（青色=首次非零）vs force（红色=接触门打开）」时间曲线（横轴即整条 episode，当前步有白色游标）。缺该通道时自动退化为只有 force 一行。最右侧是**凝胶压印面板**（GelSight 风格，照相模型在 `gel_render.py`）：接触高度场双三次上采样 + 高斯模糊后求梯度得表面法向，再用三色 LED（`GEL_LIGHTS`，窄镜面瓣 + 弱漫反射）着色——色相 = 倾斜方向。**每个通道按自己的网格 reshape**（`layouts/modality_grids` → `tactile_field.modality_grid`，缺该 attrs 的老文件回退根 `probe_grid`），高度源依次取 `tactile/height_map`（1.3 起）、`tactile/elastomer`、12 探针的 `tactile/depth`，**面板标题会写明画的是哪一个**；面板物理尺寸按**网格的 cells** 取（`gel_cells`）而不是数组形状——64×64 稠密图是覆盖 9×7 格指垫的正方形栅格，按数组长宽画会把压印拉变形，同一模型也让 `--markers` 的标记晶格精确落在图上（§4.2 末）。像素尺度由 `layouts/probe_local_pos` 的探针间距推得，故坡度是真实坡度。变换取 `world_frame: true` + `obs/link_pose` → `read_markers(..., frame="link")` **精确**逆变换（标题标 `link frame via link_pose`）；`world_frame: false` 走"已记录即 link 系"；两者都缺则退化为 `\|disp\|`（标题如实写 `world frame, no link_pose`）。真实压痕坡度只有 ~1e-3（12 µm 深 / 12 mm 垫），裸渲染看不见 → **每集算一个统一放大系数**：把峰值帧最陡处拉到 `GEL_TARGET_SLOPE` = 0.6，系数 >1.5 时写进标题（`slope x210`），两指共用同一系数（指间强弱对比是真的）。`--markers` 额外叠**标记晶格 + 位移矢量**：亚像素位移（峰值 0.056 mm ≈ 0.49 px @112 px 面板）自动放大到 `MARKER_TARGET_PX` = 3.5 px/标记，倍数写进标题（`vectors x7`），<1 px 只画淡点。**这只是渲染已记录的场，不新增空间细节**：12 探针源看起来是柔和圆顶，密网格 elastomer 才是真压印。

### 5.9 `sim/gripper.py` — 夹爪几何（独立简化版，已实现）

`sim/sensors/finger_layout.py` 是**管线用**的通用推导器（box/mesh 两条碰撞路径、标定 override、
逐模态网格、HDF5 打包）；本模块是同一夹爪几何的**独立简化版**，只回答三个问题：两指垫面
（中心/切向尺寸/内向，link 系）、闭合轴（hand 系，左指→右指）、marker 栅格（`(P, 3)`，link 系）。

| 项 | 约定 |
|---|---|
| 输入 | Panda URDF 路径（`urdf/panda_bullet/panda.urdf`）；链接名写死（`panda_leftfinger`/`panda_rightfinger`/`panda_hand`），缺链接或无 `.obj` 碰撞网格即报错 |
| 依赖 | 纯 stdlib + numpy，**不 import genesis/torch/任何其它 sim 模块**；`python -m sim.gripper --urdf <path>` 直接可跑 |
| API | `Gripper.from_urdf()`（`.closure_axis` / `.pads` / `.describe()`）；`PadFace.marker_lattice(grid=(9,7))`；常量 `PAD_TOL_M`/`PROBE_OFFSET_M`/`MARGIN_FRAC`/`DEFAULT_GRID` |
| 不做 | 运动学（关节名/开合位置/力限/PD/TCP 仍在任务 YAML 的 `robot` 节，由 `sim/config.py` 校验）；HDF5 打包；box 碰撞路径；override 旋钮 |
| 双事实来源 | **管线不 import 本模块**（`finger_layout.py` 仍是唯一事实来源）；等价性曾由 `scripts/probe_gripper_model.py` 变成可执行断言（§9-R28；该脚本已在 2026-10-20 的 scripts 清理中删除，§9-R46，断言值保留在 R28） |

---

## 6. 配置体系

```
sim/configs/
└── thermal_recognition/
    └── pick_cold_can.yaml      # 当前唯一任务配置：七体 robosuite 桌面 + 冷罐抓放 + 温度判定（§9-R42）
```

> 下文各任务小节（`pick_hot_cylinder`、`lift_object`、`pick_by_temperature`、`put_in_tray`、
> `cubes_tabletop`、`robosuite_tabletop` 等）保留**设计与实测记录**；那些 YAML 已删除
> （场景与任务并入上面的唯一配置），复现命令中的路径属于历史记录，当前可运行的只有
> `pick_cold_can.yaml`。

当前唯一配置 `pick_cold_can.yaml` 的每个键都由 `sim/config.py` 严格校验
（`load_task_config` 逐节检查键集合与类型，**未知键直接报错**；`tactile`、`temperature`
只暴露任务级选择，其余参数是 `sim/sensors/{tactile,temperature}.py` 与 `sim/obs.py`
的模块常量）。下面是多任务并存时期的历史示例，键名已不代表现状（现状见 YAML 与
`sim/config.py` 的校验表）：

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
expert:                                          # 任务契约（其余管线常量见 sim/expert/pipeline.py）
  approach: place_in_tray                        # APPROACHES 注册表键；拾取后放入托盘
  temperature_judge_enabled: false               # 打开触碰式温度判定（§9-R42）
  temperature_candidate: []                      # 判定候选物体下标；多个时每回合自动随机置换
  temperature_judge_threshold_c: [null, 15.0]    # 目标区间，null = 开端；冷=[null,15], 热=[45,null]
tactile:
  modalities: [force, elastomer]                 # 1.6：6D 合力 + 2D marker；其余参数在 sim/sensors/tactile.py
  # 常量（不再暴露到 YAML）：PROBE_GRID {default: 4×3, elastomer: 16×16}、PROBE_OFFSET_M、
  # PAD_TOL_M、PROBE_RADIUS_M、ELASTOMER_PARAMS、AS_GRID、CONTACT_LINKS、
  # TEMPORAL_REDUCTION=median、WORLD_FRAME=false
temperature:
  finger_grid: [1, 1, 1]                         # 官方最粗栅格 = 每指单格集中温度（§9-R36）；不塌缩任何轴
  object_grid: [4, 4, 4]                         # 物体
  ambient_temp_c: 21.0
  # 其余温度参数全部是 sim/sensors/temperature.py 的模块常量（唯一真源）：
  # CONVECTION_COEFFICIENT=0.0（§9-R14）、SIMULATE_ALL_LINK_TEMPERATURES=true、
  # SENSOR_TIME_CONSTANT=0.05（物体保持 0；τ 扫描定案 §9-R36）、NOISE=0.05、
  # RESOLUTION=0.1、CONTACT_DEPTH_WEIGHT=0.0（§9-R16）、
  # CONTACT_AREA_POLICY="pad_face"（全数据集定案 §9-R37）、
  # RECORD_GROUND_TRUTH=false（录 read() 实测通道）、
  # FINGER_PROPERTIES（k=0.25 物理硅胶值，材质对比经 k_eff 带出 §9-R16/R35）、
  # ENVIRONMENT_PROPERTIES
object:
  shape: cylinder
  radius_m: [0.015, 0.025]                       # 构建期冻结（每文件一次）
  height_m: [0.05, 0.08]
  mass_kg: [0.05, 0.15]
  temp_c: [60.0, 200.0]                          # 逐 episode 采样
  xy_jitter_m: 0.04
  yaw_jitter_rad: 3.14159
  # 仅渲染用的表面色（RGB 0-1，同 `table.color`）：挂在视觉 surface 上，刚体物理与
  # 温度体素化都不读它（实测：同一场景开/关颜色，物体温度、位姿、actions 逐位一致——
  # 唯一抖动的指尖通道在"同一配置跑两遍"之间也抖同样多）。null = 引擎默认灰白；
  # 多物体场景每个 body 一个色即可区分，而 `pick_by_temperature` 的孪生体必须留空
  # （外观不可分是该任务的前提，§9-R38）。
  color: null
table:
  pos: [0.4, 0.0, 0.0]
  size: [0.6, 0.8, 0.04]
  height_m: 0.4
  # 仅渲染用的表面色（RGB 0-1）；null = 引擎默认灰白（已删除的 format 1.0 语料就是这个，§9-R31）
  color: [0.55, 0.38, 0.22]
recording:
  output_dir: datasets/v1.5/hdf5
  keep_failed: true
  compression: gzip
  compression_opts: 4
  contact_burst: true
  burst_pre_s: 0.05
  burst_post_s: 0.25
  burst_max_windows: 10
  burst_modalities: null                         # null = 全部模态（含 16×16 marker 场，1.1–2.2 MiB/集）
cameras: {enabled: true, resolution: [256, 256]}  # 置 true 即启用双 256² 相机；后端 = sim/obs.py 的 RENDERER（§8）
max_episode_steps: 400                           # 20 s @ 20 Hz
lift_success_m: 0.05
stable_steps: 10
seed: 0                                          # episode i 用 seed+i
```

- 输出目录默认取 `recording.output_dir`，`scripts/generate.py --out` 可覆盖；
  查看器等运行期开关走命令行（`--viewer`）；渲染后端是 `sim/obs.py` 的 `RENDERER` 常量
  （探针脚本在 `build_scene()` 前覆写它）。
- 传感器参数文件（`sim/configs/sensors/*.yaml`）不是必需：探针布局由 URDF 程序化推导，
  触觉参数直接放在任务 YAML 的 `tactile:` 段。

**多物体任务（1.2，`sim/configs/lift_object_pair.yaml`）**：`objects:` 列表取代单
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

**网格物体（`shape: mesh`）**：物体也可以是真实资产（OBJ + MTL + 贴图）；
`robosuite_tabletop.yaml` 的 can / lemon / milk 就是三个网格体（见下）。引擎自己加载
OBJ + MTL + 贴图，**留空 `color` 即保留资产本色**。用到的资产随仓库放在
`sim/assets/`（URDF 在 `sim/assets/urdf/`，物体在 `sim/assets/robosuite/`），
`mesh_file` 写相对 `sim/assets/` 的路径；`sim/scene.py::resolve_mesh_path`
在运行时按「原样 → `sim/assets/` → Genesis assets」的顺序解析，因此仓库是自包含的，
配置里不再出现机器绝对路径。

```yaml
objects:
  - shape: mesh
    mesh_file: robosuite/models/assets/objects/meshes/lemon.obj  # 相对 sim/assets/
    mesh_scale: [1.5, 1.0, 1.0]    # MJCF scale="1.5 1 1"：非均匀，柠檬才是椭球
    place_xy: [0.12, -0.20]
    radius_m: [0.0338, 0.0338]     # mesh 时只是被抽掉占位，见下
    height_m: [0.0400, 0.0400]
    mass_kg: [0.15, 0.15]
```

- **尺寸以网格为准**：`radius_m`/`height_m` 仍按分布抽一次（随机流形状与基元一致），
  随后被 `measure_mesh` 实测的 AABB 覆盖；`sample_object_spec` 把它写进
  `ObjectSpec.aabb_min/max`，放置、元数据、槽位守卫都读实测值。
  质量独立于网格（`mass_kg` 直接给；密度 = mass / 网格体积，水密体积失败时退回凸包体积）。
- **`mesh_scale` 可以是 3 元组**（2026-10-15 加）：按轴缩放，语法与语义同 `gs.morphs.Mesh(scale=)`
  （"If a 3-tuple, it scales along each axis"），给非均匀缩放的资产用 —— robosuite 的柠檬在它自己的
  MJCF 里就是 `scale="1.5 1 1"`，不按轴缩放就只是个 45 mm 的圆球、不是柠檬。
  `measure_mesh` 的**标量路径逐字保留**（`float(scale)` 乘法 + `**3` 幂）⇒ 既有 config 的
  AABB / 体积 / 密度**逐位不变**（回归实测：改动前后既有场景的五体 r/h 与像素数逐字相同）。
- **旋转约定**（实测确定，非假设）：`mesh_euler_deg` 与 `gs.morphs.Mesh(euler=)`、
  scipy 的 `Rotation.from_euler("xyz", deg, degrees=True)` 三者一致；经非对称网格
  逐一验证（当时的酒瓶，瓶颈在 +z）。因 `reset_episode` **每回合** `set_quat(yaw_to_quat(yaw))`、
  会抹掉 morph 里烘焙的朝向，基础朝向走 `ObjectSpec.base_quat`，
  逐回合与偏航四元数左乘复合（`_quat_mul_wxyz(yaw_quat, base_quat)`）。
- **放置规则不变**：AABB 底面贴 `cfg.table.height_m`、AABB 中心对齐槽位、绕中心转 yaw；
  基元因 AABB 中心 = 几何中心，此式与旧式 `z = table + h/2` **逐位相同**（回归实测）。
- **温度通道对网格物体同样成立（2026-10-14 实测，非推断）**：物体侧的热体素格由引擎按
  link 的**世界 AABB** 切（`genesis/engine/sensors/temperature.py`：`voxel_size =
  aabb_extent / grid_size`），整条温度路径（`sim/sensors/temperature.py`、`sim/obs.py`、
  `sim/recording/`）没有任何按形状分支的代码。单集探针（当时的 LIBERO alphabet_soup 罐，真 OBJ）、
  aluminum、100 °C、34 步成功抓取 —— 指尖 20.9 → **84.2 °C**（+62.6），物体网格几乎不动
  （100.000 → 99.996 °C），`obs/temperature/fingertip_grid` 形状 `(T,2,1,1,1)` 与基元一致。
  **但热质量用的是 AABB 体积而非材料体积**（同代码 `link_volume = (aabb[1]-aabb[0]).prod()`）：
  罐头 AABB 393 vs 实心 284 cm³（1.4×，可忽略），空心体（马克杯、碗）会高估一个量级、
  升降温比真实物体慢得多。接触几何是 coacd 凸分解，薄壁结构只是近似。
- 记录：`env_args.objects[i]` / `problem_info.objects[i]` 只在 `shape: mesh` 时多记
  `mesh_file` / `mesh_scale` / `mesh_euler_deg`；基元文件的记录逐字节不变。
- 复现：`scripts/render_scene.py --config sim/configs/robosuite_tabletop.yaml --res 512`。

**立方体（`shape: cube`）**：单一几何形状的场景 —— `cubes_tabletop.yaml` 是上面那张桌子的
"全立方体"版：槽位与颜色（step 1.5 的五色，按 `place_xy` 逐格对应）不变，五个物体都是边长
**构建期随机**的立方体。

```yaml
objects:
  - shape: cube
    place_xy: [-0.03, -0.20]
    color: [0.78, 0.16, 0.14]      # 纯渲染，同 table.color 口径
    side_m: [0.030, 0.065]         # 边长（全长，米）；cube 唯一的尺寸旋钮
    mass_kg: [0.05, 0.25]
```

- **边长只抽一次**：`radius_m`/`height_m` 是**两次独立**抽样，区间写得再相等也只会在 r = h 的
  巧合上凑出正方体。`cube` 因此是唯一单独走抽样分支的形状：`side_m` 一次抽得 s，拆成
  `radius_m = s/2`、`height_m = s`，正好落进既有的 box morph `(2r, 2r, h)`
  （`sim/scene.py::sample_object_spec` / `build_scene`），体积与刚体密度也随之走 box 分支 = s³。
  非 cube 形状的随机流**逐位未变**（回归实测：同一 seed 下 `lift_object` 的 r/h/m 三个抽值相同）。
- **可夹取区间 [30, 65] mm**：夹爪全开 = 2 × `open_finger_pos` = 80 mm（资产普查用的同一判据）；
  上界压在实际抓过的 ⌀66.1 mm 罐之下 —— 立方体是平壁、没有圆柱的自定心，不能贴着开度上限，
  65 mm 留 ~7.5 mm/侧接近余量；下界让 17.5 mm 见方的指垫（`pad_face` 305.949 mm²）在面上仍留
  ≥6 mm 边。区间是每体一个键，改一行即可。
- **专家口径**：`cube` 与 `box` 同走"只能面朝抓"的判定（`sim/expert/expert.py`：偏航量化到最近的面法向
  + 物体自身偏航），否则合爪落在棱/角上。本步只出图，但这条决定了该场景日后能不能抓。
- 已知取舍：质量仍是独立抽样、不随边长走（仓库既有惯例）⇒ 密度在 ~180–9300 kg/m³ 之间浮动，
  对渲染无影响；若日后变成任务，把质量改成"抽密度 × s³"是一行。
- 复现：`scripts/render_scene.py --config sim/configs/cubes_tabletop.yaml --res 512`
  （`--seed N` 换一套边长）。实测（2026-10-15，512²，raster）：seed 0 五体
  52.3 / 31.4 / 58.5 / 51.2 / 49.0 mm，seed 7 为 51.9 / 57.1 / 40.5 / 30.2 / 57.9 mm（尺寸确实随
  seed 变、颜色不随）；两帧均"活"（std 29.9 / 43.9），五体各占全景 0.28–1.86% 像素，
  z0 = 0.400 + s/2 贴桌。

**robosuite 资产场景（`robosuite_tabletop.yaml`，2026-10-15）**：桌面场景系列第 4 步 —— 换成
robosuite 1.5.2 自带资产（已裁剪并随仓库放在 `sim/assets/robosuite/`，保留上游
`models/assets/{objects/meshes,textures}` 布局），并且第一次把
两组对照放进**同一张桌子**：

- **同一外观、不同温度**：两个可乐罐（6 / 55 °C）、两盒牛奶（55 / 6 °C）—— 同一资产建两遍，
  差别只在指尖测到的东西，"哪一罐是凉的"只能靠摸回答（「温度目标约束」任务族的底座）；
- **同一温度、不同材质**：木立方体 / 金属立方体（都 60 °C）= R38/R39 的**实测**材质对
  （balsa λ̂ 0.278、iron λ̂ 1.484，§9-R39）；
- 柠檬 = 常温对照（第三种形状/尺寸，无热信号），托盘 = 投放目标。

```yaml
material_pairs: [[aluminum, aluminum, lemon, paperboard, paperboard, balsa, iron]]  # 一具一体
objects:
  - shape: mesh
    mesh_file: robosuite/models/assets/objects/meshes/can.obj   # 相对 sim/assets/
    mesh_scale: 1.0                    # robosuite 自己的 MJCF 里 can 没有 scale
  - shape: mesh
    mesh_file: robosuite/models/assets/objects/meshes/lemon.obj # 相对 sim/assets/
    mesh_scale: [1.5, 1.0, 1.0]        # 同上：MJCF scale="1.5 1 1"，非均匀（§6 上面那条）
```

- **尺子统一取 robosuite 自己的 MJCF scale**：can 无 scale ⇒ ⌀50 × 80 mm；milk `0.9` ⇒
  39.7 × 39.7 × 144 mm；lemon `1.5 1 1` ⇒ 67.5 × 40 × 40 mm（实测值，写进各体 `radius_m`/`height_m`，
  槽位护栏与文件元数据读它们）。三条本轮实测：`can.obj` 与 `can.msh` **同尺寸**（"OBJ 是缩小版"的
  猜测不成立 —— robosuite 的 `horizontal_radius` site 0.025 也与网格一致，bottom/top site 是抓取参考
  不是几何）；三个 OBJ 都带 UV + MTL + `map_Kd`（`soda.png` / `ceramic.png` / `lemon.png`）
  ⇒ **渲出贴图**（罐是红的、柠檬是黄的）。这条差点没拿到：贴图一度**静默丢失**、整场渲成灰模
  （两个上游行为叠加：trimesh ≥ 5 的解析器根限制 + Genesis 补丁吞异常），**修法与完整证据链见
  §9-R41** —— 本场景是第一个撞上的，也是 `build_scene` 里 `install_texture_resolver_policy()`
  存在的唯一理由。几何、布局、热学身份与贴图无关（修复前后 px 数逐项相同）。
  三个都是 Z-up ⇒ 不需要 `mesh_euler_deg`（与 HOPE 那些 Y-up 资产不同）。
- **材质**：iron / balsa / aluminum 直接取 R39/R38 已实测的那几个 k；新增 `paperboard`（纸盒壁，
  k = 0.06）与 `lemon`（含水组织，k = 0.50）。两条写进 config 的物理事实：任何金属落**同一条
  k_eff 带**（铝 0.5000 / 铁 0.4984）⇒ 铝罐与铁立方体在指尖上分不开（k_pad 饱和，§9-R35）；
  `paperboard` 0.0968 vs `balsa` 0.0902 只差 **1.07×** ⇒ 纸盒与木块同样分不开 —— 真实物理
  （硬纸板 ≈ 巴尔沙木），建需要区分它们**两者**的任务之前要先知道这一条。
- **per-body 材质**走 `material_pairs[0]`（一具一体、build 期冻死，§9-R38 的机制）。顺带修掉
  `scripts/render_scene.py` 一处**静默错**：它以前无条件传 `materials[0]` 字符串给 `build_scene`，
  于是 `material_pairs[0]` 分支永远走不到、整场被建成单一材质（RGB 上看不出来，热学全错）。
- **布局**：七体两排，世界 y ≤ −0.04 全部让开停住的夹爪（世界 x 0.27–0.35 且 y > 0.05）；
  实算最小表面间隙 **0.051 m**（柠檬–罐 B，护栏 0.02）、托盘净空 **0.236 m**（牛奶 A）；
  托盘在 (0.40, 0.30)。相机沿用 step 1 的 agentview（本场景物体世界 x 跨 0.215–0.620，
  step 1 是 0.18–0.59，同量级），唯一新东西是托盘伸到 y = 0.363，需用眼睛确认它在框内。
- 复现：`scripts/render_scene.py --config sim/configs/robosuite_tabletop.yaml --res 512`
  实测（2026-10-15，512²，raster；修贴图前后各一次、另加一次内存补丁试跑，**三次逐项相同**）：
  两帧均"活"（std **31.80 / 41.74**，贴图前是 31.97 / 45.26），
  active rows 512/512、508/512，七体各占全景 **0.71–2.01%** 像素 —— 罐 A 2.01 / 罐 B 1.95 /
  牛奶 1.89 & 1.61 / 柠檬 0.93 / 金属立方体 0.78 / 木立方体 0.71，全非零 ⇒ 都在框内
  （128² 重跑同一比例到 0.02 pp ⇒ 覆盖率与分辨率无关）。AABB 底面全部贴桌（z = 0.400），
  各体**原点** z 随资产而不同：罐 0.440 / 柠檬 0.420 / 牛奶 0.461 / 立方体 0.420（资产原点
  不在 AABB 中心）。柠檬是唯一会动的一体：置位 +0.5196（`place_object` 对齐 AABB 中心，
  柠檬 AABB 中心偏原点 0.45 mm），20 步沉降后滚到 **+0.518**（`--steps 1` 时仍是 +0.520）；
  其余六体 1 步与 20 步**逐位相同**。托盘不进像素统计（脚本只数 objects）—— 已肉眼确认在框内
  （伸到 y = 0.363）且不挡停住的夹爪。
- **图**：`/tmp/robosuite_scene/{agentview,eye_in_hand,sheet}.png`（命令给 `--out` 即改）。

**托盘（`tray:`）**：一张桌子边上多一个托盘实体。Genesis 1.4.1 **没有复合基元 morph**，
托盘因此是**五个 `fixed=True` 的 box**（地板 + 四面墙），互相**下沉 `fuse_m = 1 mm`** 让
相邻面不共面（共面会让接触求解器在两个面上同时出解，是纸面厚度实体的经典伪影）。

```yaml
tray:
  enabled: true
  place_xy: [0.0, 0.30]        # 内腔中心（桌面相对）；世界系 = table.pos + 此值
  inner_x_m: 0.17              # 内腔 170 × 170 mm，对 ≤65 mm 的立方体
  inner_y_m: 0.17
  rim_m: 0.035                 # 墙高出腔内底面 35 mm
  wall_m: 0.008                # 墙厚 / 底板厚
  floor_m: 0.008
  color: [0.35, 0.35, 0.38]
```

- **几何数**：`floor_z(table) = table.height_m + floor_m − fuse_m`、`rim_z = floor_z + rim_m`；
  桌子 0.40 m ⇒ 腔底 **0.407**、墙顶 **0.442**。外廓 0.186 × 0.186 m，摆在桌面相对
  (0.0, +0.30) = 世界 (0.40, 0.30)：**远端**、距 agentview 轴线约 10°，全程在框内且不遮物体；
  内腔中心距机器人基座 **0.50 m**，与一直在抓的那个槽位半径相同。
- **物体与托盘互不侵占**：`sim/scene.py::_tray_clearance_m` 把每个槽位到托盘**外廓**的距离
  减去该槽位物体自己的外接圆半径，取最小值；`sample_episode` 的重抽条件从
  「物体间距够」变成「**两者都**够」。无托盘时该函数返回 `inf` ⇒ 既有配置的随机流**逐位不变**
  （只是把 `>= min_object_gap_m` 与 `inf` 比）。**保守**：物体按外接圆、槽位按最大
  `xy_jitter_m` 膨胀、对每个 (槽位, 物体) 对都算（因为槽位随后会被置换）。
- **托盘在物体之后建**：`build_scene` 里托盘是最后一批实体，旧配置的实体索引因此不移位。
- **墙高 35 mm 的来历**：立方体半高最多 32.5 mm，墙比它高 ⇒ 放进去的物体骑不出墙；
  同时墙顶 0.442 m 远低于搬运高度（见 §9-R39 的两段式 CARRY），进出腔时指垫不会蹭墙。
- **内腔不缩容做容差**：`tray_rest_tol_m` 只管**离地高度**，横向判据就是内腔本身 —— 曾想过
  把内腔缩一点当横向容差，算下来 65 mm 的方块合法地贴着墙放时中心离腔心 52.5 mm，任何收缩
  都会把这种合法放置判失败，故**去掉**这个旋钮。

**三材质判材质 + 放托盘（`put_in_tray.yaml`，§9-R39）**：五个随机颜色/随机边长的立方体 +
边上一个托盘，指令说材质词，专家**抓到手里靠指尖温度变化判是什么、判错放回去换一块、判对
搬到托盘放下**。三个指令 ⇒ 三个文件（`group_files: true`），每条一次生成：

| 指令（`language_template: "put the {group} cube into the tray"`） | 文件 |
|---|---|
| `put the iron cube into the tray` | `put_in_tray_iron.hdf5` |
| `put the wooden cube into the tray` | `put_in_tray_wooden.hdf5` |
| `put the plastic cube into the tray` | `put_in_tray_plastic.hdf5` |

- **`{group}` 不是 `{material}`**。指令里的词（`iron`/`wooden`/`plastic`）与热学材质名
  （`iron`/`balsa`/`nylon`）是**两套词表**：`objects[].material_group` 给出前者，
  `material_pairs` 按位给出后者。这样"木头"可以落在导热率合适的任何一种木头上，
  而不必把指令绑在 `balsa` 这个词上。
- **一种布局、三个文件**：`material_pairs: [[iron, balsa, nylon, iron, balsa]]` 一个五元组，
  按位铺到五个 `objects`（组依次 `iron / wooden / plastic / iron / wooden`）。物性在
  **build 期冻结**（R38），所以每回合的多样性来自**槽位置换**而不是换物性。
- **新旋钮（2026-10-19 起均属 `PipelineConfig`，YAML 只留 `expert.approach`）**：
  | 键 | 默认 | 作用 |
  |---|---|---|
  | `judge_lambda_thresholds` | `()` | 非空 = 多带判据：N 个材质要 N−1 个边界，**按 k_eff 从慢到快**排 |
  | `judge_min_span_k` | `0.5` | 拟合窗内指尖变化的**下限**（K）；低于此判为"没真碰上"，拟合作废 |
  | `retry_on_none` | `0` | 拟合作废时**重抓同一块**的次数上限（0 = R38 老机器，逐位不变） |
  | `judge_method` | `"fit"` | `"ratio"` = 消掉渐近线的估计量（§9-R40）：λ = ln(s1/s2)/Δt，窗可短到 1.5 s；`fit` 保留给 R38 的窗口 |
  | `judge_min_late_window_frac` | `1.0` | 判定窗口从**先碰上的指垫**起算的闸门：后手仍保有 `W×(1−frac)` 才前移。**1.0 = 关**（旧行为逐位不变），`put_in_tray` 开 `0.4` |
  | `approach` | `"place_in_tray"` | YAML 契约键：选 `PutInTrayExpert`（判对后搬运入托盘）；`APPROACHES` 注册表 |
  | `place_carry_clearance_m` | `0.10` | 搬运高度 = max(墙顶, 桌面+最高物体) + 此值 |
  | `place_carry_step_m` | `0.02` | CARRY 横移每步上限（米）。不设上限时目标直接跳到托盘，动作饱和到 `pos_scale_m`≈1 m/s，0.35 kg 的罐子会在加减速时从指垫滑出 |
  | `place_descend_tol_m` | `0.010` | 下降到位判据（再小会被 IK 残差卡住） |
  | `place_settle_steps` | `5` | 张爪后等几步再抬（等物体落稳） |

  λ 判定的候选顺序**恒为每回合随机置换**（原来的 `judge_candidate_order` 开关已删除）；
  `PickExpert` 的接近/下降/合爪/抬升常量同样在 `PipelineConfig`，探针用
  `dataclasses.replace` 覆写，不再经 YAML。
- **新旋钮（`ObjectConfig` / `TaskConfig`）**：`color_random: true`（颜色每次 build 随机，
  优先于 `color`；Genesis 建完场景**不能改色**，所以是**每文件**一套）、
  `success_criterion: place_in_tray`、`tray_rest_tol_m: 0.005`、`group_files: true`。
- **`max_episode_steps: 1800`**：判据窗口从 2 s 拉到 4 s（§9-R39 定标），每个候选取 80 步判断
  加接近/下降/合爪，最坏情况**五块都要试**（随机顺序下请求的材质排最后），再加 `retry_on_none`
  允许的重抓、搬运/下降/退回。**实测定标**：走遍五块的探针一集在 1200 步被截断且搬运尾还没跑完
  ⇒ 上限必须覆盖六次抓取 + 尾部，否则**判对的一集也会在搬运途中被砍掉**。
  上限不是缓冲区大小 —— 落盘的 episode 只存它实际走过的步数。
- 复现（三条一次跑完，或 `--group plastic` 只跑一条）：
  ```
  cd ~/ThermTacVLA && conda activate tempvla
  LD_LIBRARY_PATH=/usr/lib/wsl/lib LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
      python scripts/generate.py --config sim/configs/put_in_tray.yaml \
      --episodes 1 --cameras --out datasets/v1.5/hdf5
  ```

**冷罐抓放 + 温度判定（`sim/configs/thermal_recognition/pick_cold_can.yaml`，§9-R42/R43，2026-10-19）**：
当前仓库**唯一**的任务配置（场景与任务同文件，不设单独的 scene YAML）。桌面是 robosuite
资产场景系列第 4 步的七体布局：冷罐（0 号，`temp_c ∈ [−18, 4]`）与 **55 °C 同外观孪生罐**
（1 号）、两盒牛奶（一热一冷）、柠檬（环境温度对照）、木/金属立方体，外加远端托盘。
每回合**全部**物体按 ±10 mm / ±0.15 rad 重抽，`min_object_gap_m: 0.02` 用抖动包络做
保守间距检查（重抽 5 次仍撞则退回零抖动槽位）。两个罐子视觉完全相同，`target_index: 0`
只钉**成功目标**；专家先摸哪一罐由候选集的**每回合自动随机置换**决定（无需开关）。

- **判据不是材质判据**：没有 `requested_group`，JUDGE 相变成触碰后判断（§5.6 的冷热判定模式）：
  合爪 → 停在抓取位姿 → **每个策略步**读 measured 指尖 → 任一碰上的指进目标区间（且趋势
  不会在一个趋势窗内把读数带出区间、且该指已有斜率）就立刻搬运；**没有任何**碰上的指还能
  到达区间（平直、反向，或读数在区间内但正在离开）就立刻**原地释放**（不抬升）再摸另一罐；
  两者都没有就继续等。单指的"否"不能否决另一指的"尚未有斜率"。
  **全部候选都判**：命中即搬运，放好后继续判剩下的候选（凡命中都搬走），没有强制提交，
  判不中的留在桌上。全程不读场景真值。
- **YAML 契约（`expert:` 只有这 4 个键）**：
  | 键 | 本 config | 作用 |
  |---|---|---|
  | `approach` | `place_in_tray` | 目标控制：把目标物体放入托盘（`APPROACHES` 注册表键） |
  | `temperature_judge_enabled` | `true` | 打开温度判定；无 `requested_group` 的 place 任务从 CLOSE 进 JUDGE |
  | `temperature_candidate` | `[0, 1]` | 判定候选物体下标；互异且在范围内，多个时每回合随机置换 |
  | `temperature_judge_threshold_c` | `[null, 15.0]` | 目标区间（°C），`null` = 开端：冷 `[null,15]`、热 `[45,null]`、双侧带 `[10,20]` |

  其余常量（趋势窗 0.4 s、最小趋近速率 0.25 K/s、baseline 0.3 s、读尾 0.15 s、转场 0.10 m 等）
  全部在 `sim/expert/pipeline.py::PIPELINE`；需要时可用 `dataclasses.replace` 进程内覆写。
- **安全转场（§9-R43）**：`transit_clearance_m: 0.10` ⇒ 转场 TCP 高度
  `0.4 (桌面) + 0.144 (最高牛奶) + 0.10 = 0.644 m`。修前 seed 3 的第一次接近是一条从
  home 到罐上 approach 点的直线，在第 3 步就撞倒 4 号牛奶（xy **80.8 mm**、z **−41.2 mm**）；
  修后同集 4 号位移 **0.00 mm**，EE 先竖直到 0.643、在 0.64 高度横移、再下降，251 步成功，
  validator 0 issues。温度判定分支不变（先摸 1 号 → 判不冷释放 → 摸 0 号 → 判冷搬运）。
- **实测（2026-10-19，无相机，4 集）**：4/4 成功，步数 **166 / 163 / 167 / 212**，
  `validate_dataset.py` 0 issues。seed 3 第一次摸到的是 55 °C 孪生罐：两指 measured
  **49.93 / 48.80 °C**（baseline 20.97 / 20.82，Δ +29.0 / +28.0）⇒ 判不冷、原地释放；
  第二摸冷罐：measured **1.13 °C**（baseline 50.97，Δ −49.8）⇒ 判冷、搬运入托盘，
  `place_body: 0`，`temperature_judge_correct: True`。seed 2 的两指读数 18.57 / −11.77 °C
  （Δ −2.42 / −32.73）说明"任一碰上的指判冷"是必要的：一根指垫接触不良不能淹没另一根的证据。
  （这 4 集的步数是安全转场**之前**的数字；R43 之后同 seed 3 为 251 步。）
- **复现**：
  ```
  cd ~/ThermTacVLA && conda activate tempvla
  LD_LIBRARY_PATH=/usr/lib/wsl/lib \
      python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml \
      --episodes 4 --no-cameras --out /tmp/pick_cold_can_tempjudge4
  python scripts/validate_dataset.py /tmp/pick_cold_can_tempjudge4/*.hdf5
  ```
  （加 `--cameras` 会开双 512² 软件光栅，实测约 **39 min/集**；无相机约 50–60 s/集。）

---

## 7. 目录结构（完整规划）

```
ThermTacVLA/
├── README.md
├── pyproject.toml
├── .gitignore                       # 生成物不入库：data*/、*.hdf5、*.mp4（docs/figures 除外）等
├── docs/
│   ├── architecture.md              # 本文档（设计与实测的单一事实来源）
│   ├── status.md                    # 现状/改进点/目标 快照（2026-10-08）
│   └── figures/                     # 报告图 + 演示视频（归档副本；report_figures.py 已于 2026-10-20 删除，§9-R46）
├── sim/
│   ├── configs/
│   │   └── thermal_recognition/
│   │       └── pick_cold_can.yaml   # 当前唯一任务配置：robosuite 七体场景 + 冷罐抓放 + 温度判定（§6、§9-R42/R43）
│   ├── __init__.py
│   ├── config.py                    # YAML → 严格校验的嵌套 dict（load_task_config + helpers）
│   ├── rotations.py                 # 自实现 wxyz 四元数工具（Gate 1 交叉验证）
│   ├── expert/                      # 脚本专家包
│   │   ├── __init__.py              # 只放 docstring（保持 thermal_judge 的纯 numpy import 链）
│   │   ├── actions.py               # DeltaPoseController / tcp_pose / normalized_action_to
│   │   ├── expert.py                # PickExpert（R43 安全转场）+ PickByTemperatureExpert（边抓边判 + 冷热判定，§9-R38/R42）
│   │   │                            #   + PutInTrayExpert（判对后搬运入托盘，§9-R39）
│   │   └── thermal_judge.py         # 接触耦合率 λ 的定窗拟合判据（纯 numpy，§9-R38）
│   │                                #   decide（一个阈值）/ decide_bands（N 材质 N-1 个边界）
│   ├── scene.py                     # build_scene / SceneBundle / sample_episode / reset_episode
│   ├── obs.py                       # ObsBuilder（L1 policy 率 + fast_payload sim 率）
│   ├── gripper.py                   # 夹爪几何独立简化版（垫面/闭合轴/marker 栅格/指垫温度格盒，纯 numpy，§5.9）
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
├── tempvla/                         # （预留，未创建）未来 VLA 模型代码；2026-10-11 重组后仿真/数据层全在 sim/（§9-R29）
├── scripts/                        # 2026-10-20 两次精简后最终只保留 5 个（§9-R46 是第一次的记录）
│   ├── gate_common.py               # 配置加载/计时/内存 公共件（generate 与其余脚本共用）
│   ├── generate.py                  # 主生成入口（单 job：按 material_list 出文件）
│   ├── validate_dataset.py          # CLI 校验
│   ├── build_manifest.py            # 侧车三件套：index/splits/stats（§9-R30）
│   └── render_video.py              # HDF5 → MP4（双相机 + 温度/触觉叠加）
└── datasets/                       # 数据集根（§9-R30）：唯一入库文件是本 README.md（数据卡）
    ├── README.md                   # 布局、侧车字段规格、划分规则、注意事项
    └── v1.5/
        └── hdf5/                   # M9：1.5 栈全量重生成落点（规划；configs 的 recording.output_dir 默认）
```

> 引擎为 conda 环境内的 pip wheel（`genesis-world==1.4.1`，仓库内无引擎副本）；
> `~/tactile-genesis`（仓库外）仅作只读参考（唯一的静态拷贝物见 README）。

---

## 8. 验证关卡（Gate 0–2 + 生成）

> **2026-10-20（§9-R46）**：本节保留 Gate 0–4 与历史任务的验收记录，**其中的命令均已不可执行**——
> Gate 脚本、取证探针、演示与出图脚本已在 `scripts/` 清理中删除。Gate 0–4 于 2026-10-02/03 全部
> 通过（实测记录不变）。当前可运行的命令见根 README「运行顺序」与本节「生成」小节。

按顺序执行；**任一 Gate 失败则暂停并修订设计**。历史命令均由用户在 `tempvla` conda 环境执行
（`source activate tempvla && cd ~/ThermTacVLA`）。三个 Gate 脚本共用
`--config`（当时缺省 `sim/configs/pick_hot_cylinder.yaml`，该配置已于 2026-10-19 删除）。

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

专家管线参数（§5.6 的 `PipelineConfig`）不必手调：autotune 批量评估候选参数，
按 **成功率（主）+ 接触事件数 + 指温升温 − IK 误差惩罚** 打分
（`score = success_rate + 0.05·min(events,4)/4 + 0.05·min(heating,3°C)/3°C − 0.05·min(ik_mm,20)/20`），
先随机搜索（40%）再围绕最优做局部细化，最优参数写进输出 YAML 的
`tuning.best_pipeline_params`。

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
- 输出 YAML 的 `expert:` 段仍是这 4 个契约键（**不写调参结果**）；最优点在
  `tuning.best_pipeline_params`，另附材质、种子、试次数、top-5、耗时等溯源信息。
  应用方式：把值填进 `sim/expert/pipeline.py::PipelineConfig` 的默认值，然后重新生成。
  `tuning:` 是未知顶层键，加载器会忽略，不影响使用。
- 先跑冒烟确认链路，再跑正式；正式调参在 **Gate 2 通过之后、全量生成之前** 执行。

### 生成（冒烟 → 全量）

```bash
# 冒烟：单集、无相机（落点默认 = 配置的 recording.output_dir，现为 datasets/v1.5/hdf5，§9-R30）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml \
    --episodes 1 --no-cameras --out /tmp/data_smoke

# 全量：当前任务 50 集（落点 = recording.output_dir；--out 可覆盖，同名文件截断重写）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml --episodes 50

# 单材质 + 查看器（--material 把任务缩到含该材质的 job；当前 config 只有一个 job、七个物体）
python scripts/generate.py --config sim/configs/thermal_recognition/pick_cold_can.yaml \
    --material aluminum --episodes 5 --viewer
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
`python scripts/generate.py --config sim/configs/pick_hot_cylinder.yaml --episodes 50 --out data`，
`WALL 944 s`（含 4 次场景构建；首建 10.3 s，其余 4–5.6 s）。
实测单 episode 4.3–4.6 s（≈7.5 control steps/s），
落盘 0.063–0.065 MiB/episode（200 条合计 **12.9 MiB**），生成进程 RSS 峰值 **2.24 GiB**（预算 8 GiB 内）。
**autotune 的两个坑（留档，v2 若要用需先解决）**：(1) 随机相位的候选参数每个都要
roll 到 400 步上限（~4 min/试次），`--budget-min 30` 只够 ~6 试次，永远进不到局部细化；
(2) 打分里的 `heating` 项上限 3 °C，而实测指尖峰值每 episode 升 11–125 °C，该项对所有候选
都是满分、无区分度。建议 v2：把随机相位改为围绕 base config 的扰动 + 只做局部细化，
并把 heating 的归一化改成实测分位数。

**M5 重生成（marker 场，2026-10-04 起）**：旧栈三套（`data/` 12.9 MiB、`data_cam/` 97 MiB、
`data_lift/` 2.5 MiB）全是 **format 1.0、不含 marker 场**，2026-10-12 经用户决定**已删除**
（§9-R31）——重生成不再是"换目录"而是**从零落盘**，直接写进 `datasets/v1.4/hdf5/`（§9-R30）；
跑 `validate_dataset.py` + `build_manifest.py` 建侧车 + 每材质渲染一集抽查，**确认后才切换下游
默认**；仓库自 2026-10-08 起由 git 管理，但 `datasets/*`（除 README 数据卡）不入库（数据本身
不可回滚）。后续重生成（M9）必须在 M5–M7 与 §9-R36 的验证全过之后一次性做 1.5 栈全量。

**任务 `pick_by_temperature`（§9-R38）的生成与验收**：一个材质对 = 一个文件，`--material`
只用来把任务缩到含该材质的那些对（场景种子按对的下标，仍按 §9-R21 的警告一次跑全）。

```bash
# 冒烟（无相机最快；判据只读温度通道，与相机无关）
python scripts/generate.py --config sim/configs/pick_by_temperature.yaml --material aluminum \
    --episodes 20 --no-cameras --out /tmp/pbt_v2
# 盲判复算 + 阈值扫描（只读落盘 measured 通道，真值仅用于打分）
python scripts/judge_temperature_material.py /tmp/pbt_v2/pick_by_temperature_aluminum_abs.hdf5 \
    --sweep --json /tmp/v2_aluminum.json
# 判据验证图（图 6）
python scripts/report_figures.py --judge-json /tmp/v2_aluminum.json
```

验收口径：金属组 min λ̂ ≥ 非金属组 max λ̂ × 1.5、判定 0 错、指间 λ̂ 一致性、
以及**同一阈值在热物体与冷物体上都成立**（按温区分列的 λ̂ 带）；`*_failed.hdf5` 不是垃圾——
判错即 `success=False` 如实落盘，混淆矩阵由 attrs 免费得到。

**实测（2026-10-14，铝-ABS，20 集 32 抓）**：分离 **2.17×**、**32/32 对**、零错阈值窗口 **0.672–1.340**
（阈值 1.04 的两侧余量 1.60× / 1.35×）；**按温区拆开后两组各自都落在同一带** ——"同一阈值冷热通吃"
在 shipped 数据上成立。指间分歧、V3 孪生渲染、V4/V5 与窗口扫描的逐项数字见 §9-R38。

**任务 `put_in_tray`（§9-R39）的生成与验收**：三个材质组 = 三个文件，`group_files: true`
⇒ 一条命令写全部三条；`--group <词>` 可只跑一条。

```bash
# 定标探针（一集测遍五个物体；这条是"两个边界定在哪"的唯一依据）
python scripts/probe_lambda_bands.py --config sim/configs/put_in_tray.yaml \
    --episodes 4 --out /tmp/lambda_bands
# 三个数据集（每组一集、带相机；--group 可只跑一条）
LD_LIBRARY_PATH=/usr/lib/wsl/lib LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
    python scripts/generate.py --config sim/configs/put_in_tray.yaml \
    --episodes 1 --cameras --out datasets/v1.5/hdf5
# 场景预览图（未经专家，只看托盘和五个立方体）
LD_LIBRARY_PATH=/usr/lib/wsl/lib LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
    python scripts/render_scene.py --config sim/configs/put_in_tray.yaml --res 512
# 盲判复算（不建场景；默认复刻文件自己记录的协议并裁到在线判定步，§9-R40）
python scripts/judge_temperature_material.py datasets/v1.5/hdf5/put_in_tray_*.hdf5 --sweep
# 同一批文件按*配置里那套*协议复算（换了估计量/窗口时用这行，输出会标 counterfactual）
python scripts/judge_temperature_material.py datasets/v1.5/hdf5/put_in_tray_*.hdf5 \
    --method ratio --window 1.5 --late-window-frac 0.4 --thresholds 0.50 1.04
```

验收口径：三带的 λ̂ 相邻分离 ≥2×（R38 单边界盲判 32/32 时的实测比）、两个边界都落在
**最坏情况窗口**（慢带 max < 边界 ≤ 快带 min）内、判定 0 错、指间 λ̂ 一致、`success=True`、
validator 0 issues；外加 §9-R39 记的**捷径审计**（不看只摸能不能分）。窗口/估计量的换算与
代价见 §9-R40（`ratio` + 锚点前移把每候选的判定从 `t2+4.15 s` 压到 `t1+1.65 s`）。

---

## 9. 风险与决策记录（R1–R46）

> **一行一条**（R1–R26、R28–R46；R27 空缺，编号不回收）。"当前取值/结论"列只写执行时真正要用的数
> （阈值 / 口径 / 默认值）；机制推导、实测过程、历史读数在同编号的分节里。

| # | 风险 / 决策 | 状态 | 当前取值 / 结论 |
|---|---|---|---|
| R1 | 温度传感器类级共享参数（"last one set wins"） | 已缓解 | 全局热参数按场景固定；逐集随机化走物体初温（GT 缓冲注入） |
| R2 | 体素体积仅对轴对齐连杆精确 | 已缓解 | 物体温度以集中温度通道为主、网格为辅 |
| R3 | gs-madrona 在本机不可用（WSL2 无 NVIDIA Vulkan ICD） | 已绕过 | `renderer=raster` + llvmpipe；43–50 ms/次，全量 +12 min |
| R4 | 8 GB VRAM / 7.4 GB RAM | 已缓解 | `n_envs=1` 串行 + 流式写盘；单集 <0.5 MB |
| R5 | Quadrants 在 WSL 静默回退 CPU | 已缓解 | Gate 0 断言 GPU；生成脚本打印 steps/s 与 ETA |
| R6 | torch / gs-madrona CUDA 版本错配 → import 段错误 | 已缓解 | 独立 conda 环境锁 `genesis-world==1.4.1` |
| R7 | 脚本专家质量 ≠ 人类遥操作 | 记录 | 数据卡标注；失败集单独留；`autotune.py` 目标成功率 >80% |
| R8 | 触觉是 taxel 数组、非光学图像 | 已兑现 | GelSight 风格面板；`elastomer` 16×16 成本 +51%、压痕 17.5/11.3 µm |
| R9 | 上游 Genesis 升级破坏补丁 | 已隔离 | `compat/sensor_compat.py`；升版先跑 Gate 1 |
| R10 | 接触热流不经公开 API 暴露 | 已定级 | 不设通道；由温度网格离线差分（v2） |
| R11 | 爆发窗口存储膨胀 | 已缓解 | 默认 [−0.05, +0.25] s、上限 10 窗；开 16×16 后 1.1–2.2 MiB/集 |
| R12 | 引擎资产随源码树剥离 | 已缓解 | pip wheel 自带 `urdf/`；Gate 0 打印 sparse-checkout 恢复命令 |
| R13 | 调参只在单一材质/种子上评估 | 已缓解 | 同种子评估 + 其余材质抽验 5–10 条 |
| R14 | 引擎辐射/对流项缺面积因子（散热快 ~1e4×） | 已规避 | 全项目 `emissivity: 0` + `convection_coefficient: 0`；真 ε 值留档 |
| R15 | `panda_hand` 碰撞几何下延到指根平面 | 已缓解 | `grasp_z_offset_m ≥ h/2 + 余量 − 0.0466` |
| R16 | 接触热核显式格式且无稳定性约束；有效体积被 `gs.EPS` 接管 | 已定位 | `contact_depth_weight: 0` + 指尖 `conductivity: 0.02`；Gate 1 断言 `c < 0.5` |
| R17 | 物体网格与指尖是单向耦合 | 记录 | 受影响单元少数；数据卡标注该近似 |
| R18 | 刚体密度 ≠ 材质密度 | 记录（v2 改） | 数据卡标注 `object_mass_kg` 为随机化质量；倾向 v2 由材质导出 |
| R19 | 触觉力在"首次接触"substep 上不可逐位复现 | 已定级 | Gate 4 只对轨迹通道强断言；数据卡标注该有界暂态 |
| R20 | `eye_in_hand` 个别帧渲染尖峰 11–16 s | 已定位、未修 | 只影响耗时不影响数值；v2 试 `cameras.near: 0.01 → 0.03` |
| R21 | 材质与物体几何在数据集内完全混淆 | 已加防呆 | 不可用于"固定几何、只变材质"的对比；数据卡标注 |
| R22 | 触觉"压力突变"是接触门控 + 探针预压几何 | 已定位 | `tactile/force` 抓取段准二值（0 或 45–48 N） |
| R23 | 渐变曲线在 `proximity` 模态里 | 已实现 | 可选模态；幅值不可与 KinematicTaxel 互比 |
| R24 | 双物体场景依赖的四条假设 | 已验证 | 盒形 SDF / 逐 link 采样数 / 间隙守卫 / 无物体间热耦合 |
| R25 | 触觉对齐 UniVTAC 形态：63 点场 + 稠密高度图 | 已定案 | 接受核宽展宽（峰值 27.4 → 5.8 µm）；专家回退 `link_contact` |
| R26 | 指尖温度"4×4"不是 4×4 指垫贴图 | 记录 | 语义 = 哪根指 + 哪个 x 条带 + 多热；**不是** 2D 垫温图 |
| R28 | 夹爪几何的独立简化文件 `sim/gripper.py` | 已定案 | 管线不 import；`probe_gripper_model.py` 8/8 断言等价（该脚本已于 2026-10-20 删除，§9-R46） |
| R29 | 包结构重组 `tempvla/` → `sim/` | 已定案 | 数据兼容不变量（FORMAT_VERSION / 格式串 / conda 名）绝不动 |
| R30 | 数据集根 `datasets/` + 侧车三件套 | 已定案 | 划分是 seed 的纯函数；stats 只在 train split 的成功集上算 |
| R31 | 旧栈全量数据删除（format 1.0） | 已执行 | M9 重生成前仓库内无完整数据集；**不可回滚** |
| R32 | 热像 demo：木桌上两个立方体自然冷却 mp4 | 已实现 | 注入式牛顿项 `h = 12 W/(m²K)`、`τ = 1350 s` |
| R33 | 抓取热物体的热像 + 传感器视频 | 已实现 | 指尖热源是**自己的单格**；接触点数切换使升温率差 24–36× |
| R34 | 换材质对照：材质只从 ρcp 与 k 进入热学 | 已定案 | 物体侧可分（全网格 spread 159×）；指尖 1.031×（`k_pad=0.02` 时代） |
| R35 | 指尖材质对比的天花板是 `k_pad` 而非分辨率 | 已定案 | `conductivity = 0.25`；解析 2.248×、实测 1.96× |
| R36 | 指尖温度通道定案：官方单格 `(1,1,1)` + 非理想性，只录 measured | 已定案 | RC 0.05 s / 噪 0.05 °C / 量化 0.1 °C；V6 判别矩阵 ≥2× |
| R37 | 指尖接触面积口径补丁 `pad_face` | 已定案（全数据集，2026-10-14） | `Aᵢ = A_face/√nᵢ` → ΣAᵢ² ≡ A_face²；冻结过冲 engine 26.6% → 0.0%；代价物体网格多抽 ~281 J |
| R38 | 新任务 `pick_by_temperature`：用指尖温度变化判材质 | 已实现 | λ̂ ≥ 1.04 / 窗 2.0 s / settle 0.15 s；盲判 32/32、分离 2.17× |
| R39 | 新任务 `put_in_tray`：三材质判材质 + 托盘放置 | 已实现 + 已定标 | 铁 k=80 / 尼龙 k=0.25 / 巴尔沙木 k=0.055（k_eff 0.498 / 0.250 / 0.090）；实测 λ̂ 1.484 / 0.734 / 0.278，相邻 2.02× / 2.64×，边界 **0.50 / 1.04**（最坏情况几何中点）0 错 / 20 抓；窗 4.0 s；一材质组一文件 |
| R40 | 判定窗口 4.0 → 1.5 s：`ratio` 估计量 + 窗口锚点前移到先碰上的指垫 | 已实现；验收重生成待跑 | 每候选省 **2.50–3.10 s**（`t2+4.15 s` → `t1+1.65 s`）；旧文件反事实复算 8/8 分带正确；阈值仍 `0.50 / 1.04`；`judge_min_late_window_frac` 默认 1.0（仅 `put_in_tray` 开 0.4）⇒ R38 零回归 |
| R41 | robosuite 资产贴图**静默丢失**（trimesh ≥ 5 解析器根限制 + Genesis 补丁吞异常） | 已修（2026-10-15） | `sim/compat/mesh_load_compat.py` 把 `FilePathResolver` 的 `allow_anywhere` 默认翻成 True（= 上游 pre-5.0 语义）；只影响引用目录外贴图的资产（= 只 robosuite） |
| R42 | 温度判定（非材质）：触碰保持后按 **measured** 指尖判冷热 | 已实现（2026-10-19）；2026-10-21 改为事件驱动 + **全部候选都判、命中即搬运**（§5.6） | 候选 `[0,1]`、最初为窗 **1.5 s**（锚定后碰指垫）、冷 ⇔ reading ≤ **15 °C** 或 Δ ≤ **−3 K**（baseline 前 0.3 s）；任一碰上的指判冷即冷；判不冷**不抬升**、原地释放换下一罐；实测 4/4 成功（166/163/167/212 步），validator 0 issues；成功仍只认 `target_index: 0`。**现行为**：无固定窗口，区间即搬运、趋势到不了即释放、否则继续等；每集遍历全部候选，凡命中都搬走（托盘槽位错开），**无强制提交**；2026-10-21 评审修正：DESCEND/LIFT 门用新目标、合爪超时进 JUDGE、释放改为"无指可到达"、命中要求有斜率、metadata 用实际 pipeline、config 要求 `target_index ∈ temperature_candidate`；同日重测 4/4 成功（230 步均值）、validator 0 issues |
| R43 | APPROACH 撞倒旁物：把接近拆成「先升到最高物体之上 → 横移 → 下降」 | 已实现（2026-10-19） | `transit_clearance_m: 0.10`（默认 0 = 旧轨迹逐位不变）；七体场景转场 TCP **0.644 m**；seed 3 的 4 号牛奶 xy 位移 **80.8 → 0.00 mm**；4 集 4/4 成功（184/183/186/251 步）、validator 0 issues |
| R44 | 触觉简化（format 1.6）：只留 AT-VLA 6D 合力 + 2D marker，参数移入传感器模块 | 已实现（2026-10-20） | `force (T,2,6)` = 12 探针求和后沿指垫法向分解 `[Fn,Ft]`；`elastomer (T,512,2)` = 切平面位移；YAML 只剩 `modalities`；`world_frame=false`；不再写 depth/height_map；单集 smoke 184 步成功、validator 0 issues |
| R45 | 全景热像（format 1.7）：曾按 ThermoAct 标准合成 | **已回退（2026-10-20）** | 当日做出合成热像通道（256×192 分割上温 → 线性 20–35 °C → 8-bit → INFERNO → 256²）并 smoke 通过，随后按要求**全部回退**：通道、`cameras.thermal`、`sim/sensors/thermal_camera.py` 与格式 1.7 均已移除，格式回到 1.6。保留结论：热像分割本身 6.3 ms/次、45 步 0.3 s，不是 R20 尖峰来源 |
| R46 | `scripts/` 精简：只保留生产与数据侧工具 | 已执行（2026-10-20） | 保留 10 个（generate / gate_common / validate_dataset / build_manifest / render_video / scene_check / render_scene / autotune / probe_render / probe_camera_pose）；Gate 0–4、7 个取证探针、4 个演示/视频、3 个出图/离线判据脚本删除；历史 R# 仍为记录但不再可复现 |

### R1 — 温度传感器类级共享参数

`ambient_temperature`、`convection_coefficient`、`properties_dict`、`simulate_all_link_temperatures` 都是**类级共享**（"last one set wins"）——同一进程内多场景构建会互相覆盖。

- **影响**：逐 episode 环境温度随机化受限。
- **做法**：全局热参数按"场景构建"固定，逐集随机化走物体初温。机制 = 无公开 setter，但在 `scene.reset()` 之后直写 manager 的 ground-truth 缓冲（`_ground_truth_intermediate_cache`，转置布局 `(cols, B)`）即 `sensors/temperature.py::seed_temperature`，写入需一步 `scene.step()` 后才可见；顺序 reset → seed → step，Gate 1 压入测试已验证。材质按**文件**分组重建场景（§4.7）。

### R2 — 体素体积仅对轴对齐连杆精确

被抓物体旋转时其温度网格是近似。

- **做法**：物体温度以集中温度（`temperature/object`）为主通道，网格为辅；文档与数据卡标注网格精度边界。

### R3 — gs-madrona 在本机不可用

`backend.cpp:553 findPhysicalDevice` 报 **"Cannot find matching vulkan UUID for GPU"**——WSL2 下没有 NVIDIA Vulkan ICD（`/usr/share/vulkan/icd.d/` 只有 intel/radeon/lvp/virtio；`/usr/lib/wsl/lib/` 无 `libnvidia-vulkan-producer.so`）。装 `libvulkan1` 后仍缺 GPU ICD，只剩 lavapipe。**与 sm_120 无关**。

- **影响**：无法用 BatchRenderer 做 GPU 批量渲染。
- **做法（已实测绕过）**：`renderer=raster` + llvmpipe 出正确双相机图，43–50 ms/次、全量仅 +12 min（256²，Gate 3）。**且不需要 madrona**：raster 的 512² 也只比 128² 贵 +14% 时间——批量渲染省的是编译/驱动折腾，不是速度。

### R4 — 8 GB VRAM / 7.4 GB RAM

- **做法**：v1 `n_envs=1` 串行 + HDF5 流式写盘（单 episode 常驻内存）；单 episode < 0.5 MB；生成脚本逐 episode 释放。

### R5 — Quadrants 在 WSL 静默回退 CPU

性能塌方且难察觉。

- **做法**：Gate 0 打印 backend/device 并断言 GPU（`gs.init` gpu→cpu 回退时显式警告）；生成脚本打印 steps/s 与 ETA。

### R6 — torch / CUDA 版本错配 → import 段错误

- **做法**：独立 `tempvla` conda 环境锁 `genesis-world==1.4.1`（py3-none-any wheel 自带依赖声明）；Gate 0 先验证 import + CUDA。

### R7 — 脚本专家质量 ≠ 人类遥操作

- **做法**：数据卡如实标注；域随机化提升多样性；失败 episode 单独文件保留供分析；`scripts/autotune.py` 自动调参（§8）目标成功率 >80% 后把 `PipelineConfig` 的默认值固化。

### R8 — 触觉是 taxel 数组、非光学图像

与真实 GelSight 数据域有差距。

- **做法（2026-10-03）**：`render_video.py` 的 GelSight 风格压印面板；高度源优先 `elastomer`，退回 12 探针 `depth`，标题写明来源。
- **`elastomer` 模态实测**（16×16 = 512 标记，`lambda_d=5e4`/`lambda_s=2e4`）：① 成本 depth-only 84.4 → +elastomer **127.7 ms/step（+51%）**，几乎与探针数无关；② 记录 `(T,2P,3)` float32 = 6144 B/step 原始、gzip-4 后 ~3.1 KiB/step（44 步 episode 里 136 KiB / 2.0 MiB ≈ 7%，相机才是大头）；③ **压痕是真的**：峰值法向位移 **17.5 µm（L）/ 11.3 µm（R）**，平滑单峰；真实坡度 ~1e-3 m/m（12 µm / 12 mm 垫），渲染必须夸大（面板标题 `slope x210`）；④ **`compressibility<1` 或 `elastomer_thickness>0` 会切 FFT dilation 并崩**（`UNSUPPORTED DTYPE: complex`），只能留默认；⑤ 指垫法线在 link 系是 **y**（不是 z），高度场须投影到 `layouts/probe_local_normal` 才不是鞍形。
- **⑤ 已解决（2026-10-04）**：`obs/link_pose`（每步每指 pos+quat）+ `tactile_field.read_markers(frame="link")` 让 `world_frame: true` 的场**精确可逆**。转正为正式 schema（Q5），两处结构缺陷一并修掉：**网格不再共享**（`tactile.probe_grid` 支持逐模态覆盖，`force`/`depth` 保持 4×3 每指 12 逐字节不变，只有 `elastomer` 是 16×16）；**帧不再丢失**。新增代价：`link_pose` 56 B/step（可忽略）；burst 里每窗口 +110–218 KiB（§4.7-L4，`recording.burst_modalities` 可收口）。
- **往返一致性实测**（同 seed 跑 shipped 与 preview 两份配置）：`disp_world @ R(q)ᵀ` 与 link 系录制值最大差 **3.6e-12 m**（42 步全过 1e-6 阈值），投影到指垫法向的高度场差 **9.5e-13 m**；`force` 同法回转差 **2.4e-7 N**（旋转后的 float32 量化）；而 `depth`、`contact/*`、`link_pose`、`ee_pos`、`joint_states`、`actions`、温度网格**逐位相同**。
- **已知未解**：`probe_radius_m` = 4 mm 仍是全局共用，16×16 的 0.56 mm 间距下探针球严重重叠——preview 里场是光滑的，若下游读出过平滑再把该参数改成逐模态。另：16×16 网格会触发引擎告警 `ElastomerTaxel grid is not strictly regular ... FFT dilation will use averaged spacing`——只是告警，读数照常。

### R9 — 上游 Genesis 升级破坏补丁

- **做法**：`sensor_compat.py` 隔离在 `compat/`，升版先跑 Gate 1。

### R10 — 接触热流（contact heat）未必经公开 API 暴露

- **结论（Gate 2，已定级：不暴露）**：不设通道；由 L1 温度网格离线差分推导（`Q̇ ≈ ρcV·ΔT/Δt`，`tools/analyze_contact.py`，v2）。

### R11 — 爆发窗口存储膨胀

- **当前取值**：默认 [−0.05 s, +0.25 s]、上限 10 窗口、全部触觉模态 + 集中温度；无数值通道时 ≈0.25 MB/episode，**开启 16×16 marker 场后 1.1–2.2 MiB/episode**（当前 burst 的主要成本）。兜底：`recording.burst_modalities` 逐模态收口、`contact_burst: false` 整体关闭。

### R12 — 引擎资产随源码树剥离（vendored Genesis 只剩 textures）

- **做法**：用 pip wheel `genesis-world==1.4.1`（自带 `urdf/`）；Gate 0 检测缺失时打印 `git sparse-checkout add genesis/assets/urdf`。

### R13 — 调参默认只在单一材质/场景种子上评估

（config 第一个材质）参数可能过拟合该材质。

- **做法**：同种子评估保证与生成一致；需要时对每个材质各跑一次 `--material`，或调参后在其余材质上抽验 5–10 条；调参只动专家旋钮，不影响数据 schema。

### R14 — 引擎体素级辐射/对流项缺面积因子

`temperature.py: _radiation_convection_delta_T` 用 per-m² 通量除以 `rho*cp*voxel_volume`，**未乘面元面积 dx²** → 散热快 `1/dx²` ≈ **1e4 倍**。

- **影响**：热物体秒级掉到环境温度（实测 ε=0.1 的铝柱 τ ≈ 3 s），温度通道在 episode 内失去信息。
- **做法**：三类材质（物件 / 指尖 / 环境）全部 `emissivity: 0.0` + 全场景 `convection_coefficient: 0.0`（episode 内绝热；物理上金属件空气中 τ ≈ 1e3 s ≫ 20 s episode，故 0 反而更接近真实）。真 ε 值留档：铝 0.10 / 铜 0.30 / abs 0.90 / 钢 0.30。唯一热通道 = 接触热传导 + 内部扩散。指尖 `conductivity: 0.02`（k 同时是接触耦合增益，按数值稳定性取值，不按材料表值，见 R16）。逐 episode 由 `reset`（全部回 ambient）→ `seed_temperature`（物体采样温度）重置。

### R15 — `panda_hand` 碰撞几何向下延伸到指根平面

手架原点下方 5.84 cm = TCP 上方 4.66 cm。

- **影响**：俯视下压时**手掌先于手指**撞上物体顶面——Gate 1 trace 显示 TCP 低于物体中心 19 mm 处 244 N 冲击，臂被弹开、setpoint 跑飞。
- **做法**：下压目标须满足 `grasp_z_offset_m ≥ h/2 + 余量 − 0.0466`；本任务 h ≤ 0.08 m → dz ≥ 0 时手掌仍留 0.7–2.2 cm（实测 dz=0、h=58 mm 时 F=0）。`PipelineConfig.grasp_z_offset_m` 即该旋钮，autotune 可调；下压收敛判据须留 ≥5 mm 容差。

### R16 — 接触热核是显式格式、无稳定性约束；有效体积被 `gs.EPS` 接管

`temperature.py: _kernel_contact_heat` 单个体素 `dT = dt·k_eff·(T_other−T_cell)·A²/(V²·ρcp)`，放大因子 `c` 必须 ≪1。`A` = 接触面积缓冲（组内 count<3 时该缓冲本身就是 `Σπ·穿透`；本场景每个指尖-物体对恰好 2 接触点 → `A≈π·穿透≈6 cm²`，量级上正好等于整块指垫 45.5×53.7 mm）。**`V = voxel_volume[i_s] + gs.EPS`，`gs.EPS = 1.19e-7 m³`**（float32 eps）。

- **核心事实**：指尖的 `voxel_volume` 因 AABB 反变换 bug（只用世界 AABB 的两个角点回推 link 系，y 向 extent 塌成 −9.4e-6 m）只有 **1.435e-9 m³**（几何格容 1.17e-7 m³ = 指垫 1.87e-6 m³ ÷ 16 格，差 **81×**）——**低于 gs.EPS，所以引擎实际除的体积是 eps 本身**（1.214e-7 m³，与几何格容只差 2%）。
- **影响**：若照字段 `voxel_volume` 估算，c 被高估 `(1.214e-7/1.435e-9)² = 7160×`，任何参数看起来都不稳定；实测反演加热曲线时这 7160 恰好表现为恒定系数 `A_eff/A_buf = 1.18e-2 = 1/√7160`（曾误判为"引擎另有隐式缩放"）。
- **校核**（`scripts/probe_kernel_args.py` 包住内核、打印实参并对 scratch 缓冲跑一遍）：物体-桌面接触预测 −0.0011 K / 实测 −0.0011 K；指尖首个接触步预测 `0.529 K × 2 接触 ÷ 16 格 = 0.066 C` / 实测 `0.0667 C`（1% 内）——公式无误，差的就是 eps。真发散也确实存在：`k=10` + `π·穿透` 回退面积 → `c≈11` → 体素在 ±1000 °C 夹紧值间逐 substep 翻转（实测右指尖均温 −322.6 °C）。
- **做法**：`contact_depth_weight: 0.0` ＋ 指尖 `conductivity: 0.02`（`k_eff≈0.04`）→ **Gate 1 实测 `c = 7.9e-3`**，指垫 2 s 内爬升 +20.9 °C（= 130 °C 温差的 19%，保留速率信息，非阶跃），贴片再大 ~7× 仍稳定。Gate 1 断言 `c_max < 0.5` 并打印 `thermal \| A_all … <传感器>: A … V … c …`（`_thermal_diag`：A 按**该传感器自己的连杆对**取，V 必须含 `gs.EPS`，否则 c 恒定虚高 7160×）。k 只按数值稳定性选："按材料密度补偿热容"的思路不成立——pad 的有效热容由引擎常量 eps 决定，不由 `density/specific_heat` 决定。

### R17 — 物体网格与指尖是单向耦合

物体单元看到 `T_other = link_temps[指尖]`（指尖连杆保持 ambient），指尖单元看到 `T_other = link_temps[物体]`（保持 seed 值）——两侧都不被对方的**网格**温度反哺。

- **影响**：接触面单元被过冷到 ambient（物体侧）/ 过快贴近物体温度（指尖侧），"接触热传导"的绝对量值偏大。
- **做法**：录的是网格均温，受影响单元只占少数（物体 4×4×4 中 ~2–4 格，均值漂移几 °C，量级仍在物理范围内）；pad 侧同时受 R16 增益约束；数据卡标注该近似（v2 若要定量热流，改用离线差分）。

### R18 — 刚体密度 ≠ 材质密度

`ObjectSpec.mass_kg` 从 `object.mass_kg: [0.05,0.15]` 均匀采样、与材质解耦；`scene.py:296` 建刚体用 `rho = mass_kg/volume_m3`，而温度传感器用 `material_properties.density`（真材质值）。

- **实测**：铝柱 r=20.2 mm h=68.1 mm m=0.0971 kg → ρ_rigid = **1110**（真值 2700，2.4× 偏轻）；铜 r=20.8 mm h=69.7 mm m=0.0527 kg → **559**（真值 8960，**16× 偏轻**）；abs 1306（真值 1040，反偏重 1.26×）——金属件在数据里像塑料件一样轻。热学侧（ρcp 决定网格热惯量）用真材质值，故同一 episode 内"热的是铜、动的是塑料"。
- **做法**：v1 **不改物理**（改动会失效已通过的 Gate 2 与冒烟统计）；数据卡如实标注 `object_mass_kg`（逐 episode attr，已录）为"随机化质量"。**注意尺寸/质量是每文件一次（构建期冻结）**：文件内 pad 加热由 T0 主导（铝 r=+0.973 / 铜 +0.987 / abs +0.974 / 钢 +0.965），跨文件 mass~dT r=−0.497 纯属材质混淆（铜恰好抽到最小质量、加热最高），质量本身只影响抓取动力学、不进热通道。v1 任务实测 32/32 成功。v2 二选一：(a) `mass_kg` 由材质密度×几何导出（物理自洽，删该旋钮，**倾向**）；(b) 保留随机化但新增真值 `object_rigid_density_kg_m3` 并在数据卡声明"质量与材质解耦"。

### R19 — 触觉力读数在"首次接触"substep 上不可逐位复现

GPU 接触求解的 float32 归约顺序不稳定，`tactile/depth` 在个别 taxel 上差 ~1e-6 m，`tactile/force` 按 taxel 刚度放大成尖峰（实测 0.003–5 N），只出现在 3–4 个接触起始 policy step，之后重新收敛。

- **定级（Gate 4）**：同一场景、同一 seed、**完全关闭渲染**连跑两次，`tactile/force` 差 **4.92 N**，而同一次对比里 `ee_pos`/`actions` 逐位相同——即噪声源在引擎侧，不在相机。
- **做法**：Gate 4 只对**轨迹通道**（proprioception/actions/contact/temperature，实测逐位相同）做强断言；`tactile/*` 改判"差异只出现在接触起始、末步相同"这一有界暂态性质，并打印**无渲染对照**的噪声底（不是断言）。数据卡须标注：`tactile/force` 在接触起始步不保证跨次生成逐位一致，绝对量级为 taxel 级尖峰。

### R20 — `eye_in_hand` 相机在个别帧上渲染尖峰

同一集内中位 **24 ms**，但个别帧 **11–16 s**（~600×）。实测 ep15（aluminum）35 s 里 **29.4 s 全在 `eye_in_hand`**，集中于 step 4–6（DESCEND 下探、腕相机贴近桌面时）。

- **可复现**：同 `(scene_seed, episode_seed)` 在不同批次的集号完全一致（seed 1000 → `[11,15,20,42,44,45]` 两次逐集相同；seed 2000/3000 各复现一次；共 3 组独立复现），**关掉相机后同一集只要 4.1 s**。
- **影响**：相机开启时约 10% 的 episode 从 8.3 s 涨到 25–350 s（全量 200 集约 53 min 而非 28 min）。**只影响耗时，不影响数值**（全部 0 issues），与 R19 性质不同。
- **未修**：抬高 `near` 或把腕相机挪远都会改相机内参 → 已生成的 RGB 全部失效，故 v2 不动。将来重做相机标定时优先试 `cameras.near: 0.01 → 0.03` 并复测 ep15。
- **2026-10-20 追加（当前场景 96² 复测）**：当前唯一 config `pick_cold_can`（robosuite 重网格场景）在 **96²** 下尖峰远大于旧场景——45 步里 agentview **最多 122 s**、eye_in_hand **242 s**，单集墙钟 ~985 s。复测时临时加过合成热像通道（后已回退，见 R45）：同 seed 关掉该通道后同一批尖峰逐次复现（eye_in_hand 242.2 vs 241.8 s、agentview 120.2 vs 119.5 s），热像分割本身 6.3 ms/次、45 步合计 0.3 s。尖峰按相机计时落在 RGB 调用上，且**两种相机都会中招**——说明主因在场景更新/光栅化而非某台相机。全量 M9 前须先压掉它，否则 200 集会被尖峰支配。

### R21 — 材质与物体几何在数据集内完全混淆

`scene_seed = cfg.seed + 1000*(index+1)` 取的是**材质列表下标**，而物体半径/高度/质量都在构建期由该 seed 抽出 → 每个文件里"材质"与"这一个物体"是 1:1 绑定（aluminum 永远是 seed-1000 那根柱子）。

- **影响**：做 VLA 训练没问题（多样性足够），但**不能用于"固定几何、只变材质"的对比分析**——材质效应无法从本数据集单独辨识。
- **防呆（2026-10-03）**：`generate.py` 在 `--material` 只选子集时打印各材质的 scene seed 并与全量运行对照。此前正是分两次跑（copper 落到下标 0）产出过与 aluminum **逐位相同**的 copper 文件（actions/proprio/contact/tactile/两个 RGB 全为 0 差，仅热通道差 ~1e-3 C），已单次全量重跑修正。数据卡须标注此混淆。

### R22 — 触觉"压力突变"不是动力学，是接触门控 + 探针预压几何

① `tactile/depth`/`force` 的 SDF 查询只对 rigid solver 报过接触的 geom 运行（`_kernel_build_sensor_geom_idx` 的 geom 列表逐帧取自 `collider_state.contact_data`）→ 接触前无论多近都恒读 0；② 探针浮在指垫表面**外** `probe_offset_m` = 0.5 mm → 门一开的那个 substep，探针中心已在物体表面内 ~0.5 mm，`depth ≈ probe_radius(4.0)+0.5 = 4.5 mm` → `force ≈ 1000 N/m × 4.5 mm × 12 taxel ≈ 45 N`；③ 之后真实的"压实"只有 ~0.25 mm（≈3 N），于是该通道在抓取期近似二值（0 / 45–48 N），动态范围 ~5%。

- **实测**（2026-10-03，aluminum，3 seeds × 2 速度）：`controller.finger_speed_m_per_s=0.04`（物理闭合放慢 ~10×，episode 34→42–44 步、+25% 时间、成功率不变）下，注册接触的 substep 手指只动 **0.01 mm**、物体位置不动，`depth` 仍 **0 → 4.4688 mm** 一跳；该集全部非零 depth ∈ [4.459, 4.649] mm（基线 [4.316, 5.670]）；两个速度下 policy 级最大单步跳变都是 **~45–47 N**。
- **结论**：降低夹爪速度/换归约方式都不能得到平滑受力曲线（此前把阶跃归因于 20 Hz `median` 硬切换的结论已修正——tie-break 只解释同帧 `force>0 & depth==0` 的 2/133 帧矛盾）。要平滑只能动传感器侧：减 `probe_radius_m`/`probe_offset_m` 压低"触碰基线"（半径 4→1 mm 时台阶 45→~14 N，仍是台阶只是矮），或改用 `elastomer` 模态，或用 `proximity`（R23）。绝对幅值由 `normal_stiffness` 定标，等比缩放不改变曲线形状。数据卡须标注 force 通道的准二值性质。

### R23 — 渐变曲线在 `proximity` 模态里，且实测确实"从无到有"

`gs.sensors.ProximityTaxel` 走**被跟踪 link 的点云 BVH**（`_kernel_point_cloud_proximity_taxel_bvh`，探针球 vs 点/`sphere_intersects_aabb`，全程不读 `collider_state.contact_data`）→ 无接触门控；读数 = `stiffness × (density_scalar/N_pc) × Σ_i (R − ‖p_i − o‖)`。

- **实测**（2026-10-03，aluminum，`finger_speed_m_per_s=0.02` → 0.04 m/s，substep 5 ms）：proximity 在指垫离物体**每侧尚差 ~3.4 mm**（gap 49.7 mm，物体直径 42.8 mm）时从 0 起步——即 `probe_radius 4 mm + probe_offset 0.5 mm` 的球面触及时刻——随后 **44/44 个 substep 每个值都不同**（连续增长 0 → 0.73，无台阶），直到 44 步（0.22 s）后指垫真正接触，`depth` 才 0 → 4.42 mm、`force` 0 → 46.1 N（与 R22 一致）。采样密度 500 → 4000 几乎不改变起点（255 vs 256）与形状（顶端差 ~1.4%）——起点的 3.4 mm 是几何（球半径+预压）不是采样量化。
- **记录速率（20 Hz + median 归约）下 ramp 占 5 个 policy step（0.25 s）**：实测 HDF5 `obs/tactile/proximity` step 23 起 0.0026 → 0.084 → 0.258 → 0.458 → 0.616（step 28 才 depth 跳变）。
- **边界**：proximity 是点云近似，幅值依赖采样密度与 `density_scalar`（默认 100），**不可与 KinematicTaxel 直接互比幅值**；只对 `track_link_idx`（=物体）敏感，对指垫/桌面无响应；`probe_local_normal` 必须传指垫法线。
- **已实现**：`tactile.modalities` 可加 `proximity`（本次经 `/tmp` 实验副本验证，正式 config 未改，见 Q5）；新增 `TactileConfig.proximity_sample_points`（默认 500 = 引擎默认）；`build_finger_sensors` 对 elastomer/proximity 传 `probe_local_normal`；`obs.VECTOR_MODALITIES` 补 `proximity`。

### R24 — 双物体场景（format 1.2）依赖的四条假设

① 引擎 marker 场传感器真能同时跟踪多个压头（`track_link_idx` 复数）且 box 的 SDF 查询可用；② `n_sample_points` 传 **int** 会被引擎按 link `divmod` **均分**（两个物体 = 每个密度减半）——必须传逐 link 列表；③ 非目标物体是**热且可动**的刚体，机械臂接近路径可能撞到它；④ 物体间热耦合不存在（相距 0.24 m + `convection_coefficient: 0`）。

- **已验证（2026-10-07，`sim/configs/lift_object_pair.yaml`）**：①+box SDF — Gate 1 全绿（0 failures/0 warnings；两个温度栅格各自播种/读回：目标 130.0 C、非目标 21.0 C）；场景探针核对引擎实参 `track_link_idx=(15,16)`；**8 集生成 8/8 成功、validator 0 issues**，box 集 `tactile/elastomer` 两指 max\|disp\| = **26–80 µm**（非零 → 多压头跟踪 + box SDF 都工作）。② elastomer/proximity 均传 `(500, 500)`。③ `min_object_gap_m: 0.02` 守卫（基距 0.24 m + jitter ±0.04 下不触发）。④ 不做物体间热耦合。
- **接触形状（2026-10-08 修复）**：早期实现把 `object_yaw + jitter` **整体**取整到 90° 整数倍 → 对齐的是**世界轴**，残余角 = 物体自身 yaw（box jitter ±17°）；5 集实测残余 ≤6° → depth spread 0.10–0.67 mm；残余 10.7°/11.3° → spread 2.1–2.4 mm、marker 峰值只剩 7–8.7 µm（手指楔在棱上）。**已修**（`expert.py`：抖动先量化选面、再加物体 yaw）：同 seed 重跑 5 集，yaw −10.7°…+11.3° 全覆盖，depth spread 全部 **0.03–0.26 mm**、marker 峰值 **35–41 µm**，8/8 成功；圆柱路径逐位不变。
- **修后形状对照**（同文件相邻两集 box vs cylinder，共享刻度）：marker 场里 box 压印**铺满整块指垫**（16/16 列，峰值 36 µm，剖面平缓、两端仍有 ~17/10 µm），cylinder 只占 **11/16 列**（8.9 mm，33.5 µm，剖面更尖、两端跌到 ~6/1 µm）；刚性 depth 探针 box **0.09/0.10 mm** spread，cylinder **1.65/1.73 mm**。但 **marker 场整体相似度确实高**——gel 弹性核把压力分布平滑成宽穹顶；形状的强信号在 **depth 通道**与印记**宽度**上（8.9 vs 13.0 mm）。
- **兼容性**：9 个旧文件（404 demos，format 1.0/1.1）validate 0 issues；单物体路径 N=1 不抽 `target_index`，rng 流不变。**2026-10-08 起 `lift_object_pair.yaml` 不再记录刚性 depth**（改 elastomer-only 9×7 + 稠密高度图，R25），故该 config 的形状强信号只剩 marker 场宽度（box 铺满 7×9 格 vs cylinder 4×9 格）。

### R25 — 触觉对齐 UniVTAC 形态：63 点 marker 场 + 去刚性探针 + 稠密高度图

（format 1.3）两个必须先讲清的机制：① **弹性核的物理宽度随格距变宽**——HydroShear 的 dilation 在 FFT 格上做，核宽以**格**为单位，格距 0.81 mm → 2.04 mm（16×16 → 9×7）时同一个核摊开约 2.5×；② **稠密高度图只能是插值合成**——本项目 gel 是虚拟传感器（SDF + 网格采样），场景里没有可形变几何体给相机拍。

- **决定（2026-10-08，用户定）**：接受核宽展宽，**不重标 `lambda_d`/`lambda_s`**。机制判定：把 16×16 的场在 63 个标记**位置**上重采样，预测峰值 **26.3/36.0 µm** ≈ 原值——"采样位置"几乎不掉幅度，所以掉幅度的只能是核宽（实测峰值 27.4/36.0 µm → **5.8/7.9 µm**）。形状信号仍在（box 压印满 **7×9** 格、cylinder **4×9** 格；16×16 时是 16×16 满 vs 10×16）。代价是 63 点场绝对值小（~6–8 µm），需要绝对幅度时用 16×16 的旧 config。
- **专家不依赖探针**：接触检测回退 `contact/link_contact`，elastomer-only 实测 2/2 成功；**gate1 的两条断言已条件化**（模态没开时打印 `modality disabled` 而不是 FAIL），并新增 marker 场响应检查与轨迹表 marker 列。
- **高度图**：`tactile.height_map {enabled, size, interpolate}` 把 marker 场重采样成 `(T,L,H,W)` 米（§4.2 末）；`validate.py` **重合成比对**而不是比峰值（栅格不落在每个样本上，孤立尖峰丢 ~1.5% 峰值）；不进 burst。
- **实测（2026-10-08，aluminum，seed 5，2 集 × 46 步）**：Gate 1 全绿（marker 响应 17.81 µm）、生成 2/2、validator 0 issues、Gate 2 回归全绿（通道表含 `obs/tactile/height_map (50,2,64,64)`）。**独立复核**（自写 numpy/scipy 重推，不走 `tools/height_map.py`）：`(46,2,64,64) f32` 全有限；**从 marker 场重合成与存档逐元素 `max\|Δ\| = 0.000e+00`**（两集精确相等）；峰值就等于场峰值（5.7762/7.8944 µm）。**边距**：只给 `mode="nearest"` 时 scipy 样条预滤波的 mirror 边界会让最外 4 个像素从 5.05 µm 漂到 5.17 µm，**已改为先把采样坐标 clamp 到标记范围**——现在最外 3 行/列 `max\|Δ\| = 0`。**压印形状（峰值帧）**：cylinder 超过峰值 50% 的区域 = **4/7 列 × 9/9 行**，box = **7/7 列 × 9/9 行**。逐指峰值：cylinder 5.78/5.65 µm、box 7.89/4.92 µm（指垫 patch 14.2×13.8 mm，格距 2.033 mm 列 / 1.536 mm 行）。**体积**（仅开关 `height_map`）：4.27 vs 3.63 MiB → **+0.32 MiB/集（+17%）**。**兼容**：旧 format 1.0 文件 validator 0 issues；`pick_hot_cylinder.yaml`（force+depth+16×16）Gate 1 0/0，`depth 4.584 mm` / `force 4.584 N` 断言原样通过。**渲染**：gel 面板优先用 height map，`--markers` 晶格叠在同一张图上。

### R26 — 指尖温度"4×4"不是 4×4 指垫贴图

栅格几何完全由引擎的 link-AABB 规则决定（`voxel_size = aabb_extent / grid_size`，`temperature.py:613`；`TemperatureGrid` 选项只有 `grid_size`，**没有显式位姿/尺寸旋钮**）。

- **实测几何**（读 `_shared_metadata`）：左指 `aabb_min=(-22.732, 13.978, 53.849) mm`、`extent=(45.527, -0.0094, -53.717) mm` → `voxel=(11.382, -0.0023, -53.717) mm`（右指 y 镜像）。于是 `[4,4,1]` 实际是：x 向 4 格真间距 11.38 mm（格心 −17.042/−5.660/+5.722/+17.104 mm），**y 向 4 格总跨距只有 7.0 µm**（AABB 两角点反变换 bug，与 R16 同源），z 单格 = 整指长 53.7 mm。指垫（`probe_local_pos` x∈[−6.12,6.08]、z∈[38.8,51.09] mm）归一化后 u_x∈[1.46,2.53] → **只能落 ix∈{1,2}**；u_y≈±6170 → clamp 到端列；u_z∈[0.05,0.28] → iz=0。
- **影响**：录出的 `(T,2,4,4,1)` 只有 **1–2 行**会变（行 1/2 是垫覆盖的 x 条带，ix=2 的分界 x≈0.03 mm 恰好是垫中线，接触点落哪侧就点亮哪行），行 0/3 对着手指本体、垫接触够不到；4 列是同一 7 µm 薄片的重复。**实测**（aluminum seed 5，46 步）：cylinder demo_0 左指行 1 +21.19…+21.36 °C，行 0/2 **+0.0019/+0.0023**、行 3 −0.0003；右指仅行 1（+13.45…13.82）。box demo_1 行 1/2 仅 +0.10/+0.06（左）与 +0.56/+0.56（右）；同一"行"内 4 列差 ≤0.36 °C（≤2%）。**行 0/3 恒 21.00 且不可能被点亮**：对流/辐射全 0（R14）＋扩散长度 √(αt)≈0.17 mm ≪ 11.38 mm 格距（α = k/ρcp = 1.3e-8 m²/s）。
- **做法**：v1 **不改**——该通道语义 = "哪根指 + 指宽方向哪个条带 + 多热"的粗信号，**不是**指垫 2D 温度图；数据卡须如实标注。不要靠调 `grid_size` 修（换任何 `(nx,ny,nz)` 都仍在 AABB 三轴上，且会改数组形状、动 schema）。真 4×4 垫图需 v2：引擎侧打补丁/子类化 `TemperatureGrid`。物体侧不受影响（`object_grid [4,4,4]` 实测为正常三维格：voxel 10.68/10.68/14.52 mm、span 32/32/43.6 mm）。**1.5 起指尖改官方最粗单格、不再走 AABB 切格**（R33/R36）。

### R28 — 夹爪几何的独立简化文件 `sim/gripper.py`（2026-10-10）

管线用的 `finger_layout.derive_finger_layout` 是通用推导器（box/mesh 两条碰撞路径、`box_override`/`inner_dir_override`、joint-origin 回退、`FingerLayout`、HDF5 打包）——想单独查"这个夹爪的指垫几何"要跨模块、穿过一票管线逻辑。用户要求做一个**专门写夹爪的文件**：定位 = 独立简化版（写死 Panda、不留通用分支/override），范围 = 只几何（垫面/闭合轴/marker 栅格），运动学留在 `RobotConfig`。**管线零改动、不 import 本模块**（接受两个事实来源，代价用可执行断言对冲）。

- **等价性实测**（`scripts/probe_gripper_model.py`；2026-10-10 当时含 `PadBox` 对照共 21/21，2026-10-14 删除 pad 路线后余 **8/8**）：marker 栅格/垫面中心与尺寸/内向 vs `derive_finger_layout` **max\|Δ\| = 1.72e-9 m**（= float32 存储 ulp）、闭合轴**逐位相同**（hand 系 `[−0.000582, −1.0, 0.0]`）；每指 **63** 点、marker 栅距 2.0327 × 1.5365 mm、覆盖 patch 14.2287 × 13.8285 mm。
- **回归**：`gate1_sensors.py --config lift_object_pair.yaml` **0 failures / 0 warnings**、`c=1.22e-01`（`(1,1,1)` 单格，2026-10-14 现状；pad 时代为 `1.26e-02`）。**边界**：`describe()` 的长度/宽度顺序 = 覆盖 patch 大者在前；`PadBox`（指垫温度格盒）随 pad 路线于 2026-10-14 一并删除。

### R29 — 包结构重组：`tempvla/` → `sim/`（2026-10-11）

全部模块（config/scene/actions/expert/obs/rotations/gripper + sensors/recording/compat/tools）属**物理仿真与数据集构建**层，与未来的模型代码分层；`tempvla/` 留给 VLA 模型，仿真/数据层整体改名 **`sim/`**。

- **执行**：`git mv tempvla sim`（23 个 tracked 文件 rename 历史保留；未跟踪的 `gripper.py` 随目录移动）+ 两条 sed 全库替换——`\btempvla\.` → `sim.`、`tempvla/` → `sim/`，覆盖 ~45 个文本文件。
- **不变量（绝不随之改）**：HDF5 `FORMAT_VERSION`、文档/配置里的格式串 `tempvla-1.0…`、conda 环境名 `tempvla`——动了会破坏校验与历史记录。无 editable 安装，导入全靠脚本 `sys.path.insert(仓库根)`，**无需 pip 步骤**；pyproject 仅改发行名与 `packages.find include=["sim*"]`。
- **实测（2026-10-11）**：`compileall` 全绿；**17 个 sim 子模块逐一 import OK**；`probe_gripper_model.py` 全过（当时 21/21；2026-10-14 后 8/8）；`python -m sim.gripper --urdf panda.urdf` 输出与 R28 文档值逐项一致；`gate1_sensors.py --config lift_object_pair.yaml` **0 failures / 0 warnings**；`validate_dataset.py`（format 1.0 旧文件）**0 issues**——格式串未被触碰的直接证据。终审 grep：残留 `tempvla` 仅剩格式串、conda 环境名与 `tempvla/` 预留说明。

### R30 — 数据集根 `datasets/` + 侧车三件套（2026-10-11）

用户要求"新建文件夹放生成数据 + 便于 VLA 训练/验证/评测的格式"。**canonical 数据仍是 robomimic HDF5**（契约与 LIBERO 同构都不变），新增统一根 `datasets/`，按 format 版本分发布目录。

- **三个侧车 JSON**（`scripts/build_manifest.py`，纯 h5py/numpy/stdlib、不 import genesis）：`index.jsonl`（首行 manifest 头 + 每集一行含全部 episode attrs）、`splits.json`（规则+计数+逐 split seed 列表）、`stats.json`（归一化统计 + 通道清单 inventory）。
- **划分规则必须是 seed 的纯函数**：`u = int(sha1("temperaturevla-split-v1:<seed>")[:8],16) % 10000 / 10000`，<0.8 train / <0.9 val / else test——与目录/材质/生成批次无关，否则增量生成或按材质拆跑会把同一 seed 分进不同 split（R21 教训）。
- **stats 只在 train split 的成功 episode 上算**（val/test 泄漏防护，OpenVLA `dataset_statistics` 同口径）：小键（每样本 ≤512 元素）逐元素 mean/std/min/max/q01/q99，大键流式 absmax/nonzero_frac，非 float 只进 inventory。本期不做 RLDS/LeRobot 转换。
- **`.gitignore` 四行**：`data*/` + `!datasets/` + `datasets/*` + `!datasets/README.md`（整目录被忽略时无法反包含文件，必须先反忽略目录本身）。
- **`generate.py` 同步**：删掉旧的 `output_dir == "data"` → `data_cam` 硬编码改道，改为启动时显式列出**将被截断覆盖**的已存在文件（`h5py.File(..., "w")` 是截断写）。
- **实测（2026-10-11，当时的三套旧语料；2026-10-12 随 R31 删除——以下数字是 `build_manifest.py` 的验证证据，不再是仓库状态）**：`data/` 4 文件 200 集、seeds 44/4/2 → 集 **176/16/8**、12 个 float 统计键 + 13 个通道清单；`data_cam/` 的 seed 划分与 `data/` **逐项相同**（跨目录一致性的直接证据）、15 个通道清单；stats 语义抽查：指尖温度 q01 = 21.0 °C（环境温）、物体 60–200 °C、左指只有 2/4 条 x 条带升温（R26）。

### R31 — 旧栈全量数据删除（2026-10-12）

`data/`（pick_hot_cylinder 4 材质 × 50 = 200 集、无相机、12.9 MiB）、`data_cam/`（同 200 集 + 128² 双相机 RGB、97 MiB）、`data_lift/`（lift_object 早期试样 4 集 + 2 段演示 MP4、2.5 MiB）——全部 **format 1.0**、2026-10-02 生成——经用户决定删除。命令 `rm -rf data data_cam data_lift`（由用户执行，目录内的 9 个侧车 JSON 一并删除）。

- **理由**：1.3/1.4 栈已就绪，旧语料含 R19（触觉不可位复现）/R21/R26 等已知语义缺陷，保留两代数据的维护成本高于重生成成本。**不可回滚**（`data*/` 从未入库）。
- **后果**：M9 重生成前**仓库内没有完整数据集**（有意为之）。
- **受影响引用已全部同步**：`report_figures.py --data-dir` 默认 → `datasets/v1.4/hdf5`（图 1 在 M9 前无法重生成；`docs/figures/` 的归档图仍是旧语料的交付物）、`render_video.py`/`build_manifest.py` 示例路径、configs 注释、README/status/数据卡。本文档 §8/§9/§10 历史条目里对 `data/`、`data_cam/`、`data_lift/` 的实测描述**均指这批已删除语料**（保留为记录，不再可复现）。

### R32 — 热像 demo：木桌上两个不相触立方体的"自然冷却"mp4（2026-10-12）

新增 `sim/configs/two_cubes_wood.yaml` + `scripts/render_thermal_video.py`：与数据集 episode 同一条 `build_scene` 栈（Panda + 木桌 + 触觉/温度传感器 + 双相机，无 expert），桌上两个 4 cm 铝立方体（`place_xy` 相对**桌心** ±0.15 m → 实测 300 mm 中心距 / 243 mm 面间距，70/10 °C）在 27 °C 房间里趋于室温；每帧面板 = RGB｜THERMAL（伪红外）｜色标｜温度曲线，底部条 = 两物体 `objects_grid` 的中截面场。非任务路径（`sim_dt` 仍 0.005，demo 用 `--dt` 覆盖）。

- **为什么必须注入**：引擎自身散热项按 R14 全项目关闭，不注入则场景完全绝热；而直接打开引擎的辐射/对流也不行（缺面积因子、强 ~1e4×）。另：本 config 把 `environment_properties.conductivity` 压到 1e-9。
- **标定**：`τ = ρ·cp·V/(h·A)`，**h = 12 W/(m²·K)**（4 cm 物体静止空气自然对流 ~8–10 + 线性化辐射 ~0.6（ε=0.1）~6（ε=1），取中值）；`--h`、`--size-scale`（τ ∝ 尺寸，实测 ×1.5 → r 30 mm / 583 g / τ 2025 s、×0.5 → 675 s）可复调并打印 τ。每帧按解析解 `α = 1 − exp(−Δt/τ)` 同时施加在**物体 link 温度与其全部网格单元**上（同一因子 → 场与集总体不分家），写入路径与 `seed_temperature` 相同。
- **实测**（`--seconds 3600 --video-seconds 20 --dt 0.05`）：**τ = 1350 s**，2.67 τ 后 **70.0 → 29.99 °C、10.0 → 25.82 °C，与解析解逐位吻合**；全程 243 mm 面间距、0 xy 漂移、网格均值 == link 温度（场自身 max−min = 0.000 K）；3600 s 模拟 1119 s 墙钟（3.2× 实时），500 帧 → 20 s 视频（180× 压缩）。
- **dt 取舍**：VLA 场景 ~12–14 ms/step ≈ 68–70 steps/s 且与 dt 几乎无关（刚性求解器主导）；静止立方体 z 下沉 0.05 mm @0.005 → 0.73 @0.02 → 2.45 @0.05；机械臂 tcp 沉降 2.98 mm 在 ~150 s 内即完成（150 s 与 1000 s 实测相同 = 一次性沉降）→ 取 **dt 0.05**（dt 0.005 需 ~3 h 墙钟）。
- **关桌面接触传导的理由（实测更正）**：接触热核只作用于物体的**网格**，而 `link_temps` 只与**其他 link** 耦合、从不与自己的网格耦合——真橡木（k = 0.15）下静止立方体的网格会朝桌面温度衰减（dt 0.05 下 150 s 窗口外推 **τ ≈ 1020 s**），而**它的 link 温度纹丝不动**（150 s 内 < 1e-5 K）；开着就会让 demo 的两条通道 1 h 后差 ~1.7 K。**注意这条更正推翻了一个早先的错误前提（"引擎没有网格内扩散"）**——扩散是有的（FFT 半隐式，铝的 α = 8.4e-5 m²/s 让 4 cm 立方体 ~20 s 内均温），真正的割裂在 link 温度与网格之间（同一内核也跑在任务 episode 里；代码级推断，未单独实测）。
- **本次修掉的三个缺陷**：① 底部场条原按每格各自 auto range → 改为与 THERMAL 同尺度；② 曲线末端数值标签收敛后重叠 → 先把**所有**曲线都当障碍，再由近及远向两侧试探落位；③ 终检原拿"最后一帧渲染时刻"的采样去比 `t = sim_seconds` 的解析解（差一帧，dt 大时报假 MISMATCH）→ 改为读末态。复调已验证（`--temps 120,-40 --h 25` → τ 648 s = 1350×12/25）。

### R33 — 抓取热物体的热像 + 传感器视频（2026-10-13）

新增 `scripts/render_grasp_thermal_video.py`，复用 `sim/configs/two_cubes_wood.yaml` 的完整 VLA 场景与 R32 的注入式牛顿项，但这次由 shipped `PickExpert` 驱动 Panda 抓住 **70 °C** 的那个立方体——R32 没有接触，这一卷补上接触。画面 = 上排 RGB｜THERMAL｜色标｜曲线，下两条横贯全宽：**TEMPERATURE**（两立方体 `objects_grid` 中截面 + 两指尖单格实测读数，共用同一色标）、**TACTILE**（两指垫 64×64 `height_map` + 逐帧读数：压深/合力/接触布尔/接触点数与每点面积/指垫网格范围/finger link 温度）。默认时间轴 = 5 s 实时 + 160 s 时延保持（每帧 8 步，×10）→ 20 s 视频；RGB 面板左上角常驻 HUD 显示 `phase` 与仿真秒数。

- **① 指尖的热源是它自己的那一个网格，不是链接温度**：THERMAL 面板按每 body 的 link 温度上色，**唯独两指尖改用该单格**。实测两根 `*finger` 的 link 温度**全程印作 27.00 °C**（`link_temps` 只与**其他 link** 耦合、从不与自己的网格耦合——同一内核第二段循环确实会给 finger link 一个 `−dt·power/(ρcp·V_link)`，但物体侧 link 热容 ρcp·V = **155.5 J/K** 是指尖单格有效热容 C_engine = **0.219 J/K** 的 ~710×）。照实上色会渲染出"冷夹爪抓着热立方体"，与同帧的实测通道自相矛盾。
- **② 指尖的 τ 必须单独标定（显示项）**：引擎的有效格体积 V = 单格 `voxel_volume` 22.97 mm³ + `gs.EPS` 119.2 mm³（**引擎常数占 84%**，R16）→ C_engine = 1.54e6 × 1.422e-7 = **0.219 J/K**；"房间腿"按 h·A_face = 12 × 305.949 mm² = **3.67 mW/K** 注入 → **τ ≈ 60 s**（A_face = `pad_box_size` 非零分量之积 = 17.5 × 17.5 mm，不是 marker 栅格的 14.2 × 13.8 mm）。注入是显示项：shipped 场景 convection = 0（R14），单格也没有垫内扩散。
- **③ 左指尖"先冲高再回落"归因于引擎接触面积的 <3 点回退分支**：`_kernel_compute_contact_areas` 把接触按 link 对分组，**组内 ≥3 点**用接触点凸包多边形面积，**<3 点**回退为 `group_area = Σ_k π·depth_k`（把**穿透深度当半径**，本抓取穿透只有 0.02–0.10 mm），再除以点数得每点面积。实测左指自 t≈2.1 s 起只有 **2 个接触点**（深 70 + 90~100 µm）→ 每点分到 **A = 261.0 mm²**；t≈10.1 s 出现第 3 点后改走真三角形 → **A = 43.41 mm²**，面积降 **6.0×**。而升温率 **∝ A²** → 单格有效加热系数 **λ = 1.09 → 0.030 /s（τ 0.92 → 33 s），差 36×**（逐接触点口径；**按格内求和应为 24×，见 R37 的更正**）。这也解释了右指（全程 3 点、A = 43.410 mm²）为什么只是缓慢单调升温。
- **实测**（165 s 协议 `--no-video`，aluminum）：左指尖（2 点阶段）~3 s 内冲到 **69.50 °C**（几乎就是立方体自己的 70 °C），t≈10 s 起单调回落到 **59.35 °C**（同帧实测读数 59.30）；右指尖 **62.06 °C**（rise fit τ **9.08 s**、R² 1.0000、64 点）。两指 link 温度全程 27.00 °C；夹持力 2.18–2.22 N，物体抬起并保持（z 0.4199 → 0.5196 → 0.5181 m）；物体侧 70.0 → **65.05 °C** 与纯牛顿解析解在打印位完全一致（**contact took −0.0 mK**），cube 0 网格 mean 64.84、spread **7.0 mK**（1.4 时代同场景读 2.5–2.6 mK，R19 同源的接触记账抖动），cube 1 10.0 → 11.96 °C、spread 0.0 mK。
- **给 M9 的结论**：指尖热信号的**幅值取决于求解器报几个接触点**——那是碰撞器记账、不是物理量；同一次夹持在 2/3 点之间切换就能让升温率差 24–36×，且本抓取的前 ~8 s 恰好落在"高估"的那一支上。读数约定见 R36。

### R34 — 换材质对照：材质只从 ρcp 与 k 两个量进入热学（2026-10-13）

做法 = 同一份 `render_grasp_thermal_video.py` + 同一 `two_cubes_wood.yaml`，只换 `--material`（aluminum / copper / steel / abs），**其余全不动**：质量 173 g、几何、时间轴、种子、h = 12 W/(m²K)。

- **为什么这是一次干净的对照**：材质在整条链路上只有两个入口——① 温度传感器的 `material_properties` 提供 **ρcp**（→ 物体降温 τ = ρcp·V/(h·A)）与 **k**（→ 网格内部 α = k/ρcp，以及指垫串联导热 `k_eff = 2·k_pad·k/(k_pad+k)`）；② **刚体侧与材质无关**——`scene.py:580` 用 `rho = mass_kg/volume_m3`（R18），且 **friction 硬编码 1.0**。所以四卷动力学逐位相同：实测夹持力 **2.18..2.18 N**、z 0.4200→0.5196→0.5181 m。
- **实测**（每卷 165 s）：`aluminum`（ρcp 2.43e6）τ **1350 s** → 65.05 °C、全网格 spread **2.6 mK**；`copper`（3.45e6）τ **1916 s** → **66.45 °C**（解析预测 66.452）、spread **1.8 mK**；`steel`（3.69e6）τ **2050 s** → **66.67 °C**、spread **14.9 mK**；`abs`（1.46e6）τ **809 s** → **62.07 °C**、spread **285.9 mK**。四卷都与 `27 + 43·exp(−165/τ)` 在打印精度内重合（各自 0.05 mK 内），接触项对 link 温度的贡献 ≤ **0.2 mK**。
- **① 物体侧通道能分辨，abs vs 金属是"一眼可辨"级**：τ 跨材质 **2.53×**（abs 809 ↔ steel 2050），165 s 端点温差 **4.60 K**；全网格 spread 跨材质 **159×**（copper 1.8 → abs 285.9 mK）。机制：spread ∝ 接触格把热摊给邻居的速度 ∝ **α = k/ρcp**（copper **1.12e-4**、aluminum 8.44e-5、steel 1.22e-5、abs **1.374e-7** m²/s = 铜的 **1/815**），10 mm 体素的特征扩散时间 L²/α 分别是 **0.9 / 1.2 / 8.2 / 728 s**。实测 spread 顺序与 1/α **严格反序**；abs 在 165 s 里扩散长度只有 √(αt) ≈ **4.8 mm**，连一个体素都没跨过去。steel/aluminum 实测比 5.7× 对 α 预测 6.9× 同量级（严格 1/α 标度只在准稳态成立）。
- **② 指尖通道不能**（`k_pad = 0.02` 时代）：`k_eff` 四材质只从 **0.036364**（abs）到 **0.039998**（copper）= **1.10×**，被指垫自己的 `k_pad` 卡死；实测指尖极差 **1.031×**（aluminum = steel **逐位相同**）——而且这点小差来自**物体表面温度**跟随 ρcp，**不是 k**。
- **③ 触觉通道完全不能用**：各卷 `height_map` 压深峰值都是 **0.0069 mm**、保持段合力都是 **2.18 N**（材质不进摩擦/法向刚度）。**金属之间（Al/Cu/Steel）在 165 s 窗口内也不可分**：端温 66.67 / 66.45 / 65.05 °C 全挤在 **1.62 K** 内，铜与钢只差 **0.22 K**。
- **两条边界**：① 本对照**刻意固定质量 173 g**，而真实数据集里质量由 `object.mass_kg` 独立采样、与材质解耦并被 scene_seed 混淆（R18/R21）——"材质可从热通道读出"只在**热学量**上成立。② **视频里的物体条是 4×4×4 网格的中截面（y = 2）**，而指垫压在 y = 0/3 的面上 → **条上印的 spread 会少报**（aluminum 印 0.9 mK、abs 印 2.8 mK，全网格是 2.6 / 285.9 mK）；要靠**全网格 max−min** 读材质。
- **注（2026-10-14，图 5 按 shipped 口径重画）**：`docs/figures/material_thermal_contrast.png` 已改用 shipped 通道（指尖单格 + `k_pad = 0.25`）的四卷 165 s 同协议日志重画——全网格 spread **320.95×**（copper **4.3** → abs **1380.1 mK**，顺序仍与 1/α 严格反序）、指尖 **rise τ（tip R）1.82×**（金属 9.08–9.18 s ↔ abs 16.50 s；tip L 因 R33 的 2→3 点切换拟不出）、指尖峰值仍饱和 **1.01×**、力 **1.02×**。两处口径必须分清：① spread 幅值随 `k_pad` 走（同一扫参协议下 abs spread k=0.02 → 0.25 为 414.7 → 2280.9 mK，5.5×），故本文的 159× 与新版图的 321× 是**口径差**不是矛盾；② 上面 ② 的"指尖通道不能"是 `k_pad = 0.02` 的结论——shipped 通道下指尖 rise τ 能给出 abs/金属 ~1.8× 的间隔（R35/R36）。

### R35 — 指尖材质对比的结构性天花板：`k_pad` 而不是分辨率（2026-10-13 扫参定案）

R34 实测指尖通道跨四材质只有 **1.031×**，机制不在网格粗细而在 `k_eff = 2·k_pad·k_o/(k_pad+k_o)`：`k_pad = 0.02` 时 k_o ≫ k_pad 使 `k_eff → 2·k_pad = 0.04` 对所有金属**同时成立**，换更细的网格、更长的时间窗都改变不了。

- **扫参做法**：`render_grasp_thermal_video.py --finger-k K`，同一 config、同一时间轴（200 帧 × 45 s）、同种子，**铝与 ABS 在每个 k 下跑同一套设置**。三档 k = 0.02 / 0.10 / 0.25。
- **解析对比度**：k_p=0.25 → k_eff = 铝 **0.49939** / 钢 0.49724 / 铜 0.49967 / ABS **0.22222** / 木(0.13) **0.17105** → 铝:ABS **2.248×**、钢:ABS 2.238×、铜:钢 **1.005×**、ABS:木 1.299×；k_p=0.10 → 铝:ABS **1.499×**；k_p=0.02 → **1.100×**。
- **判据用的量与线性**：左指尖峰值升幅对 ΔT = T0−室温 归一——铝 0.0692..0.0694（T0=60/130/200 三点 spread **0.33%**）、ABS 0.0352..0.0355（**0.76%**）→ 升幅**线性于 ΔT**，比值与 T0 无关。
- **实测（当时 1.4 几何）**：k=0.25 → 铝:ABS = **1.96×**（45 s 窗；早期窗 t=2.5 s 为 **2.10×**）达到解析 2.248× 的 **87%**（余量 = 指垫自身损耗 + 逼近饱和）；铜:钢 = **1.003×** 与解析 1.005× 对上——金属间不可分。k=0.10 → **1.30×**（45 s）/**1.46×**（2.5 s）。k=0.02 → R34 那卷 165 s 协议下 **1.07×**。
- **稳定性**：k=0.25 下 gate1 `_thermal_diag` 实测最坏放大 **c = 7.12e-02 /子步**（当时 1.4 几何；shipped `(1,1,1)` 单格 2026-10-14 实测 **1.22e-01**；要求 <0.5）、无 NaN、垫温量级正常。
- **定案**：`finger_properties.conductivity = 0.25`（物理硅胶 0.2–0.3）。shipped（1,1,1）几何下的同一对比度与实测间隔见 R36 的 V6 矩阵。

### R36 — 指尖温度通道定案（format 1.5）：官方最粗栅格单格 + 官方非理想性，只录 measured（2026-10-13 用户定案）

- **实现**：`sim/config.py` 默认与 5 个 config：`finger_grid: [1,1,1]`、`sensor_time_constant: 0.05`、`noise: 0.05`、`resolution: 0.1`、`record_ground_truth: false`、`finger_properties.conductivity: 0.25`（R35）。非理想性只挂指尖（`_finger_imperfection_kwargs` 仅在 `kind == "finger"` 并入），作用顺序 = RC → random_walk → noise → bias → resolution，全部在 measured 缓冲；物体传感器保持 0/0/0，`read()` ≡ 真值逐位。
- **暖机**：引擎 reset 时把 measured 缓冲清零，所以 runner 在录第一帧前先空跑 `ceil(5τ/sim_dt)` = 50 个 sim 步（0.25 s = 5τ）暖机、再 `controller.reset()`——实测录到的 step 0 = 20.8–20.9 °C（室温 21.0，差即白噪/量化），第 2–3 步收敛；不加暖机首帧会是一段 ~0.7 °C 的假"冷启动"斜坡。
- **τ 定案依据** = 同一 20 s 数据集协议（`lift_object_pair` 专家、几何/质量/位姿钉死）在 T0∈{60,140}、2 repeats 下扫 τ = 0.15 / 0.05 / 0.00 的判别表。
- **V6 判别矩阵**（每指读数 = 金属 min ÷ 非金属 max，同指内比较、repeats 平均）：**圆柱**（r 20.2 mm h 68 mm）approach_rate：τ=0.15 **1.57–1.75×**（不达标）→ τ=0.05 **2.00 / 2.31 / 2.20 / 2.33×**（四格全 ≥2×）→ τ=0.00 2.19–2.36×；同跑 GT 列 2.26–2.42 ≈ 解析 k_eff 比 2.248×。**立方**（40×40×60 mm）：approach_rate τ=0.05 **2.24–2.29×**、GT **四格恒等于 2.24 = 解析 2.2376**（平面接触 → A² 全材质一致 → 初速比恰为 k_eff 比）；dT_300ms 2.09–2.42、dT_total **2.01–2.20**、heat_rate_max 1.98–2.19；paired 最坏格 立方 2.22、圆柱 1.75。
- **机理（τ 为什么吃掉对比）**：RC 滞后与指垫耦合时间常数级联（金属 ~0.10 s、非金属 ~0.24 s；复合 ≈ τ_c + τ_rc），把"快"的金属曲线压得比"慢"的非金属更狠——τ=0.15 恰是耦合常数本身（金属初速只剩 ~40%、非金属 ~62%），2.3× → 1.6×；τ=0.05 只压缩 ~10–25%。三次扫 τ 的 **GT 列逐位一致**（τ 只动 measured 分支）。
- **读数约定（进数据卡）**：① 材质比 = **同指内**、repeats 平均（<3 接触点回退使左右指读数可差 ~10×，R33）；② 读**初速类**特征（approach_rate；dT_300ms 可用但更弱），**不读**圆柱的幅值类（dT_total 饱和 ~1.0–1.1×、dT_late 反相 ~0.45×）；③ 单集最坏格圆柱可掉到 1.75×（R33 抽签）；④ 引擎 V = voxel_volume + gs.EPS 使有效热容 38–95% 是引擎常数。
- **冲击接触的冻结过冲伪迹**（引擎 as-built 行为，非本方案引入）：碰撞型**单子步**冲击接触会在指尖单格留下**冻结的过冲**。链条：<3 点回退 `A = ε + Σπ·穿透`（量纲上是长度却直接当 m² 用），冲击瞬间穿透 ~1.5 mm 给出 **A ≈ 4.8e-3 m²**（真实指垫接触斑 ~1e-4 m²）；升温率 ∝ A² ⇒ 显式欧拉放大 **α = dt·k_eff·A²/(V²ρc)**，**稳定边界 α=1 ⇔ A > 3.53e-3 m² ⇔ 单点穿透 ≳ 1.12 mm**（k_pad 0.25、dt 5 ms、V 1.42e-7 m³、ρc 1.54e6；ABS 侧 k_eff 小 2.25×，阈值升到 ~1.69 mm）。实测（seed 0 铝，逐子步 GT）：ctrl 10 sub 7 右指格 21.00 → **98.48** → 63.73 → 63.64，sub 10 接触消失；此后 ε=0、convection=0（R14）⇒ **无任何弛豫/散热路径，格温冻结在 63.64**（高于一切热源 62.31）达 12 个控制步。**通常不成灾**：冲击后接触若持续，后续子步继续按真实面积写该格，几个子步内自纠；只有**冲击后接触立即消失**才冻结。**频率**（16 seeds × 2 指 × {铝, ABS} = 64 指，pick_hot_cylinder）：onset 前实测升幅 >0.5 °C **8/64（12.5%）**——两种材质**同一组 seed**（0、9、10、15 各一指，4/16 集 = 25%），且**全部是后接触那根指**；GT 残差 > 物体+0.2 °C **13/64（20%）**；`approach_rate` = NaN **5/64**。左右指 onset **跨材质逐位一致**（16/16）。**约定**：① 特征读法不变——**每指 onset-relative**；② 受影响指该集可按"onset 前升幅 >1 °C"弃用（特征层过滤，不改引擎）；③ 50 集/材质的规模下 12.5% 的指级报废率不改变判别间隔的口径。**复查触发**：下游训练实测判别不足 → 先复查 τ / 特征定义；仍需更大间隔 → 启用存档的物理层方案。
- **格式与验证**：`FORMAT_VERSION = "tempvla-1.5"`；文件自描述 `layouts.attrs["temperature_sensor"]`（grid / fit / RC / noise / resolution / record=measured）；`sim/tools/validate.py::_check_finger_temperature` 校验单格形状 + 根属性（捕捉静默回退录 GT）。**V1**（管线真实性：τ 阶跃、σ、0.1 台阶、物体 read() ≡ GT）、**V2**（gate1 c<0.5）、**V6**（上表）、**V7**（`generate.py --episodes 1` + `validate_dataset.py` → "OK: no issues found"；`format_version` 1.5 + `layouts.attrs["temperature_sensor"]` 全字段 + `fingertip_grid` (49,2,1,1,1) f32 + 实测读数 100% 落 0.1 °C 台阶）**已过**。

### R37 — 指尖接触面积口径补丁：compat 策略 `pad_face`（format 1.5；2026-10-14 全数据集定案）

R33 的根因是"引擎给每个接触点的面积"根本不是物理量：`_kernel_compute_contact_areas` 按 (link_a, link_b) 分组，组内 **≥3 点**用穿透角点的 3-D 多边形面积，**<3 点**回退 `group_area = gs.EPS + Σ_k π·depth_k`（量纲上是长度却直接当 m² 用），再 `area_per_contact = group_area/(count+eps)` 平分。升温率 **∝ A²** → 同一根手指只要在 2 点/3 点之间切换，增益就跳变。**R16 的 `contact_depth_weight: 0.0` 只关掉了 `max(A, π·dw·pen)` 里的每点穿透项，从未触及这个 <3 点回退**——这是早先缓解措施没能修掉伪迹的原因。

- **两个必须一起记住的连带事实**：① 手指单格（1,1,1）里**所有接触点写进同一个格**，同一步内顺序读改写，故有效增益按 **Σ Aᵢ²** 计——**R33 行"差 36×（λ 1.09 → 0.030 /s、τ 0.92 → 33 s）"是逐接触点口径，按格内求和应为 24×（λ 2.19 → 0.091 /s，τ 0.457 → 9.31 s，与实测 G_c 0.479 → 0.0199 W/K、τ 0.45 → 9.3 s 吻合），此处更正**；② 物体侧网格与 link 温度环读的是**同一批缓冲行**，同一时刻物体的接触增益也在跳（Σ Aᵢ²：2 点 1.365e5 → 3 点 5.65e3 mm⁴ = 24×）。
- **实现**：`sim/compat/sensor_compat.py::install_contact_area_policy(finger_faces)` 包住模块级内核，在引擎写完 `contact_area_buffer` 之后、同一次 `_update_raw_data` 调用内，把**与手指 link 相连**的接触行改写为 `Aᵢ = A_face/√nᵢ`（nᵢ = 该手指本 substep 的接触点数）→ **Σ Aᵢ² ≡ A_face²**，与报几个点无关；不涉及手指的接触（物体-桌面等）保留引擎口径。A_face = `pad_box_size` 非零分量之积 = **305.949 mm²**（17.486 × 17.5 mm；新增 `FingerLayout.pad_face_area_m2`）。
- **接线**：`TemperatureConfig.contact_area_policy ∈ {engine, pad_face}`；`scene.py` **每次建场景都调用安装器**（补丁是进程全局的——不显式声明就会串到下一个场景）；`sim/recording/meta.py` 把该键写进 `env_args.sensors.temperature`（**形状不变**，只让读者能区分两份否则逐位相同的文件）。**2026-10-14 全数据集定案**：6 个 config 全部显式写 `pad_face`，且 dataclass 默认值也由 `engine` 翻转为 `pad_face`——仓库内**无一处直接构造 `SimConfig()`**（22 个 `build_scene` 调用点全部走 YAML），故翻转零风险；不翻则以后新增 config 漏写该键会静默退回旧物理，正是要避免的混合状态。**实现要点**：零拷贝取字段、clamp 只作用于 `sort_idx` 的副本、无逐子步 `.item()` 同步、`finger_faces=None` 无条件还原原始内核、`contact_depth_weight != 0` 直接 `ValueError`。
- **F1 实测**（two_cubes_wood + 300 控制步 × 20 Hz）：两指均严格 `A_face/√n`（左 2 点 **216.339 mm²**、右 3 点 **176.640 mm²**），`ΣA²/A_face² = 1.000` 每步成立、**0 行失败**（引擎口径对照：2 点 261.3 mm²、3 点 43.41 mm²）。
- **渲染实测（aluminum、同 165 s 协议；对照件 = 已交付的 engine 口径 mp4）**：**两指从接触第一刻起就同步**——峰值 69.317 / 69.307 °C、rise fit τ **0.66 / 0.66 s**（R² 1.0000，各 23 点；onset 1.15 / 1.40 s，各升 42.32 / 42.31 K），与预测 G_c = k_eff·A_face²/V = **0.329 W/K → τ 0.659 s**、平衡 **64.63 °C** 逐位吻合（t=165 s 实测 64.64）；对照 engine 卷：t=3 s 左 69.90 / 右 **29.90**（差 40 K）、t=8.5 s 69.66/47.7、t=20 s 69.23/58.55、t=100 s 才双双收敛到 60.94——即伪迹是**暂态**，修的就是这段。刚体侧逐项不变（力 2.18 N、z 0.4199 → 0.5196 → 0.5181 m）。
- **必须一起记住的代价（物体侧）**：同一批缓冲行也给物体网格用，ΣA² = A_face² 让它拿到**整块垫面**的接触导（G_obj = k_eff·A²/V_cell = 0.047 W/K，engine 的 3 点阶段只有 0.0028 W/K）→ 165 s 里物体网格从 pad 抽走 ~281 J，网格均温比自己的 link 低 **2.85 K**（engine 卷 0.21 K；M9 的 20 s episode 外推 ~0.35–0.42 K，实测 t=20 s 已是 0.42 K），全网格 spread 也由 2.6–7.0 mK 涨到 **84.9 mK**；t=165 s 时出现"指尖读数 64.60 > 物体网格 62.20 < 物体 link 65.05"的排序——这正是 R17/R32 那条**单向耦合**在 1 K 量级的显形。物体 **link** 通道不受影响（`contact took −0.0 mK`）。两个口径都不标定绝对量，pad_face 换到的是"**与报几个接触点无关**"这条不变量；代价如实记账，已随 2026-10-14 的全数据集定案接受。

- **冻结过冲的对照实测（pre-M9 翻转的依据，2026-10-14；复现脚本 `scripts/probe_freeze_overshoot.py --policy {engine,pad_face}`）**：判据 = 指尖 **GT 残差 > 物体 + 0.2 °C**（违反第二定律：过冲后接触断开、convection = 0 ⇒ 无弛豫路径，格温冻结在错值，R36）。扫 `pick_hot_cylinder`、16 seed × 2 指 × {铝, ABS} = **64 指/臂**、`record_ground_truth = true`（把 RC / 噪 / 量化移出视线）：**engine 17/64 = 26.6%**（最坏 **+4.695 °C**，aluminum seed 9 右指；R36 记的是 13/64 = 20%，同场景同判据——差的 6.6 个百分点大概率是 R19 的接触检测抖动，伪迹触发条件"冲击后接触立即消失"本身逐次运行不稳）→ **pad_face 0/64 = 0.0%**（最坏 **−7.523 °C**）。**副产物（必须一起记）**：两臂的绝对值分布完全不同——engine 的指尖在 34 步内**饱和到物体温度**（残差 ≈ ±0.3 °C，靠 <3 点回退的超大面积单子步顶上去，正是过冲的同一条机制），pad_face 则**慢而不到**（铝 −8…−59 °C、ABS −19…−97 °C）；这个落差与 ΣA² 之比 **0.687**（93,605 vs engine 2 点相 136,242 mm⁴）同量级，且 pad_face 下**两指逐位相等**（−22.276 / −22.283）而 engine 分叉（+0.071 / +3.290）。**含义**：pad_face 买到的是**相位无关 + 两指对称 + 无过冲**，代价是同一 episode 长度内幅值更小；两个口径都不标定绝对量，故 R34/R35/R38 的阈值全部不受影响（`pick_by_temperature` 本就在 pad_face 口径下定标，阈值 1.04 无需重标）。

### R38 — 新任务 `pick_by_temperature`：用指尖温度变化判材质（孪生方块 + 槽位置换 + 边抓边判）

桌上**一金属一非金属、外观完全相同的孪生方块**（不设视觉材质），指令只给材质词（`pick up the metal box`）；专家靠**抓取过程中的指尖温度变化**判手里这块是什么，**判对抬走、判错放回去换另一块**。**四项定案**：新专家任务 / **边抓边判**（不做专门轻触相）/ 每物体温度**各自独立随机** / 冷热两区**都覆盖**。

- **四个结构性障碍与解**：① 一个 HDF5 只能一种材质（`properties_dict` 按文件写入）→ **一文件一材质对**（`{task_id}_{metal}_{nonmetal}.hdf5`，`material_pairs` 三对），**不**做运行期换物性；② 槽位与材质组固定则语言单独就能定位目标槽（**捷径，任务作废**）→ **每回合槽位置换**（50/50）且专家**永远先试 object 0**，`front_object_index`（判材质时恒为 0）落 attrs 作事后检查；③ 逐 step 的 `info` 不落盘 → 判据轨迹显式接进 per-demo attrs（`judge` JSON，每次抓取一条）；④ 照搬 `PickExpert` 直接读 `bundle.object_pose()` 真值 = "凭空"走向正确那块（外观相同 → **演示不可学**）→ 判据**只读 `obs/temperature/fingertip_grid` 的 measured 通道**，专家自己测、自己判。
- **物理基础：λ 与温度无关——这是"每块温度各自随机"仍然可判的唯一理由**。接触热核 `dT_tip = dt·λ·(T_obj_cell − T_tip)`，`λ = k_eff·A²/(V²ρc)`、`k_eff = 2·k_pad·k_obj/(k_pad+k_obj)`——λ 只由材质 + 几何决定，**绝对温度不含材质信息**。`k_pad = 0.25`（R35）下 k_eff 已饱和：铜 **0.49967** / 铝 **0.49939** / 钢 **0.49724**（三者差 ≤0.5%）、ABS **0.22222** → **一个阈值覆盖所有材质对**；两种金属同场只差 **1.005×**（不可判，明确不做）。
- **判据统计量**：接触起始 `t_on` 后定窗 W 内的二参数拟合 `T(t) = T_c + (T_0−T_c)·e^{−λ(t−t_on)}`（`T_c` 由物体网格通道播种、带内闭式搜索，band 0.15–0.85）；λ̂ = 1/τ̂；**逐指各拟一次**（`link_contact` 逐指上升沿——共用一个 t_on 会把 τ_R 从 0.656 偏到 0.954、约 +45%）再取均值并**记录指间差**；`\|span\| < 0.5 K` 判不可判（返回 `None`）。**必须用 measured 标定**（RC 对两组的相对压缩不同：RC/τ 7.6% vs 3.4%），阈值只能在实测通道上定、不能用 GT（R33/R36 的教训）。模块 `sim/expert/thermal_judge.py`（纯 numpy、不 import genesis）**刻意不**统一到 `render_grasp_thermal_video.py::fit_rise_tau`——后者是整段运行 + 极值后全网搜 asymptote 的离线拟合，本判据是**定窗 + 由物体通道播种**的在线估计，合一有改动已发布 R34/R37 τ 数字的风险。
- **V1 定标（`pad_face` 口径）**：铝 **+120 / −20 °C** 两跑指尖 rise τ = **0.66 s**、ABS **1.45 s**（R² **1.0000**），**冷热两边 λ̂ 逐位相同** → 定 **W = 2.0 s**（金属 λ̂ 1.500 / ABS 0.715 = **2.10×**），阈值取几何中带 **1.04**（两侧各 1.45× 余量）；**W ≤ 0.6 s 会把两组排序反过来（禁用带）**。
- **V2 盲判**（1 文件 20 集 32 抓；只读落盘 measured 通道 + `link_contact` 自行分段）：金属 λ̂ **1.409–1.490**、非金属 **0.567–0.649** → 分离 **2.17×**；**混淆矩阵 32/32、0 错**，文件自身 `judge` 记录 vs 复算分歧 **0**；按温区拆：金属热 1.413–1.452 / 冷 1.409–1.490、非金属热 0.567–0.649 / 冷 0.629（n=1）——**一个 1.04 的阈值冷热通吃**；阈值零错窗口 **0.672–1.340**。
- **窗口扫描**（同一 32 抓复算）：W = 1.0 / 1.5 / 2.0 / 3.0 / 4.0 s → 分离 **1.19（塌）/ 1.96 / 2.17 / 2.22 / 2.22×**——1.0 s 比 V1 说的禁用带更宽（塑料被压向金属），3 s 起为内禀底，故取 **2.0 s**（更短 episode）。
- **指间 λ̂ 分歧**：均值 **4.08%**（中位 2.49%），金属 1.95% / 非金属 7.62%，最坏 **17.81%**（ABS 第二抓）；有**系统符号**（后检测到接触的那根指 λ̂ 更大，最大 6 例无一例外），与 onset 差的相关只有 0.19 → 属 **onset 定位灵敏度**（ABS 的 λW ≈ 0.69，窗内只覆盖 ~50% 弛豫）而非物理。**决定性事实**：**没有任何一根单独的指越过阈值**（金属最差指 1.399 仍在 1.04 之上 1.35×，非金属最好指 0.668 在 1.04 之下 1.56×）→ **逐指判定 64/64**，故按均值判定是安全的。
- **V3 孪生渲染**（同一手写 episode、材质按位互换，正对照改桌色）→ 两相机 **0 px 差异**、正对照 **~110 k px** = **视觉零捷径**（`Rigid` 不带 `surface=`，两体几何/质量/ρ 钉死，材质只进温度传感器的 `properties_dict`）。**V4 端到端**：20/20 `success=True`、validator "OK: no issues found"（18 通道、episode 81 / 143.3 / 186 步；单抓 81–87、双抓 180–186）。**V5**：按 `\|T0−27\|` 归一后塑料体内网格 spread 是金属的 **8.2–12.6×**（中位 **11.3×**）——金属近等温（中位 0.51 K）而 ABS 存得住梯度（中位 8.63 K），λ̂ 也各自对上材质。
- **捷径审计（如实记账）**：接触**前**物体网格的梯度增长率（按 `\|T−ambient\|` 归一）金属 **3.402–3.671** vs 非金属 **14.814–14.839** mK/s/K，同集内两体比 **4.04–4.35×** → **只看不摸的策略原则上也能分这两组**；这不是专家读的通道（专家读指尖），但读者若把它当输入，捷径就成立——`judge_temperature_material.py --json` 每次都打印这段 `channel_audit`。
- **产物与边界**：`sim/expert/thermal_judge.py` + `scripts/judge_temperature_material.py`（盲判复算，真值只用于打分、从不作输入；`--sweep` 几何阈值扫描 + 捷径审计）+ `docs/figures/material_judge.png`（λ̂ vs 播种温度，冷热同带 + 阈值 + 分离度）。金属之间不可分（1.005×）；判错即 `success=False` 如实落盘（本轮 0 错）；不做专门轻触相、不做 63 格指垫温度图；**既有任务的字节不受影响**（新字段全有默认值、既有 YAML 一字未动 → v1.5 既有文件不需重生成；M9 全量重生成仍待确认）。

---

### R39 — 新任务 `put_in_tray`：三材质判材质 + 托盘放置（2026-10-15）

桌上**五个随机颜色、随机边长的立方体**（铁 / 木 / 塑料三种材质）+ 桌边一个**托盘实体**，
室温 25 °C、每个立方体各自升到随机温度；指令是「把 X 质的立方体放进托盘」，专家
**抓到手靠指尖温度变化判材质、判错放回去换一块、判对搬到托盘放下**。三组数据集对应三条指令，
每组先只出一集 + 一张观测渲染。

- **新增的物理实体：托盘**。Genesis 1.4.1 **没有复合基元 morph**、也**没有运行期改色**
  （`engine/mesh.py` 的 `set_color` 更新不了光栅化器已建的缓冲）——两条都是实测结论，
  决定了实现形状：托盘 = 五个 `fixed=True` 的 box（地板 + 四面墙，互相下沉 1 mm 避免共面），
  颜色只能**每次 build** 定（见 `TrayConfig`、§6）。摆在桌面相对 (0.0, +0.30)：
  世界 (0.40, 0.30)，腔底 0.407 / 墙顶 0.442 m，内腔 170 × 170 mm。
- **材质三元组的可判性是**先算后选的，不是试出来的。λ 对物体**只通过**
  `k_eff = 2·k_pad·k/(k_pad+k)` 依赖（R38 已证），`k_pad = 0.25` 落在常见材料的 k 区间**中段**，
  于是「木 vs 塑料」这一类**两个都低于 k_pad** 的材料对被压缩：真实木材（松 0.13–0.15 /
  橡 0.17）与 ABS（0.20）/尼龙（0.25）的 **k_eff 比只有 1.10–1.24×**，而它们的**裸 k 比也只有
  1.33–1.43×** ⇒ **这不是窗口或传感器能修的，是材料本身的分辨率上限**。
  R38 的 32/32 是在 **2.17×** 实测分离上取得的，1.2× 远在它之下（R38 实测同材质内 spread
  可达 ~14%、最坏指间分歧 17.8%）。故三元组取 **铁 k=80 / 尼龙 k=0.25 / 巴尔沙木 k=0.055**：
  k_eff = 0.49844 / 0.25000 / 0.090164 ⇒ 相邻比 **1.99×** 与 **2.77×**。
  **这一条是用户可推翻的唯一设计选择**（备选：改用泡沫类第三材质、退回 R38 的两组、
  或接受"木/塑料"只有 1.2× 而放宽验收）。
- **判据从一界变多界**：`sim/expert/thermal_judge.py::decide_bands`，N 个材质要 N−1 个边界，
  `groups` **按 k_eff 从慢到快**排（`rank_groups` 从材质表算出来，不写在配置里，边界与场景
  因此不会漂移）。记录是 `decide` 的超集（多 `thresholds`/`band`/`margins`），
  **且只在多带路径上加键** —— R38 文件的 `judge` 记录逐字节不变。
  （**§9-R40 起例外**：每个 `judge` 记录另加一个 `"method"`，`material_judgement` 另加
  `"method"`/`"min_late_window_frac"` —— "这份文件是用哪个统计量、窗口从哪个指垫起算判的"
  是离线复算的前提。纯新增键，已落盘的文件一个字节不动。）
- **窗口 2.0 → 4.0 s**：最慢一带（巴尔沙 λ̂ ≈ 0.27 /s）在 W = 2 s 时 λW ≈ 0.54，是 R38 窗口
  扫描里"塌"的那一档（W = 1.0 s 只恢复出 1.19×）；R38 同一批 32 抓上 W = 3.0 / 4.0 s 都到
  了内禀底 2.22×。本场景实测复核：巴尔沙 λ̂ = 0.2781 /s ⇒ λW = 1.11，恢复出的相邻比 2.64×。
  代价是每候选 80 步、最坏五块都要试 + `retry_on_none` 允许的一次重抓 ⇒ `max_episode_steps: 2000`。
- **`max_episode_steps` 的定值过程（一次被自己推翻的读数，值得记下来）**：探针的五体全扫就是
  shipped 专家最坏情况的**逐位复刻**（请求的那组排在最后 ⇒ 五块都要试），但它的 `success` **永远
  不会**亮 —— `_place_candidates` 取的是**回合请求的那一组**，而扫完五块后提交的是**最后一块**，
  两者常常不同 ⇒ 监视器不成立 ⇒ 循环一路跑到上限。于是 `steps == max_steps` 是**探针的必然行为**，
  **不是**"步数不够"的证据；我一度据此把 1200 改成 1800（"五体全扫在搬运尾巴还没开始就越过 1200"），
  那是把"跑到上限"读成了"还没跑完"。真正的读数是新加的 `done_step`（专家首次报 `DONE` 的那一步）：
  **768**。据此重建上限：正常路径 ≈ 768 + `stable_steps` ≈ 780；最坏加五次同体重抓（原地再抓，
  比任何一次**换体**都短，用实测最大换体间隔 197 步封顶）= 768 + 5×197 + 10 = **1763** ⇒ 取 **2000**。
  上限不是缓冲大小（集里只存实际走过的步数），所以留宽是免费的，而**留窄不是**：
  `runner.run_episode` 把截断的回合按**失败**写盘 ⇒ `put_in_tray_iron.hdf5` 会直接不出现。
- **三条捷径各自堵上**（这是三材质任务比 R38 更容易漏的地方）：
  | 捷径 | 堵法 | 事后可查的落盘字段 |
  |---|---|---|
  | 颜色对应材质 | `color_random: true`，每次 build 重抽（色相等距 + 抖动 + 打乱） | `env_args.objects[].color` / `material_judgement.colour_encodes_material: false` |
  | 槽位对应材质 | 每回合槽位置换（R38 机制，`slots = home[rng.permutation(n)]`） | `objects_group` + `slots` |
  | 试的顺序对应材质 | `judge_candidate_order: random`，每回合置换候选顺序；修 `fixed` 后"先试哪个"是常量，语言单独就能定位 | `front_object_index`（= `order[0]`，随机）+ `judge[].body` |
  | 「只看不摸」也能分 | **没堵** —— 如实记账，见下（相邻最坏 **1.22×**，比 R38 的 4.04–4.35× 窄得多） | `obs/temperature/objects_grid`（数据里就有） |
- **一个组的多个物体**：五块是 2 铁 + 2 木 + 1 塑，「把铁块放进托盘」在桌上有两块铁时**任一块
  都算答对**。专家会提交它先摸到的那块，所以成功判据不能只查 `target_index` ——
  `EpisodeRunner._place_candidates` 取**该组全部成员**，任一在腔内即成功；判错材质仍然失败
  （搬过去的那块属于**别的组**，该组没有任何一块在托盘里）。落盘 `judge[].group_true` 与
  `place_body` 共同给出"它搬的是哪一块"。
- **成功判据是场景事实**（`success_criterion: place_in_tray`）：该组某一块**最低点离腔底 ≤5 mm**
  （读引擎的世界 AABB `get_AABB()` 的 min z，对倾倒/侧躺的物体同样成立；旧写法
  `pos_z − height/2` 假设直立，会误杀一个斜靠在托盘壁上的罐子）+ 包围盒中心在**内腔矩形内**
  + **指垫已张开**（`gripper_states[0] ≤ release_open_tol`，否则"举在同一高度的"也会判成放下）
  + 专家已到 `DONE`（退回动作因此进得了记录）。
  **内腔不缩容做横向容差**：65 mm 方块合法贴墙放时中心离腔心 52.5 mm，任何收缩都会误杀（§6）。
- **放置动作**：`PutInTrayExpert` 继承整个判材质 + 重试机器，只换尾部
  （`JUDGE -+-> CARRY -> LOWER -> DROP -> RETREAT -> DONE`）：CARRY **两段式**——
  先原地升到 `max(墙顶, 桌面+最高物体) + 0.10 m`，再按 `place_carry_step_m` 分步横移，
  搬运不会蹭到没拿的物体、也不会把罐子甩出指垫；LOWER 把物体中心降到腔底 + 半高；
  DROP 张爪并等 5 步落稳；RETREAT 退出腔口。
- **无 `requested_group` 的单纯放置任务**（`expert.approach: place_in_tray` 且
  `temperature_judge_enabled: false`、物体没有 `material_group`）：
  没有材质要判，`_on_grasped` 从 CLOSE 直接进 CARRY，继承来的 JUDGE 机器保持休眠；
  候选顺序以 `target_index` 为首（无判定时不做随机置换，避免挑错身体）。
- **物体与托盘不互相侵占**：`_tray_clearance_m` 把每个槽位到托盘**外廓**的距离减去物体外接圆
  半径取最小，与物体间距护栏**同时**成立才接受一次抽样；无托盘 ⇒ 返回 `inf` ⇒ 既有配置的
  随机流逐位不变。（保守：按外接圆、按最大 jitter 膨胀、对每个 (槽位, 物体) 对都算。）
- **实测（2026-10-15，`scripts/probe_lambda_bands.py`，4 集 × 5 体 = 20 抓）**：驱动 **shipped**
  专家与场景，只把 `requested` 置成永不匹配的哨兵，让一集走遍全部五块 ⇒ 一集 = 一次完整采样。
  实测通道（W = 4.0 s、`pad_face`、官方 `(1,1,1)` 单格 + RC/噪/量化）：

  | 材质 | 组 | n | λ̂ 均值 | min | max | 相对跨度 | r² |
  |---|---|---|---|---|---|---|---|
  | 巴尔沙木 k=0.055 | wooden | 8 | 0.2781 | 0.2666 | **0.3462** | 28.6% | 0.961 |
  | 尼龙 k=0.25 | plastic | 4 | 0.7341 | 0.7210 | 0.7452 | 3.3% | 0.995 |
  | 铁 k=80 | iron | 8 | 1.4839 | 1.4497 | 1.5015 | 3.5% | 0.994 |

  **相邻分离 2.64× / 2.02×**（都到 R38 单边界 32/32 时的 2.17× 之上或量级相当），
  实测排序 `balsa < nylon < iron` 与 k_eff 推导的 `wooden < plastic < iron` **一致**。
  两个边界的**零错窗口** = `(0.3462, 0.7210]` 与 `(0.7452, 1.4497]`；取各自**最坏情况几何中点**
  `sqrt(下带 max × 上带 min)` = **0.4996 / 1.0394** ⇒ 配置取整数 **`[0.50, 1.04]`**，
  实测 **0 错 / 20 抓**，两侧相对余量 **1.44× / 1.40×**（用均值中点 0.4519/1.0437 反而把最坏余量
  压到 1.31× —— 边界要按**最坏情况**取，不能按均值取）。
- **如实记账：慢带的尾巴，以及由它定的两个新旋钮**。20 抓里唯一的坏拟合是 seed 1003 的 body 4
  （巴尔沙木）：`λ̂ = 0.3462`（比该带其余 7 抓的 0.2666–0.2714 高 **29%**）、`r² = 0.781/0.724`、
  **窗内只走了 0.63 K**（其余抓 2.2–126 K）。即**指垫几乎没真碰上**，拟合由噪声主导。
  于是定标后补了两个旋钮（现在都在 `PipelineConfig`；默认值让 R38 **逐位不变**）：
  `judge_min_span_k: 1.5`（窗内变化不足 1.5 K ⇒ 判定"没真碰上"、拟合作废；
  把 0.63 K 与下一个最小的健康读数 2.2 K 分在两侧）与 `retry_on_none: 1`
  （拟合作废 ⇒ **重抓同一块**，而不是把这块候选浪费掉）。后者修的是一个真实缺陷：
  改之前 `None` 不等于 `requested`，专家会**放掉手里这块去试下一块**，而在**最后一块**候选上
  按"无路可走"规则**直接提交** —— 一次坏接触就可能让本来能判对的一集失败。
- **顺带修掉一个 latent 崩溃（写这两个旋钮时才发现）**：拟合作废时 `decide_bands` 返回
  `band=None`/`margins=None`，而 `_judge_now` 的多带分支对它们做了 `int()` 与 `list()`
  ⇒ **任何一次拟合失败都会抛 `TypeError` 把整集打断**（实测该值：`int(None)` / `list(None)`
  两条都抛）。它此前没被触发，只是因为**至今没有一集出现过拟合失败**（首次定标 20 抓全部有效）
  —— 也就是说，"坏接触"这条路径**从来没被执行过**。现在两处按 None 透传。
- **文件级指令按组钉住（接三组数据集时发现的第二个缺陷）**：`build_problem_info` 的语言字段
  一律取**物体自己的 `material_group`**，而 `group_files` 的三个文件共用同一套材质布局
  ⇒ `put_in_tray_wooden.hdf5` 的文件级 `language_instruction`（以及 `objects[].language_instruction`
  全部五个）会写成**物体 0 那一组的**词，实测就是 "put the iron cube into the tray"；
  每回合 attr 走 `requested_group` 一直是对的，错的是加载器/数据卡**最先读到**的那一行。
  现在 `group_files` 且 `requested_groups` 只有一个值时：五个物体的指令都用**钉住的组**，
  文件级另外落 `requested_group` 明写"这个文件要的就是它"，并把 `requested_group` 从
  `randomization.per_episode` 挪到 `per_file`（它在本文件里根本不是抽出来的）。
  条件不成立时（R38 的两组、所有旧任务）一个字节不变 —— 已逐键核对 R38 的 `problem_info`。
- **捷径审计（首次真正跑出来，如实记账）**：接触**前**读 `obs/temperature/objects_grid`（数据里本来就有
  这一路）就能分材质 —— 每体**自身**网格的 spread 增长率（按 `|T−ambient|` 归一，mK/s/K），
  **在什么都没碰之前**：

  | 材质 | n | 均值 | min | max |
  |---|---|---|---|---|
  | 铁 | 4 | 5.124 | 4.816 | 5.445 |
  | 尼龙 | 2 | 6.631 | 6.628 | 6.633 |
  | 巴尔沙木 | 4 | 22.165 | 16.038 | 28.362 |

  **排序与指尖通道相反**（铁最慢、巴尔沙最快）：良导体把底面漏进来的热迅速摊平 ⇒ 内部近等温 ⇒
  spread 长得慢；绝缘体攒出陡峭表皮 ⇒ spread 长得快。相邻**最坏**分离 **铁|尼龙 1.22×**、
  **尼龙|巴尔沙 2.42×**。对照 R38 的同类审计（两组、**4.04–4.35×**）：本任务这条捷径**更窄**，
  铁|尼龙 只有 1.22×（且尼龙只 2 个样本、巴尔沙自身带内跨度就有 55.6%）。所以它没有让
  "必须摸一下"变成空话，但**它存在**：把 `objects_grid` 当输入的策略原则上能学到一个（脆弱的）预判。
  未堵，如实记账。
- **同一场景重跑的方差（顺带得到，直接解释了 `judge_min_span_k`）**：seed 1001 在两个进程里各跑一遍
  （同场景、同回合种子、同候选顺序 `[4,1,0,2,3]`）。**物理几乎逐位复现**（窗内 span 两次差 ≤0.3%：
  42.5/42.63、117.53/117.6、16.97/16.93、−106.97/−106.97），**λ̂ 的复现精度却随 span 单调变差**：

  | 体 | 材质 | span (K) | λ̂ 两次 | 相对差 |
  |---|---|---|---|---|
  | 0 | 铁 | 117.5 | 1.4848 / 1.4862 | 0.09% |
  | 3 | 铁 | 107.0 | 1.4983 / 1.4958 | 0.17% |
  | 4 | 巴尔沙 | 42.5 | 0.2666 / 0.2658 | 0.3% |
  | 2 | 尼龙 | 16.9 | 0.7192 / 0.7376 | 2.6% |
  | 1 | 巴尔沙 | **2.25** | 0.2817 / 0.3027 | **7.5%** |

  ⇒ 噪声**不在仿真里，在拟合的条件数里**：span 越小，`T_cell` 与 `λ` 越难分离（`plateau_rel` 也越大）。
  这既是 `judge_min_span_k` 的直接依据，也给出门槛附近方差的量级：**刚好在 1.5 K 门槛之上的 2.25 K，
  两次读数差 7.5%**；而巴尔沙带离 0.50 边界的余量是 0.53–0.60（40% 以上）⇒ 这个量级的抖动无害。
- **重抓分支的验证方式**：坏接触是**稀有事件**（20 抓里 1 次），全集采样很可能一次都碰不到
  ⇒ 这一支不能靠跑 episode 证明。`scripts/probe_judge_retry.py` 换掉物理而不是采样物理：
  `_judge_now` 替换成一串脚本化裁决、`_advance` 只记录相位，不建场景、不碰 GPU、约 1 秒跑完，
  逐条断言：`= 0` 精确复现改动前的机器（`None` ⇒ 换下一块、从不重抓）、`= 1` 重抓**同一块**、
  次数受旋钮约束、候选推进时计数清零、抓取预算恰为 `n_obj × (1 + retry)`、最后一块重抓一次后
  仍按老规则提交、命中时两种设置都立即提交。末段还拿**真的 `_judge_now`**（不是桩）喂一条
  零升幅轨迹，断言 `group/band/margins` 都是 None 而不抛异常，并用一条正常升幅轨迹做阳性对照。
  **它不能证明的是**"真实接触确实会产生 `None`" —— 那是 `probe_lambda_bands.py` 的事。
- **回读口径**：判错即 `success=False` 如实落盘。`scripts/judge_temperature_material.py` 的三带盲判
  复算**已于 §9-R40 接入**（`--thresholds` / `--method ratio` / 按文件自己的协议复刻 + 裁到在线
  判定步）；本轮的定标证据仍全部来自 `probe_lambda_bands.py`，两者在 R40 那节对过账。
- **实测出的一个真缺陷：三个文件的回合随机流是同一条（2026-10-15 生成后复查发现，已修，文件待重跑）**。
  一次命令写三个文件、每个 `--episodes 1`，而 `generate.py` 里**回合种子是 `cfg.seed + i`、与
  `scene_seed` 无关**：`cfg.seed = 0`、`i = 0` ⇒ 三个文件的 `episode_seed` **都是 0**。
  场景种子 1000/2000/3000 只管**建场期**抽样（边长/质量/颜色，实测三份确实各不相同），
  回合期抽样共用一条 rng 流。后果是**三份文件逐位相同**：`objects_xy`、`objects_temp_c`、
  `objects_yaws`、槽位置换、候选顺序，以及 `front_object_index = 4`。
  纯 numpy 复现（不需建场景）：`default_rng(0)` 耗 20 个 double（5 体 × (xy 抖动 2 + 偏航 1 + 温度 1)），
  `permutation(5)` → 槽位置换 `[3 2 0 1 4]`，`integers(0,1)`，再 `permutation(5)` → 候选顺序
  `[4 3 2 1 0]` —— **与三份文件的 `objects_xy` 及 `judge` 日志逐项吻合**
  （铁档先摸 body 4 判 wooden 再摸 body 3；塑料档依次摸 4/3/2，三次判决全对）。
  ⇒ 这**正好废掉槽位置换存在的理由**（本 config 与 §9-R38 的原话：不置换则指令直接点名槽位、
  温度通道永远用不上）：在这份数据里三个材质词各自**恒定**对应一个桌面位置，策略可以完全跳过
  温度通道。**修法**：`group_files` 模式下回合种子改用 `scene_seed + i`。`group_files` 目前
  只有 `put_in_tray.yaml` 用 ⇒ **零回归**（材质扫描仍走 `cfg.seed + i`——那里 episode i 本就该是
  每个文件同一次抽样）。修后 1000/2000/3000 给出三套不同的槽位置换与候选顺序。
  **附带说明**：`--episodes 1` 时每个文件只有**一次**置换抽样，去捷径靠的是**文件之间**不同；
  文件内要有多样性，`--episodes` 就得大于 1。

---

### R40 — 判材质窗口 4.0 s → 1.5 s：渐近线不用估了，窗口从**先碰上的那个指垫**起算（2026-10-15）

用户提出「接触判断时间过长，能否改进」。先把代价拆开（8 个抓取，实测步数）：一个候选 =
`闭合→t1` 0.40–0.60 s + `t1→t2` 0.45–0.60 s + `t2→判定` 4.15 s（R39 的 `settle 0.15 + W 4.0`）
⇒ **窗口占一个候选的 82%**。所以两条路：把窗口挪早（(1)），或把窗口本身缩短（(3)）。本轮两条都落。

- **(3) 换估计量：`ratio`（`sim/expert/thermal_judge.py::estimate_coupling_ratio`）**。引擎的接触响应是
  `dT/dt = λ(T_cell − T(t))`，即 `λ·D₀·exp(−λ·Δt)` —— **`T_cell` 只是斜率的尺度**，在两个半窗
  斜率的**比值**里消掉：
  `λ = ln(s1/s2)/(t2 − t1)`。R39 之所以要 4.0 s，是因为 `fit` 要**估出** `T_cell`，而
  `T_cell` 与 `λ` 在小窗口里分不开（R38 的窗口扫描：巴尔沙 λW < ~1.4 就塌）。**把参数消掉
  而不是估出来，窗口就只剩"装得下两个可分斜率"这一个要求**。热/冷方向自动无关：`D₀` 的符号
  在两个斜率里都带着，比值里约掉 —— 这正是 R38「一个阈值同时覆盖热件与冷件」的同一个原因。
  `t_cell_c` 仍按 `v0 + s1/λ` 反解出来，但**只是诊断字段**，没有任何判据读它。
- **(1) 窗口锚点：从 `min(onsets)` 起算**。这里**我先说错过一次，值得记下来**：我最初把它
  描述成「三行、零代价」，读了 `estimate_coupling` 才发现**每个指垫是按自己的 onset 开窗的**
  ⇒ 锚点前移会**截断后碰上那个指垫的窗口**，代价真实存在。所以它有个闸门
  `judge_min_late_window_frac`（**新旋钮**，见 §6）：后碰的指垫仍保有 `W×(1−frac)` 才前移，
  否则锚点留在后碰的那个上。两指先后差 0.25–0.60 s（立方体停在中线外 ~1.5 mm）。
- **实测（三个已落盘文件，`scripts/judge_temperature_material.py`，8 抓/16 指，同一批文件两种协议）**：

  | 协议 | wooden (λ̂) | plastic | iron | iron/wooden | 几何中点 | 最坏余量 | 可用指尖 |
  |---|---|---|---|---|---|---|---|
  | `fit` W=4.0，锚点在后手（**R39 的shipped**） | 0.2682–0.2704 | 0.7216–0.7400 | 1.4881–1.4912 | 5.50× | 0.4417 / 1.0494 | 1.63× / 1.42× | 16/16 |
  | `ratio` W=1.5，锚点前移（**本轮**） | 0.2629–0.2704 | 0.7451–0.7581 | 1.4914–1.4979 | 5.52× | 0.4488 / 1.0633 | 1.66× / 1.40× | 16/16 |
  | `ratio` W=1.5，锚点在后手 | 0.2629–0.2781 | 0.7403–0.7454 | 1.4938–1.4970 | 5.37× | 0.4537 / 1.0552 | 1.63× / 1.42× | 16/16 |
  | `ratio` W=1.0，锚点前移 | 0.2464–0.2841 | 0.7527–0.7696 | 1.4905–1.4995 | 5.25× | 0.4624 / 1.0710 | 1.63× / 1.39× | **12/16** |

  **三条带在两种估计量下是同一批**，分离度没动，8/8 抓取分带全对。
- **W 的下限不是"条件数"，是采样数**（`W = 1.0` 那一行的 12/16 就是它）。两个半窗各需
  `max(min_samples=10, 2×min_half_samples=8)` 个采样；锚点前移后，**后碰的指垫先花掉了两指
  间隔 g 才开窗**，只剩 `W − g`。实测**这是一道崖不是一道坡**：`W = 1.0`、`g = 600 ms` 时
  后手恰好落在 **9** 个采样上 ⇒ 估计量返回 `None`，4/16 个指尖被丢掉；而丢掉的不只是一个读数
  —— **两指平均没了**，于是 wooden 从 0.2629–0.2781（W=1.5，两指）张到 0.2464–0.2841
  （W=1.0，单指）。`W = 1.5` 比门槛多 3 个采样。**闸门本身在边界上是平滑的**：`g = 0.9 s`
  （闸门极限）时后手仍有 13 个采样。唯一的例外是那个 `g = 3.75 s` 的离群抓取，它正确地回落到
  后手锚点。
- **为什么是 W = 1.5 而不是 1.0**：`ratio` W=1.0 锚点前移（t1+1.75 s）并不比 `ratio` W=1.5
  锚点前移（t1+1.65 s）更快 —— **1.5 反而早 0.1 s**，且 16/16 全可用、带宽更紧。原因是
  W=1.0 的前移版本要在"后手只剩 0.4 s"的条件下工作，那个条件下它已经掉采样了。
- **每候选实测省 2.50–3.10 s**（从记录步数直接读，不是估算）：`t2+4.15 s` → 闸门命中时
  `t1+1.65 s`（8 抓里 7 抓，g = 0.25–0.60 s），未命中时 `t2+1.65 s`（那个 g = 3.75 s 的离群）。
- **阈值不动，且理由记下来**。两条边界的**新中点**是 `sqrt(0.2704×0.7451) = 0.4489` 与
  `sqrt(0.7581×1.4914) = 1.0633`，按 R39 自己那条「取最坏情况几何中点」的规则本应挪到
  `0.45 / 1.06`（最坏余量 1.66×/1.40× vs 现在 1.49×/1.37×）。**不挪**：三条带与 R39 的
  20 抓包络**重叠**（该包络是 `[0.2666, 0.3462] / [0.7210, 0.7452] / [1.4497, 1.5015]`），
  `1.04` 还是 `pick_by_temperature` 共用的那个数，为了 +1% 的余量去重验另一个任务不划算；
  而 `0.50` 是"没有任何东西靠近的那条缝"上更保守的一侧。**这是判断，不是测量**。
- **`judge_min_late_window_frac` 默认 = 1.0（关）**，只有 `put_in_tray.yaml` 显式开 0.4。
  锚点逻辑在**共享的** `_act_judge` 里，默认开就会**顺带改掉 `pick_by_temperature` 的判定时刻**
  —— 那里用的是 `fit`，后手被 g 截断的窗口正好落在 R38 量过的弱条件数区（`fit` 要 λW ≳ 1.2–1.4
  才看得到弯曲）。**1.0 ⇒ 闸门条件 `t_anchor − early ≤ 0` 只在与两指同刻接触时才成立，而那一步
  `early == max(onsets)`，与旧行为逐位相同** ⇒ R38 及其余任务**零回归**（已逐配置核对：没有
  别的 config 出现这两个新键）。
- **回读口径（本节顺带修掉的两个工具缺陷，都是"复算与在线不一致"这一类）**：
  `scripts/judge_temperature_material.py` 现在**默认复刻文件自己记录的那套协议**
  （`method`/`window`/`settle`/`min_late_window_frac`/`min_span_k`/`thresholds`，全部从
  `problem_info.material_judgement` 读，见 §10-M10 的 meta 增补），并且**把轨迹裁到在线判定
  真正发生的那一步**（`_anchor_window`，镜像 `_act_judge`）—— 不裁的话，锚点前移的分支下
  复算会拿到专家根本没有的采样。裁点**与专家自己落的 `step` 逐位一致：旧文件 8/8 抓全中**
  （这是"这边的锚点规则 == 真的跑过的规则"的直接证据）。两个顺带查出的缺陷：
  (a) 逐帧 `cand` 是**候选序号不是物体下标**（`judge_candidate_order: random` 下两者差一个置换，
  而落盘的只有 `front_object_index`），原来那句「attribution differs」的告警是**假的**（在
  `put_in_tray_iron` 上 4/4 全报）—— 改成与该文件唯一断言的物体下标 `place_body` 比对；
  (b) `min_span_k` 没有透传 ⇒ `judge_fingers` 用**估计量的默认 0.5** 而不是任务的 1.5
  （`**kwargs` 直落估计量，"没传"不等于"不设闸门"）。**如实在账：这三个文件的 8 抓里没有一根
  手指落在 0.5–1.5 K 之间，所以这个 bug 在本轮数字上零影响** —— 它是个定时炸弹，不是本轮
  误差的来源。上面那张表里"13/16"那一行的宽度变化来自裁点差一个采样（早期草稿脚本多了 +1
  样本），不是闸门。
- **本轮没做的**：8 抓样本很小（4 木 / 2 塑 / 2 铁），且**没有重跑生成**——上面所有数字都是
  在旧文件上**反事实复算**出来的。要做到 §8 那种"验收"，得用新配置重生成三个文件，再核对
  verdict 与 λ̂ 不变、episode 缩短。`probe_lambda_bands.py` 的 20 抓也应该在新协议下再跑一遍。

### R41 — robosuite 资产贴图**静默丢失**：trimesh ≥ 5 的解析器根限制 + Genesis 补丁吞异常（2026-10-15）

第 4 个桌面场景（`sim/configs/robosuite_tabletop.yaml`）第一次用 robosuite 自带网格，渲出来**全是灰模**
——罐、牛奶、柠檬都是 `Kd 0.8 0.8 0.8` 的单色体。查下来不是资产的问题，是**两个上游行为叠加**：

- **trimesh ≥ 5.0 收紧了路径解析**：`resolvers.FilePathResolver(source)` 默认
  `allow_anywhere=False` —— 资产引用的文件**不许逃出网格自身目录**。robosuite 的 MTL 恰恰相反：
  `map_Kd ../../textures/soda.png`（贴图统一放 `assets/textures/`，网格在
  `assets/objects/meshes/`）。于是 `absolute()` 抛 `ValueError`，`get()` 依次试
  `name` / `lstrip('/')` / `basename` 三个候选**全部落空**。这是 5.0 的**有意收紧**，不是回归：
  它的 docstring 把 `../textures/thing.png` 明确列为"pre-5.0 behavior"。
- **Genesis 的 OBJ 补丁把失败吞了**：`ext/_trimesh_patch.py::parse_mtl` 里 `resolver.get()` 包在
  `except BaseException: log.debug("failed to load image")` 中 ⇒ **无警告、无异常**，材质静默退回
  MTL 的 `Kd`，渲染器那侧看不出任何异常。

**本轮证据链**：`genesis.utils.mesh.load_mesh(can.obj)` → `TextureVisuals(uv=(1555,2))` 但
`SimpleMaterial.image is None`；直接调解析器 `get('../../textures/soda.png')` ⇒
`ValueError: escapes resolver root`，`allow_anywhere=True` ⇒ **627792 B = `soda.png` 整文件**；
把默认值翻过来重渲同一场景，agentview std 31.97 → 31.80、罐变红、柠檬变黄（几何与像素覆盖不变）。

- **影响范围（逐项查过，不是"应该没事"）**：只有引用**目录外**贴图的资产，即
  `robosuite/models/assets/**`。LIBERO 资产**一个 `.mtl` 都没有**（颜色在顶点上）⇒ step 1–3 的
  图、`docs/figures`、已落盘数据集全不受影响；Panda（`assets/urdf/panda_bullet/`）的
  `map_Kd colors.png` 与网格同目录 ⇒ 也不受影响。几何、碰撞（coacd 走顶点）、布局、热学身份
  与贴图无关。全仓只有 `sim/configs/robosuite_tabletop.yaml` 引用 robosuite 网格（`grep -rl` 实测）。
- **做法**：`sim/compat/mesh_load_compat.py::install_texture_resolver_policy()` —— 把
  `FilePathResolver.__init__` 的**默认值**翻成 `allow_anywhere=True`（显式传 `False` 仍优先；
  trimesh < 5 无此参数 ⇒ no-op；幂等，装过就返回），`build_scene` 在建**第一个 morph 之前**调用。
  这是恢复上游文档写明的 pre-5.0 语义，不是自造行为。
- **回归**：改动前后的对照场景重渲**逐字相同**（物体行 / 位姿 / liveness / 像素数全等，
  仍只有刻意改格式的 `scene:` 行差异）—— "没有其它资产受影响"的直接证据。

### R42 — 温度判定（不是材质判定）：先碰、再读 measured 指尖，判不冷就不抬（2026-10-19）

> **2026-10-21 修订**：定窗判据已被**事件驱动**判据取代——任一碰上的指读数进区间就立刻搬运，
> 任一碰上的指趋势到不了区间（平直/反向）就立刻释放换下一个，否则继续等；没有固定窗口，
> 也没有相对臂（见 §5.6）。下面 1.5 s 窗口 / 3 K 相对臂的描述与实测数字保留为 R42 当日的
> 记录，不再代表当前行为。验证（2026-10-21，4 集无相机）：4/4 成功、validator 0 issues；
> 每次触碰都在合爪后的首个可判定步结束（`hold_s` = 0.50 s）——冷罐 `in_interval` 直接搬运，
> 温罐以 +3.1 / +13.7 K/s 的正斜率 `wrong_trend` 释放后换下一个；负温罐的"一指平直
> （−0.12 K/s）、另一指已进区间"现场也判中，即正证据优先。
>
> **2026-10-21 第二次修订（候选遍历协议）**：原行为是"判中即搬运并结束、最后一个候选无路可退
> 就强制提交"。用户要求每集**遍历全部候选**：判中即搬运，但放好之后机器回到 APPROACH，
> 继续判剩下未摸过的候选；**凡满足区间的候选都要搬走**——托盘首个落点占中心，之后沿 ±x 错开
> 0.055 m（理由见 §5.6）。**不再有任何强制提交**：最后一个候选被否时
> RELEASE / RETREAT 直接结束于 `DONE`，所以一集可搬运 0、1 或多块。`problem_info` 的
> `protocol` 改为 `touch_all_candidates_event_driven`；每个 demo 新增 `place_bodies`（JSON，
> 逐次搬运）与 `place_count`，`place_body`/`place_material`/`place_rest_z` 保留并写**最后一次**
> 搬运，兼容只读单次搬运的下游。下面"最后一个提交"与 `committed`/`forced` 的描述保留为 R42
> 当日记录，不再代表当前行为。当前 config 下只有 0 号冷罐能满足 `[null, 15]`，正常每集搬运
> 数应为 1，多搬运路径要在两个候选都能命中时才会走到。
>
> **第一次遍历冒烟（2026-10-21，4 集无相机）**：3/4 成功，暴露两个缺陷。① **区间"穿过"被
> 判成命中**：搬完冷罐的指垫只有 ~2 °C，再摸 55 °C 孪生罐时读数在 +52 K/s 的升温中掠过
> 9.97 °C，旧规则（区间内即命中）立即判中并搬运，`all correct: False`；另一集同样以
> +47 K/s 的升温掠过 9.33 °C，被误搬进托盘。② **槽距不足**：0.045 m 让两枚 25 mm 半径
> 罐的心距只有 45 mm（< 50 mm），第二次 LOWER 与托盘里的第一块相撞、该集 400 步卡死
> （`success: False`）。**修法**：命中条件加"趋势不把读数带出区间"（`reading + slope ×
> trend_window_s` 仍在区间内；§5.6 ①），区间内但正在离开的指按"到不了"释放；槽位改
> ±x 0.055 m。复测（同日，同样 4 集无相机）**4/4 成功**：每集 touches = 2、搬运 1、
all correct True，validator 0 issues；温罐四条全部以 `wrong_trend` 释放（斜率 +52.19 /
+31.60 / +47.33 / +14.05 K/s，其中一条是先摸温罐再摸冷罐），唯一的搬运都落在中心槽
`(+0.400, +0.300)`，不再有 LOWER 卡死。

> **2026-10-21 第三次修订（代码评审，行为改变）**：评审专家管线时修掉五处设计缺陷。
> ① **DESCEND / LIFT 的到位门用了旧目标**：`pos_err` 在分派前对*上一步*的 `target_pos`
> 计算，而分支里刚把目标改成抓取点/抬升点；`pos_tol_m`(0.01) < `descend_tol_m`(0.015)，
> 于是 APPROACH 一结束 DESCEND 就立刻放行——**下降实际发生在 CLOSE 里**（指垫在抓取点上方
> 8 cm 处开始合拢），`descend_tol_m` 是死旋钮；LIFT 同理，`lift_tol_m` 门首步即过，抬升由
> HOLD 执行（"抬升到位"门并不存在）。现在两处都对**刚赋的新目标**测距。
> ② **CLOSE 超时是死路**：无接触超时 `_advance(LIFT)` 后停在 HOLD，永不 DONE，而 runner 的
> 成功要求 DONE——目标此前已搬进托盘的一集也会被判失败并耗到 400 步。新增钩子
> `_on_close_timeout()`：基类仍进 LIFT；温度判定机器进 JUDGE（无 onset ⇒ 既有的 unmeasured
> 分支 → RELEASE → 下一候选）；托盘专家在判定关闭时进 RELEASE → DONE。
> ③ **metadata 写全局常量**：`temperature_judge_*` 的窗长/baseline 读 `PIPELINE` 而非 runner
> 实际用的 pipeline；现在 runner 用 `self.pipeline`，`build_problem_info` 增加 `pipeline=`
> 参数由 `generate.py` 传入。
> ④ **配置漏校验**：判定开启而 `target_index` 不在 `temperature_candidate` 时无法成功，loader
> 却接受；现在直接拒绝。
> ⑤ **否决/命中的不对称**：旧规则中任一指的"否"会否决另一指的"未定"（晚碰上的指还没有
> 斜率），趋势窗未满的指也能凭"区间内"单独命中。现在命中要求 `trend_ok is True`，释放要求
> **没有**任何指 `trend_ok is not False`。
> 同步清理：`PlaceState` 改由 `_new_state()` 工厂构造（不再手工逐字段复制）；删掉只写不读的
> `trace_c`、无用局部变量与不可达分支；unmeasured 记录补 `hold_s: None`；place 记录的
> `step` 改名 `lower_steps`。**2026-10-21 之前生成的数据都带旧的 DESCEND/LIFT 语义**
> （下降在 CLOSE 内、HOLD 承担抬升）；仓库内无依赖旧语义的在库数据集（旧格式语料已删、M9
> 未跑）。**重测（同日，4 集无相机）4/4 成功**，steps 233/232/235/219（均值 230，比修正前的
> 211–227 略增，符合"LIFT 现在真的爬 10 cm"的预期），validator 0 issues；每集 touches = 2、
> 搬运 1、all correct True；温罐四条仍全部 `wrong_trend`（斜率 +52.26 / +31.60 / +47.79 /
> +14.00 K/s，demo_3 先摸温罐），冷罐 `in_interval`（−36.02 / −16.90 / −34.45 / −24.12 K/s），
> 每次触碰仍在合爪后首个可判定步结束（`hold_s` = 0.50 s），搬运都落在中心槽
> `(+0.400, +0.300)`、rest z 0.447。

`pick_cold_can` 的任务是"把冷的那罐放进托盘"：桌上有**两个视觉完全相同**的罐子
（0 号冷罐 `−18…4 °C`、1 号 55 °C 孪生罐），没有 `material_group`、没有 λ 阈值、也没有
槽位要置换。材质判据（R38–R40）回答的是"这是什么材质"，在这里是错的问题；但"先碰一下、
读指尖温度、再决定抓不抓"这个行为仍然需要——这就是本轮给专家管线加的**触碰逻辑**。

- **状态机接入点（代码零分支复制）**：复用 `PickByTemperatureExpert` 的 JUDGE/RELEASE 机器。
  `_order_candidates` 在 `temperature_judge_enabled` 时返回 YAML 的 `temperature_candidate`
  （多个候选时**总是**用回合 RNG 重新置换，原来的 `judge_candidate_order` 开关已删除）；
  `PutInTrayExpert._on_grasped` 在"无 `requested_group` 且
  `temperature_judge_enabled`"时返回 JUDGE 而不是 CARRY；`_act_judge` 分派到
  `_act_temperature_judge`。材质判据的 `judge_log` 与冷热判定的 `temperature_log` **分开**，
  互不污染；`n_cand` 用 `len(self.order) or self.n_obj`，候选只有两个而场景有七体时
  最后一块/下一块的判定仍然正确。
- **触碰保持**：合爪后 `target_pos = grasp_pos` 原地保持；窗口从**后碰上的指垫**的 onset
  起算（两指到达时间可差数百毫秒，从先碰的起算会让另一指少采一截），持续
  `PIPELINE.temperature_judge_window_s = 1.5 s`。逐指 baseline 取**自己** onset 前
  `PIPELINE.temperature_judge_baseline_s = 0.3 s` 的均值，reading 取窗口末
  `PIPELINE.temperature_read_tail_s = 0.15 s` 的均值。判中规则（2026-10-19 同日重构为区间）：
  `temperature_judge_threshold_c: [null, 15]` 内 **或** `reading − baseline ≤ −3 K`
  （`PIPELINE.temperature_judge_drop_c`）；**任一碰上的指判中即整块判中**。
  绝对臂与相对臂互为补充——绝对臂抓住"本来就冷"（不管 baseline 多高），相对臂抓住
  "指垫还带着上一块的热、但这一块明显把它拉下去了"；OR 跨指则防止一根指垫接触不良
  淹没另一根的证据。`[45, null]` 的热目标镜像这一规则，双侧区间 `[lo, hi]` 只有绝对臂。
- **分支与失败面**：判冷 → `_on_positive_verdict()`（`PutInTrayExpert` 返回 CARRY，进入
  R39 的两段式搬运/下降/释放）；判不冷 → 从**同一触碰位姿** `RELEASE`，不抬升、不搬运，
  回到 APPROACH 摸下一罐。最后一罐无路可退 ⇒ `committed`/`forced` 强制提交，搬运它；
  成功判据仍只认 `target_index: 0` 在托盘内，所以**强制搬错的一集是失败**（落
  `*_failed.hdf5`），不会把错误藏成成功。窗口内没有任何接触 onset ⇒ 无测量：按
  `PIPELINE.retry_on_none` 重摸同一块，或用同一套"不中/最后一个提交"规则前进，
  **绝不从无接触状态直接进 LIFT**。
- **记录**：`problem_info.temperature_judgement` 写 `protocol: touch_then_commit`、统计量、
  传感器通道（`obs/temperature/fingertip_grid`，measured）、候选/顺序/阈值区间/目标类型/
  窗长/baseline；`randomization.per_episode` 追加 `temperature_candidate_order`。每个 demo 的
  `temperature_judge`（JSON）逐次触碰记 `cand`/`body`/`verdict_is_target`/`measured_c`/
  `baseline_c`/`delta_c`/`per_finger`/`threshold_c`/`target_kind`/`drop_c`/`window_s`/`onsets_s`，
  另有 `temperature_judge_touches`、`temperature_judge_first_body`、
  `temperature_judge_correct`；runner 落盘前补 `is_target`/`correct`
  （`verdict_is_target` = 专家实测判定，`is_target` = 真值标签，二者不同名）。
  **没有任何字段读取场景真值**；`objects_temp_c`
  仍按数据集惯例作为真值记录，供下游做温度回归，与专家输入无关。
- **实测（2026-10-19，无相机，4 集，seed 0–3）**：**4/4 成功**，步数
  **166 / 163 / 167 / 212**，`validate_dataset.py` **0 issues**。seed 3 是第一次出现
  **拒绝支路**：候选顺序把 55 °C 孪生罐排在前面 → 两指 measured **49.93 / 48.80 °C**
  （baseline 20.97 / 20.82，Δ +28.97 / +27.98）⇒ `cold: False`、`committed: False`，
  原地释放；第二摸冷罐 measured **1.13 °C**（baseline 50.97，Δ −49.83）⇒ `cold: True`
  并搬运，`place_body: 0`，`temperature_judge_correct: True`。seed 2 的两指 reading
  **18.57 / −11.77 °C**（Δ −2.42 / −32.73）：一根指垫只勉强接触时，另一根仍然投出冷票，
  正对应上面的 OR 规则。4 集 `temperature_judge_correct` 全 True。
  （R43 的安全转场改变了接近轨迹，同 seed 的 measured 读数与步数会略有不同；
  判定链与成功率不变，见 R43 的 4 集回归。）
- **代价**：冷热判定比材质判据轻——没有拟合、没有 λ 扫描、没有真值播种，只多一个
  `1.5 s` 的保持窗（30 个策略步）；无相机 4 集共 4.2 MB，单集约 50–60 s。
  若开双 512² 相机，实测约 **39 min/集**（软件光栅，R3/R20 的已知代价）。

### R43 — APPROACH 撞倒旁物：先升到最高物体之上，再横移（2026-10-19）

用户报告「夹爪运动过程中会碰倒别的东西」。对着 seed 3 的相机录像（212 步）逐物体查位移：

- **4 号牛奶**（起点 `(0.356, −0.037, 0.461)`，高 0.144 m ⇒ 顶面 **0.544 m**）从**第 3 步**
  开始动，最终 xy 位移 **80.8 mm**、z **−41.2 mm**（撞倒）；其余物体除被抓目标与柠檬
  3.9 mm 的初始沉降外均 ≤4 mm。
- EE 起点 `(0.306, 0, 0.483)`，1 号罐的 approach 点在 `(0.403, −0.202, 0.52)`。原 APPROACH
  是一条从当前位姿直达 approach 点的**直线**，在 z≈0.48–0.52 斜穿 4 号牛奶所在区域
  （x≈0.33–0.36，y≈−0.03）——碰撞是路径本身，不是夹爪尺寸或抓取位姿。

修复（`sim/expert/expert.py::PickExpert._act_approach`）：

- 新旋钮 `PipelineConfig.transit_clearance_m`，**默认 0.10**（2026-10-19 前默认 0.0）：
  0 时回到原来的单目标 APPROACH（旧轨迹逐位不变）；pick_cold_can 一直用 0.10。
- >0 时 APPROACH 拆成三个子阶段（状态存在 `ExpertState.approach_stage`）：
  0. 在**当前 xy** 竖直上升到 `_transit_z()`；
  1. 在转场高度横移到 approach 点的 xy（只要求位置到点，姿态在下段收敛）；
  2. 竖直下降到 approach 点，再按原判据（`pos_tol_m` + `rot_tol_rad`）进 DESCEND。
- `_transit_z() = max(桌面 `table.height_m` + 最高物体高度, `tray.rim_z`) + transit_clearance_m`。
  与 R39 的 CARRY 不同，这里不把「物体半高」再加一遍：空手时**手指尖只在 TCP 下方
  7.2 mm**（URDF finger mesh 在 hand 系 z∈[0.0584, 0.1122]，TCP 在 0.105），所以 TCP
  高度就是真实净空。七体场景最高物体是牛奶 0.144 m、桌面 0.4 m ⇒ 转场 TCP **0.644 m**。
- `_advance(APPROACH)` 会把子阶段清零，所以**释放后换候选**、`rejudge` 重摸同一块都自动
  走同一条安全路径；R39 的两段式 CARRY 不变。
- IK 可达性先探过再定值：两个罐正上方 z = 0.604 / 0.624 / 0.644 的 top-down 目标，
  位置残差 **≤0.50 mm**、姿态残差 ≤1e-4 rad，0.10 m 的 clearance 不需要下调。

验证（seed 3，无相机，`/tmp/pick_cold_can_transit`）：

- 4 号牛奶 xy 位移 **0.00 mm**（修前 80.8 mm），3 号牛奶 **0.00 mm**；2 号柠檬
  **3.92 mm**（与修前同一初始沉降，非碰撞）；1 号孪生罐 **0.32 mm**；5/6 号立方体 0.00 mm。
- EE 轨迹：第 0–12 步从 `(0.306, 0, 0.483)` 竖直到 `(0.313, −0.001, 0.643)`；第 12–24 步
  在 z≈0.64 横移到 `(0.404, −0.199)`；随后下降。全程最高 z **0.6439**，横移段离牛奶顶面
  ~9 cm。
- 温度判定链不变：先摸 1 号罐（measured 28.2 / 51.1 °C，判不冷）→ 原地释放 → 摸 0 号罐
  （−1.4 °C，判冷）→ 搬运入托盘；`place_body: 0`、`temperature_judge_correct: True`。
- **251 步成功**（修前 212；两次 APPROACH 合计多 39 步），`validate_dataset.py` 0 issues。
- **4 集回归（seed 0–3，无相机）**：4/4 成功，步数 **184 / 183 / 186 / 251**（修前
  166 / 163 / 167 / 212），validator 0 issues；4 集里 3/4 号牛奶与 5/6 号立方体的
  xy 位移均 **0.00 mm**，柠檬仍是 ~4 mm 的初始沉降（与抓取路径无关）。

代价：每次接近多 ~20 个策略步（≈1 s 仿真）。开双 512² 相机时这些步也按 ~11 s/步计，
单集墙钟相应增加；无相机单集仍在 1–2 min 量级。

### R44 — 触觉简化（format 1.6）：只留 AT-VLA 6D 合力 + 2D marker，参数移入传感器模块（2026-10-20）

用户目标（原话）：「触觉信息有两个，一个是 2D marker 信号，一个是 AT-VLA 的 6D 合力；
简化传感器的配置，选择 `modalities: [force, elastomer]`，剩下的具体参数在传感器 py 文件中实现」。

- **口径**：`force` = 每指 12 个 `KinematicTaxel` 探针力求和后沿指垫外法线
  `n = -inner_dir` 分解成 `[Fn(3), Ft(3)]`（AT-VLA 的 6D = 3D 法向 + 3D 切向合力，
  **不是** 力 + 力矩），压缩为正；`elastomer` = 每 marker 的切平面位移 `(u,v)`，
  基由 16×16 未变形网格的首行/首列差分构造并对 `n` 正交化（`u` = 网格轴 0、`v` = 轴 1），
  **法向分量丢弃** —— 与 AT-VLA 的 N×2 marker 格式对齐。
- **配置收缩**：`tactile:` 只剩 `modalities`；`PROBE_GRID`/`PROBE_OFFSET_M`/`PAD_TOL_M`/
  `PROBE_RADIUS_M`/`ELASTOMER_PARAMS`/`AS_GRID`/`CONTACT_LINKS`/`TEMPORAL_REDUCTION`/
  `WORLD_FRAME` 全部成为 `sim/sensors/tactile.py` 的模块常量（`TactileConfig` 用只读
  property 懒加载，旧调用点不改）。旧 YAML 若还带这些键，loader 直接报错而不是静默忽略。
- **格式**：`FORMAT_VERSION = tempvla-1.6`；`force (T,2,6)`、`elastomer (T,512,2)`，
  均为 **link / pad 系**，`world_frame=false`；`obs/link_pose (2,7)` 保留作世界系参考。
  `tactile/depth` 与 `tactile/height_map` 不再写入；`layouts.attrs["channel_formats"]`
  声明每个通道的 `{features, frame, shape}`，`validate.py` 按它校验（旧文件回退到老规则）。
- **兼容**：format ≤1.5 的 3D marker / height map / GelSight 渲染仍可读；新 2D 数据喂给
  `height_map` / `gel_render` 会显式 `ValueError`。`contact/link_force` 不变，作为求解器级
  合力与 6D 探针合力对照。
- **验证**（`/tmp/tactile6d_smoke`，单集、无相机、seed 0）：184 步成功、2 个接触事件、
  validator **0 issues**；`force (184,2,6) f32`、`elastomer (184,512,2) f32`、
  `format_version=tempvla-1.6`、`channel_formats` 含 `force/{6,link}` 与
  `elastomer/{2,pad_tangent}`；`import sim.config` 不拉 torch / genesis。
- **不重跑 M9**：本轮只做代码 + 单集 smoke；全量重生成与正式数据卡仍待用户确认。

### R45 — 全景热像（format 1.7）：曾实现，2026-10-20 全部回退

用户先要求「在全景相机处额外加一个热成像仪……标准按照 ThermoAct 的热成像图来」，
按 ThermoAct 标准做出了合成热像通道（agentview 位姿 256×192 分割渲染 + 逐像素上温
→ 线性 20–35 °C → 8-bit → INFERNO → 256²），格式升到 1.7 并做过单集 smoke
（validator 0 issues）。随后用户决定「先不进行热成像渲染这条通道，全部回退」：
`cameras.thermal` 配置、`obs/agentview_thermal`、`sim/sensors/thermal_camera.py`、
校验器分支与格式 1.7 **均已移除**，`FORMAT_VERSION` 回到 `tempvla-1.6`。

- 保留的实测结论：热像相机只多一次 **256×192** 分割渲染（**6.3 ms/次**中位，45 步
  合计 0.3 s），**不是** R20 渲染尖峰的来源——同 seed 关掉热像后尖峰逐次复现
  （eye_in_hand 242.2 vs 241.8 s、agentview 120.2 vs 119.5 s）。
- 若重启：Genesis 1.4.1 没有热渲染器（R17/R36），只能走「分割图 + 温度状态合成」；
  上温优先级 物体网格均值 > 实测指尖单格 > 连杆温度 > ambient，编码为线性 20–35 °C
  → 8-bit → INFERNO → 256²（TMC256B 256×192 / 竖 FOV 42°）。按这些要点足以重建，
  不再保留半成品代码。

### R46 — `scripts/` 精简：只保留生产与数据侧工具（2026-10-20）

用 2026-10-20 的审计把 `scripts/` 的 29 个文件按用途分成 A–E 五组（生产 / Gate / 探针 /
演示 / 出图），用户决定「除了 A 保留之外，其他全部删除，并根据仓库修改文档」。执行结果：

- **保留 10 个**：`generate.py` + `gate_common.py`（唯一生产链）、`validate_dataset.py` /
  `build_manifest.py` / `render_video.py`（数据侧）、`scene_check.py` / `render_scene.py` /
  `probe_camera_pose.py` / `probe_render.py`（场景与渲染检查）、`autotune.py`。
- **删除 19 个**：Gate 0–4（5）；取证探针 `probe_contact_areas` / `probe_contact_heat` /
  `probe_freeze_overshoot` / `probe_judge_retry` / `probe_kernel_args` / `probe_lambda_bands` /
  `probe_thermal_discrimination`（7）；演示与视频 `demo_contact_heat` / `render_thermal` /
  `render_thermal_video` / `render_grasp_thermal_video`（4）；出图与离线判据 `report_figures` /
  `judge_temperature_material` / `thermal_features`（3）。
- **删除前已归档**：19 个文件打包在仓库外
  `/tmp/ThermTacVLA_removed_scripts_2026-10-20.tar.gz`（`/tmp`，跨重启不保证保留；仓库内不留副本）。
  `sim/`、`sim/tools/`、`sim/compat/`、`datasets/README.md` 与 `docs/figures/` 的归档媒体均未删除。
- **语义**：R32–R40、R44–R45 等分节与 §8 里引用的这些脚本命令**保留为历史记录**（数字与结论不变），
  但从 2026-10-20 起**不再可复现**；需要重跑某个历史实验时，按对应 R# 分节的描述重写一次性探针
  （模式不变：覆写模块常量 → `build_scene()`，见 `sim/sensors/temperature.py` 的 docstring）。
- **同步的文档**：根 `README.md`（任务/状态/目录/运行顺序/数据格式）、`docs/status.md`、
  `datasets/README.md`、`docs/group_meeting_2026-10-15.md`、`sim/README.md`，以及本文件
  §5.6 / §5.8 / §5.9 / §7 / §8 里对这些脚本的引用（改为「历史 / 已删除」）。
- **边界**：本记录不改 M9 计划——「全量重生成需用户明确确认」这条约束不变；也不删除任何数据、
  `docs/figures/` 文件或 `sim/` 实现。

---

## 10. 里程碑（工作量量级）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境搭建 + Gate 0/1 | ✅ `tempvla` 环境（genesis-world 1.4.1 + cu128 torch）；Gate 0/1/2 在 GPU 后端全绿（failures 0 / warnings 0），见 §9-R16 |
| M1 | 单 episode 最小链路（无相机、单 env、HDF5 写出 + 校验） | ✅ 代码完成（`gate2_scene.py`） |
| M2 | 每材质 50 demo + 成功判定 + 失败分离 | ✅ **完成**：`data/` 下 4 文件 × 50 demo = **200/200 成功、0 失败、0 error**（15.7 min，`WALL 944 s`）；逐文件 3.17–3.25 MiB，合计 12.9 MiB；`validate_dataset.py data/*.hdf5` 四个文件全部 "OK: no issues found"；每文件 50 demos、`total` attr = 50 一致。逐材质：步数 32–36、接触事件 1.8–1.9、T0 62–196 °C、指垫峰值升温 铝 35.4 / 铜 61.9 / abs 33.6 / 钢 45.9 °C |
| M3 | 相机接入（Gate 3 + Gate 4 均已通过）+ 双相机 128² | ✅ **代码完成 + 全量数据已生成**：`renderer=raster` + llvmpipe；`cameras.enabled=true` 即写入 `obs/agentview_rgb` + `obs/eye_in_hand_rgb`，渲染不扰动仿真（Gate 4）。代价 +124 ms/步（含首帧着色器编译口径）、+0.49 MiB/episode；`data_cam/` **200/200 成功、97 MiB、53 min**（干净集 8.3 s/ep；尖峰见 R20）。**后续（2026-10-03）**：分辨率提到 256²——`data_cam/` 仍为 128²，`sim/configs/*.yaml` 已改为 256²，未重新出全量 |
| M4 | 任务套件 v1（3 任务）+ replay/analyze + 数据卡 | ⏳ v2（v1 交付 `pick_hot_cylinder` 一个任务、4 个材质文件） |
| M5 | **marker 场转正**（format `tempvla-1.1`：逐模态网格 + `obs/link_pose` + `elastomer` 进正式 schema，Q5） | ✅ **代码 + 七项验证全过（2026-10-04）**：`config.py`/`scene`/`obs`/`sensors`/`recording`/`tools/tactile_field.py`/`gel_render.py`/`render_video.py --markers`/`validate.py`；Gate 1（0 failures/0 warnings，两套网格各按指列出）、Gate 2（0 issues）、`generate.py` 冒烟（`/tmp/gelship`，通道形状逐项核对）、**往返一致性**（R8）、`render_video.py --markers`、Gate 4（0 failures，`link_pose`/`elastomer` 逐位一致）、旧语料向后兼容（12 个 format 1.0 文件 validate 0 issues + 渲染回退正常）。**待做：全量重生成** —— `data/`、`data_cam/`、`data_lift/` 现均为 format 1.0、**不含 marker 场**；先写暂存目录、抽查后再替换旧目录（旧目录改名保留；`data*/` 不入库，git 也回滚不了数据本身） |
| M6 | **双物体任务**（format `tempvla-1.2`：`objects:` 列表 + box morph + 逐物体通道/attrs + box 面抓专家） | ✅ **代码 + 验证（2026-10-07，R24 汇总）**：`sim/configs/lift_object_pair.yaml`（圆柱 + 立方体、按 episode 二选一）；Gate 1 / Gate 2 全绿（0 failures/0 warnings，validator 0 issues）；demo 生成 8 集 8/8 成功、两种目标都覆盖；box 面抓 4/4 成功且 marker 场两指有响应（26–80 µm）；`render_video.py --markers` 两集视频 + 圆柱 vs box 压印对照图；旧语料 9 文件 validate 0 issues。**2026-10-08 修 box 面抓残余角缺陷**（R24）：修正后 box 抓取 depth spread 0.03–0.26 mm（任意物体 yaw），圆柱 vs box 的形状对照在同一文件相邻两集成立。**未做**：双物体全量数据集（与 M5 的全量重生成一起，待定） |
| M7 | **触觉形态对齐 UniVTAC**（format `tempvla-1.3`：63 点 marker 场 + 去刚性探针 + 合成 64×64 height map） | ✅ **代码 + 验证（2026-10-08，R25 汇总）**：`sim/configs/lift_object_pair.yaml` 改 9×7 elastomer-only（间距 2.03/1.54 mm，覆盖 14.2×13.8 mm 指垫）；`sim/tools/height_map.py` 新模块（cell-centred 栅格 + 坐标 clamp 防 mirror 边界 ringing），`validate.py` 以重合成比对校验（独立复核 `max\|Δ\| = 0.000e+00`）；Gate 1（含新 marker 响应检查，depth/force 显示 `modality disabled`）/ Gate 2 全绿，生成 2/2 成功、validator 0 issues（含 `obs/tactile/height_map (50,2,64,64)`）；体积 +0.32 MiB/episode（+17%）；`render_video.py` 优先以 height map 作 gel 面板源；单物体两 config 与全部旧文件不受影响。**未做**：1.3 全量数据集（旧 data/、data_cam/ 仍为 1.0 / 128²） |
| M8 | **任务 `pick_by_temperature`**（format 1.5：一文件一材质对 + 每回合槽位置换 + 边抓边判专家 `PickByTemperatureExpert` + 盲判复算工具，§9-R38） | ✅ **代码 + V1–V5 全过（2026-10-14）**：新增 `sim/configs/pick_by_temperature.yaml`、`sim/expert/thermal_judge.py`、`scripts/judge_temperature_material.py`；`sim/expert/expert.py` 增 `JUDGE`/`RELEASE` 阶段与 `_set_goal` 重构（对既有任务逐位兼容）、`sim/recording/{runner,meta}.py` 增判据 attrs、`scripts/report_figures.py` 增图 6。**逐项实测数字（定标 / 盲判 32/32 与 2.17× / 零错窗口 / V3–V5）见 §9-R38**。**未做**：铜-ABS / 钢-ABS 两个材质对、全量生成（与 M9 一起，跑前确认） |
| M9 | **format 1.5 全量重生成**（指尖改官方最粗单格 `(1,1,1)` + 官方 RC/噪声/量化、只录 measured —— §9-R36；落点 `datasets/v1.5/hdf5/`） | ⏳ **代码与验证就绪**（`FORMAT_VERSION` 1.5 + validator 1.5 分支 + 5 个 config 全切换 + `sim/tools/validate.py::_check_finger_temperature`；V1/V2/V6/V7 全过，见 R36）；**待执行**：全量 `generate.py` → `validate_dataset.py` + `build_manifest.py` 侧车 → 抽查渲染；跑前确认（R31 后仓库内无完整数据集） |
| M10 | **任务 `put_in_tray`**（托盘实体 + 三材质多带判据 + 搬运放置专家 `PutInTrayExpert`，§9-R39） | ⏳ **代码 + 定标（2026-10-15）**：新增 `sim/configs/put_in_tray.yaml`、`TrayConfig`/`script` 托盘构建、`thermal_judge.decide_bands`/`rank_groups`、`PutInTrayExpert`、`generate.py` 的 `group_files`/`--group` 轴、`scripts/probe_lambda_bands.py`、`scripts/probe_judge_retry.py`。**已做**：三带定标 + 两条边界 + **接触前捷径审计**（见 R39 实测）、`judge_min_span_k`/`judge_retry_on_none` 两个旋钮与重试分支的 V1 验证、`max_episode_steps` 由实测 `done_step = 768` 定为 **2000**、**三条指令各一集 + 观测渲染**（`datasets/v1.5/hdf5/put_in_tray_{iron,wooden,plastic}.{hdf5,mp4}`：3/3 成功、validator 0 issues、抓取判材 **5/5 全对**），并因此查出并修掉**回合随机流三文件共用**的缺陷（§9-R39，文件待重跑）。**未做**：批量生成。**2026-10-15 同日追加（§9-R40）**：判定窗口 4.0 → **1.5 s**（新估计量 `ratio` 消掉渐近线 + 窗口锚点前移到先碰上的指垫），每候选省 **2.50–3.10 s**；`judge_temperature_material.py` 补齐多边界盲判 + 按文件自身协议复刻 + 裁到在线判定步；阈值经复核**不动**。**待做**：用新配置**重生成**三个文件做验收（现有数字都是在旧文件上的反事实复算）、`probe_lambda_bands.py` 在新协议下重跑 |

---

## 11. 待决问题（已决 = 已实现进 v1）

| # | 问题 | 结论 |
|---|---|---|
| Q1 | 本体：浮动手 vs 手臂+手 | ✅ **已决：Franka Panda 单臂 + 平行夹爪**（`fixed=True`；动作语义与浮动手同构，底层为 IK + 关节 PD） |
| Q2 | 手型号 | ✅ **已决：Franka 平行夹爪**（两指、指尖温度网格）。探针网格**逐模态**（2026-10-04）：`force`/`depth`/`proximity` 每指 12（4×3），`elastomer` 每指 256（16×16）——见 Q5 |
| Q3 | v1 相机 | ✅ **已决：v1 数据无相机**（`cameras.enabled=false`）；**v2 相机链路已完成并验证**（Gate 3 渲染 + Gate 4 落盘/不扰动仿真，§8）：置 `enabled: true` 即写 `obs/agentview_rgb` / `obs/eye_in_hand_rgb`。madrona/BatchRenderer 在本机被缺失的 Vulkan loader 挡住（`sudo apt install libvulkan1` 可解，非 sm_120 问题） |
| Q4 | v1 任务清单 | `pick_hot_cylinder`（✅ 已实现，v1/v2 数据即它）/ `lift_object`（✅ 已实现，最小任务：同一专家去掉"hot/pick"语义，传感栈不裁剪，`sim/configs/lift_object.yaml`）/ **`pick_by_temperature`（✅ 已实现，2026-10-14：新专家任务——外观完全相同的孪生方块 + 每回合槽位置换，专家靠抓取过程中的**指尖温度变化**判材质组、判错放回去换另一块，§9-R38）** / `place_hot_object` / `push_hot_object`（v2） |
| Q5 | 触觉模态记录集合 | ✅ **已决（2026-10-04）：`force` + `depth` + `elastomer`**（2026-10-20 由 §9-R44 收窄：当前唯一 config 只留 `force` + `elastomer`，本条保留为当时的设计依据） —— `elastomer` 从实验配置**转正为正式 schema**，它是唯一能给出真压痕形状的通道（R8）。转正时把当初列的两条代价都从根上修掉，而不是接受：① **网格共享 → 逐模态网格**（`tactile.probe_grid` 支持 `{default: [4,3], elastomer: [16,16]}`，`force`/`depth` 的 4×3 与语义逐字节不变，`layouts/` 加一层 `grids/<tag>/` 别名层保持老读者可读，§4.2）；② **帧丢失 → `obs/link_pose`**（每步每指 pos+quat，让 `world_frame: true` 精确可逆，`v_link = R(q)ᵀ·v_world`），因此**不必**把 `world_frame` 记成 false —— force 的世界系语义与已有语料保持可比。代价如实记账：时间 +51%（无相机路径 84.4→127.7 ms/step，R8）、L1 ~3.1 KiB/step、**burst 每窗口 +110–218 KiB（约 1.1–2.2 MiB/集，`recording.burst_modalities` 可收口）**；语料规模见 §2.5。`proximity` **仍待定**：实测给出 `force`/`depth` 没有的接触前渐变（R23），代价 +1 个通道与点云采样开销，本次**不随 marker 场一起进**（配置层已支持，需要时单独开）。实验夹具 `sim/configs/lift_gel_preview.yaml`（`world_frame: false`）**保留**：它是往返一致性测试的对照件（用 `tactile_field` 断言两种帧下的高度场一致）。 |
| Q6 | 失败 episode 策略 | ✅ **已决：`*_failed.hdf5` 单独保留**（`keep_failed: true`）；全成功时该文件在 `close()` 里被删除——避免 `data/*.hdf5` 通配符把空文件喂给 `validate_dataset.py` 报 "no demo_N groups found" |
| Q7 | 分辨率 / 步数上限 | ✅ **已决：256×256**（2026-10-03 由 128² 上调；LIBERO 用 128²，但那是 Isaac RTX + 照片级资产，llvmpipe 下 128² 的物体只有 188 px ~14×14，夹爪-接触区难以分辨。256² → 764 px ~28×28，时间只 +9%、磁盘 2.4×；详见 §8 分辨率裁决表）；T ≤ 400（20 s，✅ 已实现） |
| Q8 | 接触记录默认档位 | ✅ **已决：L1+L3 常开；L4 默认开**（[−0.05 s, +0.25 s]、上限 10）。模态随 `tactile.modalities` 全带（含 16×16 marker 场，2026-10-04）；要压体积就设 `recording.burst_modalities: [force, depth]`——**窗口数与帧数不变**，只换模态集 |
