# sim/ simulation environment generation

用于生成VLA训练测评仿真数据集，所有可执行入口都在 [scripts/](../scripts/)

```
sim/
├── assets/                      随仓库分发的资产（机器人 URDF、物体网格 + MTL + 贴图）
│   ├── urdf/panda_bullet/       Franka Panda（Genesis 1.4.1 同款）
│   └── robosuite/               robosuite_tabletop.yaml 用的 can/lemon/milk + 贴图
├── configs/                     定义具体任务
|   └── thermal_recognition      冷热物体识别
|       ├── pick_cold_can.yaml      抓取冷饮罐
|       └── pick_lemon.yaml         抓取柠檬（一个柠檬 + 两个罐子 + 两盒牛奶；不判温）
├── config.py                    任务配置：YAML → 严格校验的嵌套 dict（无 dataclass）
├── scene.py                     场景构建
├── expert/                      专家管线
│   ├── pipeline.py              专家管线常量（PipelineConfig / PIPELINE）
│   ├── expert.py                专家接口
│   ├── actions.py               7维 delta 动作
│   └── thermal_judge.py         热学判断器
├── obs.py                       相机rgb渲染
├── rotations.py                 几何变换
├── sensors/                     传感器
│   ├── tactile.py               触觉装配 + 6D/2D 派生常量（唯一参数源）
│   └── temperature.py           温度传感器
├── recording/                   记录与落盘
│   ├── runner.py
│   ├── hdf5_writer.py           流式 LIBERO 风格 HDF5
│   ├── events.py                L3/L4 纯逻辑
│   └── meta.py                  文件级元数据
├── tools/                       无引擎离线工具
│   ├── validate.py              纯 h5py/numpy 契约校验
│   ├── tactile_field.py         marker 场读取：2D 切平面位移（1.6）/ 3D 位置（≤1.5）
│   └── gel_render.py            高度场 → GelSight 风格光度图（仅 ≤1.5 数据）
└── compat/                      钉版引擎的补丁（build 前安装，幂等）
    ├── sensor_compat.py         sensor reset 修复 + pad_face 接触面积
    └── mesh_load_compat.py
```

- **引擎边界**：只有 `scene`、`sensors/*`、`compat/*` 直接 import genesis（且全部函数级懒加载，
  `meta` 另有一处取版本号的懒加载）；模块级 import torch 的只有 `sensors/temperature.py` 与
  `compat/sensor_compat.py`。`sensors/tactile.py` 模块级不碰 torch（`_extract` 内局部 import，
  仅 `force_torque` 用），`sensors/__init__.py` 是 PEP 562 懒导出。
- **`compat/` 是唯一触碰引擎行为的地方**（幂等补丁，build 前由入口安装）。
- 「纯 numpy 档」（不装 Genesis 也能 import）：`config`、`rotations`、`gripper`、`thermal_judge`、
  `expert/pipeline.py`、`tools/*`、`recording/events.py`、`recording/hdf5_writer.py` —— 见「依赖分档」。

## 数据流（一集的生命周期）

```
configs/<task>.yaml
   │ load_config()                     config.py：YAML → 嵌套 dict（严格校验）
   ▼
TaskConfig（dict）
   │ build_scene()                     scene.py（入口先装 compat/ 的三个引擎补丁）
   ▼
SceneBundle {env, robot, objects, 传感器 handles}
   │ 每回合：scene.reset() → seed_temperature() → expert 循环（≤ max_steps）
   ▼
EpisodeRunner                          recording/runner.py：一集端到端，四层记录
   ├─ expert/expert.py → expert/actions.py（7 维 delta → DLS IK → PD） → 引擎
   ├─ obs.py      ← sensors/（触觉 6D 合力 + 2D marker / 温度实测通道），每控制步打包
   ├─ events.py   L3 接触事件 + L4 burst 窗口（200 Hz，ring buffer）
   └─ meta.py     env_args / problem_info / 语言指令（JSON 可序列化）
   ▼
HDF5Writer                             recording/hdf5_writer.py（流式，一集一 flush）
   ▼
datasets/v1.5/hdf5/<task_id>_<group>.hdf5
   ▼
sim/tools/validate.py                  scripts/validate_dataset.py 包 CLI；纯 h5py，不需要 Genesis
```

## 模块地图（行数为 2026-10-20 的 `wc -l`）

### 顶层 —— 配置 / 场景 / 行为

| 文件 | 行数 | 职责 |
|---|---|---|
| `config.py` | 575 | 任务配置：一个 YAML 端到端描述一个任务（机器人/桌子/物体/传感器/热学/随机化/成功判据/记录）；`load_task_config` 返回**普通嵌套 dict**（无 dataclass、无默认值），加载时逐节严格校验键集合与类型，未知/缺失键直接报错，全库直接 `cfg["section"]["key"]` 取值。顶层 `temperature_groups` 指定每集置换 `temp_c` 温度档的同外观物体组（空列表 = 旧固定分布）。YAML 只保留任务级选择（`tactile.modalities`、`temperature` 三键）；其余参数是 `sensors/tactile.py` / `sensors/temperature.py` 与 `obs.py` 的模块常量 |
| `scene.py` | 1269 | 场景构建（机器人/桌子/物体/传感器注册）+ 每回合 reset/播种；**build 期 vs 回合期的契约都在这** |
| `obs.py` | 283 | 观测组装：本体状态/位姿 + `tactile/*`（1.6：6D 合力 / 2D marker）+ `contact/*` + `temperature/*` + 双相机；burst payload；**相机渲染后端常量 `RENDERER`（唯一源，不是 YAML 键）** |
| `gripper.py` | 353 | 夹爪几何的**独立第二推导**（无 Genesis/torch）：指垫面/闭合轴/marker 格；等价性曾由 `scripts/probe_gripper_model.py` 断言（<1e-6 m），该脚本已于 2026-10-20 清理时删除（§9-R46） |
| `rotations.py` | 107 | 纯 numpy 四元数（wxyz/Hamilton）；动作数学不依赖引擎广播规则 |

### `expert/` —— 专家管线

| 文件 | 行数 | 职责 |
|---|---|---|
| `expert/pipeline.py` | 125 | 冻结的 `PipelineConfig` 单例 `PIPELINE`：全部专家管线常量（接近高度、R43 安全转场、抓取抖动、CLOSE/SQUEEZE 时序、放置余量、温度判定窗口/基线/相对落差、λ 窗口与阈值）。YAML 的 `expert:` 只保留任务契约（`approach` + 三个温度开关）；需要时可用 `dataclasses.replace(PIPELINE, ...)` 在进程内覆写并注入 `EpisodeRunner(pipeline=...)` |
| `expert/expert.py` | 1072 | 脚本专家（输出与策略同构的 7 维动作）：`PickExpert`（内含 R43 安全转场：先升到最高物体之上再横移）、`PickByTemperatureExpert`（CLOSE→JUDGE→判错 RELEASE 换下一个，R38；另含 R42 冷热判定——触碰保持读 measured 指尖，判不冷原地释放）、`PutInTrayExpert`（JUDGE→CARRY→DROP→RETREAT，R39）；`APPROACHES` 注册表 + `build_expert` 按 `expert.approach` 构造；只消费 policy-rate 观测 |
| `expert/actions.py` | 160 | 7 维 delta 动作管线（LIBERO 语义）：平移/旋转归一化 + 夹爪协同 → DLS IK（暖启动）→ PD；TCP = 指间夹取中心 |
| `expert/thermal_judge.py` | 518 | 判材质统计量：`dT/dt = λ(T_cell − T)`；**λ 不含温度**（同一阈值判冷热，R38/R39）；在 shipped 实测通道上估计（RC/噪声/量化在环内）；`fit`/`ratio` 两个估计器（R40） |

### `sensors/` —— 传感器装配

| 文件 | 行数 | 职责 |
|---|---|---|
| `finger_layout.py` | 559 | 从 URDF 运行期推导指垫探针布局：mesh 顶点取 pad slab + 局部高度吸附（AABB 会悬空/提前闭合）；网格由 `probe_grid` 指定（1.6 主用 force 4×3 / elastomer 16×16）+ `layouts/` HDF5 packing |
| `tactile.py` | 424 | 触觉传感器装配 + **1.6 派生的唯一参数源**：`PROBE_GRID`/偏移/半径/弹性体常数等模块常量、`force_6d`（探针求和 → `[Fn,Ft]`）、`marker_2d`（切平面投影）、`pad_tangent_basis`、`channel_format`（`layouts.attrs["channel_formats"]`）；`history_length` = 控制减速 + 中值折叠；模块级不 import torch |
| `temperature.py` | 400 | TemperatureGrid 装配 + 每回合 GT 播种；**温度参数的唯一源**（模块常量：接触面积口径/非理想性/材料/链路温度耦合，YAML 只留两组 grid + 室温）；指尖 = 官方 `(1,1,1)` 实测通道 + 官方非理想性；物体网格零噪声；共享元数据 last-wins / sticky / properties 不刷新 / 世界 AABB 切割等引擎坑都在这 |

### `recording/` —— 记录与落盘

| 文件 | 行数 | 职责 |
|---|---|---|
| `runner.py` | 601 | `EpisodeRunner`：一集端到端（L1 策略率 + L3 事件 + L4 burst）；一集一 flush，生成期 RAM 恒定 |
| `hdf5_writer.py` | 239 | 流式 LIBERO 风格 HDF5（robomimic 超集）；`FORMAT_VERSION = tempvla-1.6`；组布局见模块 docstring 与 docs §4 |
| `events.py` | 209 | L3/L4 纯逻辑（numpy，无 Genesis）：上升沿接触事件 + 200 Hz burst 窗口；可单测、可离线复用 |
| `meta.py` | 624 | 文件级元数据：`env_args` / `problem_info` / 语言指令；训练脚本只凭它就应能重建观测/动作契约；genesis 仅在取引擎版本号时懒加载（try/except） |

### `tools/` —— 无引擎的离线工具（训练/dataloader 侧可直接用）

| 文件 | 行数 | 职责 |
|---|---|---|
| `validate.py` | 621 | 纯 h5py/numpy 契约校验（结构/形状/dtype/时间轴/非有限值），返回 Report；按 `layouts.attrs["channel_formats"]` 校验 1.6 的 `force (T,L,6)` / `elastomer (T,L·P,2)`，缺 attr 时回退旧规则 |
| `tactile_field.py` | 311 | marker 场读取：1.6 = 2D 切平面 `(T,F,ny,nx,2)` + `reference_positions_2d`；≤1.5 = 3D 位移 → 位置 → link 系（height map 路径） |
| `height_map.py` | 179 | **仅 ≤1.5**：3D marker 场 → 64×64 样条重采样（纯重采样，不加信息）；1.6 的 2D 输入显式报错 |
| `gel_render.py` | 152 | 高度场 → GelSight 风格光度图（三色 LED 编码表面法向）；Pillow 懒加载 |

### `compat/` —— 钉版引擎的补丁（build 前安装，幂等）

| 文件 | 行数 | 职责 |
|---|---|---|
| `sensor_compat.py` | 150 | `install_sensor_reset_fix`（传感器 reset 修复）+ `install_contact_area_policy`（**pad_face 接触面积策略**，R37；策略值在 `sensors/temperature.py` 的 `CONTACT_AREA_POLICY`，`scene` 每次建场景都安装） |
| `mesh_load_compat.py` | 45 | trimesh≥5 外部贴图解析策略修复（robosuite `map_Kd ../../…` 会静默失贴图，R41） |

## 三条贯穿全包的约定

1. **build 期 vs 回合期**：几何/材质/尺寸在 `scene.build()` 时定死（温度传感器把 link AABB 与体素
   冻结在 build 期）⇒ 物体按 canonical 位姿建、每回合用 root DOF 挪位；每回合温度在 `reset()`
   之后、第一步之前经 `seed_temperature()` 写入。改场景先想这条，否则会撞上引擎的冻结语义。
2. **数据契约（传感器口径）**：触觉 = **6D 合力**（每指一个 `[Fn,Ft]`，link 系）+
   **2D marker** 切平面位移（format 1.6；≤1.5 是 3D 场 + 可选 height map）；
   指尖温度 = 官方单格 `(1,1,1)` **实测**通道（RC/噪声/量化，R36）+ `pad_face` 接触面积（R37，
   全数据集定案）。
   口径全量记进 `env_args` 与 `layouts.attrs["channel_formats"]`，事后可比对。
3. **策略率消费 + 流式落盘**：expert 与一切判定只消费 policy-rate 观测（绝不裸读引擎）；
   HDF5 一集一 flush，生成期内存恒定。

## 依赖分档

- **纯 numpy/h5py/yaml 档**（不装 Genesis 也能 import）：`config`、`rotations`、`gripper`、
  `thermal_judge`、`tools/*`、`recording/events.py`、`recording/hdf5_writer.py`。
- **引擎档**：`scene`、`sensors/*`、`compat/*`。genesis 的 import 全部是**函数级懒加载**
  （`scene`、`sensors/*`、`meta` 取版本号），模块级 import torch 的只有
  `sensors/temperature.py` 与 `compat/sensor_compat.py`（`sensors/tactile.py` 在
  `_extract` 内局部 import，`sensors/__init__.py` 是 PEP 562 懒导出 ⇒
  `import sim.config` 不拉 torch / genesis）。
- `expert` / `actions` / `obs` / `runner` 不直接 import 引擎——它们只操作传入的句柄与数据。

## 怎么读（推荐顺序）

1. 主线：`config.py` → `scene.py` → `expert/expert.py` → `recording/runner.py` → `recording/hdf5_writer.py`
2. 判材质任务：加 `thermal_judge.py`（R38/R39 的判定核心）
3. 想改传感器：`sensors/finger_layout.py`（几何推导）→ `sensors/tactile.py` / `temperature.py`（装配）
4. 想用数据而不建场景：`tools/` + `recording/events.py`（纯 h5py/numpy 档）
5. 引擎行为不对时：先看 `compat/`，再查 docs §9 里对应的 R#

## 维护约定

新增/删除/改名模块时同步更新本文件的两张表（模块地图、依赖分档）与两张图（架构图、数据流）；
具体的实测数字、契约细节、风险只写进 docs/architecture.md，本文件不复制。
