# TemperatureVLA 现状、改进点与目标（2026-10-08）

| | |
|---|---|
| 本文定位 | **状态快照**：现在到哪了、哪些是改进点、目标与差距。设计与实测依据以 `docs/architecture.md` 为单一事实来源；文中的 R#/Q#/M# 均指该文档 §9/§11/§10 |
| 代码状态 | 触觉栈已到 format `tempvla-1.3`（63 点 marker 场 + 合成 height map），`configs/lift_object_pair.yaml` 已切换；Gate 1/2、生成、校验全部实测通过 |
| 数据状态 | 全量数据集仍是旧栈：`data/`（13 MiB）、`data_cam/`（98 MiB）、`data_lift/` 均为 **format 1.0**（无 marker 场，`data_cam/` 还是 128² 旧分辨率）；1.3 只有 2 集 preview |
| 图 | 本文件引用的图在 `docs/figures/`，由 `scripts/report_figures.py` 生成 |
| 版本管理 | 2026-10-08 起由 git 管理（首次提交 = 基线快照）；`data*/`、`*.hdf5`、`*.mp4`（`docs/figures/` 交付视频除外）等生成物不入库 |

---

## 1. 一句话现状

生成"拾取热物体"的 LIBERO 同构数据集，在 RGB/本体之外提供**温度**与**接触触觉**两条通道，
目标是训练能融合二者、并可迁移到真机的 VLA。
目前：**数据管线与触觉表征已经成型并逐项验证；传感器保真度只做到"结构正确"，幅值未标定；
融合基线与真机迁移尚未开始。**

---

## 2. 已完成（都有实测证据）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0–M1 | 环境 + Gate 0/1/2 + 最小 episode 链路 | ✅ |
| M2 | `data/`：4 材质 × 50 = **200/200 成功、0 失败**，12.9 MiB，944 s | ✅ |
| M3 | 相机 Gate 3/4；`data_cam/` 200/200（128²，97 MiB，~53 min）；配置已升 256² | ✅ |
| M4 | 任务套件（3 个任务配置已有：`pick_hot_cylinder` / `lift_object` / `lift_object_pair`）；缺 replay/analyze 与数据卡 | ⏳ 部分 |
| M5 | format 1.1：逐模态探针网格 + `obs/link_pose`，`elastomer` 进正式 schema | ✅ |
| M6 | format 1.2：双物体任务（`objects:` 列表、box 面抓专家、逐物体通道） | ✅ |
| M7 | format 1.3：63 点 marker 场 + 去刚性探针 + 合成 height map（见 §3） | ✅ 代码+验证；全量数据未重生成 |

数据消费接口：`tempvla/tools/tactile_field.py`（读 marker 场）、`height_map.py`（合成/读高度图）、
`gel_render.py`（GelSight 风格成像）、`validate.py`（校验）——均为纯 h5py+numpy(+scipy)，
训练侧 dataloader 不需要 import genesis。

---

## 3. 本阶段改进：format 1.3（2026-10-08）

三件事，全部落在 `configs/lift_object_pair.yaml`；`pick_hot_cylinder.yaml` / `lift_object.yaml`
保持 force+depth+16×16 不变，是"绝对幅值重要时"的退路。

### 3.1 63 点 marker 场（9×7，gsmini/UniVTAC 形态）
- 每指 63 个 3D 位移标记，间距 2.03 mm（列）× 1.54 mm（行），覆盖 14.2×13.8 mm 的指垫；
  `modalities: [elastomer]`，不再采样任何刚性探针。
- 端到端：Gate 1（含新加的 marker 响应检查）、Gate 2 全绿；preview 2 集 2/2 成功；validator 0 issues。

### 3.2 去刚性探针（force/depth 不再记录）
- 专家不依赖它：depth 缺失时回退到 `obs/contact/link_contact`（`contact_links: true` 独立构建）。
- 顺带规避了 R22 的"近二值"缺陷——旧 force 通道 0→45 N 阶跃，本来就是不良监督信号。
- **代价**：形状信号变弱。刚性 depth 曾提供 box（面接触，0.09 mm 展差）vs 圆柱（曲率，1.65 mm）的强区分；
  现在只剩 marker 场的 footprint 宽度（见 §4.1-R25 与图 2）。

### 3.3 合成 height map（64×64，`(T,F,H,W)` float32 米）
- 我们的 gel 是**虚拟传感器**（SDF 查询 + 网格采样），场景里没有可形变几何体——没有东西可以"拍"，
  所以 map 是从 63 点场**纯样条重采样**得到的（语义 = UniVTAC 的 `press_depth`，不增加信息）。
- 栅格为 cell-centred（覆盖 marker 网格的"格"，外缘半格平延），与渲染器的 `grid_m_per_px` 同一范围模型；
  坐标 clamp 到样本范围（仅 `mode="nearest"` 会因 scipy 的 mirror 预滤波在边缘 ringing，实测 5.05→5.17 µm）。
- 校验方式：**从存下的 marker 重合成再逐元素比对**（独立复核 `max|Δ| = 0.000e+00`），不比峰值
  （孤立尖峰在 9×7→64×64 下会丢 ~1.5% 峰值）。
- 体积代价：+0.32 MiB/episode（4.27 vs 3.63 MiB / 2 集，**+17%**）；burst 窗口不带它。

---

## 4. 改进点（已知缺陷，按对"训练/迁移"的影响排序）

图 3（`kernel_grid_coupling.png`）和第 4 节的表格一起看。

### 4.1 传感器保真度（最影响"迁移到真机"的一组）

| # | 缺陷 | 实测/机制 | 改进方向 |
|---|---|---|---|
| R25 | **弹性核宽度随网格间距变宽**（核在 FFT 格上做，以"格"为单位） | 同一抓取同 seed：峰值 16×16 **27.4/36.0 µm** → 9×7 **5.8/7.9 µm**；把 256 场在 63 个位置上重采样预测 **26.3/36.0 µm** → 掉幅值的是核不是采样；形状被抹平成宽穹顶（footprint：圆柱 4/7 列 × 9/9 行，盒 7/7 × 9/9） | ① 按 ~1/dx² 重标 `lambda_d/lambda_s`；② 或启用 `compressibility<1` / `elastomer_thickness>0` 的精确谱解（R8：FFT complex 崩溃，1.4.1 未重试） |
| R22 | 刚探针 force/depth **近二值**（0 → 45 N 阶跃，位置由探针几何决定） | 已实测：把合拢速度降 10× 也不能平滑 | pair config 已去掉该通道；旧数据仍有 |
| R14+R16 | 温度**绝热**（conv/rad 关闭）且接触热增益受 `gs.EPS` 支配 | 绝对升温速率**不可信**；只有结构与相对量可用 | 对标真机（热电偶/热像）；或至少量化 `gs.EPS` 依赖并给出换算约定 |
| R19 | tactile/* 不可位复现（GPU 归约顺序） | ee_pos/actions 位级一致；触觉有 ~1e-3 相对量级漂移，接触瞬间另有力通道 0.003–5 N 的尖峰差（本 session 复核：同一 config 两次生成，9×7 场 \|disp\| 峰值 16.49 vs 16.45 µm） | 训练/断言用容差与统计性质，勿逐位比对 |
| R8 | engine 自带的一批真实物理旋钮**未启用** | `elastomer_thickness`+`compressibility`（谱解）、`hysteresis_strength/tau`（Maxwell 粘弹）、`noise`、`probe_radius_noise`、`dead_taxel_*`、Schmitt 接触阈值、`read()`/`read_ground_truth()` 双分支 | 逐项开启 → 就是"更真实的触觉模拟"，对应格式 1.4 |

### 4.2 数据工程

| 缺陷 | 说明 | 改进方向 |
|---|---|---|
| 旧全量数据未重生成 | `data/`、`data_cam/`、`data_lift/` 是 format 1.0：无 marker 场、无多物体、`data_cam/` 还是 128² | M8：用新栈一次性重生成（见 §7） |
| R20 | eye-in-hand 相机 ~10% episode 卡顿（8 s → 25–350 s），只费墙钟不坏数据 | `cameras.near: 0.01→0.03`（会作废旧 RGB，择机） |
| R21 | 材质与几何 1:1 混淆（`scene_seed` 用列表下标） | 约定：一次生成全部材质，不要按材质拆跑 |
| height map +17% 体积 | 全量 200 集会多 ~64 MiB | 预算可接受；必要时 `burst_modalities` 或按需记录 |

### 4.3 验证方法

| 缺口 | 现状 | 改进方向 |
|---|---|---|
| height map | 已有精确重合成比对（`validate.py`） | ✅ 已闭环 |
| 温度通道 | **无自动校验**（如"接触前恒定、接触后单调、幅值有限"） | 加 validator 规则（小工作量） |
| 物理正确性 | 全部校验都是"自洽性"（sim 对 sim），没有真值对标 | 真机数据进来之前无法闭环（见 §5 的 sim-to-real 层） |

---

## 5. 目标与差距

**目标**：一个可训练、可评测、可迁移的"温度 + 接触触觉"VLA 数据与基线体系——
在仿真里提供带两条物理通道的 LIBERO 同构数据，训练融合策略，并最终在真机（gsmini 触觉 +
指端温度）上验证迁移。

里程碑（M0–M7 见 §2；M8 起为本文提出的增量，尚未写进 architecture.md §10）：

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M4 | 任务套件补齐：replay/analyze + 数据卡 | ⏳ 部分 |
| M8 | 用 1.3 栈重生成全量数据（4 材质 × 50，256²） | 未开始 |
| M9 | **融合基线**：温度 + 触觉双编码器的最小策略，做通道消融（有/无触觉、有/无温度） | 未开始 |
| M10 | **触觉 sim-to-real**：gsmini 对齐（间距/幅值/标定），仿真与真机的 marker 场可比 | 未开始 |
| M11 | **温度 sim-to-real**：真机热电偶/热像对标（R14/R16 结案的前提） | 未开始 |
| M12 | **真机机械臂迁移** | 未开始 |

五层差距（图 4 右；百分比是**主观**就绪度估计，只用于定位，不是指标）：

| 层 | 就绪度 | 说明 |
|---|---|---|
| 数据管线（仿真侧） | ~85% | 多物体、双相机、marker 场、height map 都已闭环；全量数据未重生成 |
| 触觉保真度 | ~40% | 场结构正确；幅值/核宽未标定（R25），engine 的粘弹/噪声/死点等真实旋钮未启用（R8） |
| 温度保真度 | ~25% | 通道因果、局部、随物体温度变化（图 1）；绝热 + 增益未标定（R14/R16） |
| 仿真→真机对齐 | ~5% | 无真机数据、无硬件在环 |
| 融合 VLA 基线 | 0% | 未训练任何策略 |

---

## 6. 本次生成的可视化（`docs/figures/`）

| 文件 | 看什么 |
|---|---|
| `thermal_channel.png` | **温度通道**。(a) 4 材质指尖升温中位曲线——接触（步 11）前严格为零，接触后单调上升（绝热 R14：只靠接触导热）；(b) 200 个 episode 的"最终升温 vs 物体初温"散点，r=0.77——通道确实携带物体温度信息（幅值不可信，R16）；(c) copper 一集的最终 4×4 指尖温度网格——**只有接触带在升温**（86/54 °C），其余格仍为环境 21 °C；物体自身几乎不降温（200 集中位 <0.05 °C，最大 0.23 °C——物体热质量远大于指尖） |
| `tactile_field_and_map.png` | **触觉表征**。上/下 = 圆柱/立方体，左/右 = 63 点 marker 场 / 由它合成的 64×64 height map，共用绝对色标 0–8 µm，青 "+" 是 marker 节点。要点：map 与场是同一数据的两种视图（不增加信息）；形状差异只剩 footprint 宽度（>50% 峰值：圆柱 4/7 列 × 9/9 行 = 单侧条带，盒 7/7 × 9/9 = 满垫） |
| `kernel_grid_coupling.png` | **R25 缺陷**。(a) 16×16（256 样本）实测场，峰值 27.4 µm；(b) 把 (a) 只在 63 个位置重采样——峰值 26.4 µm（说明"少采样"几乎不损失）；(c) 同一抓取、改用 9×7 实测——峰值只剩 5.8 µm，穹顶被展宽。下方柱状图：4 个"抓取 × 手指"组，measured/predicted ≈ **22%**。结论：掉幅值的是**弹性核随格距变宽**，不是采样 |
| `milestones.png` | **目标与差距**。左：M0–M12 状态；右：五层就绪度 |
| `gel_panel_with_heightmap.mp4` / `gel_panel_field_only.mp4` | 同一条 preview episode（pair, 256² 双相机 + 指尖温度网格 + GelSight 风格触觉面板）的对照回放：前者触觉面板源为 **64×64 height map**，后者为 63 点场。注意标题里的 slope exaggeration（场只有 ~8 µm，渲染放大 ×567 才可见）。演示集的桌子为暖木色（`table.color`，2026-10-08 起）：**仅表面渲染，不进物理也不进任何通道**——同 seed 重生成后 R25 峰值（5.78/5.69、7.89/4.92 µm）与 height_map max ≡ marker 峰值逐位复现；旧数据集的灰白桌对应 `color: null`（默认） |

复现：

```bash
cd ~/TemperatureVLA
conda activate tempvla
python scripts/report_figures.py         # 图 1–4 → docs/figures/（缺 /tmp preview 文件时对应图跳过）
# preview 数据本身（视频用的就是这对 episode，seed 5 → demo_0/ep5、demo_1/ep6）：
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe python scripts/generate.py \
    --config configs/lift_object_pair.yaml --episodes 2 --seed 5 --out /tmp/hm3
# 视频本身：
python scripts/render_video.py /tmp/hm3/lift_object_pair_aluminum.hdf5 \
    --demo 1 --markers --out docs/figures/gel_panel_with_heightmap.mp4
```

---

## 7. 建议的下一步（最短路径）

1. **触觉物理校准实验**（半自动、一天内）：同 seed 扫 `lambda_d/lambda_s`（按 ~1/dx² 重标）并重试
   `compressibility/elastomer_thickness` 谱解路径 → 目标：63 点场峰值回到 20–40 µm 且穹顶不过宽，
   与 16×16 的形态一致 → 闭合 R25。
2. **M8 全量重生成**：4 材质 × 50 @ 256²，pair 任务；按 preview 实测 2.14 MiB/episode 外推
   → **~430 MiB**、约 1–1.5 h 墙钟。这是所有下游训练的前提。
3. **温度通道对标**（可与 2 并行）：把 R16 的 `gs.EPS` 依赖量化成换算表，或在真机上取一组
   指尖热电偶数据做幅值标定 → 决定温度通道是"标定后绝对量"还是"相对量"进训练。
4. **M9 融合基线**：小双编码器（温度网格 + marker 场/height map）+ 动作头，先做通道消融，
   量化"触觉/温度到底带来多少成功率/鲁棒性增益"。
5. **M10/M11 硬件**：gsmini（触觉）+ 指端热电偶（与触觉同址，R 级风险）→ 采集真机数据做 sim-to-real。
