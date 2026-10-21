# datasets/ — 数据集根

本项目所有生成数据的统一根目录。**canonical 数据格式是 robomimic HDF5**
（`sim/recording/hdf5_writer.py`，`format_version = tempvla-1.x`，与 LIBERO 同构）；本目录只负责
**存放布局 + 侧车元数据**，不改 HDF5 内部结构（设计依据：[docs/architecture.md §9-R30](../docs/architecture.md)）。

**当前状态**：目录内只有本 README。旧栈（format 1.0）三套语料已于 2026-10-12 删除
（[§9-R31](../docs/architecture.md)）；全量数据将由 M9 用 1.6 栈重生成进 `v1.5/hdf5/`
（指尖温度 = 官方单格 (1,1,1) 实测通道，[§9-R36](../docs/architecture.md)；触觉 =
6D 合力 + 2D marker，[§9-R44](../docs/architecture.md)；合成热像（1.7）已回退，
[§9-R45](../docs/architecture.md)）。
**版本管理**：本目录只有本 README 入库（`.gitignore` 排除 `datasets/*` 的其余内容）——
数据文件、侧车、MP4 均为生成物。

## 存放布局

```
datasets/
├── README.md            # 本文件（数据卡，唯一入库）
└── v1.5/                # M9 落点。目录名沿用 1.5 时代（config 的 output_dir），**内容格式
    └── hdf5/            #   以文件内 attrs["format_version"] 为准**（当前 tempvla-1.6），
                         #   不要用目录名判断格式
```

## 侧车三件套（每个数据目录一份）

数据目录里除 `*.hdf5` 外有三个 JSON 侧车，由

```bash
conda activate tempvla && cd ~/ThermTacVLA
python scripts/build_manifest.py datasets/v1.5/hdf5                  # 侧车默认写到同一目录
python scripts/build_manifest.py datasets/v1.5/hdf5 --val-frac 0.1 --test-frac 0.1
```

生成（纯 h5py/numpy/stdlib，**不 import genesis**，训练侧机器可直接跑）：

| 文件 | 内容 | 用途 |
|---|---|---|
| `index.jsonl` | 首行 `kind=manifest`（format_version / generator / task_id / 各文件集数与失败标记）；其后**每集一行**：`file` `demo` `seed` `split` + 该集全部 HDF5 attrs（success / object_material / object_temp_c / …） | 一行读完整个数据集清单；训练侧按 `split` 过滤、按 `file+demo` 打开 HDF5，不需要自己 glob 文件 |
| `splits.json` | 划分**规则**（salt/hash/mod/fracs）、seed 与 episode 计数、逐 split 的 seed 列表 | 复现与审计：规则是 seed 上的纯函数，任何时候可重放出同一划分 |
| `stats.json` | 归一化统计 + 通道清单 `inventory`（shape/dtype/覆盖集数） | 训练归一化、通道自检；**只在 train split 的成功 episode 上计算**（val/test 泄漏防护，同 OpenVLA `dataset_statistics` 口径） |

### 划分规则（固定；不随目录/批次改变）

```
u = int(sha1("temperaturevla-split-v1:" + str(seed))[:8], 16) % 10000 / 10000
u < 0.8 -> train     0.8 <= u < 0.9 -> val      u >= 0.9 -> test
```

- **只依赖 `episode_seed`**，与数据目录、文件名、材质、生成批次无关 → 同一 seed 在任何目录、
  任何时间、增量生成下都分到同一 split（`build_manifest.py` 的跨目录一致性实测即此规则的行为证据）。
- `split` 是逐 seed 取值而非切分数据集：数据集越大越接近 80/10/10。
- 失败文件（`*_failed.hdf5`）不参与划分与统计。

### 统计口径

- **只在 train split + 成功 episode** 上统计（`stats.json.computed_on` 字段写明）。
- 小键（单个样本 ≤ 512 个元素，如 `actions`、`obs/tactile/force`）：逐元素
  mean/std/min/max/q01/q99（另附 absmax/nonzero_frac）；大键（如相机帧）走流式
  mean/std/min/max/absmax/nonzero_frac、无分位数；非 float（bool 接触、uint8 图像）
  只进 `inventory`，不归约。
- 统计形状 = 单个样本的 obs 形状（如 `obs/temperature/fingertip_grid` 是 `[2,1,1,1]`
  的逐元素统计——每指一个**实测**集中温度，§9-R36）。换 format 版本重生成后必须重跑本脚本，**不要跨格式复用 stats**。

## 注意

- **一个数据集 = 一次调用**：`scene_seed` 按材质在列表中的下标分配，按材质拆跑会改变物体
  几何/质量与 seed 的对应（R21）；`generate.py` 对子集运行会打印警告。
- 落盘路径 = config 的 `recording.output_dir`（现为 `datasets/v1.5/hdf5`），`--out` 可覆盖；
  同名文件是**截断重写**，启动时会先打印将被覆盖的文件清单。
- 做正式评测时显式给 `--val-frac/--test-frac`（规则记录在 `splits.json`，随时可复现）；
  小数据集下默认 10% 可能只落到个位数 seed。

## 任务 `pick_by_temperature` 的文件布局与读法（§9-R38；历史）

> **2026-10-20 注**：本节是历史记录——`pick_by_temperature.yaml` 已在 2026-10-19 的单配置清理中删除，
> 文中引用的 `scripts/judge_temperature_material.py` 与 `scripts/thermal_features.py` 也已在
> 2026-10-20 的 scripts 清理中删除（§9-R46）。判定统计量本身仍在 `sim/expert/thermal_judge.py`。

**一个材质对 = 一个文件**：`pick_by_temperature_<metal>_<nonmetal>.hdf5`（`sim/configs/pick_by_temperature.yaml`
的 `material_pairs`；`<metal>` 落 body 0、`<nonmetal>` 落 body 1）。与其余任务"每材质一个文件"同构，
区别是**两个物体材质不同、外观完全相同**（孪生方块），指令只给材质词。

读法要点（每集 attrs 自描述）：

- `requested_group` = 指令点名的组（`metal` / `plastic`），语言标注 `"pick up the {group} {shape}"`；
  `objects_group` / `objects_material` / `objects_temp_c` 是真值（顺序 = body 0/1）。**每体温度各自独立
  随机**（`temp_c_regimes` 各 50/50），所以"温度高低"本身不含材质信息。
- **槽位 ↔ 材质组每集置换（50/50）**：per-demo `slots` 记录本集哪个体落在哪个槽（canonical 槽位 y = ∓0.15）。
  **不要**用"目标总在某个槽"之类的先验——那正是这个任务要防的捷径。
- **专家永远先抓 body 0**（`front_object_index` 判材质时恒为 0），判对了抬走、判错了放回去换另一体。
  因此 `success` 已蕴含判据正确性，而 per-demo `judge`（JSON，每次抓取一条）给出中间过程：
  `lambda_s` / `per_finger`（逐指 λ̂）/ `onset_s` / `group_true` / `correct` / `committed` / `cand`；
  混淆矩阵直接由它统计。抓取条数 = 1（第一抓即对）或 2（第一抓判错）。
- 判据通道 = `obs/temperature/fingertip_grid`（measured，`(T,2,1,1,1)`）+
  `obs/contact/link_contact`（`(T,2,1)`，逐指取接触上升沿）。**盲判复算**：曾用
  `python scripts/judge_temperature_material.py <file.hdf5> [--sweep] [--json out.json]`
  （真值只用于打分；该工具已于 2026-10-20 删除，§9-R46）。
- **捷径警告**：`objects_grid` 在接触**前**就带 4.04–4.35× 的组间梯度增长率差（按 `\|T−ambient\|` 归一）
  → **"只看不摸"的策略原则上也能分出这两组**。拿它当输入前先想清楚这是不是你要的（专家本身只读指尖）。

实测（2026-10-14，铝-ABS，20 集 / 32 抓）：盲判 **32/32 对**、分离 **2.17×**（阈值 1.04 两侧各 1.60×/1.35×
余量）；金属 λ̂ 1.409–1.490 / 非金属 0.567–0.649，**冷热两区各自都落在同一带**（见
`docs/figures/material_judge.png`）；指间 λ̂ 分歧均值 4.08%、最坏 17.81%（但**没有任何一根单独的指越过
阈值** → 逐指判定 64/64）。

## 待办（M9 时补）

- 1.5 全量生成后用同一 `scripts/build_manifest.py` 建侧车，并在本节补**正式数据卡**：
  各任务/材质规模、温度与位姿分布、成功率、已知缺陷（R19 触觉不可位复现、R26 温度场语义边界、
  R33 单格指温的接触点记账依赖、R36 指尖通道读数约定与实测判别间隔、**R36 冲击接触的冻结过冲伪迹**
  ——engine 口径下实测 ~12.5% 的指（全部是左右指 onset 非对称集里的后触指）onset 前窗口被污染，
  用法 = 每指 onset-relative 特征 + 按"onset 前升幅 >1 °C"过滤该指该集，见 R36 右列。
  **该伪迹已随 2026-10-14 的全 `pad_face` 翻转消除**（对照实测 26.6% → 0/64，§9-R37），
  故 1.5 数据卡里应记为"engine 口径的历史缺陷"，onset-relative 读法本身仍保留（R33 的相位依赖尚在）。
- **接触面积口径（§9-R37）**：**全数据集统一 `pad_face`**（2026-10-14 定案）——compat 把与手指相连的接触
  改写为 `A_i = A_face/sqrt(n_i)`（A_face = 305.949 mm²），两指逐位对称、与求解器报几个接触点无关。
  每个文件的 `env_args.sensors.temperature.contact_area_policy` 仍逐文件记录该口径（数组形状相同，
  只有该键能区分），保留它作为自描述证据。对照实测（`pick_hot_cylinder`，GT 残差 > 物体 +0.2 °C）：
  冻结过冲伪迹 **engine 17/64 = 26.6% → pad_face 0/64 = 0.0%**。**读数据须知**：pad_face 下指尖
  在同一 episode 长度内**达不到**物体温度（残差 −8…−97 °C，engine 则饱和到 ≈0），幅值不可跨口径比较。
- 重生成后建议补一张指级体检图：逐集画出两指的 measured 前窗口升幅（>1 °C 即伪迹），
  作为数据卡的已知缺陷分布证据（`scripts/thermal_features.py` 曾提供 onset/pre 口径，
  已于 2026-10-20 删除，§9-R46）。
- **`pick_by_temperature` 的三个材质对**（铝/铜/钢 × ABS）与其余任务一起生成；每对是**独立文件**，
  数据卡须逐对标注。判据的窗口与阈值由铝-ABS 定标，其余两对靠 `k_eff` 在 `k_pad = 0.25` 下饱和
  （三金属差 ≤0.5%，§9-R35/R38）继承同一组值——生成后每对应各跑一次
  离线盲判扫描确认分离度仍在 1.5× 以上（原 `judge_temperature_material.py --sweep`，
  工具已于 2026-10-20 删除，§9-R46）。
- 评测协议（在 sim 里对 test split 跑 closed-loop、开环离线指标）待模型侧接线后写入本节。
