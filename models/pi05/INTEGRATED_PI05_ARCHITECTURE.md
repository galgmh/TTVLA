<!-- 本文件是用户给出的 Thermal-π0.5 主干集成规格，逐字保存，作为实现与测试的唯一依据。 -->

# Thermal-π0.5：热觉空间记忆与主动接触策略的主干集成规格

## 1. 任务定义

在 OpenPI 的 PyTorch π0.5 上增加三个可训练模块，并让它们直接影响 Action Expert 的 flow-matching 动作生成：

1. `ThermalContactEncoder`：将每次短时接触的温度/力觉序列编码为局部热觉 token；
2. `ThermalSpatialMemory`：将任意数量的历史接触投影到物体表面候选点，形成空间热场表示；
3. `ActiveStrategyModule`：根据语言、视觉、机器人状态和空间热场选择继续探测还是抓取，并产生可微的目标 token。

这三个模块不是 π0.5 外部规划器。它们产生的 token 必须加入 π0.5 的 prefix，由 PaliGemma 处理并通过逐层 prefix K/V 提供给 Action Expert。最终执行信号仍然是 π0.5 的 7D action chunk。

首版只实现 PyTorch 路径，不实现 JAX/Flax 对等版本。

## 2. π0.5 原结构与插入位置

OpenPI 当前 PyTorch π0.5 的核心路径是：

```text
images ──SigLIP──┐
                 ├── prefix embeddings ── PaliGemma/VLM tower ── prefix K/V
language tokens ─┘                                             │
                                                               ▼
noisy action chunk ─ action_in_proj ─ Action Expert tower ─ velocity prediction
flow timestep ───────────────────────── adaRMSNorm condition
```

集成后改为：

```text
历史K次热接触 [B,K,L,C]
        │
        ▼
ThermalContactEncoder ───────────────► K个接触token
                                             │
N个表面候选点/法向 ─ SurfacePointEncoder ────┤
                                             ▼
                                  ThermalSpatialMemory
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
                N个表面热场token                         M个压缩空间token
                         │                                       │
图像/语言全局特征 ────────┤                                       │
                         ▼                                       │
               ActiveStrategyModule                              │
             skill + target distribution                         │
                         │                                       │
                differentiable goal token                        │
                         └───────────────────┬───────────────────┘
                                             ▼
images | language | M spatial tokens | 1 goal token
                                             │
                                      π0.5 prefix tower
                                             │ prefix K/V
                                             ▼
                                      π0.5 Action Expert
                                             │
                                      7D action chunk
```

### 主干集成点

修改 `PI0Pytorch.embed_prefix()`：在原有 image embeddings 和 language embeddings 后追加：

```text
[thermal_spatial_token_1 ... thermal_spatial_token_M, active_goal_token]
```

这些 token 必须投影到 PaliGemma prefix width。默认真实模型为 2048 维，但代码必须从 `paligemma_config.width` 读取，禁止写死。

所有新增 prefix token：

- `pad_mask=True` 时有效；
- 与 image/language 处于同一 prefix attention block，`att_mask=0`；
- Action Expert 通过现有 PaliGemma-with-expert 层间 K/V 使用它们；
- 不改变 noisy action suffix 和 π0.5 的 adaRMS timestep 注入方式。

## 3. 训练样本的时间语义

一个训练样本对应控制时刻 `t`：

```text
输入：t时刻图像、语言、机器人状态
     t之前已经完成的K次接触窗口
     t时刻可见物体表面候选点

监督：t开始的H步动作块
     t时刻高层意图（PROBE/GRASP/OTHER）
     下一目标接触点或抓取点
     仿真器提供的局部/表面温度标签
```

严禁未来信息泄漏：正在执行的 probe 产生的温度窗口，只有 probe 完成后的训练样本才能进入历史 memory。发起 probe 之前的样本不能包含该次接触结果。

## 4. Observation 新字段

扩展 OpenPI `Observation`，所有字段为可选；`thermal_enabled=False` 时原模型行为必须不变。

```python
# 历史接触原始序列
thermal_history: Tensor | None          # [B,K,L,C]
thermal_history_time_mask: Tensor | None# [B,K,L], bool
contact_mask: Tensor | None             # [B,K], bool
contact_xyz_obj: Tensor | None          # [B,K,3]
contact_normal_obj: Tensor | None       # [B,K,3]
contact_age_s: Tensor | None            # [B,K,1]

# 当前物体表面候选点
surface_xyz_obj: Tensor | None          # [B,N,3]
surface_normal_obj: Tensor | None       # [B,N,3]
surface_mask: Tensor | None             # [B,N], bool
surface_reachable: Tensor | None        # [B,N], bool
surface_grasp_score: Tensor | None      # [B,N,1]

# 机器人与任务的附加连续量
ee_xyz_obj: Tensor | None               # [B,3]
probe_budget_remaining: Tensor | None   # [B,1]
```

默认形状：

```yaml
K: 8                 # 历史接触上限；允许0到8个有效接触
L: 25                # 0.5 s × 50 Hz
C: 14                # 双指温度、差分、斜率、力、接触状态等
N: 64                # 当前候选表面点
D: 256               # 新模块内部维度
M: 8                 # 压缩后送入π0.5的空间token数
H: 10                # action horizon
A: 7                 # LIBERO式动作维度
```

`link_id`、link 名称、上下/把手区域标签及真实 link 温度不得进入 Observation。

## 5. 监督字段

监督字段不放入模型 Observation，放在训练 batch 的 `thermal_supervision` 中：

```python
@dataclass
class ThermalSupervision:
    contact_temp_c: Tensor          # [B,K,1]
    contact_valid: Tensor           # [B,K,1]
    surface_temp_c: Tensor          # [B,N,1]
    surface_temp_mask: Tensor       # [B,N], bool
    surface_safe: Tensor            # [B,N,1]
    strategy_type: Tensor           # [B], 0=OTHER,1=PROBE,2=GRASP
    target_surface_index: Tensor    # [B], OTHER时为-1
```

其中 `surface_temp_c`、`surface_safe` 和 contact 对应 link 温度都是仿真 privileged 标签，只参与 loss。

## 6. 新增神经网络模块

新增目录：

```text
src/openpi/models_pytorch/thermal/
  __init__.py
  config.py
  contact_encoder.py
  point_encoder.py
  spatial_memory.py
  strategy.py
  integrated_encoder.py
  outputs.py
  losses.py
```

### 6.1 ThermalContactEncoder

输入：

```text
thermal_history          [B,K,L,14]
thermal_history_time_mask[B,K,L]
contact_mask             [B,K]
```

实现：

1. reshape 为 `[B*K,14,L]`；
2. 4个 causal residual Conv1D block，hidden channels `64,64,128,128`；
3. dilation `1,2,4,8`，kernel size 3；
4. masked temporal mean + last-valid pooling；
5. MLP 投影到 `[B,K,128]`；
6. 输出温度均值、log variance、contact-valid logit。

输出：

```python
@dataclass
class ContactEncoderOutput:
    latent: Tensor          # [B,K,128]
    temp_mean_c: Tensor     # [B,K,1]
    temp_logvar: Tensor     # [B,K,1]
    valid_logit: Tensor     # [B,K,1]
```

热编码器不能接收接触坐标和候选点信息，以防用位置代替热响应完成温度预测。

### 6.2 SurfacePointEncoder

输入每个候选点的：

```text
normalized xyz_obj(3)
normal_obj(3)
reachable(1)
grasp_score(1)
distance_to_ee(1)
```

坐标使用 6-band Fourier encoding，再经过 MLP：

```text
input → Linear(256) → SiLU → Linear(256) → LayerNorm
```

输出 `[B,N,256]`。训练时随机打乱候选点顺序。

### 6.3 ContactTokenBuilder

将局部热 latent 与空间属性结合：

```text
contact token = MLP([
  thermal_latent,
  Fourier(contact_xyz_obj),
  contact_normal_obj,
  predicted_temp_mean,
  predicted_temp_std,
  valid_probability,
  normalized_age
])
```

输出 `[B,K,256]`。它不使用固定区域槽。

### 6.4 ThermalSpatialMemory

这是可训练的神经记忆读取器，而不是在网络内部保存跨 episode 状态。原始历史接触由 rollout/dataset 组装，网络每次前向重新编码，因此支持 batch、DDP 和离线训练。

结构：

1. 2层 Set Transformer 编码 K 个 contact tokens；
2. N 个 surface tokens 作为 query，对 contact tokens 做 2层 cross-attention；
3. 得到 `[B,N,256]` 的 `surface_thermal_features`；
4. 每个候选点预测 `temp_mean_c`、`temp_logvar`、`safe_logit`；
5. 使用 M=8 个可学习 latent query，对全部 surface thermal features 做 Perceiver resampling；
6. 得到 `[B,M,256]` 的 `spatial_latents`。

K=0 必须合法：加入一个可学习 `NO_CONTACT` token，同时预测较高方差。所有 attention 必须正确应用 contact/surface mask。

安全性 head 要以 language summary 为条件：

```text
safe_logit_i = SafeMLP(surface_thermal_feature_i, language_summary)
```

这样同一温度点在“寻找冷水”和“寻找可安全握持位置”中可得到不同任务相关判断。

### 6.5 ActiveStrategyModule

输入：

```text
surface_thermal_features [B,N,256]
masked mean image embeddings → [B,256]
masked mean language embeddings → [B,256]
robot state projection → [B,256]
probe budget embedding → [B,256]
```

输出：

```python
@dataclass
class StrategyOutput:
    skill_logits: Tensor         # [B,3]: OTHER/PROBE/GRASP
    probe_logits: Tensor         # [B,N]
    grasp_logits: Tensor         # [B,N]
    selected_distribution: Tensor# [B,N]
    goal_feature: Tensor         # [B,256]
```

约束：

- unreachable 点同时 mask probe/grasp；
- grasp score 低于阈值的点 mask grasp；
- probe budget 为0时 mask PROBE；
- `surface_mask=False` 的 logits 设为 dtype 可表示的最小值。

可微选点：

```text
训练前期：softmax point distribution
训练后期：straight-through gumbel-softmax(hard=True)
评测：masked argmax
```

目标特征：

```text
probe_goal = Σ probe_distribution_i * surface_feature_i
grasp_goal = Σ grasp_distribution_i * surface_feature_i
goal_feature = skill_prob_probe * probe_goal
             + skill_prob_grasp * grasp_goal
             + learned_skill_embedding
```

该 `goal_feature` 经投影后作为一个 `active_goal_token` 加入 π0.5 prefix。于是主动策略的输出在同一次前向中直接条件化 Action Expert，而不仅是辅助分类结果。

### 6.6 IntegratedThermalEncoder

统一入口：

```python
class IntegratedThermalEncoder(nn.Module):
    def forward(
        self,
        observation,
        image_embeddings,       # π0.5已算出的原始image embeddings
        language_embeddings,    # π0.5已算出的原始language embeddings
    ) -> IntegratedThermalOutput:
        ...
```

输出：

```python
@dataclass
class IntegratedThermalOutput:
    prefix_tokens: Tensor       # [B,M+1,paligemma_width]
    prefix_mask: Tensor         # [B,M+1]
    contact: ContactEncoderOutput
    field_temp_mean_c: Tensor   # [B,N,1]
    field_temp_logvar: Tensor   # [B,N,1]
    field_safe_logit: Tensor    # [B,N,1]
    strategy: StrategyOutput
```

`prefix_tokens` 的构成为：

```text
Linear(256 → paligemma_width)(8 spatial latents)
+ Linear(256 → paligemma_width)(1 active goal feature)
+ learned token-type embeddings
```

## 7. 对 PI0Pytorch 的具体修改

### 7.1 配置

在 `Pi0Config` 增加可冻结配置 `thermal: ThermalPi0Config | None = None`，至少包含：

```python
enabled: bool = False
history_size: int = 8
window_size: int = 25
signal_dim: int = 14
num_surface_points: int = 64
hidden_dim: int = 256
thermal_latent_dim: int = 128
num_spatial_tokens: int = 8
strategy_gumbel_tau: float = 1.0
min_grasp_score: float = 0.3
```

若 OpenPI 当前 frozen dataclass/序列化不方便嵌套配置，可以使用以 `thermal_` 开头的平铺字段，但不得硬编码在模型中。

### 7.2 `__init__`

在 `PI0Pytorch.__init__` 中，当 `config.thermal_enabled` 时实例化：

```python
self.thermal_encoder = IntegratedThermalEncoder(
    config=...,
    paligemma_width=paligemma_config.width,
    action_expert_width=action_expert_config.width,
)
```

checkpoint 加载必须允许 base π0.5 checkpoint 缺少 `thermal_encoder.*` 参数；只允许这些新 key 缺失，其他缺失或多余 key 仍报错。

### 7.3 `embed_prefix`

将签名改为可接收完整 observation，或在调用处额外传入 thermal fields。建议：

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, observation):
```

伪代码：

```python
image_embs = embed_all_images(images)
lang_embs = embed_language(lang_tokens)

embs = [*image_embs, lang_embs]
masks = [*image_masks, lang_masks]
att_masks = all_zeros_for_prefix

thermal_output = None
if self.thermal_enabled:
    thermal_output = self.thermal_encoder(
        observation,
        image_embeddings=concat(image_embs),
        language_embeddings=lang_embs,
    )
    embs.append(thermal_output.prefix_tokens)
    masks.append(thermal_output.prefix_mask)
    att_masks.extend([0] * (M + 1))

return cat(embs), cat(masks), att_masks, thermal_output
```

### 7.4 `forward`

保留原 flow-matching 路径：

```text
x_t = t * noise + (1-t) * action
u_t = noise - action
Action Expert predicts v_t
L_flow = MSE(v_t, u_t)
```

新增 loss：

```text
L_contact_temp = GaussianNLL(contact temperature)
L_contact_valid = BCE(contact validity)
L_field_temp = GaussianNLL(surface temperature)
L_field_safe = BCE(task-conditioned surface safety)
L_skill = CE(OTHER/PROBE/GRASP)
L_target = CE(probe/grasp target candidate)
```

总损失建议初值：

```text
L = 1.0 L_flow
  + 0.2 L_contact_temp
  + 0.1 L_contact_valid
  + 0.3 L_field_temp
  + 0.1 L_field_safe
  + 0.2 L_skill
  + 0.3 L_target
```

实现 `compute_loss_dict()` 返回各分项和 `total`。若现有 trainer 只能接收 `[B,H,A]` loss，保留兼容 `forward()`，把每个样本的 aux scalar 加到其 flow loss tensor 上；同时由新的训练 wrapper 调用 `compute_loss_dict()` 记录指标。

### 7.5 `sample_actions`

必须与训练使用完全相同的 prefix 构建路径。禁止训练时加入 thermal tokens、推理时遗漏。若原实现缓存 prefix K/V，则 thermal history 或 surface candidates 变化时必须重新计算缓存。

返回值仍为 `[B,H,A]`，可选通过 `return_diagnostics=True` 额外返回：

```python
{
  "skill_probability": ...,
  "selected_surface_index": ...,
  "field_temp_mean_c": ...,
  "field_temp_std_c": ...,
}
```

## 8. 动作与主动策略的关系

`StrategyOutput` 不直接向机器人发送 `PROBE(point)` 命令。其 `active_goal_token` 条件化 π0.5 Action Expert，由 Action Expert 输出到达、接触、保持、回撤或抓取所需的连续 7D chunk：

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper_position]
```

因此训练数据必须让动作轨迹与策略标签一致：

- `strategy=PROBE` 时，action chunk 朝选定点移动并进行轻触/保持；
- probe 完成后的 chunk 先回撤或转移；
- `strategy=GRASP` 时，action chunk 对齐选定抓点并闭合；
- 接触力仍由底层阻抗与 force guard 限制，π0.5 不直接输出力。

策略 head 在论文中既是可解释决策输出，也是通过 goal token 影响动作的主干组成部分。

## 9. 训练阶段

### Stage 0：原模型回归测试

- `thermal_enabled=False`；
- 加载 π0.5 base checkpoint；
- 固定输入和 noise，修改前后 flow loss/action 输出数值一致；
- 这是后续合并的硬门槛。

### Stage 1：预训练热觉与空间记忆

冻结整个 π0.5，仅训练：

```text
ThermalContactEncoder
SurfacePointEncoder
ThermalSpatialMemory
safe/temp heads
```

使用随机 K 个历史观测预测未观测候选点温度。batch 必须包含 K=0、1、2、4、8。

### Stage 2：预训练 ActiveStrategyModule

冻结 π0.5 与 Stage 1 模块，使用 privileged oracle 产生策略类型和下一目标点标签。训练 strategy/target loss。

### Stage 3：接入 π0.5 Action Expert

训练：

- 所有新增热觉模块；
- thermal-to-prefix projection；
- goal-to-prefix projection；
- Action Expert LoRA 或 Action Expert 全量参数；

冻结：

- SigLIP；
- PaliGemma VLM 主干，首轮保持冻结。

同时优化 flow loss 和辅助 losses。

### Stage 4：联合微调

只在 Stage 3 闭环成功后：

- 对 PaliGemma 后若干层启用 LoRA；
- 新模块学习率约 `1e-4`；
- Action Expert/LoRA 学习率约 `5e-5`；
- PaliGemma LoRA 学习率约 `1e-5`；
- 保留各辅助 loss，避免策略退化为纯行为克隆捷径。

## 10. 需要修改的 OpenPI 文件

Claude 必须先记录 OpenPI commit hash，再根据实际版本调整，但目标文件至少包括：

```text
src/openpi/models/model.py
  - Observation 增加可选 thermal/surface 字段
  - from_dict/tree mapping/预处理保持这些字段

src/openpi/models/pi0_config.py
  - 增加 thermal feature flag 与维度配置
  - 测试直接构造 Observation（PyTorch 路径不需要 JAX 输入规格）

src/openpi/models_pytorch/preprocessing_pytorch.py
  - resize 图像时不丢弃 thermal/surface 字段
  - dtype/device/mask 标准化

src/openpi/models_pytorch/pi0_pytorch.py
  - 初始化 IntegratedThermalEncoder
  - embed_prefix 追加 tokens
  - forward/sample_actions 共用 prefix 路径
  - compute_loss_dict

src/openpi/training/config.py
  - 新增 pi05_thermal_libero / pi05_thermal_genesis 配置

src/openpi/transforms.py 或项目数据 transform
  - dataset key → Observation 字段

src/openpi/models_pytorch/thermal/*
  - 全部新增神经模块
```

## 11. 测试要求

### 形状与 mask

- `B=1/3`、`K=0/1/8`、`N=16/64`；
- 所有输出无 NaN；
- padding contact 和 padding surface 不影响有效输出；
- empty memory 可前向与反向。

### 泛化结构

- 打乱 K 个接触顺序，spatial latents 基本不变；
- 打乱 N 个候选点，field/strategy logits 按相同置换变化；
- 模型 forward 中不存在 link ID；
- K、N 与杯子 link 数量完全解耦。

### π0.5 集成

- `thermal_enabled=False` 与原 π0.5 数值回归一致；
- `thermal_enabled=True` 时 prefix 长度恰好增加 `M+1`；
- thermal token 参数能从 flow loss 获得非零梯度；
- strategy goal projection 能从 flow loss 获得非零梯度；
- Action Expert 的 suffix 长度、action horizon、adaRMS condition 不变；
- base checkpoint 仅缺失新增 thermal keys；
- training forward 与 `sample_actions` 使用相同 thermal prefix。

### 因果性

- 删除未来 probe 窗口后，训练样本仍可构建；
- 在同一 pre-probe 状态中改变未来温度记录不能改变模型输入；
- rollout 只有在接触窗口结束后更新 memory。

## 12. Claude 的实施任务拆分

不要让 Claude 一次完成所有模块。按以下任务提交，每个任务完成后测试再继续。

### Task A：OpenPI 扩展接口与回归基线

```text
阅读 INTEGRATED_PI05_ARCHITECTURE.md 和当前 OpenPI 源码，记录 commit hash。
只实现 ThermalPi0Config、Observation 可选字段、preprocessing 字段保留、
fake observation 以及 thermal_enabled=False 的回归测试。
不要实现神经网络。保证现有 π0.5 单元测试全部通过。
```

### Task B：热接触编码器

```text
实现 thermal/contact_encoder.py、输出 dataclass 和独立损失。
支持 [B,K,L,C]、K=0/padding mask、causal convolution 和反向传播。
添加合成温度序列 overfit 测试。不要修改 PI0Pytorch.forward。
```

### Task C：空间记忆

```text
实现 point_encoder、contact token builder、ThermalSpatialMemory 和 Perceiver resampler。
测试 contact permutation invariance、surface permutation equivariance、empty memory、
masked Gaussian NLL。不要接入 π0.5。
```

### Task D：主动策略与可微 goal token

```text
实现 ActiveStrategyModule、action masks、soft/gumbel/argmax 三种选点模式。
实现 strategy/target loss，验证 goal_feature 能对目标点选择产生梯度。
```

### Task E：接入 π0.5 prefix

```text
实现 IntegratedThermalEncoder，并修改 PI0Pytorch.embed_prefix，使 M个spatial tokens和
1个active goal token追加到image/language prefix。修改forward和sample_actions共用该路径。
验证prefix长度、attention mask、checkpoint加载和从flow loss到thermal/strategy模块的梯度。
thermal_enabled=False 必须与原模型数值一致。
```

### Task F：联合 loss 与训练配置

```text
实现 compute_loss_dict、训练指标、分阶段冻结规则和 pi05_thermal_genesis 配置。
用 dummy π0.5 variant 跑一个 batch 的 forward/backward/optimizer step/checkpoint reload。
```

### Task G：Genesis 数据与闭环推理

```text
实现每个控制时刻的因果 contact history 构建、候选点采样、监督标签映射，
接入 sample_actions diagnostics。在一个短 episode 中验证：主动选择接触点、
接触完成后写入memory、下一次动作随热觉观测变化、最终选择抓取点。
```

## 13. 交给 Claude 的总提示词

```text
你正在修改 Physical-Intelligence/openpi 的 PyTorch π0.5，实现 Thermal-π0.5。
完整规格见 INTEGRATED_PI05_ARCHITECTURE.md。

先检查当前仓库 commit 和实际接口；源码与规格的文件行号不一致时，以语义插入点为准，
但不得改变总体数据流：raw contact history → thermal encoder → spatial memory →
active strategy → M+1 continuous prefix tokens → PaliGemma prefix → Action Expert。

禁止实现固定四区域槽，禁止把 link_id/link_name/真实温度放入 Observation，
禁止把主动策略做成 π0.5 外部规划器。策略必须产生 active_goal_token，并通过
π0.5 prefix 条件化 Action Expert。最终执行输出仍为 7D action chunk。

当前只完成我指定的 Task，不提前实现下一 Task。保持 thermal_enabled=False 的原模型
行为和 checkpoint 兼容性。每个新模块必须支持 batch、padding mask、K=0、CPU测试和反向传播。
完成后运行相关测试，报告修改文件、张量形状、参数量、测试结果以及未完成项。
```
