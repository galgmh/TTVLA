# Thermal Model Architecture

`thermal_model/` 是独立的热触觉模型包，目录组织参考同级的
`pi05_model/`。当前版本**尚未接入** `pi05_model`、`finetune_new.py` 或
`server_policy.py`，可以单独开发、训练和检查。

## 目录

```text
thermal_model/
  __init__.py                 包级公开接口
  config.py                   ThermalPi0Config
  observation.py              最终在线模型输出容器
  preprocessing.py            原始 episode -> 12 通道热觉窗口

  model.py                    最终接入 π0.5 的热觉模型
  contact_encoder.py          因果热触觉序列编码器
  memory_bank.py              接触事件状态机和长期记忆
  spatial_memory.py           ContactTokenBuilder + Spatial Memory
  prefix_adapter.py           热觉 token -> π0.5 宽度

  data/
    loader.py                 热编码器和空间预训练 Dataset

  encoder_pretraining/
    config.py                 直接模型配置与 build_encoder()
    losses.py                 单触点温度 NLL 和有效性 BCE
    prepare_dataset.py
    train.py
    evaluate.py

  spatial_pretraining/
    field_decoder.py          训练期空间热场辅助解码器
    strategy_aux.py           训练期全局技能辅助头
    config.py                 空间表示模型配置与编码器权重加载
    losses.py                 查询点温度 Gaussian NLL
    prepare_dataset.py        v3 -> thermal-spatial-v2
    train.py                  训练 builder/memory/field decoder
    evaluate.py

  pi05_training/
    losses.py                 π0.5 接入阶段的训练侧辅助损失
    labels.py                 仿真标签 -> ThermalSupervision
    transforms.py             数据批次 -> 热觉模块输入
    step.py                   训练 wrapper
    finetune.py               分阶段冻结和学习率分组
    rollout.py                闭环协议

  configs/
    encoder_pretraining.yaml
    spatial_pretraining.yaml
    training/pi05_thermal_stage2,3,5.yaml
```

## 与 pi05_model 的对应关系

```text
pi05_model/config.py          <-> thermal_model/config.py
pi05_model/observation.py     <-> thermal_model/observation.py
pi05_model/preprocessing.py   <-> thermal_model/preprocessing.py
pi05_model/model.py           <-> thermal_model/model.py
```

热觉分支的最终在线路径保留接触编码、记忆、空间压缩和 prefix adapter。
热场 decoder 与技能辅助头只存在于 `spatial_pretraining/`，不进入最终模型。

## 外部运行时输入

热觉模型不接受区域编号、link 名称、候选点或接触次数。单步输入约定为：

```text
thermal_window             [B, L, 12] float
thermal_window_time_mask   [B, L]     bool
state                      [B, 8]     float
```

`state` 为七个机械臂关节角和夹爪宽度：

```text
[q1, q2, q3, q4, q5, q6, q7, gripper_width]
```

12 通道顺序：

```text
left_temperature
right_temperature
left_delta_temperature
right_delta_temperature
left_temperature_rate
right_temperature_rate
left_force_x
left_force_y
left_force_z
right_force_x
right_force_y
right_force_z
```

模型不再单独接收传感器 xyz/normal。`state` 通过固定的机器人正向运动学得到：

```text
state [B,8]
  -> arm FK
  -> hand/TCP pose
  -> left/right finger transforms
  -> left/right pad positions [B,2,3]
  -> left/right pad normals   [B,2,3]
```

训练、Genesis 部署和真实机器人必须注入同一个 `FingertipGeometryProvider`。
当前 `sim/model_env.py` 提供 Genesis FK 对应实现；真实机器人应使用相同 URDF
和指垫几何实现该接口。

## 当前窗口编码

```text
thermal_window [B,L,12]
  -> ThermalContactEncoder
     - 因果膨胀卷积
     - masked mean pooling
     - last-valid pooling
  -> contact latent [B,D]
  -> temperature mean / logvar [B,1]
  -> contact-valid logit [B,1]
```

因果卷积保证时刻 `t` 只使用 `t` 及以前的帧，`time_mask=False` 的 padding
不会参与卷积或 pooling。

## 当前接触 token

```text
contact latent
  + Fourier(left/right pad positions)
  + left/right pad normals
  -> ContactTokenBuilder
  -> current contact token [B,D]
```

两个指垫位置使用 `12 * fourier_bands` 维 Fourier 特征，两个法向共 6 维。
因此 ContactTokenBuilder 输入维度为：

```text
thermal_latent_dim + 12 * fourier_bands + 6
```

位置和法向用于表达“热觉读数来自左右哪个指垫、在何处”，不包含热区类别标签。
温度均值、方差和有效性仍由编码器独立输出，分别用于辅助损失、接触事件状态机
和诊断。

## 接触事件和长期记忆

`ThermalMemoryBank` 维护每个 batch 行独立的接触状态：

```text
NO_CONTACT
  --连续 N_on 帧 valid_probability > on_threshold-->
CONTACT_ACTIVE
  --连续 N_off 帧 valid_probability < off_threshold-->
CONTACT_ENDED
```

行为：

- 接触进行中，使用指数滑动平均维护 temporary token。
- temporary token 在当前控制周期立即参与推理。
- 一次持续接触结束后，才向长期记忆写入一个 token。
- 最多保存 `K=memory_size` 个完成接触。
- 容量满后 FIFO 删除最旧 token。
- batch size、device 或 dtype 改变时安全重建状态。
- 训练前向不持久化状态；部署推理才更新。

## 空间记忆

```text
K 个完成 contact tokens + 1 个 temporary token
  -> masked Set Transformer
  -> M 个 learned queries
  -> spatial latent [B,M,D]
```

没有历史接触时仍提供一个可学习的 `NO_CONTACT` token，避免全 mask 导致 NaN。

## Prefix Adapter

```text
spatial latent [B,M,D]
  -> LayerNorm + MLP
  -> thermal type embedding
  -> sigmoid gate
  -> thermal prefix [B,M,W]
```

`W` 是 PaliGemma 隐藏宽度。默认 gate 初始化为 `0.01`，未来接入时应从接近
原模型行为的输出开始训练。

## 训练边界

空间预训练在 `spatial_pretraining/` 中完成：

```text
冻结接触编码器
  -> ContactTokenBuilder
  -> ThermalSpatialMemory
  -> SpatialFieldDecoder
  -> field temperature Gaussian NLL
```

训练完成后只把 `contact_builder` 和 `spatial_memory` 的在线表示接入最终模型。
`SpatialFieldDecoder` 和 `GlobalSkillAuxHead` 保留在预训练包中，不进入
π0.5 在线前向。π0.5 接入后只使用最终模型输出的 prefix，并可将接触温度和
有效性作为训练侧辅助项。

## 数据流

```text
sim ThermalContact HDF5
  -> thermal_model/preprocessing.py
  -> 12 通道窗口
  -> 51 帧组 / 每组 25 个长度 26 的窗口
  -> thermal_model/data/loader.py
  -> thermal_model/encoder_pretraining
  -> contact / memory / spatial / prefix
```

当前预训练数据由 `encoder_pretraining/prepare_dataset.py` 生成
`thermal-encoder-v3`。每个 episode 固定产生 4 个接触组和 4 个非接触组，
所有窗口直接来自源数据，`mask` 全为 True。非接触样本的温度标签为 `NaN`，
只监督接触有效性。

空间预训练由 `spatial_pretraining/prepare_dataset.py` 在 v3 窗口末端补充
左右指垫位置/法向、接触区域和四个热区查询点，生成 `thermal-spatial-v2`。训练加载
已训练的接触编码器，默认冻结编码器，只训练 ContactTokenBuilder、
ThermalSpatialMemory 和训练期辅助 SpatialFieldDecoder。
数据按 episode 和 `end_sensor_index` 排序去重，`episode_offsets` 给出每条
完整序列。训练时逐 episode 重置 MemoryBank，再按时间顺序更新和读取记忆。
`thermal-spatial-v2` 还保存当前接触区域；field loss 只选择该区域对应的
query，并将 current token 显式加入 SpatialMemory。历史 token 由 MemoryBank
维护，当前 token 不等待接触结束即可参与当前推理。

接触温度和空间查询温度都输出有界 `logvar`：

```text
logvar = logvar_min + (logvar_max - logvar_min) * sigmoid(raw_logvar)
logvar ∈ [-4, 3]
sigma  ∈ [0.135, 4.48] °C
```

Gaussian NLL 直接使用该有界值，不再在 loss 内依赖零梯度 `clamp`。

## 当前集成边界

当前不会发生：

```text
pi05_model -> thermal_model
finetune_new.py -> thermal_model
server_policy.py -> thermal_model
```

未来接入点应保持为：

```text
pi05_model
  prefix = [image tokens | language tokens | thermal tokens]

Action Expert
  -> action output [B,H,32]
```

热觉分支只增加 prefix token 和辅助 loss，不改变 Action Expert 输出维度。

## 独立运行

```bash
cd /home/ca/TTVLA/models

python -m thermal_model.encoder_pretraining.prepare_dataset
python -m thermal_model.encoder_pretraining.train
python -m thermal_model.encoder_pretraining.evaluate
```

这里的训练和评测只读取 `thermal-encoder-v3` HDF5，没有 `--dataset`
参数。数据集 YAML 只由 `prepare_dataset.py` 在生成派生数据时使用。

普通 π0.5 训练和推理不导入本包：

```bash
python finetune_new.py --config_name your_config
python server_policy.py --config_name your_config --checkpoint_dir ...
```
