# 热学引导主动触碰 VLA：在线最终方案

## 设计原则

VLA外部输入只有两路RGB、语言指令、机器人状态和当前热触觉传感器最近L帧。
系统不知道也不要求输入“第几次接触”。历史接触由每个推理episode内部的
`ThermalMemoryBank`自行维护。

不增加候选表面点模型。下一触点由π0.5根据视觉、语言、当前触觉和内部热记忆，
通过连续动作隐式产生。

## 模型路径

```text
thermal_window [B,L,C]
  -> ThermalContactEncoder
     -> latent / temperature mean / logvar / validity
  -> ContactTokenBuilder(+ 当前传感器基座系位置和法向)
  -> current contact token [B,D]
  -> ThermalMemoryBank(max K completed tokens + current temporary token)
  -> ThermalSpatialMemory [B,M,D]
  -> gated ThermalPrefixAdapter [B,M,paligemma_width]
  -> [RGB tokens | instruction tokens | thermal tokens]
  -> PaliGemma prefix KV
  -> π0.5 Action Expert
  -> actions [B,H,32]
```

`K`只是内部长期记忆容量；`M`是固定prefix token数。

## 运行时数据契约

```text
thermal_window             [B,L,C] float
thermal_window_time_mask   [B,L]   bool
sensor_xyz_base            [B,3]   float
sensor_normal_base         [B,3]   float
```

传感器位置和法向由机器人状态/FK获得，不是环境给出的候选点。新episode开始必须
调用`reset_thermal_memory()`。

## 接触事件与记忆写入

有效性概率由ContactEncoder预测。状态机使用`on > off`双阈值和连续帧计数抑制
抖动：

```text
NO_CONTACT --连续N_on帧超过on阈值--> CONTACT_ACTIVE
CONTACT_ACTIVE --连续N_off帧低于off阈值--> CONTACT_ENDED
```

接触进行中，token以指数滑动平均更新并作为临时token参与当前prefix；接触结束时
才向长期MemoryBank写入一次。容量达到K后使用FIFO淘汰最旧token。

## 训练监督

```text
contact_temp_c          [B,1]
contact_valid           [B,1]
field_query_xyz_base    [B,Q,3]
field_query_temp_c      [B,Q,1]
field_query_mask        [B,Q]
strategy_type           [B]
```

查询坐标只进入训练/评测FieldDecoder，不能进入运行时Observation或控制prefix。

损失：

```text
L = L_flow
  + λct L_contact_temperature_NLL
  + λcv L_contact_valid_BCE
  + λft L_field_temperature_NLL
  + λsk L_global_skill_CE
```

## 训练状态规则

- 普通随机batch默认不更新在线MemoryBank，避免不同episode串状态。
- 单触点Stage 1直接训练当前窗口编码器及温度/不确定性/有效性。
- 联合VLA训练若要学习跨接触记忆，应按episode顺序展开短片段，并在片段开始重置
  或因果重建MemoryBank。
- MemoryBank保存detach后的历史token，不跨控制周期保留无限计算图。
- π0.5内部动作维度固定为32；机器人动作只在数据边界pad/crop。

## 分阶段训练

1. 当前窗口ContactEncoder、ContactTokenBuilder、SpatialMemory和FieldDecoder。
2. 冻结主干，用flow loss训练gated PrefixAdapter。
3. 联合训练热学模块、PrefixAdapter和Action Expert。
4. 加入PROBE/GRASP/RETRACT/ABORT全局辅助分类。
5. 仅在语义对齐不足时解冻PaliGemma最后1～2层。

## 验收条件

- 外部Observation中不存在接触次数、历史K窗口、候选点或目标点索引。
- 同一次持续接触只写入一个长期token。
- 当前接触在结束前也能通过临时token影响动作。
- 新episode能可靠清空记忆；batch size/device/dtype变化会安全重建状态。
- 训练前向不隐式污染在线记忆；部署推理才更新记忆。
- prefix只增加M个thermal token，Action Expert仍输出官方32维动作。
