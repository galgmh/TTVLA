# Online Thermal-π0.5 实现结构

## 外部输入

每个控制时刻只输入两路RGB、指令、机器人状态和当前热触觉滑动窗口：

```text
thermal_window             [B,L,C]
thermal_window_time_mask   [B,L]
sensor_xyz_base            [B,3]
sensor_normal_base         [B,3]
```

模型不要求接触编号或接触次数。`K=memory_size`仅是内部记忆容量。

## 在线路径

```text
当前L帧 -> ThermalContactEncoder
         -> latent / temperature mean / logvar / valid probability
         -> ContactTokenBuilder(+ sensor xyz/normal)
         -> current contact token
                    |
                    v
        ThermalMemoryBank事件状态机
        - 滞回判定一次持续接触
        - 接触中：临时token参与当前推理
        - 接触结束：只写入一次
        - 最多K个完成token，满时FIFO
                    |
                    v
        ThermalSpatialMemory [B,M,D]
                    |
                    v
        gated ThermalPrefixAdapter [B,M,W]
                    |
                    v
[image tokens | language tokens | thermal tokens]
                    |
                    v
              π0.5 Action Expert -> [B,H,32]
```

训练前向默认不持久化MemoryBank，防止随机batch和不同episode串状态。部署时
`sample_actions()`启用在线更新，并在每个新episode开始调用
`reset_thermal_memory()`。

`SpatialFieldDecoder`只在训练/评测时读取thermal spatial latent和查询坐标，
不把查询点送入prefix。`GlobalSkillAuxHead`只提供四分类辅助损失和诊断，
不控制环境、不输出目标点。
