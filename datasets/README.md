# datasets/ — 数据输出目录

本目录**不入库**（见仓库根 `.gitignore`），只放生成物。

约定：

```
datasets/thermal_contact_v1/
  pilot/           每集一个 HDF5（<episode_id>.hdf5）
  train/ validation/ test_id/ test_finger_history/
  test_sensor_shift/ test_mixed_contact/
  runs/            训练 checkpoint 与历史
```

- 划分由 `scripts/generate_contact_manifests.py` 在 `data/manifests/` 里
  冻结；`<split>` 子目录里的文件必须与清单一一对应。
- 一份文件 = 一集 = 按某一全排列顺序完成的四次受控触碰，字段契约见
  `data/schema/thermal_contact_v1.py`。
- 校验：`data.validation.validate_thermal_contact.validate_directory`。

## 单 episode HDF5 格式

```text
attrs
  episode_id                  # 全局唯一 episode 编号
  random_seed                 # 复现该 episode 的随机种子
  split                       # train/validation/test 等划分
  genesis_version             # Genesis 版本
  coordinate_convention       # 坐标系与四元数约定
  episode_metadata            # JSON：指令、接触顺序/时长/姿态等可审计参数

sensor/
  time                        # [Ts]，100 Hz

tactile/
  fingertip_temp              # [Ts,2]，左(0)/右(1)指尖温度
  contact_force_link          # [Ts,2,3]，指尖 link 坐标系三轴力
  contact_normal_force        # [Ts,2]，法向力
  contact_area                # [Ts,2]，接触面积
  contact_mask                # [Ts,2]，是否有物理接触
  contact_event_id            # [Ts]，0=无接触，1..N=接触事件
  sensor_position_base        # [Ts,2,3]，传感器在机器人 base 系的位置
  sensor_normal_base          # [Ts,2,3]，指垫法向在机器人 base 系的方向

control/
  time                        # [Tc]，20 Hz
  sensor_index                # [Tc]，不晚于当前时刻的最新传感器帧
  eef_target                  # [Tc,7]，[x,y,z,qw,qx,qy,qz]
  gripper_target              # [Tc]，m
  target_force                # [Tc]，N

robot/
  eef_pose                    # [Tc,7]，实测 TCP 位姿
  joint_position              # [Tc,9]，7 rad + 2 m
  joint_velocity              # [Tc,9]，7 rad/s + 2 m/s
  gripper_width               # [Tc]

camera/
  base_0_rgb                  # [Tc,H,W,3]
  wrist_0_rgb                 # [Tc,H,W,3]
  base_0_intrinsics           # [3,3]
  wrist_0_intrinsics          # [3,3]
  base_0_pose                 # [7]
  wrist_0_pose                # [Tc,7]
  hand_to_wrist_camera        # [4,4]

privileged/
  link_temp                   # [Ts,link_number]
  contact_link_mask           # [Ts,2,link_number]
  contact_valid               # [Ts,2]

contact_events/
  start_sensor_index          # [N]
  end_sensor_index_exclusive  # [N]
```

`contact_events` 中每条记录一次计划接触；小于 50 ms 的力触觉抖动由
记录器滞回合并，真正离开杯子后才结束事件。所有位姿四元数均使用 `wxyz`。

当前完整主集为 `5 温度情景 × 24 接触顺序 × 12 平衡时长方案 = 1440`
episode。指令固定为“触碰杯子并感受其温度分布”，左右指尖每集均从 25°C
开始。
