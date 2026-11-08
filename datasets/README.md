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
- 一份文件 = 一集 = 一次单区域受控触碰，字段契约见
  `data/schema/thermal_contact_v1.py`。
- 校验：`data.validation.validate_thermal_contact.validate_directory`。
