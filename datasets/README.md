# Dataset conversion

Convert raw thermal HDF5 episodes into the LeRobot v3 layout read by
`models/data/data_loader.py`.

Install the fixed conversion dependency in the active environment:

```bash
python -m pip install "lerobot==0.4.3"
```

LeRobot 0.4.3 has no `[dataset]` extra; the package itself installs the dataset
dependencies used by this converter.

Conversion accepts one or more source directories, an output directory, and a
repo id:

```bash
python datasets/convert_hdf5_to_lerobot.py \
  --source datasets/lift_with_cup_rim/train datasets/lift_with_cup_wall/train \
  --root datasets/lerobot/lift \
  --repo-id thermtacvla/lift \
  --fps 20 \
  --overwrite
```

Every source directory is searched recursively. If source directories overlap,
the same HDF5 file is converted only once.

The converter recursively reads every HDF5 file under the source. Each file is
treated as a valid episode; success filtering belongs to the generation stage,
not to format conversion.

Example source layout:

```text
datasets/
  lift_with_cup_rim/train/*.hdf5
  lift_with_cup_wall/train/*.hdf5
  lift_with_handle/train/*.hdf5
```

and writes:

```text
datasets/lerobot/thermal_manipulation/
  data/chunk-000/file-000.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  videos/observation.images.up/chunk-000/file-000.mp4
  videos/observation.images.side/chunk-000/file-000.mp4
```

Explicit paths:

```bash
python datasets/convert_hdf5_to_lerobot.py \
  --source /path/to/raw_hdf5 \
  --root /path/to/lerobot_dataset \
  --repo-id my-org/my-dataset \
  --fps 20 \
  --overwrite
```

The converter requires exactly LeRobot 0.4.3 and uses
`lerobot.datasets.lerobot_dataset.LeRobotDataset`.

For `data_loader.py`, set:

```python
DataConfig(
    repo_id="thermtacvla/thermal_manipulation",
    local_root="/path/to/lerobot_dataset",
    video_backend="pyav",  # local fallback; use the working backend in the cloud
)
```
