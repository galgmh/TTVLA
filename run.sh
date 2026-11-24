#!/usr/bin/env bash
set -e   # 任一命令失败即退出

python -m sim.generate \
  --task sim/configs/tasks/lift_with_cup_wall.yaml \
  --scene sim/configs/scene/cup_on_table.yaml \
  --split train --start 0 --num_per_ins 10 --out datasets/

python -m sim.generate \
  --task sim/configs/tasks/lift_with_cup_rim.yaml \
  --scene sim/configs/scene/cup_on_table.yaml \
  --split train --start 0 --num_per_ins 10 --out datasets/