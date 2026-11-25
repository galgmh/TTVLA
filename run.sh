#!/usr/bin/env bash
set -e   # 任一命令失败即退出

#python -m sim.generate \
#  --task sim/configs/tasks/lift_with_cup_wall.yaml \
#  --split train --start 0 --num_per_ins 10 --out datasets/

#python -m sim.generate \
#  --task sim/configs/tasks/lift_with_cup_rim.yaml \
#  --split train --start 0 --num_per_ins 10 --out datasets/


python -m sim.generate   --task sim/configs/tasks/thermal_lift_hot_cup.yaml   --split train   --start 0   --num_per_ins 10   --out /home/ca/TTVLA/datasets/
python -m sim.generate   --task sim/configs/tasks/thermal_lift_cool_cup.yaml   --split train   --start 0   --num_per_ins 10   --out /home/ca/TTVLA/datasets/
python -m sim.generate   --task sim/configs/tasks/thermal_lift_hot_handle_cup.yaml   --split train   --start 0   --num_per_ins 10   --out /home/ca/TTVLA/datasets/