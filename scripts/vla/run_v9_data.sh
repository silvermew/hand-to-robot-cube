#!/bin/bash
# 13 Oct, v9 data: 100 demos for each of the 6 push tasks of make_sim_demos.TASKS (the gripper closed, the cube on its
# cross turned within 20 deg of my cube), each task converted to its own LeRobot dataset in parallel, then all merged
# with the v8 dataset (the 13 pick-and-place tasks) into one. Log: outputs/vla/v9/data.log
# sim_push.py then had no pause, and the fingers came down 2.6 cm behind the face (4.6 cm from v10 on).
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
mkdir -p outputs/vla/v9
TASKS="push_AB push_BC push_CB push_BA push_AC push_CA"
seed=200
for t in $TASKS; do
    seed=$((seed + 1))
    nice -n 10 ~/.venvs/robot-sim/bin/python -m src.make_sim_demos --data data/raw --task $t --n 100 --seed $seed \
        --out data/vla_demos_v9/$t > outputs/vla/v9/demos_$t.log 2>&1 &
done
wait
for t in $TASKS; do say "$t: $(grep '^kept' outputs/vla/v9/demos_$t.log)"; done
for t in $TASKS; do
    nice -n 10 ~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v9/$t --root data/lerobot/v9_parts/$t \
        --clock > outputs/vla/v9/to_lerobot_$t.log 2>&1 &
done
wait
for t in $TASKS; do
    grep -q '^wrote' outputs/vla/v9/to_lerobot_$t.log || { say "conversion of $t FAILED"; exit 1; }
done
~/.venvs/lerobot/bin/python -m src.merge_datasets data/lerobot/cube_sim_v9 data/lerobot/cube_rotate_sim_v8 \
    $(for t in $TASKS; do echo data/lerobot/v9_parts/$t; done) > outputs/vla/v9/merge.log 2>&1 || { say "merge FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v9/merge.log)"
