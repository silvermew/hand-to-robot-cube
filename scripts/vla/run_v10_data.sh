#!/bin/bash
# 14 Oct, v10 data: as v9, with the push demos regenerated (sim_push.py: a 0.4 s pause above the start point, and
# the fingers come down 4.6 cm behind the face instead of 2.6). 100 demos for each of the 6 push tasks, each task
# converted to its own LeRobot dataset in parallel, then all merged with the v8 dataset (the 13 pick-and-place
# tasks) into one. Log: outputs/vla/v10/data.log
# sim_push.py then had PRE_ROLL 1.5 s and HOVER_S 0.4 s (now 2.0 and 0.6, the v11 values).
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
mkdir -p outputs/vla/v10
TASKS="push_AB push_BC push_CB push_BA push_AC push_CA"
seed=200
for t in $TASKS; do
    seed=$((seed + 1))
    nice -n 10 ~/.venvs/robot-sim/bin/python -m src.make_sim_demos --data data/raw --task $t --n 100 --seed $seed \
        --out data/vla_demos_v10/$t > outputs/vla/v10/demos_$t.log 2>&1 &
done
wait
for t in $TASKS; do say "$t: $(grep '^kept' outputs/vla/v10/demos_$t.log)"; done
for t in $TASKS; do
    nice -n 10 ~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v10/$t --root data/lerobot/v10_parts/$t \
        --clock > outputs/vla/v10/to_lerobot_$t.log 2>&1 &
done
wait
for t in $TASKS; do
    grep -q '^wrote' outputs/vla/v10/to_lerobot_$t.log || { say "conversion of $t FAILED"; exit 1; }
done
~/.venvs/lerobot/bin/python -m src.merge_datasets data/lerobot/cube_sim_v10 data/lerobot/cube_rotate_sim_v8 \
    $(for t in $TASKS; do echo data/lerobot/v10_parts/$t; done) > outputs/vla/v10/merge.log 2>&1 || { say "merge FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v10/merge.log)"
