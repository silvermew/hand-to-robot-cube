#!/bin/bash
# 15 Oct, v11 data: as v10, with the push demos regenerated (sim_push.py: a 2.0 s approach and a 0.6 s pause, so the
# descent starts in the policy's second 50-step chunk, as the grasps' does). 100 demos for each of the 6 push tasks,
# each task converted to its own LeRobot dataset in parallel, then all merged with the v8 dataset into one.
# Log: outputs/vla/v11/data.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
mkdir -p outputs/vla/v11
TASKS="push_AB push_BC push_CB push_BA push_AC push_CA"
seed=200
for t in $TASKS; do
    seed=$((seed + 1))
    nice -n 10 ~/.venvs/robot-sim/bin/python -m src.make_sim_demos --data data/raw --task $t --n 100 --seed $seed \
        --out data/vla_demos_v11/$t > outputs/vla/v11/demos_$t.log 2>&1 &
done
wait
for t in $TASKS; do say "$t: $(grep '^kept' outputs/vla/v11/demos_$t.log)"; done
for t in $TASKS; do
    nice -n 10 ~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v11/$t --root data/lerobot/v11_parts/$t \
        --clock > outputs/vla/v11/to_lerobot_$t.log 2>&1 &
done
wait
for t in $TASKS; do
    grep -q '^wrote' outputs/vla/v11/to_lerobot_$t.log || { say "conversion of $t FAILED"; exit 1; }
done
~/.venvs/lerobot/bin/python -m src.merge_datasets data/lerobot/cube_sim_v11 data/lerobot/cube_rotate_sim_v8 \
    $(for t in $TASKS; do echo data/lerobot/v11_parts/$t; done) > outputs/vla/v11/merge.log 2>&1 || { say "merge FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v11/merge.log)"
