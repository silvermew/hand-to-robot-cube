#!/bin/bash
# 13 Oct, v8 data: 100 demos for each of the 13 tasks of make_sim_demos.TASKS (the crosses in the scene, the cube
# starting on its cross, carries keeping their direction), v6's approach and random face near switches; then one
# LeRobot dataset with the clock. Log: outputs/vla/v8/data.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
TASKS="ccw cw AB BC AC CA BA CB AB_ccw BC_cw AC_ccw BA_ccw CB_cw"
seed=100
for t in $TASKS; do
    seed=$((seed + 1))
    nice -n 10 ~/.venvs/robot-sim/bin/python -m src.make_sim_demos --data data/raw --task $t --n 100 --seed $seed \
        --random-face 30 --approach-h 0.15 --hover 0.4 --out data/vla_demos_v8/$t > outputs/vla/v8/demos_$t.log 2>&1 &
done
wait
for t in $TASKS; do say "$t: $(grep '^kept' outputs/vla/v8/demos_$t.log)"; done
mkdir -p data/vla_demos_v8_all
for t in $TASKS; do cp -al data/vla_demos_v8/$t/. data/vla_demos_v8_all/; done
say "demos: $(ls data/vla_demos_v8_all | wc -l)"
nice -n 10 ~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v8_all --root data/lerobot/cube_rotate_sim_v8 \
    --clock > outputs/vla/v8/to_lerobot.log 2>&1 || { say "conversion FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v8/to_lerobot.log)"
