#!/bin/bash
# 14 Oct, v10: the v9 30k checkpoint fine-tuned 10k more steps on the v10 dataset (v9's push demos regenerated: the
# v9 policy copied a descent that landed the gripper on the cube), then the 10k checkpoint on every task (20 cube
# poses each). Waits for the v10 dataset and for the v9 30k tests to finish, then stops the v9 run before its 20k
# tests to free the GPU. Log: outputs/vla/v10/train_run.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
TASKS="push_AB push_BC push_CB push_BA push_AC push_CA ccw cw AB BC AC CA BA CB AB_ccw BC_cw AC_ccw BA_ccw CB_cw"
say "waiting for the v10 dataset and the v9 30k tests"
until grep -q "wrote \|FAILED" outputs/vla/v10/data.log 2>/dev/null; do sleep 30; done
grep -q FAILED outputs/vla/v10/data.log && { say "no dataset"; exit 1; }
# Patterns anchored at the start of the command line: an unanchored one also matched the shell that launched this
# script (its command line held this whole file), and the wait never ended.
until grep -q "030000 CB_cw" outputs/vla/v9/train_run.log; do sleep 30; done
pkill -f "^/bin/bash scripts/vla/run_v9_train.sh"
sleep 2
pkill -f "^/home/orion/.venvs/lerobot/bin/python -m src.eval_vla --policy outputs/vla/v9/"
sleep 10
say "training"
scripts/vla/train_smolvla.sh data/lerobot/cube_sim_v10 outputs/vla/v10/smolvla_cube_v10 10000 \
    outputs/vla/v9/smolvla_cube_v9/checkpoints/030000/pretrained_model > outputs/vla/v10/train.log 2>&1
say "training over"
mkdir -p outputs/vla/v10/test
for t in $TASKS; do
    demos=data/vla_demos_v8/$t
    [ -d data/vla_demos_v10/$t ] && demos=data/vla_demos_v10/$t
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy outputs/vla/v10/smolvla_cube_v10/checkpoints/010000/pretrained_model \
        --task $t --episodes 20 --seed 2026 --device cuda --demos $demos --video 2 \
        --video-dir outputs/vla/v10/test/videos_010000 > outputs/vla/v10/test/010000_$t.txt 2>&1
    say "010000 $t: $(grep '^policy' outputs/vla/v10/test/010000_$t.txt)"
done
say "done"
