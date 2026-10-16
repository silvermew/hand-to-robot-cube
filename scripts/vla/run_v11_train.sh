#!/bin/bash
# 15 Oct, v11: the v10 10k checkpoint fine-tuned 10k more steps on the v11 dataset (push demos whose descent starts
# in the policy's second 50-step chunk, as the grasps' does), then the 10k checkpoint on every task (20 cube poses
# each). Waits for the v11 dataset only: start it when the GPU is free. Log: outputs/vla/v11/train_run.log
#   setsid nohup scripts/vla/run_v11_train.sh > outputs/vla/v11/train_run.log 2>&1 < /dev/null &
# Stop with: kill $(cat outputs/vla/v11/train.pid)
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
TASKS="push_AB push_BC push_CB push_BA push_AC push_CA ccw cw AB BC AC CA BA CB AB_ccw BC_cw AC_ccw BA_ccw CB_cw"
until grep -q "wrote \|FAILED" outputs/vla/v11/data.log 2>/dev/null; do sleep 30; done
grep -q FAILED outputs/vla/v11/data.log && { say "no dataset"; exit 1; }
say "training"
scripts/vla/train_smolvla.sh data/lerobot/cube_sim_v11 outputs/vla/v11/smolvla_cube_v11 10000 \
    outputs/vla/v10/smolvla_cube_v10/checkpoints/010000/pretrained_model > outputs/vla/v11/train.log 2>&1
say "training over"
mkdir -p outputs/vla/v11/test
for t in $TASKS; do
    demos=data/vla_demos_v8/$t
    [ -d data/vla_demos_v11/$t ] && demos=data/vla_demos_v11/$t
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy outputs/vla/v11/smolvla_cube_v11/checkpoints/010000/pretrained_model \
        --task $t --episodes 20 --seed 2026 --device cuda --demos $demos --video 2 \
        --video-dir outputs/vla/v11/test/videos_010000 > outputs/vla/v11/test/010000_$t.txt 2>&1
    say "010000 $t: $(grep '^policy' outputs/vla/v11/test/010000_$t.txt)"
done
say "done"
