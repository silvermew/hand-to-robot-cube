#!/bin/bash
# 13 Oct, v9: v7-10k fine-tuned 30k steps on the 19 tasks (v8's 13 pick-and-place tasks and the 6 pushes), then the
# 30k and 20k checkpoints on every task (20 cube poses each). Replaces v8, whose data it contains. Waits for the
# v9 dataset and until no detector training has run for 10 minutes (the 8 GB card cannot hold both).
# Log: outputs/vla/v9/train_run.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
say "waiting for the v9 dataset and for the GPU to be free of detector training"
until grep -q "wrote \|FAILED" outputs/vla/v9/data.log 2>/dev/null; do sleep 60; done
grep -q FAILED outputs/vla/v9/data.log && { say "no dataset"; exit 1; }
free=0
while [ $free -lt 10 ]; do
    if pgrep -f "boss[.]cli[.]train_detector" >/dev/null; then free=0; else free=$((free + 1)); fi
    sleep 60
done
say "training"
scripts/vla/train_smolvla.sh data/lerobot/cube_sim_v9 outputs/vla/v9/smolvla_cube_v9 30000 \
    outputs/vla/v7/smolvla_cube_v7/checkpoints/010000/pretrained_model > outputs/vla/v9/train.log 2>&1
say "training over"
mkdir -p outputs/vla/v9/test
for step in 030000 020000; do
    for t in push_AB push_BC push_CB push_BA push_AC push_CA ccw cw AB BC AC CA BA CB AB_ccw BC_cw AC_ccw BA_ccw CB_cw; do
        demos=data/vla_demos_v8/$t
        [ -d data/vla_demos_v9/$t ] && demos=data/vla_demos_v9/$t
        ~/.venvs/lerobot/bin/python -m src.eval_vla --policy outputs/vla/v9/smolvla_cube_v9/checkpoints/$step/pretrained_model \
            --task $t --episodes 20 --seed 2026 --device cuda --demos $demos --video 2 \
            --video-dir outputs/vla/v9/test/videos_$step > outputs/vla/v9/test/${step}_$t.txt 2>&1
        say "$step $t: $(grep '^policy' outputs/vla/v9/test/${step}_$t.txt)"
    done
done
say "done"
