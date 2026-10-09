#!/bin/bash
# 8 Oct, v5: v4's step-12k policy fine-tuned 10k more steps on 900 clip_45 demos = v4's 600 + 300 for cube yaws
# within ~15 deg of a grasp-face switch, with the face drawn at random among the two equally good ones
# (make_sim_demos --random-face 30 --ambiguous-only). Then every checkpoint on the 20 selection poses (GPU),
# and the best one on the 50 test poses, twice. Log: outputs/vla/v5/run.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
for seed in 51 52 53 54; do
    ~/.venvs/robot-sim/bin/python -m src.make_sim_demos --data data/raw --n 75 --clip clip_45_rot+60_B --seed $seed \
        --random-face 30 --ambiguous-only --out data/vla_demos_v5_new > outputs/vla/v5/demos_$seed.log 2>&1 &
done
wait
say "new demos: $(ls data/vla_demos_v5_new | wc -l); $(grep -h '^kept' outputs/vla/v5/demos_*.log | paste -sd' ')"
mkdir -p data/vla_demos_v5
cp -al data/vla_demos_v4/. data/vla_demos_v5/ && cp -al data/vla_demos_v5_new/. data/vla_demos_v5/
say "demos: $(ls data/vla_demos_v5 | wc -l)"
~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v5 --root data/lerobot/cube_rotate_sim_v5 --clock \
    > outputs/vla/v5/to_lerobot.log 2>&1 || { say "conversion FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v5/to_lerobot.log)"
scripts/vla/train_smolvla.sh data/lerobot/cube_rotate_sim_v5 outputs/vla/v5/smolvla_cube_v5 10000 \
    outputs/vla/v4/smolvla_cube_v4/checkpoints/012000/pretrained_model > outputs/vla/v5/train.log 2>&1
say "training over"
mkdir -p outputs/vla/v5/eval
for ckpt in $(ls -d outputs/vla/v5/smolvla_cube_v5/checkpoints/0* | sort -r); do
    step=$(basename "$ckpt")
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy "$ckpt/pretrained_model" --episodes 20 --device cuda \
        --demos data/vla_demos_v5 > "outputs/vla/v5/eval/$step.txt" 2>&1
    say "$step: $(grep '^policy' outputs/vla/v5/eval/$step.txt)"
done
best=$(for f in outputs/vla/v5/eval/0*.txt; do echo "$(grep -o 'success *[0-9.]*' $f | awk '{print $2}') $(basename $f .txt)"; done | sort -n | tail -1 | awk '{print $2}')
say "best on the selection poses: $best; 50 test poses, two runs"
for run in 1 2; do
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy outputs/vla/v5/smolvla_cube_v5/checkpoints/$best/pretrained_model \
        --episodes 50 --seed 2026 --device cuda --demos data/vla_demos_v5 --video 50 \
        --video-dir outputs/vla/final_test/videos_v5_run$run > outputs/vla/final_test/v5_${best}_run$run.txt 2>&1
    say "test run $run: $(grep '^policy' outputs/vla/final_test/v5_${best}_run$run.txt)"
done
say "done"
