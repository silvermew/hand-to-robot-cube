#!/bin/bash
# 8 Oct, v6: 900 new clip_45 demos with a higher, paused approach (15 cm above the grasp point, 0.4 s hover) and
# the random face choice near switches (uniform cube yaws). Fine-tuned 10k steps from v5's best checkpoint once
# v5's pipeline is over; then 20 selection poses per checkpoint and the 50 test poses twice. Log: outputs/vla/v6/run.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
mkdir -p data/vla_demos_v6
for seed in 61 62 63 64; do
    nice -n 10 ~/.venvs/robot-sim/bin/python -m src.make_sim_demos --data data/raw --n 225 --clip clip_45_rot+60_B \
        --seed $seed --random-face 30 --approach-h 0.15 --hover 0.4 --out data/vla_demos_v6 > outputs/vla/v6/demos_$seed.log 2>&1 &
done
wait
say "demos: $(ls data/vla_demos_v6 | wc -l); $(grep -h '^kept' outputs/vla/v6/demos_*.log | awk '{s += $2; a += $7} END {print s " kept of " a " attempts"}')"
nice -n 10 ~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v6 --root data/lerobot/cube_rotate_sim_v6 --clock \
    > outputs/vla/v6/to_lerobot.log 2>&1 || { say "conversion FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v6/to_lerobot.log)"
while pgrep -f "run_v5[.]sh" > /dev/null; do sleep 60; done
best5=$(for f in outputs/vla/v5/eval/0*.txt; do echo "$(grep -o 'success *[0-9.]*' $f | awk '{print $2}') $(basename $f .txt)"; done | sort -n | tail -1 | awk '{print $2}')
say "v5 is over; starting from v5 $best5"
scripts/vla/train_smolvla.sh data/lerobot/cube_rotate_sim_v6 outputs/vla/v6/smolvla_cube_v6 10000 \
    outputs/vla/v5/smolvla_cube_v5/checkpoints/$best5/pretrained_model > outputs/vla/v6/train.log 2>&1
say "training over"
mkdir -p outputs/vla/v6/eval
for ckpt in $(ls -d outputs/vla/v6/smolvla_cube_v6/checkpoints/0* | sort -r); do
    step=$(basename "$ckpt")
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy "$ckpt/pretrained_model" --episodes 20 --device cuda \
        --demos data/vla_demos_v6 > "outputs/vla/v6/eval/$step.txt" 2>&1
    say "$step: $(grep '^policy' outputs/vla/v6/eval/$step.txt)"
done
best=$(for f in outputs/vla/v6/eval/0*.txt; do echo "$(grep -o 'success *[0-9.]*' $f | awk '{print $2}') $(basename $f .txt)"; done | sort -n | tail -1 | awk '{print $2}')
say "best on the selection poses: $best; 50 test poses, two runs"
for run in 1 2; do
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy outputs/vla/v6/smolvla_cube_v6/checkpoints/$best/pretrained_model \
        --episodes 50 --seed 2026 --device cuda --demos data/vla_demos_v6 --video 50 \
        --video-dir outputs/vla/final_test/videos_v6_run$run > outputs/vla/final_test/v6_${best}_run$run.txt 2>&1
    say "test run $run: $(grep '^policy' outputs/vla/final_test/v6_${best}_run$run.txt)"
done
say "done"
