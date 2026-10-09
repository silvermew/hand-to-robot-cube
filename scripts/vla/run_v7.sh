#!/bin/bash
# 9 Oct, v7: one policy, three instructions, each taught by one of my clips (cross B), 300 demos each, with v6's
# approach (15 cm, 0.4 s hover) and random face near switches:
#   counter-clockwise: clip_45 (+55 deg; 300 of v6's demos, hand turn)
#   clockwise:         clip_31 (-57 deg; the gripper follows the cube's turn: the hand alone under-turns by 1/3)
#   no turn:           clip_38 (+4 deg)
# Fine-tuned 10k steps from v6-10k. The last checkpoint (no selection) is tested per instruction on the same 50
# cube poses as v4-v6. Log: outputs/vla/v7/run.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
CCW="pick up the cube, rotate it counter-clockwise and put it down"
CW="pick up the cube, rotate it clockwise and put it down"
STILL="pick up the cube and put it back down without turning it"
common="--data data/raw --n 75 --random-face 30 --approach-h 0.15 --hover 0.4"
for seed in 71 72 73 74; do
    ~/.venvs/robot-sim/bin/python -m src.make_sim_demos $common --seed $seed --clip clip_31_rot-60_B --turn cube \
        --instruction "$CW" --out data/vla_demos_v7_cw > outputs/vla/v7/demos_cw_$seed.log 2>&1 &
    ~/.venvs/robot-sim/bin/python -m src.make_sim_demos $common --seed $((seed + 10)) --clip clip_38_rot+0_B \
        --instruction "$STILL" --out data/vla_demos_v7_still > outputs/vla/v7/demos_still_$seed.log 2>&1 &
done
mkdir -p data/vla_demos_v7_ccw
ls data/vla_demos_v6 | shuf --random-source=<(yes) -n 300 | while read f; do ln "data/vla_demos_v6/$f" "data/vla_demos_v7_ccw/$f"; done
wait
say "demos: ccw $(ls data/vla_demos_v7_ccw | wc -l), cw $(ls data/vla_demos_v7_cw | wc -l), still $(ls data/vla_demos_v7_still | wc -l)"
mkdir -p data/vla_demos_v7
for t in ccw cw still; do cp -al data/vla_demos_v7_$t/. data/vla_demos_v7/; done
~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v7 --root data/lerobot/cube_rotate_sim_v7 --clock \
    > outputs/vla/v7/to_lerobot.log 2>&1 || { say "conversion FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v7/to_lerobot.log)"
scripts/vla/train_smolvla.sh data/lerobot/cube_rotate_sim_v7 outputs/vla/v7/smolvla_cube_v7 10000 \
    outputs/vla/v6/smolvla_cube_v6/checkpoints/010000/pretrained_model > outputs/vla/v7/train.log 2>&1
say "training over"
ckpt=outputs/vla/v7/smolvla_cube_v7/checkpoints/010000/pretrained_model
mkdir -p outputs/vla/v7/test
test() {  # task, instruction, template clip, turn mode
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy $ckpt --episodes 50 --seed 2026 --device cuda \
        --demos data/vla_demos_v7_$1 --instruction "$2" --clip $3 --turn $4 --video 10 \
        --video-dir outputs/vla/v7/test/videos_$1 > outputs/vla/v7/test/$1.txt 2>&1
    say "$1: $(grep '^policy' outputs/vla/v7/test/$1.txt) | $(grep 'target turn' outputs/vla/v7/test/$1.txt)"
}
test ccw "$CCW" clip_45_rot+60_B hand
test cw "$CW" clip_31_rot-60_B cube
test still "$STILL" clip_38_rot+0_B hand
say "done"
