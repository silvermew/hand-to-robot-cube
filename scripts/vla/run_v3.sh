#!/bin/bash
# 7 Oct, v3: 300 clean demos replaying one clip (clip_45), the clock in the state, 15k steps, every checkpoint
# evaluated (10 episodes, CPU). Log: outputs/vla/v3/run.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
mkdir -p outputs/vla/v3
say() { echo "$(date +%H:%M) $*"; }
while pgrep -f "python -m src[.]make_sim_demos" > /dev/null; do sleep 15; done
say "demos: $(ls data/vla_demos_v3 | wc -l)"
~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v3 --root data/lerobot/cube_rotate_sim_v3 --clock \
    > outputs/vla/v3/to_lerobot.log 2>&1 || { say "conversion FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v3/to_lerobot.log)"
scripts/vla/train_smolvla.sh data/lerobot/cube_rotate_sim_v3 outputs/vla/v3/smolvla_cube_v3 15000 > outputs/vla/v3/train.log 2>&1 &
sleep 60
cp outputs/vla/v3/train.pid outputs/vla/train.pid
say "training (pid $(cat outputs/vla/train.pid))"
scripts/vla/eval_checkpoints.sh outputs/vla/v3/smolvla_cube_v3 outputs/vla/v3/eval data/vla_demos_v3
say "done: outputs/vla/v3/eval/<step>.txt"
