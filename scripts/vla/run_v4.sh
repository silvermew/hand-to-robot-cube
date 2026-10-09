#!/bin/bash
# 7 Oct, v4: v3's step-15k policy fine-tuned 15k more steps on 600 clip_45 demos (v3's 300 + 300 new), clock
# in the state; checkpoints evaluated on the CPU during training (10 episodes), then all on the GPU (20 episodes).
# Log: outputs/vla/v4/run.log
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
say() { echo "$(date +%H:%M) $*"; }
while pgrep -f "python -m src[.]make_sim_demos" > /dev/null; do sleep 15; done
mkdir -p data/vla_demos_v4
cp -al data/vla_demos_v3/. data/vla_demos_v4/ && cp -al data/vla_demos_v4_new/. data/vla_demos_v4/  # hard links
say "demos: $(ls data/vla_demos_v4 | wc -l)"
~/.venvs/lerobot/bin/python -m src.to_lerobot --demos data/vla_demos_v4 --root data/lerobot/cube_rotate_sim_v4 --clock \
    > outputs/vla/v4/to_lerobot.log 2>&1 || { say "conversion FAILED"; exit 1; }
say "$(grep '^wrote' outputs/vla/v4/to_lerobot.log)"
scripts/vla/train_smolvla.sh data/lerobot/cube_rotate_sim_v4 outputs/vla/v4/smolvla_cube_v4 15000 \
    outputs/vla/v3/smolvla_cube_v3/checkpoints/015000/pretrained_model > outputs/vla/v4/train.log 2>&1 &
sleep 60
cp outputs/vla/v4/train.pid outputs/vla/train.pid
say "training (pid $(cat outputs/vla/train.pid))"
scripts/vla/eval_checkpoints.sh outputs/vla/v4/smolvla_cube_v4 outputs/vla/v4/eval_cpu data/vla_demos_v4
say "training over; 20-episode evaluations on the GPU"
for ckpt in $(ls -d outputs/vla/v4/smolvla_cube_v4/checkpoints/0* | sort -r); do
    step=$(basename "$ckpt")
    out=outputs/vla/v4/eval/$step.txt
    mkdir -p outputs/vla/v4/eval
    [ -f "$out" ] && continue
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy "$ckpt/pretrained_model" --episodes 20 --device cuda \
        --video 2 --video-dir outputs/vla/v4/eval --demos data/vla_demos_v4 > "$out.tmp" 2>&1 && mv "$out.tmp" "$out"
done
say "done: outputs/vla/v4/eval/<step>.txt"
