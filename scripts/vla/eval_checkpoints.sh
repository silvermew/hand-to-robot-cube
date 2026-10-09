#!/bin/bash
# Evaluates every SmolVLA checkpoint of a run closed-loop as training saves it (same 10 cube poses each time, CPU,
# so the GPU stays with training). Results: <eval dir>/<step>.txt, videos of 2 episodes per checkpoint.
#   scripts/vla/eval_checkpoints.sh [run dir] [eval dir] [demos dir]   (defaults: the first run)
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl OMP_NUM_THREADS=6
RUN=${1:-outputs/vla/smolvla_cube}
EVAL=${2:-outputs/vla/eval}
DEMOS=${3:-data/vla_demos}
TRAIN_PID=$(cat outputs/vla/train.pid)
mkdir -p "$EVAL"

evaluate_new() {
    for ckpt in "$RUN"/checkpoints/[0-9]*; do
        step=$(basename "$ckpt")
        [ -f "$EVAL/$step.txt" ] && continue
        # skip a checkpoint that may still be being written
        [ -n "$(find "$ckpt/pretrained_model/model.safetensors" -mmin +2 2>/dev/null)" ] || continue
        nice -n 10 ~/.venvs/lerobot/bin/python -m src.eval_vla --policy "$ckpt/pretrained_model" --episodes 10 \
            --device cpu --video 2 --video-dir "$EVAL" --demos "$DEMOS" > "$EVAL/$step.txt.tmp" 2>&1 \
            && mv "$EVAL/$step.txt.tmp" "$EVAL/$step.txt"
    done
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    evaluate_new
    sleep 60
done
sleep 180  # the last checkpoint
evaluate_new
