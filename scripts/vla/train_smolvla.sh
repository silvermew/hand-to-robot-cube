#!/bin/bash
# SmolVLA fine-tune on sim demos, local RTX 2080 (8 GB). Batch 8 without AMP: ~4.6 GB peak, ~1.2 s/step
# (measured 6 Oct). Checkpoint every 2k steps.
#   scripts/vla/train_smolvla.sh [dataset root] [run dir] [steps] [start policy]
#   (defaults: the first run, 20000 steps, lerobot/smolvla_base; a checkpoint's pretrained_model to continue from it)
# Stop at any time with: kill $(cat <run dir>/train.pid); the latest checkpoint stays usable.
cd "$(dirname "$0")/../.."
unset PYTHONPATH
DATASET=${1:-data/lerobot/cube_rotate_sim}
RUN=${2:-outputs/vla/smolvla_cube}
STEPS=${3:-20000}
START=${4:-lerobot/smolvla_base}
mkdir -p "$(dirname "$RUN")"
echo $$ > "$(dirname "$RUN")/train.pid"  # this shell becomes lerobot-train (exec): the PID to kill
exec ~/.venvs/lerobot/bin/lerobot-train --policy.path="$START" \
  --dataset.repo_id=local/$(basename "$DATASET") --dataset.root="$DATASET" \
  --batch_size=8 --steps=$STEPS --save_freq=2000 --log_freq=100 \
  --output_dir="$RUN" --job_name=$(basename "$RUN") \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false
