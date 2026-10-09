#!/bin/bash
# 8 Oct: final test on 50 cube poses never used for choosing a checkpoint (seed 2026; selection used seed 123).
# The chosen checkpoint of each version, each scored against its own demos' typical turn.
cd "$(dirname "$0")/../.."
unset PYTHONPATH
export MUJOCO_GL=egl
run() {  # name, checkpoint, demos, videos
    out=outputs/vla/final_test/$1.txt
    [ -f "$out" ] && return
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy "$2/pretrained_model" --episodes 50 --seed 2026 --device cuda \
        --demos "$3" --video "$4" --video-dir outputs/vla/final_test/videos_$1 > "$out.tmp" 2>&1 && mv "$out.tmp" "$out"
}
run v4_012000 outputs/vla/v4/smolvla_cube_v4/checkpoints/012000 data/vla_demos_v4 5
run v3_015000 outputs/vla/v3/smolvla_cube_v3/checkpoints/015000 data/vla_demos_v3 2
run v1_018000 outputs/vla/smolvla_cube/checkpoints/018000 data/vla_demos 2
