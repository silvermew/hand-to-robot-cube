"""Convert the successful sim demos into a LeRobotDataset (run with the lerobot venv's python).

    ~/.venvs/lerobot/bin/python -m src.to_lerobot [--demos data/vla_demos] [--root data/lerobot/cube_rotate_sim]

Checked against lerobot 0.6.1 (LeRobotDataset.create / add_frame / save_episode / finalize).
Cameras are named like smolvla_base's inputs, so lerobot-train needs no rename map:
  observation.images.camera1  agentview (LIBERO-style pose), 256x256 RGB, upright
  observation.images.camera2  wrist camera, 256x256 RGB
  observation.state  (6)      see src/vla_format.py
  action             (5)      delta end-effector action, see src/vla_format.py
One task string per demo (the fixed instruction), 20 fps (the sim control rate).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from src.vla_format import ACTION_NAMES, demo_states_actions, state_names

ROOT = Path(__file__).resolve().parent.parent
FPS = 20


def features(clock=False):
    image = dict(dtype="video", shape=(256, 256, 3), names=["height", "width", "channel"])
    return {"observation.images.camera1": image, "observation.images.camera2": dict(image),
            "observation.state": dict(dtype="float32", shape=(len(state_names(clock)),), names=state_names(clock)),
            "action": dict(dtype="float32", shape=(len(ACTION_NAMES),), names=ACTION_NAMES)}


def main():
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--demos", default=str(ROOT / "data" / "vla_demos"))
    p.add_argument("--root", default=str(ROOT / "data" / "lerobot" / "cube_rotate_sim"))
    p.add_argument("--repo-id", default="local/cube_rotate_sim")
    p.add_argument("--clock", action="store_true", help="add the time since the episode started to the state")
    args = p.parse_args()
    demos = sorted(Path(args.demos).glob("*.npz"))
    if not demos:
        raise SystemExit(f"no demos in {args.demos}; run src.make_sim_demos first")
    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)  # create() refuses an existing folder; this one is fully derived from the demos
    # Frames are encoded to H.264 as they are added (the default writes PNG files, then AV1 per episode), which is
    # also faster to decode while training
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features(args.clock), root=root, robot_type="panda",
                               use_videos=True, streaming_encoding=True,
                               rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=23))
    n_frames = 0
    for path in demos:
        d = np.load(path)
        if not bool(d["success"]):
            continue
        states, actions = demo_states_actions(d, args.clock)
        task = str(d["instruction"]) if "instruction" in d.files else "pick up the cube, rotate it and put it down"
        agent, wrist = d["agentview"], d["wrist"]  # once: each d[key] decompresses the whole array again
        for k in range(len(actions)):
            ds.add_frame({"observation.images.camera1": agent[k], "observation.images.camera2": wrist[k],
                          "observation.state": states[k], "action": actions[k], "task": task})
        ds.save_episode()
        n_frames += len(actions)
    ds.finalize()
    print(f"wrote {ds.meta.total_episodes} episodes, {n_frames} frames to {root} (repo_id {args.repo_id})")


if __name__ == "__main__":
    main()
