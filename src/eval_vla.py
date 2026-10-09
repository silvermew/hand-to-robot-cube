"""Closed-loop evaluation of a policy in the robosuite scene, against the scripted replay.

    python -m src.eval_vla --policy random [--data data/raw] [--episodes 20]
    ~/.venvs/lerobot/bin/python -m src.eval_vla --policy outputs/vla/checkpoints/last/pretrained_model
        (the lerobot venv also has robosuite; see requirements-vla.txt)

Each episode starts the cube at a random reachable pose (same sampler as make_sim_demos, another
seed, so poses the demos never had), runs the policy for MAX_STEPS at 20 Hz, and reports
separately: lifted (>= 2 cm), placed (lifted, then put down within 3 cm of where it started; the
demos move it 1.0 cm median, 2.6 cm at most) and turned (yaw change within 15 deg of the demos'
median turn: the instruction gives no angle, so the policy is held to what it was taught);
success = all three. The scripted replay of a randomly drawn +60 bin clip is scored the same
way from the same poses. Policies see 256x256 agentview and wrist images (upright), the 6-D
state of src/vla_format.py and the fixed instruction. --video N saves the first N policy
episodes (agentview | wrist) to outputs/vla/eval/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.common import ROOT, data_dirs, is_synthetic
from src.geometry import gripper_down
from src.make_sim_demos import INSTRUCTION, bin_clips, sample_cube_pose
from src.sim_replay import (LIFT_MIN, POS_TOL, WRIST_CAMERA, YAW_TOL_DEG, cube_pose, make_env, render, run_replay,
                            set_cube_pose, sim_state, with_cube_turn)
from src.vla_format import CLOCK_NAME, FPS, action_to_target, make_state

MAX_STEPS = 400


def random_policy(seed=0):
    rng = np.random.default_rng(seed)

    def act(obs):
        return np.r_[rng.normal(0, 0.01, 3), rng.normal(0, 0.05), rng.choice([-1.0, 1.0])]
    return act


def lerobot_policy(path, device=None, n_action_steps=None):
    """SmolVLA (or any lerobot policy) from a checkpoint folder; imported lazily (lerobot venv only).
    n_action_steps: how many of each predicted 50-step chunk to execute before predicting again (default: the
    trained 50, i.e. 2.5 s open loop)."""
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(path).to(device).eval()
    policy.config.device = device
    if n_action_steps:
        policy.config.n_action_steps = n_action_steps
    # the saved preprocessor moves inputs to the training device; follow lerobot-eval and override it
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    policy.reset()

    def act(obs):
        batch = {"observation.images.camera1": torch.from_numpy(obs["agentview"]).permute(2, 0, 1)[None].float() / 255,
                 "observation.images.camera2": torch.from_numpy(obs["wrist"]).permute(2, 0, 1)[None].float() / 255,
                 "observation.state": torch.from_numpy(obs["state"])[None], "task": [obs["task"]]}
        with torch.no_grad():
            action = post(policy.select_action(pre(batch)))
        return action[0].cpu().numpy()
    act.reset = policy.reset
    act.clock = trained_with_clock(path)
    return act


def trained_with_clock(path):
    """Whether the checkpoint's training data had the clock in the state (to_lerobot --clock). The saved policy
    config keeps the base model's input features (6-D state, 3 cameras), so read the dataset's own metadata."""
    cfg = json.loads((Path(path) / "train_config.json").read_text())
    root = Path(cfg["dataset"]["root"])
    info = json.loads(((root if root.is_absolute() else ROOT / root) / "meta" / "info.json").read_text())
    return CLOCK_NAME in info["features"]["observation.state"]["names"]


def demo_turn(demo_dir):
    """Median cube turn (deg) of the training demos."""
    turns = [float(np.load(p)["metric_sim_dyaw_deg"]) for p in sorted(Path(demo_dir).glob("*.npz"))]
    if not turns:
        raise SystemExit(f"no demos in {demo_dir}: run src.make_sim_demos first")
    return float(np.median(turns)), len(turns)


def run_policy_episode(env, act, pos0, yaw0, frames=None, instruction=INSTRUCTION):
    env.reset()
    set_cube_pose(env, pos0, yaw0)
    if hasattr(act, "reset"):
        act.reset()
    z0, max_z = pos0[2], pos0[2]
    yaws = [yaw0]
    clock = getattr(act, "clock", False)
    for step in range(MAX_STEPS):
        s = sim_state(env)
        obs = dict(agentview=render(env, "agentview").copy(), wrist=render(env, WRIST_CAMERA).copy(),
                   state=make_state(s["eef_pos"], s["eef_yaw"], s["gripper_opening"], step / FPS if clock else None)[0],
                   task=instruction)
        if frames is not None:
            frames.append(np.hstack([obs["agentview"], obs["wrist"]]))
        pos, heading, grip = action_to_target(s["eef_pos"], s["eef_yaw"], act(obs))
        env.step(np.r_[pos, Rotation.from_matrix(gripper_down(heading)).as_rotvec(), grip])
        cp, cy = cube_pose(env)
        max_z = max(max_z, cp[2])
        yaws.append(cy)
    final_pos, _ = cube_pose(env)
    return dict(lift_cm=100 * (max_z - z0), final_pos=final_pos, dyaw_deg=np.degrees(np.unwrap(yaws)[-1] - yaw0))


def score(lift_cm, final_pos, dyaw_deg, pos0, target_dyaw):
    lifted = lift_cm >= 100 * LIFT_MIN
    # "placed" needs a lift first: an untouched cube would otherwise count as put back
    placed = bool(lifted and np.linalg.norm(final_pos[:2] - pos0[:2]) < POS_TOL and abs(final_pos[2] - pos0[2]) < 0.01)
    yaw_ok = abs(dyaw_deg - target_dyaw) < YAW_TOL_DEG
    return dict(lifted=bool(lifted), placed=placed, yaw_ok=bool(yaw_ok), success=bool(lifted and placed and yaw_ok),
                lift_cm=round(float(lift_cm), 1), moved_cm=round(100 * float(np.linalg.norm(final_pos[:2] - pos0[:2])), 1),
                dyaw_deg=round(float(dyaw_deg), 1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--policy", default="random", help="'random' or a lerobot checkpoint folder")
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--demos", default=None, help="the training demos (default <data>/../vla_demos)")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default=None, help="cuda / cpu for a lerobot policy (default: cuda if available)")
    p.add_argument("--video", type=int, default=0, help="save the first N policy episodes as videos")
    p.add_argument("--n-action-steps", type=int, default=None, help="replan after this many steps of each chunk")
    p.add_argument("--video-dir", default=str(ROOT / "outputs" / "vla" / "eval"))
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--clip", default=None, help="the scripted reference replays this clip (default: a random +60 bin clip)")
    p.add_argument("--turn", choices=("hand", "cube"), default="hand", help="the scripted reference's turn mode")
    args = p.parse_args()
    dirs = data_dirs(args.data)
    clips = [(args.clip, dict(np.load(dirs["processed"] / f"{args.clip}.npz")))] if args.clip else bin_clips(dirs)
    if not clips:
        raise SystemExit("no processed clips in the rotation bin; run the pipeline first")
    if args.turn == "cube":
        clips = [(n, with_cube_turn(c)) for n, c in clips]
    target, n_demos = demo_turn(args.demos or dirs["base"] / "vla_demos")
    act = random_policy(args.seed) if args.policy == "random" else lerobot_policy(args.policy, args.device, args.n_action_steps)
    env = make_env(render=True)
    rng = np.random.default_rng(args.seed)
    video_dir = Path(args.video_dir)
    rows = {"policy": [], "scripted": []}
    for ep in range(args.episodes):
        pos0, yaw0 = sample_cube_pose(rng)
        frames = [] if ep < args.video else None
        r = run_policy_episode(env, act, pos0, yaw0, frames, args.instruction)
        rows["policy"].append(score(r["lift_cm"], r["final_pos"], r["dyaw_deg"], pos0, target))
        if frames:
            import imageio.v2 as imageio
            video_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_n{args.n_action_steps}" if args.n_action_steps else ""
            imageio.mimsave(video_dir / f"{Path(args.policy).parent.name}{tag}_episode{ep + 1}.mp4", frames, fps=20)
        name, clip = clips[rng.integers(len(clips))]
        m, _, log = run_replay(env, clip, pos0, yaw0, True, record=False)
        rows["scripted"].append(score(m["lift_cm"], log["final_cube_pos"], m["sim_dyaw_deg"], pos0, target))
        print(f"episode {ep + 1}: policy {rows['policy'][-1]} | scripted ({name}) {rows['scripted'][-1]}")
    env.close()
    tag = "SYNTHETIC DATA, " if is_synthetic(args.data) else ""
    print(f"\n{tag}{args.episodes} cube poses; target turn {target:+.1f} deg +-{YAW_TOL_DEG:.0f} (median of {n_demos} demos)")
    for who, rs in rows.items():
        print(f"{who:9s}: " + "  ".join(f"{k} {100 * np.mean([r[k] for r in rs]):5.1f}%"
                                        for k in ("lifted", "placed", "yaw_ok", "success")))


if __name__ == "__main__":
    main()
