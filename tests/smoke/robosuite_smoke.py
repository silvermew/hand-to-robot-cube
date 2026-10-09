"""Smoke test: robosuite Lift (Panda + cube), offscreen rendering, ~100 steps.

Saves test_frame.png and test_video.mp4 next to this script and prints the
cube pose at the start and end. Set HEADLESS_STATE_ONLY=1 to skip rendering.
"""
import os
import sys
from pathlib import Path

# Headless GPU rendering by default; override with MUJOCO_GL=osmesa if EGL fails.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np
import robosuite as suite

OUT_DIR = Path(__file__).resolve().parent
N_STEPS = 100
CAMERA = "agentview"
STATE_ONLY = os.environ.get("HEADLESS_STATE_ONLY") == "1"


def cube_pose(obs):
    # robosuite reports quaternions as (x, y, z, w)
    return np.round(obs["cube_pos"], 4), np.round(obs["cube_quat"], 4)


def main():
    print(f"robosuite {suite.__version__}, MUJOCO_GL={os.environ['MUJOCO_GL']}, state_only={STATE_ONLY}")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=not STATE_ONLY,
        use_camera_obs=not STATE_ONLY,
        camera_names=CAMERA,
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        horizon=N_STEPS + 10,
    )
    obs = env.reset()
    pos, quat = cube_pose(obs)
    print(f"[start] cube_pos={pos}  cube_quat(xyzw)={quat}")

    rng = np.random.default_rng(0)
    low, high = env.action_spec
    frames = []
    for _ in range(N_STEPS):
        action = 0.1 * rng.uniform(low, high)  # small random actions
        obs, reward, done, _ = env.step(action)
        if not STATE_ONLY:
            # Offscreen images come out upside down (OpenGL convention)
            frames.append(obs[f"{CAMERA}_image"][::-1].copy())

    pos, quat = cube_pose(obs)
    print(f"[end]   cube_pos={pos}  cube_quat(xyzw)={quat}  reward={reward:.4f}")
    print(f"robot eef_pos={np.round(obs['robot0_eef_pos'], 4)}  sim_time={env.sim.data.time:.2f}s")

    if not STATE_ONLY:
        import imageio.v2 as imageio

        imageio.imwrite(OUT_DIR / "test_frame.png", frames[-1])
        # A good agentview frame (mostly white table) averages ~215-220; far off means a garbled render
        print(f"saved {OUT_DIR / 'test_frame.png'} (mean pixel {frames[-1].mean():.1f}, expect ~215-220)")
        try:
            imageio.mimsave(OUT_DIR / "test_video.mp4", frames, fps=20)
            print(f"saved {OUT_DIR / 'test_video.mp4'} ({len(frames)} frames)")
        except Exception as e:  # video is optional
            print(f"video not saved: {e}", file=sys.stderr)

    env.close()
    print("OK")


if __name__ == "__main__":
    main()
