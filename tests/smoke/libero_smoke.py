"""Smoke test: one LIBERO task, offscreen rendering, ~50 zero-action steps.

Run in the separate `libero` venv (LIBERO needs robosuite 1.4, not 1.5).
Saves libero_frame.png next to this script.
"""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OUT_DIR = Path(__file__).resolve().parent
SUITE, TASK_ID, N_STEPS = "libero_object", 0, 50


def main():
    bench = benchmark.get_benchmark_dict()[SUITE]()
    task = bench.get_task(TASK_ID)
    print(f"{SUITE} task {TASK_ID}: '{task.language}'")
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(bench.get_task_init_states(TASK_ID)[0])  # official initial state #0

    for _ in range(N_STEPS):
        obs, reward, done, _ = env.step([0.0] * 7)

    obj_keys = [k for k in obs if k.endswith("_pos") and not k.startswith("robot")][:3]
    for k in obj_keys:
        print(f"{k} = {np.round(obs[k], 4)}")
    print(f"robot eef_pos={np.round(obs['robot0_eef_pos'], 4)}  reward={reward}  done={done}")

    frame = obs["agentview_image"][::-1]  # offscreen images come out upside down
    imageio.imwrite(OUT_DIR / "libero_frame.png", frame)
    print(f"saved {OUT_DIR / 'libero_frame.png'} (mean pixel {frame.mean():.1f})")
    env.close()
    print("OK")


if __name__ == "__main__":
    main()
