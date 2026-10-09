"""Sim demos for the VLA stretch goal: replay the human +60 deg clips from random cube start poses.

    python -m src.make_sim_demos [--data data/raw] [--n 100] [--max-attempts 300] [--noise] [--out DIR] [--seed 0] [--clip NAME]
        [--random-face DEG] [--ambiguous-only] [--approach-h M] [--hover S] [--instruction TEXT] [--turn hand|cube]

Uses the processed clips whose intended angle is in the rotation bin (default 45..75 deg), not the transports,
samples a cube start pose inside the reachable area (x within -0.10..+0.06 m of the cube
home: further out, the carry leaves the arm's reach) with any yaw, replays a
randomly chosen clip with the scripted controller, and keeps the successful replays (images,
state, actions) in <data>/../vla_demos/. These are scripted demos derived from the human
trajectories; a policy trained on them imitates that script.
"""
import argparse
from pathlib import Path

import numpy as np

from src import constants as C
from src.common import ROOT, data_dirs
from src.geometry import yaw_of
from src.sim_replay import (CUBE_HOME_XY, DT, PHASES, TABLE_Z, demo_path, eef_pose, face_costs, make_env,
                            run_replay, save_demo, with_cube_turn)

ANGLE_BIN = (45, 75)
MAX_CARRY = 0.05  # m; transports end at another cross, which the instruction does not ask for
X_RANGE, Y_RANGE = (-0.10, 0.06), (-0.10, 0.10)
INSTRUCTION = "pick up the cube, rotate it counter-clockwise and put it down"
NOISE = np.array([0.025, 0.025, 0.010, np.radians(8)])  # drift std: x, y, z (m), heading (rad); the clean
                                                        # demos already trail their targets by 2-3 cm
NOISE_S = 0.5                                           # its correlation time
NOISE_GAIN = {"to_start": 1.0, "approach": 1.0, "closing": 0.0, "carry": 1.0, "releasing": 0.0, "retreat": 1.0,
              "settle": 0.0}


def perturbation(rng):
    """Drift for --noise demos (DART-style): smooth random offsets on the executed targets, while the logged action
    stays the planned target, so the demos show the arm being brought back on course. The first policy pushed the
    cube once it was a few cm off, a state no perfect demo contains (see the README). Off while the gripper
    closes or opens, fading out over the descent onto the cube, and off at the end."""
    def offsets(plan):
        n, k = len(plan["pos"]), int(NOISE_S / DT)
        smooth = np.stack([np.convolve(rng.normal(size=n + k - 1), np.ones(k) / np.sqrt(k), "valid") for _ in range(4)], 1)
        names = [PHASES[i] for i in plan["phase"]]
        gain = np.array([NOISE_GAIN[name] for name in names])
        approach = np.flatnonzero(np.array(names) == "approach")
        gain[approach] = np.linspace(1, 0, len(approach))
        gain = np.convolve(np.pad(gain, k // 2, mode="edge"), np.ones(k) / k, "valid")[:n]  # no jumps at phase changes
        return smooth * NOISE * gain[:, None]
    return offsets


def sample_cube_pose(rng):
    xy = CUBE_HOME_XY + [rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)]
    return np.r_[xy, TABLE_Z + C.CUBE_SIDE / 2], rng.uniform(-np.pi, np.pi)


def bin_clips(dirs, angle_bin=ANGLE_BIN):
    out = []
    for p in sorted(dirs["processed"].glob("*.npz")):
        clip = dict(np.load(p))
        carry = np.linalg.norm(clip["cube1"][:2] - clip["cube0"][:2])
        if angle_bin[0] <= float(clip["intended_angle"]) <= angle_bin[1] and carry < MAX_CARRY:
            out.append((p.stem, clip))
    return out


def main(raw_dir=None, n=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(raw_dir or ROOT / "data" / "raw"))
    p.add_argument("--n", type=int, default=n or 100, help="successful demos to keep")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", action="store_true", help="add drift the scripted controller corrects (see perturbation)")
    p.add_argument("--out", default=None, help="demo folder (default <data>/../vla_demos)")
    p.add_argument("--clip", default=None, help="replay only this processed clip (one consistent template), any angle")
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--turn", choices=("hand", "cube"), default="hand",
                   help="gripper turns like the hand, or like the cube (sim_replay.with_cube_turn)")
    p.add_argument("--random-face", type=float, default=0.0,
                   help="deg: draw the grasp face at random among those within this of the best (0: always the best)")
    p.add_argument("--approach-h", type=float, default=0.10, help="m above the grasp point before going down")
    p.add_argument("--hover", type=float, default=0.0, help="s to pause there before going down")
    p.add_argument("--ambiguous-only", action="store_true",
                   help="only cube poses where two grasp faces are within --random-face of each other")
    args = p.parse_args([] if raw_dir else None)
    dirs = data_dirs(args.data)
    clips = [(args.clip, dict(np.load(dirs["processed"] / f"{args.clip}.npz")))] if args.clip else bin_clips(dirs)
    if args.turn == "cube":
        clips = [(n, with_cube_turn(c)) for n, c in clips]
    if not clips:
        print(f"no processed clips with an intended angle in {ANGLE_BIN} deg")
        return
    out = Path(args.out) if args.out else dirs["base"] / "vla_demos"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = make_env(render=True)
    env.reset()
    heading_home = yaw_of(eef_pose(env)[1])
    margin = np.radians(args.random_face)
    kept, attempts = 0, 0
    for attempts in range(1, (args.max_attempts or 3 * args.n) + 1):
        name, clip = clips[rng.integers(len(clips))]
        pos0, yaw0 = sample_cube_pose(rng)
        while args.ambiguous_only and np.sort(face_costs(clip, yaw0, heading_home)[1])[1] > \
                np.sort(face_costs(clip, yaw0, heading_home)[1])[0] + margin:
            pos0, yaw0 = sample_cube_pose(rng)
        metrics, _, log = run_replay(env, clip, pos0, yaw0, True, record=True, video=False,
                                     perturb=perturbation(rng) if args.noise else None,
                                     face_rng=rng if margin > 0 else None, face_margin=margin,
                                     approach_h=args.approach_h, hover_s=args.hover)
        if metrics["success"]:
            log["instruction"] = args.instruction
            save_demo(demo_path(out, name, pos0, yaw0, True), log, metrics, name, pos0, yaw0, True)
            kept += 1
        print(f"attempt {attempts}: {name} cube ({pos0[0]:+.3f}, {pos0[1]:+.3f}) yaw {np.degrees(yaw0):+5.0f} -> "
              f"{'kept' if metrics['success'] else 'failed'} ({kept}/{args.n})")
        if kept >= args.n:
            break
    env.close()
    print(f"kept {kept} successful demos out of {attempts} attempts in {out}")


if __name__ == "__main__":
    main()
