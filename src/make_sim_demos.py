"""Sim demos for the VLA stretch goal: replay the human +60 deg clips from random cube start poses.

    python -m src.make_sim_demos [--data data/raw] [--n 100] [--max-attempts 300] [--noise] [--out DIR] [--seed 0] [--clip NAME]
        [--random-face DEG] [--ambiguous-only] [--approach-h M] [--hover S] [--instruction TEXT] [--turn hand|cube] [--task NAME]

With a push task (push_AB, ...), the template is one of my push moves (src/push_extract.py) replayed by
src/sim_push.py with the gripper closed, from a cube on its start cross turned within PUSH_YAW_JITTER of how mine was.

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
from src.sim_push import load_move, make_push_env, run_push_replay, save_push_demo, start_pose
from src.sim_replay import (CUBE_HOME_XY, DT, FRAME_YAW, PHASES, SIM_CROSSES, TABLE_Z, demo_path, eef_pose,
                            face_costs, make_env, run_replay, save_demo, with_cube_turn)

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


# One sentence per task, each taught by one of my clips: (sentence, clip, start cross or None for any, gripper turns
# like my hand or like the cube). Carries use the cube's turn: my hand drifts ~20 deg on some plain carries. Not used:
# C -> A with a -96 deg turn (clip 91): the scripted arm lags 8 cm behind that fast carry and misses by 5 cm.
CARRY = "move the cube from cross {} to cross {}"
TASKS = {
    "ccw": ("pick up the cube, rotate it counter-clockwise and put it down", "clip_45_rot+60_B", None, "hand"),
    "cw": ("pick up the cube, rotate it clockwise and put it down", "clip_31_rot-60_B", None, "cube"),
    "AB": (CARRY.format("A", "B"), "clip_76_rot+0_A", "A", "cube"),
    "BC": (CARRY.format("B", "C"), "clip_77_rot+0_B", "B", "cube"),
    "AC": (CARRY.format("A", "C"), "clip_78_rot+0_A", "A", "cube"),
    "CA": (CARRY.format("C", "A"), "clip_79_rot+0_C", "C", "cube"),
    "BA": (CARRY.format("B", "A"), "clip_80_rot+0_B", "B", "cube"),
    "CB": (CARRY.format("C", "B"), "clip_81_rot+0_C", "C", "cube"),
    "AB_ccw": (CARRY.format("A", "B") + " and rotate it counter-clockwise", "clip_88_rot+60_A", "A", "cube"),
    "BC_cw": (CARRY.format("B", "C") + " and rotate it clockwise", "clip_89_rot-60_B", "B", "cube"),
    "AC_ccw": (CARRY.format("A", "C") + " and rotate it counter-clockwise", "clip_90_rot+90_A", "A", "cube"),
    "BA_ccw": (CARRY.format("B", "A") + " and rotate it counter-clockwise", "clip_92_rot+30_B", "B", "cube"),
    "CB_cw": (CARRY.format("C", "B") + " and rotate it clockwise", "clip_93_rot-30_C", "C", "cube"),
}
# Pushes, each taught by one of my straight pushes (the one the scripted replay follows best, outputs/push/replay.csv);
# none of my turning pushes replays in the sim (see the README)
PUSH = "push the cube from cross {} to cross {}"
PUSH_TASKS = {"push_AB": "push_09_AB_straight", "push_BC": "push_06_BC_straight", "push_CB": "push_15_CB_straight",
              "push_BA": "push_08_BA_straight", "push_AC": "push_29_AC_straight", "push_CA": "push_30_CA_straight"}
TASKS.update({k: (PUSH.format(k[5], k[6]), v, k[5], "push") for k, v in PUSH_TASKS.items()})
CROSS_JITTER = 0.02  # m around a cross
PUSH_YAW_JITTER = np.radians(20)  # my cubes sat square to the line of crosses; a push turns a cube set at 45 deg


def sample_cube_pose(rng):
    xy = CUBE_HOME_XY + [rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)]
    return np.r_[xy, TABLE_Z + C.CUBE_SIDE / 2], rng.uniform(-np.pi, np.pi)


def sample_cross_pose(rng, cross=None):
    """The cube on a cross (any of the three if None), within CROSS_JITTER of it, at any yaw."""
    cross = cross or "ABC"[rng.integers(3)]
    xy = SIM_CROSSES[cross] + rng.uniform(-CROSS_JITTER, CROSS_JITTER, 2)
    return np.r_[xy, TABLE_Z + C.CUBE_SIDE / 2], rng.uniform(-np.pi, np.pi)


def is_push(task):
    return task is not None and TASKS[task][3] == "push"


def task_clip(dirs, task):
    """(clip name, the clip as replayed: with the cube's turn if the task says so) for a task of TASKS; for a push
    task, the move."""
    _, name, _, turn = TASKS[task]
    if turn == "push":
        return name, load_move(dirs["base"] / "push" / "processed" / f"{name}.npz")
    clip = dict(np.load(dirs["processed"] / f"{name}.npz"))
    return name, with_cube_turn(clip) if turn == "cube" else clip


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
    p.add_argument("--task", choices=sorted(TASKS), default=None,
                   help="one of TASKS: its sentence, clip and turn mode; the cube starts on its cross; crosses drawn, "
                        "carries keep their direction (sim_replay.FRAME_YAW)")
    p.add_argument("--ambiguous-only", action="store_true",
                   help="only cube poses where two grasp faces are within --random-face of each other")
    args = p.parse_args([] if raw_dir else None)
    dirs = data_dirs(args.data)
    if args.task:
        clips = [task_clip(dirs, args.task)]
        args.instruction = TASKS[args.task][0]
    else:
        clips = [(args.clip, dict(np.load(dirs["processed"] / f"{args.clip}.npz")))] if args.clip else bin_clips(dirs)
        if args.turn == "cube":
            clips = [(n, with_cube_turn(c)) for n, c in clips]
    sample = (lambda r: sample_cross_pose(r, TASKS[args.task][2])) if args.task else sample_cube_pose
    if is_push(args.task):
        sample = lambda r: start_pose(clips[0][1], r, CROSS_JITTER, PUSH_YAW_JITTER)
    if not clips:
        print(f"no processed clips with an intended angle in {ANGLE_BIN} deg")
        return
    out = Path(args.out) if args.out else dirs["base"] / "vla_demos"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = make_push_env() if is_push(args.task) else make_env(render=True, crosses=bool(args.task))
    env.reset()
    heading_home = yaw_of(eef_pose(env)[1])
    margin = np.radians(args.random_face)
    kept, attempts = 0, 0
    for attempts in range(1, (args.max_attempts or 3 * args.n) + 1):
        name, clip = clips[rng.integers(len(clips))]
        pos0, yaw0 = sample(rng)
        while args.ambiguous_only and np.sort(face_costs(clip, yaw0, heading_home)[1])[1] > \
                np.sort(face_costs(clip, yaw0, heading_home)[1])[0] + margin:
            pos0, yaw0 = sample(rng)
        if is_push(args.task):
            metrics, _, log = run_push_replay(env, clip, pos0, yaw0, record=True, video=False)
        else:
            metrics, _, log = run_replay(env, clip, pos0, yaw0, True, record=True, video=False,
                                         perturb=perturbation(rng) if args.noise else None,
                                         face_rng=rng if margin > 0 else None, face_margin=margin,
                                         approach_h=args.approach_h, hover_s=args.hover,
                                         frame_yaw=FRAME_YAW if args.task else None)
        if metrics["success"]:
            log["instruction"] = args.instruction
            if args.task:
                log["task"] = args.task
            path = demo_path(out, name, pos0, yaw0, True)
            if is_push(args.task):
                save_push_demo(path, log, metrics, name, pos0, yaw0)
            else:
                save_demo(path, log, metrics, name, pos0, yaw0, True)
            kept += 1
        print(f"attempt {attempts}: {name} cube ({pos0[0]:+.3f}, {pos0[1]:+.3f}) yaw {np.degrees(yaw0):+5.0f} -> "
              f"{'kept' if metrics['success'] else 'failed'} ({kept}/{args.n})")
        if kept >= args.n:
            break
    env.close()
    print(f"kept {kept} successful demos out of {attempts} attempts in {out}")


if __name__ == "__main__":
    main()
