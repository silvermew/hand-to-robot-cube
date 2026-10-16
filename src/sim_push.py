"""Replay a human push move (src/push_extract.py) on the simulated Panda with its gripper closed.

    python -m src.sim_push data/push/processed/<move>.npz [--out DIR] [--no-video]
    python -m src.sim_push --all [--workers 8]      (every move, results in outputs/push/replay.csv)

A scripted controller driven by the human data, like src/sim_replay.py for the grasps. My fingertips are not
tracked (the hand card sits 12 cm behind them, on the back of the hand), so the contact point is set from the cube:
on the face away from where the cube goes, at mid-height, centred for a straight push and E_TURN to one side for a
turning push (the side that turns the cube the way mine turned). The point is fixed on the cube, so during a push
the fingers follow the path that point took on my cube (my cube's motion, moved onto the sim table with
sim_replay.FRAME_YAW), and the gripper turns with it. Before each push the contact point is re-anchored on the sim
cube where it actually is, as a grasp is anchored on the cube, so errors do not carry over from push to push.
Between pushes the fingers back off, lift over the cube and come down behind it again, as my hand did. They come
down APPROACH_BACK behind the face after pausing HOVER_S above that point: the first push policy (v9) copied
demos that came down 2.6 cm behind it while the arm still trailed its sideways move, and its gripper landed on the
cube's top edge and rolled it over in most episodes of four of the six push tasks (the same failure as the
grasps before v6). The approach is timed like the grasps' (PRE_ROLL + HOVER_S = 2.6 s): the policy runs its
50-step (2.5 s) chunks open loop, so the descent must start in its second chunk, predicted from above the start
point, not in the first one, predicted from the home pose (v10's 1.5 + 0.4 s put most of the descent there).

The cube slides on the table with friction SLIDE_FRICTION instead of robosuite's 1.0. At 1.0 a 5.7 cm cube tips over
its front edge when pushed above 2.85 cm, and the controller holds the gripper about 6 deg off vertical, so the
fingers touch the top of the face first: every push tipped the cube by 14 deg, which my cube on the tablecloth
never did. Grasps are unchanged (the finger pads have their own friction, 2.0), and the pick-and-place scenes keep 1.0.

The sim arm sags 3-4 cm toward its base and 1 cm down from a still target near the table (robosuite's OSC with the
default gains): an integral term on the target position takes that out, since a push 3 cm off-centre turns the cube.
"""
import argparse
import csv
import multiprocessing
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src import constants as C
from src.common import label_frame
from src.geometry import gripper_down, rotz, wrap
from src.sim_replay import (DT, FRAME_YAW, ROOT, SIM_CROSSES, TABLE_Z, WRIST_CAMERA, annotate, cube_pose, eef_pose,
                            make_env, ramp, render, save_video, set_cube_pose, sim_state, to_sim_xy)

PUSH_Z = TABLE_Z + 0.028        # grip site height while pushing: fingertips 1.6 cm above the table. Lower, they
                                # wedge under the cube's edge and the pad friction holds it tilted
CLEAR_Z = TABLE_Z + C.CUBE_SIDE + 0.04   # height to move over the cube between pushes
FINGER_HALF = 0.014             # grip site to the closed fingers' front, with the closing axis along the face
PRESS = 0.004                   # m the target sits inside the face, so contact holds while the arm trails it
BACK_OFF = 0.03                 # m the fingers back away from the face after a push
APPROACH_BACK = 0.05            # m behind the face where they come down before a push
HOVER_S = 0.6                   # s still above that point before coming down
E_TURN = 0.02                   # m off-centre for a turning push
TURNING_DEG = 30                # a push turning my cube more than this is a turning push
KI = np.array([0.15, 0.15, 0.05])      # integral gain per step: weaker in z, where it made the descent overshoot
I_MAX = np.array([0.06, 0.06, 0.02])   # its limit (m)
PRE_ROLL, LIFT_S, TRAVEL_S, DESCEND_S, GLIDE_S, BACK_S, POST_ROLL = 2.0, 0.4, 0.8, 0.8, 0.5, 0.3, 1.0
PHASES = ("to_start", "approach", "push", "back_off", "settle")
ACTION_FORMAT = "absolute world-frame OSC target: pos (3), rotvec (3), gripper (+1 close, -1 open)"
SLIDE_FRICTION = 0.4            # plastic on cloth is about 0.3-0.5 (not measured)
POS_TOL, TURN_TOL_DEG, MAX_RISE, MAX_TILT_DEG = 0.03, 25.0, 0.01, 10.0


def make_push_env(render=True):
    return make_env(render=render, crosses=True, slide_friction=SLIDE_FRICTION)


def load_move(path):
    return dict(np.load(path))


def sim_goal(move, pos0):
    """Where the sim cube should end: its start plus my cube's displacement, turned onto the sim table."""
    return np.r_[pos0[:2] + rotz(FRAME_YAW)[:2, :2] @ (move["cube1"] - move["cube0"]), pos0[2]]


def human_turn_deg(move):
    return float(np.degrees(move["cube_yaw1"] - move["cube_yaw0"]))


def sample_path(move, k):
    """My cube's xy and yaw over push k, at the control rate, relative to the push start, in the sim's axes."""
    t0, t1 = move["pushes"][k]
    ts = np.arange(t0, t1 + DT / 2, DT)
    xy = np.stack([np.interp(ts, move["t"], move["cube_xy"][:, i]) for i in range(2)], 1)
    yaw = np.interp(ts, move["t"], move["cube_yaw"])
    return (xy - xy[0]) @ rotz(FRAME_YAW)[:2, :2].T, yaw - yaw[0]


def contact(sim_yaw, path_xy, turn):
    """Contact point and outward face normal in the cube frame for a push going along path_xy (sim axes) from a cube
    at yaw sim_yaw: the face whose outward normal points most against the motion; off-centre to the left of the
    motion for a clockwise turn (a push left of the centre turns the cube clockwise), to the right for
    counter-clockwise."""
    far = path_xy[np.argmax(np.linalg.norm(path_xy, axis=1))]
    u = rotz(-sim_yaw)[:2, :2] @ (far / max(np.linalg.norm(far), 1e-9))  # motion direction, cube frame
    axes = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    normal = axes[np.argmin(axes @ u)]
    left = rotz(np.pi / 2)[:2, :2] @ -normal
    e = 0.0 if abs(np.degrees(turn)) < TURNING_DEG else (E_TURN if turn < 0 else -E_TURN)
    return C.CUBE_SIDE / 2 * normal + e * left, normal


def finger_pose(centre, yaw, point, normal):
    """Grip-site xy and heading with the fingers on the contact point of a cube at (centre, yaw), just outside the
    face, the closing axis along the face."""
    R = rotz(yaw)[:2, :2]
    tangent = R @ (rotz(np.pi / 2)[:2, :2] @ normal)
    return centre + R @ (point + normal * (FINGER_HALF - PRESS)), np.arctan2(tangent[1], tangent[0])


def nearest_heading(h, ref):
    """The gripper is symmetric under a half turn: the heading equivalent to h closest to ref."""
    return ref + wrap(2 * (h - ref)) / 2


def run_push_replay(env, move, pos0, yaw0, record=True, video=True):
    """Returns metrics, video frames and the step log (images only if record), in sim_replay's log format."""
    env.reset()
    set_cube_pose(env, pos0, yaw0)
    log = {k: [] for k in list(sim_state(env)) + ["action", "agentview", "wrist", "phase", "clip_time"]}
    frames = []
    integral = np.zeros(3)
    at = dict(pos=eef_pose(env)[0], heading=float(np.arctan2(eef_pose(env)[1][1, 0], eef_pose(env)[1][0, 0])))

    def step(pos, heading, phase, clip_t):
        nonlocal integral
        s = sim_state(env)
        integral = np.clip(integral + KI * (pos - s["eef_pos"]), -I_MAX, I_MAX)
        action = np.r_[pos + integral, Rotation.from_matrix(gripper_down(heading)).as_rotvec(), 1.0]
        for k, v in s.items():
            log[k].append(v)
        log["action"].append(action)
        log["phase"].append(PHASES.index(phase))
        log["clip_time"].append(clip_t)
        if record:
            agent, wrist = render(env, "agentview"), render(env, WRIST_CAMERA)
            log["agentview"].append(agent)
            log["wrist"].append(wrist)
            if video:
                human = np.degrees(np.interp(clip_t, move["t"], move["cube_yaw"]) - move["cube_yaw0"])
                frames.append(label_frame(annotate(np.hstack([agent, wrist, render(env, "cubeview")]), [
                    f"clip t={clip_t:4.1f}s  {phase}",
                    f"cube turn: sim {np.degrees(np.unwrap(log['cube_yaw'])[-1] - yaw0):+6.1f}  human {human:+6.1f} deg"]),
                    False))
        env.step(action)
        at.update(pos=pos, heading=heading)

    def go(to, heading, seconds, phase, clip_t):
        n = max(1, int(round(seconds / DT)))
        for p, h in zip(ramp(at["pos"], to, n), np.linspace(at["heading"], heading, n)):
            step(p, h, phase, clip_t)

    for k, (t0, t1) in enumerate(move["pushes"]):
        path_xy, path_yaw = sample_path(move, k)
        c, yaw = cube_pose(env)
        point, normal = contact(yaw, path_xy, float(move["push_turn"][k]))
        xy, heading = zip(*[finger_pose(c[:2] + d, yaw + dy, point, normal) for d, dy in zip(path_xy, path_yaw)])
        heading = nearest_heading(heading[0], at["heading"]) + (np.unwrap(heading) - heading[0])
        start = xy[0] + rotz(yaw)[:2, :2] @ normal * APPROACH_BACK
        if k == 0:
            go(np.r_[start, CLEAR_Z], heading[0], PRE_ROLL, "to_start", t0)
        else:
            go(np.r_[at["pos"][:2], CLEAR_Z], at["heading"], LIFT_S, "approach", t0)
            go(np.r_[start, CLEAR_Z], heading[0], TRAVEL_S, "approach", t0)
        go(np.r_[start, CLEAR_Z], heading[0], HOVER_S, "approach", t0)
        go(np.r_[start, PUSH_Z], heading[0], DESCEND_S, "approach", t0)
        go(np.r_[xy[0], PUSH_Z], heading[0], GLIDE_S, "approach", t0)
        for i, (p, h) in enumerate(zip(xy, heading)):
            step(np.r_[p, PUSH_Z], h, "push", t0 + i * DT)
        away = rotz(yaw + path_yaw[-1])[:2, :2] @ normal * BACK_OFF
        go(np.r_[xy[-1] + away, PUSH_Z], heading[-1], BACK_S, "back_off", t1)
    end_t = float(move["t"][-1])
    go(np.r_[at["pos"][:2], CLEAR_Z], at["heading"], LIFT_S, "back_off", end_t)
    for _ in range(int(POST_ROLL / DT)):
        step(at["pos"], at["heading"], "settle", end_t)

    final = sim_state(env)
    log = {k: np.array(v) for k, v in log.items() if len(v)}
    metrics = push_metrics(log, final, move, pos0, yaw0)
    log.update({f"final_{k}": v for k, v in final.items()})
    return metrics, frames, log


def push_metrics(log, final, move, pos0, yaw0):
    yaw = np.unwrap(np.r_[log["cube_yaw"], final["cube_yaw"]])
    z = np.r_[log["cube_pos"][:, 2], final["cube_pos"][2]]
    up = Rotation.from_quat(np.vstack([log["cube_quat"], final["cube_quat"]])).as_matrix()[:, 2, 2]
    m = dict(pos_err_cm=100 * float(np.linalg.norm(final["cube_pos"][:2] - sim_goal(move, pos0)[:2])),
             human_turn_deg=human_turn_deg(move), sim_turn_deg=float(np.degrees(yaw[-1] - yaw0)),
             rise_cm=100 * float(z.max() - pos0[2]), tilt_deg=float(np.degrees(np.arccos(np.clip(up, -1, 1))).max()),
             pushes=len(move["pushes"]), steps=len(log["action"]))
    m["turn_err_deg"] = m["sim_turn_deg"] - m["human_turn_deg"]
    m["success"] = bool(m["pos_err_cm"] < 100 * POS_TOL and abs(m["turn_err_deg"]) < TURN_TOL_DEG
                        and m["rise_cm"] < 100 * MAX_RISE and m["tilt_deg"] < MAX_TILT_DEG)
    return m


def start_pose(move, rng=None, cross_jitter=0.0, yaw_jitter=0.0):
    """The sim cube where mine started (moved onto the sim table), turned as mine was; with rng, anywhere within
    cross_jitter of the sim copy of my start cross, turned within yaw_jitter of mine."""
    xy = to_sim_xy(move["cube0"])
    yaw = float(move["cube_yaw0"]) + FRAME_YAW
    if rng is not None:
        xy = SIM_CROSSES[str(move["start_cross"])] + rng.uniform(-cross_jitter, cross_jitter, 2)
        yaw += rng.uniform(-yaw_jitter, yaw_jitter)
    return np.r_[xy, TABLE_Z + C.CUBE_SIDE / 2], yaw


def save_push_demo(path, log, metrics, move_name, pos0, yaw0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **log, success=metrics["success"], source_clip=move_name, sim_cube_pos0=pos0,
                        sim_cube_yaw0=yaw0, dt=DT, phase_names=np.array(PHASES), action_format=ACTION_FORMAT,
                        **{f"metric_{k}": v for k, v in metrics.items()})


def format_metrics(m):
    return (f"pos err {m['pos_err_cm']:.1f} cm | turn sim {m['sim_turn_deg']:+.0f} vs human {m['human_turn_deg']:+.0f} deg"
            f" | rise {m['rise_cm']:.1f} cm | tilt {m['tilt_deg']:.0f} deg | {m['pushes']} pushes, {m['steps']} steps"
            f" | success {m['success']}")


def _replay(path):
    move = load_move(path)
    env = make_push_env(render=False)
    m, _, _ = run_push_replay(env, move, *start_pose(move), record=False)
    env.close()
    return dict(move=Path(path).stem, kind=str(move["kind"]), **{k: round(v, 1) if isinstance(v, float) else v
                                                                  for k, v in m.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("move", nargs="?", help="processed move, e.g. data/push/processed/push_01_AB_straight.npz")
    p.add_argument("--all", action="store_true", help="replay every move without video")
    p.add_argument("--processed", default=str(ROOT / "data" / "push" / "processed"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=str(ROOT / "outputs" / "push" / "sim_replay"))
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    if args.all:
        with multiprocessing.Pool(args.workers) as pool:
            rows = pool.map(_replay, sorted(Path(args.processed).glob("push_*.npz")))
        table = ROOT / "outputs" / "push" / "replay.csv"
        with open(table, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for kind in ("straight", "cw", "ccw"):
            rs = [r for r in rows if r["kind"] == kind]
            print(f"{kind:8s}: {sum(r['success'] for r in rs)}/{len(rs)} succeed | median pos err "
                  f"{np.median([r['pos_err_cm'] for r in rs]):.1f} cm | median turn sim/human "
                  f"{np.median([r['sim_turn_deg'] / r['human_turn_deg'] for r in rs]):.2f} | max tilt "
                  f"{max(r['tilt_deg'] for r in rs):.0f} deg")
        print(f"wrote {table}")
        return
    move = load_move(args.move)
    env = make_push_env(render=not args.no_video)
    metrics, frames, _ = run_push_replay(env, move, *start_pose(move), record=not args.no_video,
                                         video=not args.no_video)
    print(Path(args.move).stem, format_metrics(metrics))
    if frames:
        print("saved", save_video(frames, args.out, Path(args.move).stem, gif=False))
    env.close()


if __name__ == "__main__":
    main()
