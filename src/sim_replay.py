"""Replay a processed human clip on a simulated Panda (robosuite Lift, OSC controller).

    python -m src.sim_replay data/processed/<clip>.npz [--cube-xy X Y] [--cube-yaw DEG] [--raw-wrist]

This is a scripted controller driven by the human data, not a learned policy: the
retargeted gripper pose is sent each step as an absolute world-frame target to
robosuite's OSC controller, and the gripper closes / opens at the human grasp /
release times. The human data drives grasp to release (the carry and the turn); the
approach and retreat are scripted, straight down onto the grasp point and straight up,
because a real hand comes in from the side at table height (see the README). Every replay, including failures, is saved to <data>/sim_demos/ with
agentview and wrist images, robot and cube state, and the action sent at each step.
The input is a processed clip from extract.py (poses in the table frame, yaws unwrapped,
NaN where untracked).
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import cv2
import imageio.v2 as imageio
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation

from src import constants as C
from src.common import label_frame
from src.geometry import gripper_down, rotz, wrap, yaw_of

ROOT = Path(__file__).resolve().parent.parent
DT = 0.05            # 20 Hz control
DWELL = 0.6          # s the arm holds still at grasp and release while the fingers move
PRE_ROLL = 2.0       # s to move from the robot's home pose to above the grasp point
APPROACH_H = 0.10    # m: the gripper comes straight down onto the grasp point from this height, and leaves the same way
DESCEND_S = 1.0      # s for that descent (and for the retreat)
POST_ROLL = 1.0      # s to let the cube settle before measuring
CUBE_MASS = 0.09     # kg, typical for a Rubik's cube (not weighed)

# Scene geometry and agentview copied from LIBERO's tabletop scenes
# (LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py:190): a 1.0 x 1.2 m
# table with its top at z = 0.9, i.e. 1.2 cm below the robot base (robosuite's Lift: 11.2 cm).
TABLE_SIZE = (1.0, 1.2, 0.05)
TABLE_Z = 0.9
ROBOT_BASE_XY = np.array([-0.66, 0.0])  # robosuite places the Panda 0.16 m behind the table edge
AGENTVIEW_POS = (0.6586131746834771, 0.0, 1.6103500240372423)
AGENTVIEW_QUAT = (0.6380177736282349, 0.3048497438430786, 0.3048497438430786, 0.6380177736282349)  # wxyz
WRIST_CAMERA = "robot0_eye_in_hand"     # identical in LIBERO (robosuite 1.4) and robosuite 1.5.2
CUBE_HOME_XY = np.array([-0.10, 0.0])   # 0.56 m in front of the base, as in robosuite's default Lift

MIN_TARGET_Z = TABLE_Z + 0.02
MAX_REACH = 0.70     # horizontal; beyond ~0.75 m the arm nears full extension and OSC tracking degrades
LIFT_MIN, POS_TOL, YAW_TOL_DEG = 0.02, 0.03, 15.0
IMAGE_SIZE = 256
CUBEVIEW_POS = np.r_[CUBE_HOME_XY + [0.40, -0.32], TABLE_Z + 0.32]
CUBEVIEW_TARGET = np.r_[CUBE_HOME_XY, TABLE_Z + 0.05]
PHASES = ("to_start", "approach", "closing", "carry", "releasing", "retreat", "settle")
ACTION_FORMAT = "absolute world-frame OSC target: pos (3), rotvec (3), gripper (+1 close, -1 open)"
# Face colours in the cube frame: +z white, +x red (ID 2), +y blue (ID 3), -x orange, -y green, -z yellow
FACES = [(2, 1, "1 1 1 1"), (0, 1, "0.8 0.1 0.1 1"), (1, 1, "0.1 0.3 0.9 1"),
         (0, -1, "1 0.5 0 1"), (1, -1, "0.1 0.7 0.2 1"), (2, -1, "1 0.9 0 1")]


def customise_xml(xml):
    """Resize Lift's cube to the real cube, set its mass, colour each face so yaw shows in renders,
    match LIBERO's table texture and agentview pose, and add a close camera on the cube."""
    root = ET.fromstring(xml)
    for tex in root.iter("texture"):
        if tex.get("name") == "tex-ceramic":  # light wood like LIBERO; a white table hides the cube's white top
            tex.set("file", str(Path(tex.get("file")).with_name("light-wood.png")))
    for cam in root.iter("camera"):
        if cam.get("name") == "agentview":
            cam.set("pos", " ".join(map(str, AGENTVIEW_POS)))
            cam.set("quat", " ".join(map(str, AGENTVIEW_QUAT)))
    forward = (CUBEVIEW_TARGET - CUBEVIEW_POS) / np.linalg.norm(CUBEVIEW_TARGET - CUBEVIEW_POS)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    ET.SubElement(root.find("worldbody"), "camera", name="cubeview", pos=" ".join(map(str, CUBEVIEW_POS)),
                  xyaxes=" ".join(map(str, np.r_[right, np.cross(right, forward)])), fovy="40")
    h = C.CUBE_SIDE / 2
    for geom in root.iter("geom"):
        if geom.get("name") in ("cube_g0", "cube_g0_vis"):
            geom.set("size", f"{h} {h} {h}")
        if geom.get("name") == "cube_g0":
            geom.set("density", str(CUBE_MASS / C.CUBE_SIDE ** 3))
    body = next(b for b in root.iter("body") if b.get("name") == "cube_main")
    for i, (axis, sign, rgba) in enumerate(FACES):
        pos, size = np.zeros(3), np.full(3, 0.9 * h)
        pos[axis], size[axis] = sign * (h + 0.0005), 0.0005
        ET.SubElement(body, "geom", name=f"cube_face{i}", type="box", pos=" ".join(map(str, pos)),
                      size=" ".join(map(str, size)), rgba=rgba, contype="0", conaffinity="0", group="1", mass="0")
    return ET.tostring(root, encoding="unicode")


def make_env(render=True):
    cfg = load_composite_controller_config(robot="Panda")
    cfg["body_parts"]["right"].update(input_type="absolute", input_ref_frame="world")
    env = suite.make("Lift", robots="Panda", controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=render, use_camera_obs=False, ignore_done=True,
                     control_freq=int(round(1 / DT)), initialization_noise=None, table_full_size=TABLE_SIZE)
    # Lift hard-codes its table height, so set it and rebuild the model as a hard reset would.
    # This creates new robot objects; reset() only re-binds observables to them while hard_reset is on.
    env.table_offset = np.array([0.0, 0.0, TABLE_Z])
    env._load_model()
    env.reset_from_xml_string(customise_xml(env.model.get_xml()))
    env.hard_reset = False  # later resets keep this model instead of rebuilding the default one
    assert np.allclose(env.sim.data.get_body_xpos("robot0_base")[:2], ROBOT_BASE_XY)
    return env


def set_cube_pose(env, pos, yaw):
    qpos = env.sim.model.get_joint_qpos_addr("cube_joint0")
    qvel = env.sim.model.get_joint_qvel_addr("cube_joint0")
    env.sim.data.qpos[qpos[0]:qpos[1]] = np.r_[pos, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    env.sim.data.qvel[qvel[0]:qvel[1]] = 0.0
    env.sim.forward()


def cube_pose(env):
    i = env.sim.model.body_name2id("cube_main")
    return env.sim.data.body_xpos[i].copy(), yaw_of(env.sim.data.body_xmat[i].reshape(3, 3))


def eef_pose(env):
    site = env.robots[0].eef_site_id["right"]
    return env.sim.data.site_xpos[site].copy(), env.sim.data.site_xmat[site].reshape(3, 3).copy()


def sim_state(env):
    """Robot and cube state as logged per step (quaternions x, y, z, w)."""
    eef_pos, eef_R = eef_pose(env)
    fingers = env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes["right"]]
    i = env.sim.model.body_name2id("cube_main")
    cube_R = env.sim.data.body_xmat[i].reshape(3, 3)
    return dict(eef_pos=eef_pos, eef_quat=Rotation.from_matrix(eef_R).as_quat(), eef_yaw=yaw_of(eef_R),
                gripper_opening=fingers[0] - fingers[1],
                cube_pos=env.sim.data.body_xpos[i].copy(), cube_quat=Rotation.from_matrix(cube_R).as_quat(),
                cube_yaw=yaw_of(cube_R))


def human_signals(clip):
    """Rest poses before grasp / after release (from extract.py) and the tracked yaw series."""
    return dict(cube_yaw=clip["cube_yaw"], hand_yaw=clip["hand_yaw"], cube0=clip["cube0"],
                cube_yaw0=float(clip["cube_yaw0"]), cube1=clip["cube1"], cube_yaw1=float(clip["cube_yaw1"]))


def hold_mask(clip):
    return (clip["t"] >= float(clip["t_grasp"])) & (clip["t"] <= float(clip["t_release"]))


def face_costs(clip, sim_cube_yaw, heading_home):
    """The four grasp headings (one per face pair orientation) and each one's largest distance from the home
    heading over the hold."""
    t, hv = clip["t"], clip["hand_visible"]
    hand_yaw = np.interp(t, t[hv], clip["hand_yaw"][hv])
    dyaw = hand_yaw - np.interp(float(clip["t_grasp"]), t, hand_yaw)
    faces = sim_cube_yaw + np.arange(4) * np.pi / 2
    return faces, np.array([np.max(np.abs(wrap(f + dyaw[hold_mask(clip)] - heading_home))) for f in faces])


def retarget(clip, sim_cube_pos, sim_cube_yaw, heading_home, use_grasp_point=True, face_rng=None,
             face_margin=0.0):
    """Gripper targets (pos (N,3), closing-axis heading (N,), clamped (N,)) at the clip's sample times.

    Grasp-anchored, 4-DoF: the tracked point's displacement from the grasp instant and the
    hand yaw change, expressed in the human cube's start frame, are applied around the sim
    cube. The tracked point is the grasp point (fixed to the hand, at the cube centre at
    grasp) or, with use_grasp_point=False, the wrist marker itself.
    Frames without a hand detection are linearly bridged (bridged_pct in the metrics).
    """
    h = human_signals(clip)
    t, hv = clip["t"], clip["hand_visible"]
    hand_pos = np.stack([np.interp(t, t[hv], clip["hand_pos"][hv, k]) for k in range(3)], axis=1)
    hand_yaw = np.interp(t, t[hv], h["hand_yaw"][hv])
    tg = float(clip["t_grasp"])
    hp_g = np.array([np.interp(tg, t, hand_pos[:, k]) for k in range(3)])
    hy_g = np.interp(tg, t, hand_yaw)

    if use_grasp_point:
        offset = rotz(-hy_g) @ (h["cube0"] - hp_g)  # cube centre in the hand's yaw frame at grasp
        point = hand_pos + np.einsum("nij,j->ni", np.array([rotz(y) for y in hand_yaw]), offset)
    else:
        point = hand_pos - hp_g + h["cube0"]
    disp = point - h["cube0"]  # zero at the grasp instant
    R = rotz(sim_cube_yaw) @ rotz(-h["cube_yaw0"])
    pos = sim_cube_pos + disp @ R.T
    radial = pos[:, :2] - ROBOT_BASE_XY
    reach = np.linalg.norm(radial, axis=1)
    clamped = (reach > MAX_REACH) | (pos[:, 2] < MIN_TARGET_Z)
    pos[:, :2] = ROBOT_BASE_XY + radial * np.minimum(1.0, MAX_REACH / reach)[:, None]
    pos[:, 2] = np.maximum(pos[:, 2], MIN_TARGET_Z)

    # Grasp on the cube face whose heading keeps the hold closest to the home heading. With face_rng, any face
    # within face_margin of the best is drawn at random: near the cube yaws where the best face switches, two
    # grasps 90 deg apart are equally good, and demos that always took one made the policy average them
    # into a corner grasp (see the README).
    dyaw = hand_yaw - hy_g
    faces, cost = face_costs(clip, sim_cube_yaw, heading_home)
    best = int(np.argmin(cost))
    if face_rng is not None:
        best = int(face_rng.choice(np.flatnonzero(cost <= cost.min() + face_margin)))
    heading = heading_home + wrap(faces[best] - heading_home) + dyaw  # continuous, near home
    return pos, heading, clamped


def with_rotation(clip, phi):
    """The template clip with its hand rotation rescaled to phi radians. The wrist positions are moved so the
    grasp point (the hand point at the cube centre at grasp, see sim_replay.retarget) keeps its path:
    rescaling the yaw alone would swing that point around the wrist and misplace the cube."""
    out = dict(clip)
    t, hv = clip["t"], clip["hand_visible"]
    tg = float(clip["t_grasp"])
    yaw_g = np.interp(tg, t[hv], clip["hand_yaw"][hv])
    pos_g = np.array([np.interp(tg, t[hv], clip["hand_pos"][hv, k]) for k in range(3)])
    offset = rotz(-yaw_g) @ (clip["cube0"] - pos_g)
    new_yaw = yaw_g + phi * clip["hand_yaw_rel"] / float(clip["hand_yaw_change"])
    new_pos = np.full_like(clip["hand_pos"], np.nan)
    for i in np.flatnonzero(hv):
        point = clip["hand_pos"][i] + rotz(clip["hand_yaw"][i]) @ offset
        new_pos[i] = point - rotz(new_yaw[i]) @ offset
    out["hand_yaw"] = np.where(hv, new_yaw, np.nan)
    out["hand_pos"] = new_pos
    return out


MIN_HAND_TURN = np.radians(10)  # below this the hand's own turn is too small to rescale


def with_cube_turn(clip):
    """The clip with the hand's turn while held rescaled to the cube's measured turn (from its rest poses, so it is
    known even when the cube is hidden while held): the robot reproduces the object's turn with the hand's timing.
    The human fingers add 16% (counter-clockwise) to 51% (clockwise) to the hand's turn, which a parallel gripper
    copying the hand cannot (see the README)."""
    if abs(float(clip["hand_yaw_change"])) < MIN_HAND_TURN:
        return clip
    return with_rotation(clip, float(clip["cube_yaw_change"]))


def hand_span(clip):
    """First and last time the hand was tracked."""
    t = clip["t"][clip["hand_visible"]]
    return float(t[0]), float(t[-1])


def hold_schedule(clip):
    """Clip time and gripper state of the human-driven steps: grasp to release, with dwells at both ends while
    the fingers move."""
    tg, tr = float(clip["t_grasp"]), float(clip["t_release"])
    n = int(round(DWELL / DT))
    ht = np.arange(tg, tr, DT)
    return np.r_[np.full(n, tg), ht, np.full(n, tr)], np.r_[np.ones(n + len(ht), bool), np.zeros(n, bool)]


def phase_name(h, closed, tg, tr):
    if h < tg:
        return "approach"
    if h == tg:
        return "closing"
    if closed:
        return "carry"
    return "releasing" if h == tr else "retreat"


def annotate(frame, lines):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def render(env, camera):
    return env.sim.render(width=IMAGE_SIZE, height=IMAGE_SIZE, camera_name=camera)[::-1]  # OpenGL rows are flipped


def ramp(a, b, n):
    """n cosine-eased steps on the straight line from a to b."""
    w = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n))
    return a + np.multiply.outer(w, np.asarray(b) - a)


def plan_steps(env, clip, sim_cube_pos, sim_cube_yaw, use_grasp_point, face_rng=None, face_margin=0.0,
               approach_h=APPROACH_H, hover_s=0.0):
    """Per-step targets: home to above the grasp point, straight down, close; the retargeted human hold; open,
    straight up, and hold while the cube settles. hover_s pauses above the grasp point before going down: the arm
    trails its target by 2-3 cm, so without it the fingers start down while still coming in from the side, which
    a policy copying the demos turns into pushing the cube (see the README)."""
    eef0, R0 = eef_pose(env)
    heading_home = yaw_of(R0)
    pos_t, heading_t, clamped = retarget(clip, sim_cube_pos, sim_cube_yaw, heading_home, use_grasp_point, face_rng,
                                         face_margin)
    ht, closed = hold_schedule(clip)
    tg, tr = float(clip["t_grasp"]), float(clip["t_release"])
    hold_pos = np.stack([np.interp(ht, clip["t"], pos_t[:, k]) for k in range(3)], axis=1)
    hold_heading = np.interp(ht, clip["t"], heading_t)
    up = np.array([0.0, 0.0, approach_h])
    grasp, release = hold_pos[0], hold_pos[-1]
    n_pre, n_move, n_post = int(PRE_ROLL / DT), int(DESCEND_S / DT), int(POST_ROLL / DT)
    n_hover = int(round(hover_s / DT))
    n_in = n_hover + n_move
    phases = (["to_start"] * n_pre + ["approach"] * n_in + [phase_name(h, c, tg, tr) for h, c in zip(ht, closed)]
              + ["retreat"] * n_move + ["settle"] * n_post)
    return dict(
        pos=np.vstack([ramp(eef0, grasp + up, n_pre), np.repeat([grasp + up], n_hover, 0), ramp(grasp + up, grasp, n_move),
                       hold_pos, ramp(release, release + up, n_move), np.repeat([release + up], n_post, 0)]),
        heading=np.r_[ramp(heading_home, hold_heading[0], n_pre), np.full(n_in, hold_heading[0]), hold_heading,
                      np.full(n_move + n_post, hold_heading[-1])],
        closed=np.r_[np.zeros(n_pre + n_in, bool), closed, np.zeros(n_move + n_post, bool)],
        clip_time=np.r_[np.full(n_pre + n_in, tg), ht, np.full(n_move + n_post, tr)],
        phase=np.array([PHASES.index(p) for p in phases]),
        run=slice(n_pre + n_in, n_pre + n_in + len(ht)),
        clamped_pct=100 * clamped[hold_mask(clip)].mean())


def bridged_pct(clip):
    t0, t1 = hand_span(clip)
    inside = (clip["t"] >= t0) & (clip["t"] <= t1)
    return 100 * (1 - clip["hand_visible"][inside].mean())


def run_replay(env, clip, sim_cube_pos, sim_cube_yaw, use_grasp_point=True, record=True, synthetic=False, perturb=None,
               video=True, face_rng=None, face_margin=0.0, approach_h=APPROACH_H, hover_s=0.0):
    """Returns metrics, video frames (if record and video) and the step log (images only if record).
    perturb(plan) -> (N, 4) offsets (dx, dy, dz, dheading) added to the executed targets only; the logged action
    stays the planned target, so it points back on course (make_sim_demos --noise)."""
    env.reset()
    set_cube_pose(env, sim_cube_pos, sim_cube_yaw)
    plan = plan_steps(env, clip, sim_cube_pos, sim_cube_yaw, use_grasp_point, face_rng, face_margin, approach_h, hover_s)
    offsets = perturb(plan) if perturb else np.zeros((len(plan["pos"]), 4))
    h = human_signals(clip)
    visible = clip["cube_visible"]

    log = {k: [] for k in list(sim_state(env)) + ["action", "agentview", "wrist"]}
    frames = []
    for i in range(len(plan["pos"])):
        rotvec = Rotation.from_matrix(gripper_down(plan["heading"][i])).as_rotvec()
        action = np.r_[plan["pos"][i], rotvec, 1.0 if plan["closed"][i] else -1.0]
        executed = np.r_[plan["pos"][i] + offsets[i, :3],
                         Rotation.from_matrix(gripper_down(plan["heading"][i] + offsets[i, 3])).as_rotvec(), action[6]]
        for k, v in sim_state(env).items():  # observation before the action is applied
            log[k].append(v)
        log["action"].append(action)
        if record:
            agent, wrist = render(env, "agentview"), render(env, WRIST_CAMERA)
            log["agentview"].append(agent)
            log["wrist"].append(wrist)
        if record and video:
            yaw = np.unwrap(log["cube_yaw"])
            hum = np.interp(plan["clip_time"][i], clip["t"][visible], h["cube_yaw"][visible]) - h["cube_yaw0"]

            frames.append(label_frame(annotate(np.hstack([agent, wrist, render(env, "cubeview")]), [
                f"clip t={plan['clip_time'][i]:4.1f}s  {PHASES[plan['phase'][i]]}",
                f"cube yaw change: sim {np.degrees(yaw[-1] - yaw[0]):+5.1f}"
                f"  human {np.degrees(hum):+5.1f} deg"]), synthetic))
        env.step(executed)

    final = sim_state(env)
    log = {k: np.array(v) for k, v in log.items() if len(v)}
    metrics = replay_metrics(log, final, h, sim_cube_pos, sim_cube_yaw, plan)
    metrics["bridged_pct"] = bridged_pct(clip)
    log.update(phase=plan["phase"], clip_time=plan["clip_time"], **{f"final_{k}": v for k, v in final.items()})
    return metrics, frames, log


def replay_metrics(log, final, h, sim_cube_pos, sim_cube_yaw, plan):
    start = 0  # the cube at rest, before the approach: a push while approaching counts against the replay
    cube_yaw = np.unwrap(np.r_[log["cube_yaw"], final["cube_yaw"]])
    expected = sim_cube_pos + rotz(sim_cube_yaw) @ rotz(-h["cube_yaw0"]) @ (h["cube1"] - h["cube0"])
    human_dyaw = np.degrees(h["cube_yaw1"] - h["cube_yaw0"])
    sim_dyaw = np.degrees(cube_yaw[-1] - cube_yaw[start])
    held = np.flatnonzero(plan["closed"])
    # Slip: how much the cube's yaw lagged the gripper's yaw while held
    eef_yaw = np.unwrap(log["eef_yaw"])
    slip = (eef_yaw[held[-1]] - eef_yaw[held[0]]) - (cube_yaw[held[-1]] - cube_yaw[held[0]])
    eef_after_step = np.vstack([log["eef_pos"][1:], final["eef_pos"]])
    m = dict(
        lift_cm=100 * (max(log["cube_pos"][:, 2].max(), final["cube_pos"][2]) - log["cube_pos"][start, 2]),
        pos_err_cm=100 * np.linalg.norm(final["cube_pos"] - expected),
        human_dyaw_deg=human_dyaw,
        sim_dyaw_deg=sim_dyaw,
        slip_deg=np.degrees(slip),
        track_err_mm=1000 * np.mean(np.linalg.norm(eef_after_step[plan["run"]] - plan["pos"][plan["run"]], axis=1)),
        clamped_pct=plan["clamped_pct"],
    )
    m["lifted"] = bool(m["lift_cm"] >= 100 * LIFT_MIN)
    m["success"] = bool(m["lifted"] and m["pos_err_cm"] < 100 * POS_TOL
                        and abs(sim_dyaw - human_dyaw) < YAW_TOL_DEG)
    return m


def demo_path(demo_dir, clip_name, sim_cube_pos, sim_cube_yaw, use_grasp_point):
    tag = f"x{sim_cube_pos[0]:+.3f}_y{sim_cube_pos[1]:+.3f}_yaw{np.degrees(sim_cube_yaw):+04.0f}"
    return Path(demo_dir) / f"{clip_name}__{tag}{'' if use_grasp_point else '_rawwrist'}.npz"


def save_demo(path, log, metrics, clip_name, sim_cube_pos, sim_cube_yaw, use_grasp_point):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **log, success=metrics["success"], source_clip=clip_name,
                        sim_cube_pos0=sim_cube_pos, sim_cube_yaw0=sim_cube_yaw, use_grasp_point=use_grasp_point,
                        dt=DT, phase_names=np.array(PHASES), action_format=ACTION_FORMAT,
                        **{f"metric_{k}": v for k, v in metrics.items()})


def save_video(frames, out_dir, stem, gif=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out / f"{stem}.mp4", frames, fps=int(round(1 / DT)))
    if gif:
        small = [cv2.resize(f, (f.shape[1] * 3 // 4, f.shape[0] * 3 // 4), interpolation=cv2.INTER_AREA)
                 for f in frames[::3]]
        imageio.mimsave(out / f"{stem}.gif", small, duration=3 * DT * 1000, loop=0)  # duration in ms per frame
    return out / f"{stem}.mp4"


def format_metrics(m):
    return (f"lift {m['lift_cm']:.1f} cm | pos err {m['pos_err_cm']:.1f} cm | cube yaw change sim "
            f"{m['sim_dyaw_deg']:+.1f} vs human {m['human_dyaw_deg']:+.1f} deg | slip {m['slip_deg']:+.1f} deg | "
            f"eef tracking {m['track_err_mm']:.1f} mm | clamped {m['clamped_pct']:.0f}% | hand bridged "
            f"{m.get('bridged_pct', 0):.0f}% | success {m['success']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("clip", help="processed clip, e.g. data/processed/clip_07_rot+60_B.npz")
    p.add_argument("--cube-xy", type=float, nargs=2, default=tuple(CUBE_HOME_XY), help="sim cube start, metres")
    p.add_argument("--cube-yaw", type=float, default=0.0, help="sim cube start yaw, degrees")
    p.add_argument("--raw-wrist", action="store_true", help="anchor on the wrist marker, not the grasp point")
    p.add_argument("--out", default=None, help="video folder (default outputs/<real|synthetic>/sim_replay)")
    args = p.parse_args()

    from src.common import is_synthetic
    clip_path = Path(args.clip).resolve()
    synthetic = is_synthetic(clip_path)
    out = Path(args.out) if args.out else ROOT / "outputs" / ("synthetic" if synthetic else "real") / "sim_replay"
    clip_name = clip_path.stem
    clip = dict(np.load(clip_path))
    env = make_env()
    sim_cube_pos = np.array([*args.cube_xy, TABLE_Z + C.CUBE_SIDE / 2])
    sim_cube_yaw = np.radians(args.cube_yaw)
    metrics, frames, log = run_replay(env, clip, sim_cube_pos, sim_cube_yaw, not args.raw_wrist, synthetic=synthetic)
    demo = demo_path(clip_path.parent.parent / "sim_demos", clip_name, sim_cube_pos, sim_cube_yaw, not args.raw_wrist)
    save_demo(demo, log, metrics, clip_name, sim_cube_pos, sim_cube_yaw, not args.raw_wrist)
    mp4 = save_video(frames, out, clip_name + ("" if not args.raw_wrist else "_rawwrist"))
    print(format_metrics(metrics))
    print(f"saved {demo} ({demo.stat().st_size / 1e6:.1f} MB, {len(log['action'])} steps) and {mp4} (+ .gif)")
    env.close()


if __name__ == "__main__":
    main()
