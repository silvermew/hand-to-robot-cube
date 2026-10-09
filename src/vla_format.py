"""State / action format shared by the demo converter (lerobot venv) and the closed-loop evaluator.

numpy only, so it imports in both environments.
  state  (6): end-effector x, y, z (robosuite world, m), sin and cos of the gripper heading, gripper opening (m),
              plus with clock=True the time since the episode started (s): the scripted demos keep a fixed
              schedule (stop above the cube, then go down), which one image cannot tell apart
  action (5): dx, dy, dz (m), d heading (rad) toward the next target, gripper (+1 close, -1 open)
Roll and pitch are fixed (gripper pointing down), as in the scripted replay. Absolute rotation
vectors are not used as targets: for a gripper pointing down they sit near pi and flip sign.
"""
import numpy as np

STATE_NAMES = ["eef_x", "eef_y", "eef_z", "heading_sin", "heading_cos", "gripper_opening"]
CLOCK_NAME = "time_s"
FPS = 20  # the sim control rate
ACTION_NAMES = ["dx", "dy", "dz", "dheading", "gripper"]
MAX_STEP_POS, MAX_STEP_HEADING = 0.05, 0.3


def rotvec_to_heading(rotvec):
    """Heading of the x axis of the rotation (Rodrigues), i.e. the gripper heading for gripper_down()."""
    rotvec = np.asarray(rotvec, float)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    k = np.divide(rotvec, theta, out=np.zeros_like(rotvec), where=theta > 1e-12)
    c, s = np.cos(theta[..., 0]), np.sin(theta[..., 0])
    # first column of R = cos(t) e_x + sin(t) (k x e_x) + (1 - cos(t)) k_x k
    x = np.stack([c + (1 - c) * k[..., 0] ** 2, s * k[..., 2] + (1 - c) * k[..., 0] * k[..., 1]], -1)
    return np.arctan2(x[..., 1], x[..., 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def state_names(clock=False):
    return STATE_NAMES + [CLOCK_NAME] if clock else STATE_NAMES


def make_state(eef_pos, eef_yaw, gripper_opening, t=None):
    """States (N, 6), or (N, 7) with the time t (s) since the episode started."""
    eef_pos = np.atleast_2d(eef_pos)
    eef_yaw, gripper_opening = np.atleast_1d(eef_yaw), np.atleast_1d(gripper_opening)
    cols = [eef_pos, np.sin(eef_yaw), np.cos(eef_yaw), gripper_opening] + ([np.atleast_1d(t)] if t is not None else [])
    return np.column_stack(cols).astype(np.float32)


def demo_states_actions(demo, clock=False):
    """(T, 6 or 7) states and (T, 5) delta actions from a saved replay (absolute OSC targets were logged)."""
    t = np.arange(len(demo["eef_pos"])) / FPS if clock else None
    states = make_state(demo["eef_pos"], demo["eef_yaw"], demo["gripper_opening"], t)
    target_heading = rotvec_to_heading(demo["action"][:, 3:6])
    actions = np.c_[demo["action"][:, :3] - demo["eef_pos"], wrap(target_heading - demo["eef_yaw"]),
                    demo["action"][:, 6]].astype(np.float32)
    return states, actions


def action_to_target(eef_pos, eef_yaw, action):
    """Absolute target position, target heading and gripper command for a delta action (clipped)."""
    action = np.asarray(action, float)
    pos = eef_pos + np.clip(action[:3], -MAX_STEP_POS, MAX_STEP_POS)
    heading = eef_yaw + np.clip(action[3], -MAX_STEP_HEADING, MAX_STEP_HEADING)
    return pos, heading, 1.0 if action[4] > 0 else -1.0
