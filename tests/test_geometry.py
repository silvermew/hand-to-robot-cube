"""Rotation-math checks: yaw conventions, wrapping / unwrapping, quaternions, frame changes.

    python -m pytest tests -q
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src import constants as C
from src.geometry import (align_quat_signs, angle_between, gripper_down, hand_yaw_of, heading, invert_pose,
                          make_pose, marker_in_cube, mean_rotation, quat_from_R, R_from_quat, rotz,
                          smooth_quats, unwrap_valid, wrap, yaw_of)


def test_yaw_is_counter_clockwise_from_above():
    assert np.isclose(yaw_of(rotz(np.radians(30))), np.radians(30))
    assert np.isclose(heading(np.array([0.0, 1.0, 0.0])), np.pi / 2)  # +y is 90 deg CCW from +x


def test_hand_yaw_is_heading_of_marker_top_edge():
    # marker +y (top edge, toward the fingertips) along table +x -> hand yaw 0
    R = rotz(-np.pi / 2)
    assert np.allclose(R @ [0, 1, 0], [1, 0, 0])
    assert np.isclose(hand_yaw_of(R), 0.0)
    assert np.isclose(hand_yaw_of(rotz(0.3)), np.pi / 2 + 0.3)


def test_wrap_range():
    a = np.linspace(-10, 10, 101)
    w = wrap(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))


def test_unwrap_valid_crosses_pi_and_keeps_gaps():
    true = np.linspace(0, np.radians(400), 50)
    valid = np.ones(50, bool)
    valid[20:24] = False
    out = unwrap_valid(wrap(true), valid)
    assert np.all(np.isnan(out[20:24]))
    assert np.allclose(out[valid], true[valid])


def test_quaternion_round_trip_and_xyzw_order():
    R = Rotation.from_euler("zyx", [40, 10, -20], degrees=True).as_matrix()
    q = quat_from_R(R)
    assert np.allclose(R_from_quat(q), R)
    assert np.allclose(quat_from_R(np.eye(3)), [0, 0, 0, 1])  # scalar last


def test_pose_inverse_and_frame_change():
    T = make_pose(rotz(0.7), [0.1, -0.2, 0.3])
    assert np.allclose(T @ invert_pose(T), np.eye(4))
    p_local = np.array([0.05, 0.0, 0.0, 1.0])
    assert np.allclose((T @ p_local)[:3], [0.1, -0.2, 0.3] + rotz(0.7) @ [0.05, 0, 0])


@pytest.mark.parametrize("marker_id", C.CUBE_IDS)
def test_cube_markers_are_proper_rotations_on_their_faces(marker_id):
    T = marker_in_cube(marker_id)
    R = T[:3, :3]
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(T[:3, 3], R[:, 2] * C.CUBE_SIDE / 2)  # face centre along the outward normal


def test_cube_marker_layout_convention():
    up = np.array([0, 0, 1.0])
    assert np.allclose(marker_in_cube(1)[:3, 1], [-1, 0, 0])  # ID 1 top edge toward the ID 4 face (as glued)
    for i in C.CUBE_SIDE_IDS:
        assert np.allclose(marker_in_cube(i)[:3, 1], up)      # side markers upright
    normals = [heading(marker_in_cube(i)[:3, 2]) for i in C.CUBE_SIDE_IDS]
    assert np.allclose(np.diff(np.unwrap(normals)), np.pi / 2)  # 2 -> 3 -> 4 -> 5 counter-clockwise


def test_gripper_down_points_down_with_given_heading():
    R = gripper_down(0.4)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], [0, 0, -1])
    assert np.isclose(yaw_of(R), 0.4)


def test_quaternion_smoothing_handles_sign_flips():
    q = np.array([quat_from_R(rotz(a)) for a in np.linspace(0, 0.5, 20)])
    q[1::2] *= -1  # same rotations, alternating signs
    s = smooth_quats(q, 5)
    angles = [yaw_of(R_from_quat(x)) for x in s]
    assert np.all(np.diff(angles) > 0)
    assert np.allclose(np.linalg.norm(s, axis=1), 1.0)
    assert np.all(np.sum(align_quat_signs(q)[1:] * align_quat_signs(q)[:-1], axis=1) > 0)


def test_mean_rotation_and_angle_between():
    Rs = [rotz(a) for a in np.radians([9, 10, 11])]
    assert np.isclose(np.degrees(angle_between(mean_rotation(Rs), rotz(np.radians(10)))), 0.0, atol=1e-6)


def test_vla_delta_actions_round_trip_to_logged_targets():
    from src.vla_format import action_to_target, demo_states_actions, rotvec_to_heading
    rng = np.random.default_rng(0)
    headings = rng.uniform(-np.pi, np.pi, 20)
    rotvecs = np.array([Rotation.from_matrix(gripper_down(h)).as_rotvec() for h in headings])
    assert np.allclose(np.cos(rotvec_to_heading(rotvecs) - headings), 1.0)
    eef_pos = rng.uniform(-0.2, 0.2, (20, 3))
    eef_yaw = headings + rng.uniform(-0.2, 0.2, 20)
    target = eef_pos + rng.uniform(-0.04, 0.04, (20, 3))
    demo = dict(eef_pos=eef_pos, eef_yaw=eef_yaw, gripper_opening=np.full(20, 0.05),
                action=np.c_[target, rotvecs, np.ones(20)])
    states, actions = demo_states_actions(demo)
    assert states.shape == (20, 6) and actions.shape == (20, 5)
    for i in range(20):
        pos, heading, grip = action_to_target(eef_pos[i], eef_yaw[i], actions[i])
        assert np.allclose(pos, target[i], atol=1e-6) and np.isclose(np.cos(heading - headings[i]), 1.0) and grip == 1.0
