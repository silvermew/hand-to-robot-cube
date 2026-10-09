"""Rotation and pose helpers shared by tracking, retargeting and the simulator.

Quaternions are (x, y, z, w) everywhere (scipy's convention). Poses are 4x4 matrices
T_a_b mapping coordinates in frame b to frame a.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from src import constants as C


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def heading(v):
    """Angle of a vector projected on the table plane, positive counter-clockwise from above."""
    return np.arctan2(v[..., 1], v[..., 0])


def yaw_of(R):
    """Heading of the frame's x axis (the cube yaw convention)."""
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def hand_yaw_of(R):
    """Heading of the wrist marker's +y axis (its top edge, toward the fingertips)."""
    return np.arctan2(R[..., 1, 1], R[..., 0, 1])


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unwrap_valid(angle, valid):
    """Unwrap over the valid samples only; invalid samples become NaN (never filled in)."""
    out = np.full(len(angle), np.nan)
    out[valid] = np.unwrap(angle[valid])
    return out


def gripper_down(heading_):
    """Panda grip-site orientation pointing straight down, finger-closing axis (site x) at `heading_`."""
    c, s = np.cos(heading_), np.sin(heading_)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])  # columns: x, y = z cross x, z = down


def make_pose(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def invert_pose(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_pose(R.T, -R.T @ t)


def pose_from_rvec(rvec, tvec):
    return make_pose(Rotation.from_rotvec(np.ravel(rvec)).as_matrix(), tvec)


def quat_from_R(R):
    return Rotation.from_matrix(R).as_quat()


def R_from_quat(q):
    return Rotation.from_quat(q).as_matrix()


def angle_between(Ra, Rb):
    """Geodesic angle between rotations, radians."""
    return Rotation.from_matrix(np.swapaxes(Ra, -1, -2) @ Rb).magnitude()


def mean_rotation(Rs):
    return Rotation.from_matrix(np.asarray(Rs)).mean().as_matrix()


def align_quat_signs(q):
    """Flip quaternions so consecutive ones are in the same hemisphere (q and -q are the same rotation)."""
    q = np.array(q, float)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smooth_quats(q, window):
    """Moving average of sign-aligned unit quaternions, renormalised. `q` is one contiguous run."""
    if len(q) < 3 or window < 2:
        return np.array(q, float)
    q = align_quat_signs(q)
    pad = window // 2
    padded = np.pad(q, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(padded[:, k], kernel, mode="valid")[:len(q)] for k in range(4)], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def marker_in_cube(marker_id):
    """Pose T_cube_marker of a cube face marker (see constants.CUBE_MARKER_AXES)."""
    axes = np.array(C.CUBE_MARKER_AXES[marker_id], float)  # rows: marker x, y, z in cube coordinates
    return make_pose(axes.T, axes[2] * C.CUBE_SIDE / 2)


def marker_corners(size):
    """3D corners in the marker frame, in ArUco order: top-left, top-right, bottom-right, bottom-left."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
