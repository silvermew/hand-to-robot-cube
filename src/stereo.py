"""Stereo helpers for the calibrated robot head (rig from src.import_rig: X_right = T_lr @ X_left)."""
import cv2
import numpy as np
from scipy.optimize import least_squares

from src.geometry import pose_from_rvec

MAX_EPIPOLAR_PX = 1.5


def triangulate(c_left, c_right, cams, T_lr):
    """3D points in the left camera frame (metres) from matching pixel points (N, 2)."""
    nl = cv2.undistortPoints(c_left.reshape(-1, 1, 2), cams["left"]["K"], cams["left"]["dist"]).reshape(-1, 2)
    nr = cv2.undistortPoints(c_right.reshape(-1, 1, 2), cams["right"]["K"], cams["right"]["dist"]).reshape(-1, 2)
    X = cv2.triangulatePoints(np.eye(3, 4), T_lr[:3], nl.T, nr.T)
    return (X[:3] / X[3]).T


def epipolar_px(c_left, c_right, cams, T_lr):
    """Largest distance (right-image px) of a point from the epipolar line of its left-image match."""
    E = np.array([[0, -T_lr[2, 3], T_lr[1, 3]], [T_lr[2, 3], 0, -T_lr[0, 3]], [-T_lr[1, 3], T_lr[0, 3], 0]]) @ T_lr[:3, :3]
    nl = cv2.undistortPoints(c_left.reshape(-1, 1, 2), cams["left"]["K"], cams["left"]["dist"]).reshape(-1, 2)
    nr = cv2.undistortPoints(c_right.reshape(-1, 1, 2), cams["right"]["K"], cams["right"]["dist"]).reshape(-1, 2)
    lines = (E @ np.c_[nl, np.ones(len(nl))].T).T
    d = np.abs(np.sum(np.c_[nr, np.ones(len(nr))] * lines, axis=1)) / np.hypot(lines[:, 0], lines[:, 1])
    return float(d.max() * cams["right"]["K"][0, 0])


def plane_frame(P):
    """Centre and unit normal (toward the camera) of 4 marker corners in ArUco order."""
    centre = P.mean(0)
    n = np.cross(P[1] - P[0], P[3] - P[0])
    return centre, -n / np.linalg.norm(n)  # (TR - TL) x (BL - TL) points into the face


def marker_pose(P):
    """T_cam_marker from 4 triangulated corners in ArUco order (x toward TR, y toward the top edge, z out)."""
    x = P[1] - P[0]
    y = P[0] - P[3]
    x /= np.linalg.norm(x)
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, np.cross(x, y)], axis=1)
    T[:3, 3] = P.mean(0)
    return T


def project(obj, T_cam_obj, cam):
    rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
    return cv2.projectPoints(obj, rvec, T_cam_obj[:3, 3], cam["K"], cam["dist"])[0].reshape(-1, 2)


def view_errors(T_left_obj, view, cams, T_lr):
    """Corner reprojection errors (N, 2) in px of one view: (camera "left" or "right", marker id, object points
    (N, 3) in the object frame, detected image points (N, 2))."""
    camera, _, obj, img = view
    return project(obj, T_left_obj if camera == "left" else T_lr @ T_left_obj, cams[camera]) - img


def rms(errors):
    return float(np.sqrt(np.mean(np.sum(np.reshape(errors, (-1, 2)) ** 2, axis=1))))


def fit_views(T0, views, cams, T_lr):
    """T_left_obj minimising the corner reprojection error over all views of both cameras, from T0; and its RMS."""
    def residuals(x):
        T = pose_from_rvec(x[:3], x[3:])
        return np.concatenate([view_errors(T, v, cams, T_lr).ravel() for v in views])

    rvec, _ = cv2.Rodrigues(T0[:3, :3])
    sol = least_squares(residuals, np.r_[rvec.ravel(), T0[:3, 3]], method="lm")
    return pose_from_rvec(sol.x[:3], sol.x[3:]), rms(sol.fun)
