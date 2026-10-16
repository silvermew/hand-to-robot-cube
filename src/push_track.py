"""Track the phone videos of pushes (one camera) in the table frame.

    python -m src.push_track [--videos data/videos] [--out data/push/tracks] [--workers 6]

The robot's cameras were down on 13 Oct, so the pushes were filmed with a phone on a stand, from about where the
robot head stands. One camera gets a marker's direction right but its distance 2-3 cm off. A pushed cube stays on
the table, so its top marker lies in the plane CUBE_SIDE above marker 13: each of its corners is where the camera
ray through that corner meets the plane. When the hand hides the top marker, the pose from the side markers is slid
along its camera ray to the same height. The hand card has no such constraint and keeps the one-camera position.

There is no checkerboard video of the phone, so each video's focal length is the value that puts the top marker,
from its own one-marker pose, at the cube's height above the table (principal point at the image centre, no
distortion). With it the cube's rest spots land within 1 cm of the crosses measured with the stereo rig.
Writes <out>/<video>.npz: t (s, from the frame timestamps: the phone's frame rate varies), cube_pos, cube_yaw,
cube_source (0 none, 1 top marker, 2 side markers), hand_pos, hand_yaw, hand_visible, focal, T_table_cam.
"""
import argparse
import multiprocessing
from pathlib import Path

import cv2
import numpy as np

from src import constants as C
from src.common import ROOT
from src.geometry import hand_yaw_of, heading, invert_pose, marker_corners, marker_in_cube, yaw_of
from src.track import cube_pose, detect, hand_pose, make_detector, marker_solutions, table_pose

FOCAL_START = 1500.0    # px, a first guess for a phone's main camera at 1920x1080
MAX_TOP_FIT_MM = 3.0    # RMS distance of the top marker's corners on the plane from a 42 mm square


def camera(focal, w, h):
    return dict(K=np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]]), dist=np.zeros(5), placeholder=False)


def read_video(path):
    """(timestamps in s, detections) for every frame."""
    detector = make_detector()
    cap = cv2.VideoCapture(str(path))
    t, dets, size = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        dets.append(detect(detector, frame))
        size = frame.shape[1], frame.shape[0]
    cap.release()
    return np.array(t), dets, size


def top_marker_height(det, cam, T_table_cam):
    """Height of the top marker above the table from its own one-marker pose."""
    T, _ = min(marker_solutions(det[C.CUBE_TOP_ID], C.CUBE_TOP_MARKER, cam), key=lambda s: s[1])
    return (T_table_cam @ T)[2, 3]


def estimate_focal(dets, size, iterations=3):
    """The focal length that puts the top marker at the cube's height: a one-marker depth scales with the focal
    length, so the height does too; a few rescalings converge."""
    focal = FOCAL_START
    for _ in range(iterations):
        cam = camera(focal, *size)
        T_cam_table, _, _ = table_pose(dets, cam)
        T_table_cam = invert_pose(T_cam_table)
        h = [top_marker_height(d, cam, T_table_cam) for d in dets if C.CUBE_TOP_ID in d][::3]
        focal *= C.CUBE_SIDE / np.median(h)
    return focal


def on_plane(points, cam, T_table_cam, z):
    """Table-frame points where the camera rays through the image points meet the plane at height z."""
    rays = T_table_cam[:3, :3] @ np.linalg.solve(cam["K"], np.c_[points, np.ones(len(points))].T)
    origin = T_table_cam[:3, 3]
    return (origin[:, None] + (z - origin[2]) / rays[2] * rays).T


def cube_from_top(corners, cam, T_table_cam):
    """(cube centre, cube yaw, fit RMS in mm) from the top marker's corners on the plane of the top face."""
    pts = on_plane(corners, cam, T_table_cam, C.CUBE_SIDE)[:, :2]
    model = marker_corners(C.CUBE_TOP_MARKER)[:, :2]
    a, b = model - model.mean(0), pts - pts.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)  # 2-D Kabsch: rotation taking the marker's square onto the corners
    R = u @ np.diag([1, np.linalg.det(u @ vt)]) @ vt
    rms = 1000 * np.sqrt(np.mean(np.sum((a @ R.T - b) ** 2, 1)))
    marker_x = marker_in_cube(C.CUBE_TOP_ID)[:3, 0]  # the marker's x axis in the cube frame
    yaw = np.arctan2(R[1, 0], R[0, 0]) - heading(marker_x)
    return np.r_[pts.mean(0), C.CUBE_SIDE / 2], yaw, rms


def cube_from_sides(det, cam, T_table_cam):
    """(cube centre slid along its camera ray to the cube's resting height, cube yaw) from the side markers."""
    T, _, _ = cube_pose({i: det[i] for i in C.CUBE_SIDE_IDS if i in det}, cam, T_table_cam)
    if T is None:
        return None, np.nan
    Tt = T_table_cam @ T
    origin, ray = T_table_cam[:3, 3], Tt[:3, 3] - T_table_cam[:3, 3]
    return origin + (C.CUBE_SIDE / 2 - origin[2]) / ray[2] * ray, yaw_of(Tt[:3, :3])


def track_video(path):
    t, dets, size = read_video(path)
    focal = estimate_focal(dets, size)
    cam = camera(focal, *size)
    T_cam_table, n_table, note = table_pose(dets, cam)
    T_table_cam = invert_pose(T_cam_table)
    n = len(dets)
    out = dict(t=t, focal=focal, image_size=np.array(size), K=cam["K"], T_table_cam=T_table_cam, table_note=note,
               cube_pos=np.full((n, 3), np.nan), cube_yaw=np.full(n, np.nan), cube_source=np.zeros(n, int),
               top_fit_mm=np.full(n, np.nan), hand_pos=np.full((n, 3), np.nan), hand_yaw=np.full(n, np.nan),
               hand_visible=np.zeros(n, bool))
    for k, det in enumerate(dets):
        if C.CUBE_TOP_ID in det:
            pos, yaw, rms = cube_from_top(det[C.CUBE_TOP_ID], cam, T_table_cam)
            out["top_fit_mm"][k] = rms
            if rms <= MAX_TOP_FIT_MM:
                out["cube_pos"][k], out["cube_yaw"][k], out["cube_source"][k] = pos, yaw, 1
        if out["cube_source"][k] == 0 and any(i in det for i in C.CUBE_SIDE_IDS):
            pos, yaw = cube_from_sides(det, cam, T_table_cam)
            if pos is not None:
                out["cube_pos"][k], out["cube_yaw"][k], out["cube_source"][k] = pos, yaw, 2
        T, _ = hand_pose(det, cam, T_table_cam)
        if T is not None:
            Tt = T_table_cam @ T
            out["hand_pos"][k], out["hand_yaw"][k], out["hand_visible"][k] = Tt[:3, 3], hand_yaw_of(Tt[:3, :3]), True
    return out


def _job(args):
    path, out_dir = args
    out = track_video(path)
    np.savez(Path(out_dir) / f"{path.stem}.npz", **out)
    src = out["cube_source"]
    return (f"{path.stem}: focal {out['focal']:.0f} px, {len(src)} frames, cube from top {np.mean(src == 1):.0%} / "
            f"sides {np.mean(src == 2):.0%}, hand {out['hand_visible'].mean():.0%}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--videos", default=str(ROOT / "data" / "videos"))
    p.add_argument("--pattern", default="2026101[3-9]_*.mp4", help="the phone videos in --videos")
    p.add_argument("--out", default=str(ROOT / "data" / "push" / "tracks"))
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    paths = sorted(Path(args.videos).glob(args.pattern))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(args.workers) as pool:
        for line in pool.imap(_job, [(q, args.out) for q in paths]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
