"""Marker tracking: per-frame poses of the cube and the hand in the table frame.

    python -m src.track [--data data/raw] [--clips NAME ...] [--stride 1] [--no-debug-video] [--workers 8]

Per frame, DICT_4X4_50 markers are detected and each marker's pose is solved with
SOLVEPNP_IPPE_SQUARE, keeping both planar solutions. A single marker is ambiguous (the two
solutions can both fit the corners), so the solution whose up-axis is closest to the table's
up is used: the hand is palm-down and the cube stays upright. The cube pose is one PnP over
the corners of all visible cube markers, seeded by the best single marker. Poses with a high
reprojection error or a one-frame rotation flip are rejected and flagged, never filled in.

The table pose (camera in the table frame) is the median over the first frames where the
table marker is visible and the hand is not yet in view.

With the robot head's two cameras (<clip>_cam1 / _cam2 and data/rig.json), markers are detected in
both views; each camera's own pose seeds one fit of the cube (or hand) pose to the corners from
both views, which fixes the depth (one camera at 1.4 m is 2-3 cm off) and keeps tracking when only
the second camera sees a marker. Poses stay in the left (cam1) camera frame.

Writes <data>/../tracks/<clip>.npz (raw and smoothed tracks, visibility flags, errors) and,
unless --no-debug-video, <out>/debug/<clip>_track.mp4 with the tracked axes drawn.
"""
import argparse
import multiprocessing
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from src import constants as C
from src.calibrate import load_camera
from src.common import ROOT, clip_stem, data_dirs, is_synthetic, label_frame, list_videos
from src.geometry import (angle_between, hand_yaw_of, invert_pose, make_pose, marker_corners, marker_in_cube,
                          mean_rotation, pose_from_rvec, quat_from_R, R_from_quat, smooth_quats, yaw_of)
from src.import_rig import load_rig
from src.stereo import MAX_EPIPOLAR_PX, epipolar_px, fit_views, marker_pose, rms, triangulate, view_errors
from src.video import find_video, read_frames, video_fps

MAX_REPROJ_PX = 2.5      # RMS corner error above which a pose is rejected
MAX_REPROJ_TWO_VIEW_PX = 4.0  # per view, when both cameras are in the fit: a fingertip on a marker's edge moves
                              # a corner differently in each view, and the joint fit still has the right depth
MAX_TILT_DEG = 50        # cube / back of hand must be within this of table-up
FLIP_DEG = 25            # a one-frame rotation jump larger than this is a planar flip, not motion
TABLE_FRAMES = 30
SMOOTH_S = 0.1           # moving-average window for the smoothed tracks
DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
TABLE_DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.TABLE_DICT))


def make_detectors(dictionary):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def make_detector():
    """Detectors for the cube and hand markers and for the table marker (the same one if one dictionary)."""
    main = make_detectors(DICTIONARY)
    return main, main if C.TABLE_DICT == C.ARUCO_DICT else make_detectors(TABLE_DICTIONARY)


def canonical_patch(gray, corners, size):
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    patch = cv2.warpPerspective(gray, H, (size, size))
    thresh, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return patch, thresh, H


def border_is_dark(gray, corners, size=120, strip=3):
    """True if a thin strip just inside the quad is dark, as on a real marker's black border.

    A white margin thinner than about half a marker cell, on a darker background (black table, carton),
    makes its outer edge a second square that also decodes as the marker, and OpenCV may return that
    square, too large by twice the margin. Its inside strip is the white margin."""
    patch, thresh, _ = canonical_patch(gray, corners, size)
    ring = np.ones((size, size), bool)
    ring[strip:-strip, strip:-strip] = False
    return patch[ring].mean() < thresh


def inner_square(gray, corners, size=240):
    """Corners of the black square inside a quad whose rim is a white margin: from each side, the first
    mostly-dark line of the rectified patch, mapped back to the image and refined to sub-pixel."""
    patch, thresh, H = canonical_patch(gray, corners, size)
    dark = patch < thresh
    mid = slice(size // 4, 3 * size // 4)  # the border cells are black along their whole length

    def first_dark(profile):
        return int(np.argmax(profile > 0.5))

    top, left = first_dark(dark[:, mid].mean(1)), first_dark(dark[mid, :].mean(0))
    bottom = size - 1 - first_dark(dark[::-1, mid].mean(1))
    right = size - 1 - first_dark(dark[mid, ::-1].mean(0))
    inner = np.float32([[left - 0.5, top - 0.5], [right + 0.5, top - 0.5], [right + 0.5, bottom + 0.5],
                        [left - 0.5, bottom + 0.5]])
    back = cv2.perspectiveTransform(inner[None], np.linalg.inv(H)).reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(gray, back, (5, 5), (-1, -1), criteria).reshape(4, 2)


def collect(detector, gray, ids_wanted, found):
    """Detect; replace a square whose inside strip is light (a white-margin edge) by the black square
    inside it. Per ID keep the largest dark-bordered square (the closer of two copies)."""
    corners, ids, _ = detector.detectMarkers(gray)
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        c, i = c.reshape(4, 2), int(i)
        if i not in ids_wanted:
            continue
        if not border_is_dark(gray, c):
            c = inner_square(gray, c)
            if not border_is_dark(gray, c):
                continue
        if i not in found or cv2.contourArea(c) > cv2.contourArea(found[i]):
            found[i] = c
    return found


def detect(detector, frame):
    """{marker id: (4, 2) corners} for the pipeline's markers."""
    main, table = detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if table is main:
        return collect(main, gray, set(C.MARKER_SIZE), {})
    found = collect(main, gray, set(C.MARKER_SIZE) - {C.TABLE_ID}, {})
    return collect(table, gray, {C.TABLE_ID}, found)


def marker_solutions(corners, size, cam):
    """Both IPPE solutions as (T_cam_marker, RMS reprojection error in px)."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(marker_corners(size), corners.astype(np.float64), cam["K"],
                                                cam["dist"], flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return [(pose_from_rvec(r, t), float(e)) for r, t, e in zip(rvecs, tvecs, np.ravel(errs))]


def reprojection_rms(obj, img, T, cam):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    proj, _ = cv2.projectPoints(obj, rvec, T[:3, 3], cam["K"], cam["dist"])
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def upright(T_cam_obj, T_table_cam):
    """Cosine between the object's z axis and table-up."""
    return float((T_table_cam[:3, :3] @ T_cam_obj[:3, :3])[2, 2])


def stereo_videos(dirs, stem):
    """The raw (cam1, cam2) videos of a clip if both exist and the rig is calibrated (data/rig.json), else None."""
    left, right = (next(iter(sorted(dirs["raw"].glob(f"{stem}_cam{k}.*"))), None) for k in (1, 2))
    if left is None or right is None or not (dirs["base"] / "rig.json").exists():
        return None
    return left, right


def stereo_table_pose(left, right, cams, T_lr):
    """T_cam_table (left camera) from marker 13 triangulated in both cameras, median over its first TABLE_FRAMES
    well-matched frames. A single camera has two mirror-image pose solutions for one marker and at ~1.4 m
    picks the wrong one in some clips (tilted table frame)."""
    corners = [triangulate(a[C.TABLE_ID], b[C.TABLE_ID], cams, T_lr) for a, b in zip(left, right)
               if C.TABLE_ID in a and C.TABLE_ID in b
               and epipolar_px(a[C.TABLE_ID], b[C.TABLE_ID], cams, T_lr) < MAX_EPIPOLAR_PX][:TABLE_FRAMES]
    if len(corners) < 5:
        return None, len(corners)
    return marker_pose(np.median(np.array(corners), axis=0)), len(corners)


def table_pose(detections, cam):
    """T_cam_table from one camera: the first TABLE_FRAMES detections before the wrist marker first appears."""
    first_hand = next((k for k, d in enumerate(detections) if C.WRIST_ID in d), len(detections))
    frames = [d[C.TABLE_ID] for d in detections[:first_hand] if C.TABLE_ID in d][:TABLE_FRAMES]
    note = "single camera"
    if len(frames) < 5:
        frames = [d[C.TABLE_ID] for d in detections if C.TABLE_ID in d][:TABLE_FRAMES]
        note = "single camera; table marker seen in < 5 frames before the hand appeared, used frames with the hand in view"
    if not frames:
        return None, 0, "table marker never detected"
    poses = [min(marker_solutions(c, C.TABLE_MARKER, cam), key=lambda s: s[1])[0] for c in frames]
    R = mean_rotation([T[:3, :3] for T in poses])
    keep = [T for T in poses if np.degrees(angle_between(T[:3, :3], R)) < 3.0] or poses
    T = make_pose(mean_rotation([T[:3, :3] for T in keep]), np.median([T[:3, 3] for T in keep], axis=0))
    return T, len(keep), note


def hand_pose(det, cam, T_table_cam):
    if C.WRIST_ID not in det:
        return None, np.nan
    sols = [s for s in marker_solutions(det[C.WRIST_ID], C.WRIST_MARKER, cam) if s[1] <= MAX_REPROJ_PX]
    if not sols:
        return None, np.nan
    T, err = max(sols, key=lambda s: upright(s[0], T_table_cam))
    if upright(T, T_table_cam) < np.cos(np.radians(MAX_TILT_DEG)):
        return None, np.nan
    return T, err


def cube_corner_points(marker_id):
    """Corners of a cube marker in the cube frame, in ArUco order."""
    return (marker_in_cube(marker_id) @ np.c_[marker_corners(C.MARKER_SIZE[marker_id]), np.ones(4)].T).T[:, :3]


def cube_pose(det, cam, T_table_cam):
    """Fused cube pose from all visible cube markers, seeded by the best single-marker solution."""
    ids = [i for i in C.CUBE_IDS if i in det]
    candidates = []
    for i in ids:
        T_marker_cube = invert_pose(marker_in_cube(i))
        sols = [(T @ T_marker_cube, e) for T, e in marker_solutions(det[i], C.MARKER_SIZE[i], cam)]
        T, e = max(sols, key=lambda s: upright(s[0], T_table_cam))
        if e <= MAX_REPROJ_PX and upright(T, T_table_cam) >= np.cos(np.radians(MAX_TILT_DEG)):
            candidates.append((T, e, i))
    if not candidates:
        return None, np.nan, []
    seed, seed_err, seed_id = min(candidates, key=lambda c: c[1])
    if len(ids) == 1:
        return seed, seed_err, [seed_id]
    obj = np.vstack([cube_corner_points(i) for i in ids])
    img = np.vstack([det[i] for i in ids]).astype(np.float64)
    rvec, _ = cv2.Rodrigues(seed[:3, :3])
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam["K"], cam["dist"], rvec, seed[:3, 3].copy(), True,
                                  cv2.SOLVEPNP_ITERATIVE)
    if ok:
        T = pose_from_rvec(rvec, tvec)
        err = reprojection_rms(obj, img, T, cam)
        if err <= MAX_REPROJ_PX and upright(T, T_table_cam) >= np.cos(np.radians(MAX_TILT_DEG)):
            return T, err, ids
    return seed, seed_err, [seed_id]  # a misdetected marker spoils the joint fit; fall back to the best one


CUBE_POINTS = [(i, cube_corner_points(i)) for i in C.CUBE_IDS]
HAND_POINTS = [(C.WRIST_ID, marker_corners(C.WRIST_MARKER))]


def views_of(points, det_l, det_r):
    """(camera, marker id, object points, image corners) for every listed marker that each camera detected."""
    return [(camera, i, obj, det[i].astype(np.float64)) for camera, det in (("left", det_l), ("right", det_r))
            for i, obj in points if i in det]


def seeds_of(pose_fn, det_l, det_r, cams, T_lr, T_table_cam):
    """Each camera's own single-view pose (cube_pose or hand_pose), in the left camera frame."""
    T_rl = invert_pose(T_lr)
    seeds = []
    for det, camera, T_left_cam in ((det_l, "left", np.eye(4)), (det_r, "right", T_rl)):
        T = pose_fn(det, cams[camera], T_table_cam @ T_left_cam)[0]
        if T is not None:
            seeds.append(T_left_cam @ T)
    return seeds


def two_view_pose(seeds, views, cams, T_lr, T_table_cam):
    """The best seed refitted to the corners from both cameras, where the second view fixes the depth that one
    camera gets wrong by 2-3 cm at 1.4 m. A view (one marker in one camera) that still fits worse than
    MAX_REPROJ_TWO_VIEW_PX (MAX_REPROJ_PX once one camera is left) is taken as a misdetection: dropped, and the
    rest refitted. (T_left_obj, RMS px, views used)."""
    while views and seeds:
        T, err = min((fit_views(s, views, cams, T_lr) for s in seeds), key=lambda f: f[1])
        per_view = [rms(view_errors(T, v, cams, T_lr)) for v in views]
        both = len({v[0] for v in views}) == 2
        if max(per_view) <= (MAX_REPROJ_TWO_VIEW_PX if both else MAX_REPROJ_PX):
            if upright(T, T_table_cam) < np.cos(np.radians(MAX_TILT_DEG)):
                break
            return T, err, views
        views = [v for v, e in zip(views, per_view) if e < max(per_view)]
    return None, np.nan, []


def track_cube(det_l, det_r, cam, rig, T_table_cam):
    """Cube pose in the left camera: (T or None, RMS px, marker ids used, cameras used). rig = (cams, T_lr), or
    None for the left camera alone."""
    if rig is None:
        T, err, used = cube_pose(det_l, cam, T_table_cam)
        return T, err, used, {"left"}
    cams, T_lr = rig
    T, err, views = two_view_pose(seeds_of(cube_pose, det_l, det_r, cams, T_lr, T_table_cam),
                                  views_of(CUBE_POINTS, det_l, det_r), cams, T_lr, T_table_cam)
    return T, err, {v[1] for v in views}, {v[0] for v in views}


def track_hand(det_l, det_r, cam, rig, T_table_cam):
    """Hand-marker pose in the left camera: (T or None, RMS px, cameras used)."""
    if rig is None:
        T, err = hand_pose(det_l, cam, T_table_cam)
        return T, err, {"left"}
    cams, T_lr = rig
    T, err, views = two_view_pose(seeds_of(hand_pose, det_l, det_r, cams, T_lr, T_table_cam),
                                  views_of(HAND_POINTS, det_l, det_r), cams, T_lr, T_table_cam)
    return T, err, {v[0] for v in views}


def reject_flips(R, valid):
    """Drop single frames that disagree with both valid neighbours while the neighbours agree."""
    idx = np.flatnonzero(valid)
    flips = 0
    for a, b, c in zip(idx[:-2], idx[1:-1], idx[2:]):
        if (np.degrees(angle_between(R[a], R[b])) > FLIP_DEG and np.degrees(angle_between(R[b], R[c])) > FLIP_DEG
                and np.degrees(angle_between(R[a], R[c])) < FLIP_DEG):
            valid[b] = False
            flips += 1
    return flips


def reject_implausible(pos, valid, min_z=-0.03, max_z=0.6, jump=0.08):
    """Drop poses below the table or far above it, and single-frame position jumps (a bad pose between two
    neighbours that agree). Returns the number dropped."""
    bad = valid & ((pos[:, 2] < min_z) | (pos[:, 2] > max_z))
    valid &= ~bad
    idx = np.flatnonzero(valid)
    jumps = 0
    for a, b, c in zip(idx[:-2], idx[1:-1], idx[2:]):
        if (np.linalg.norm(pos[b] - pos[a]) > jump and np.linalg.norm(pos[b] - pos[c]) > jump
                and np.linalg.norm(pos[a] - pos[c]) < jump):
            valid[b] = False
            jumps += 1
    return int(bad.sum()) + jumps


def runs(valid):
    """(start, stop) of each run of consecutive valid samples."""
    edges = np.diff(np.r_[0, valid.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def smooth_track(pos, quat, valid, window, pos_valid=None):
    """Moving averages: positions over runs of pos_valid frames (default: valid), rotations over runs of valid
    frames; NaN elsewhere."""
    pos_s, quat_s = np.full_like(pos, np.nan), np.full_like(quat, np.nan)
    for a, b in runs(valid if pos_valid is None else pos_valid):
        if b - a >= 3 and window > 1:
            pad = window // 2
            padded = np.pad(pos[a:b], ((pad, pad), (0, 0)), mode="edge")
            pos_s[a:b] = np.stack([np.convolve(padded[:, k], np.ones(window) / window, mode="valid")[:b - a]
                                   for k in range(3)], axis=1)
        else:
            pos_s[a:b] = pos[a:b]
    for a, b in runs(valid):
        quat_s[a:b] = smooth_quats(quat[a:b], window)
    return pos_s, quat_s


def track_clip(dirs, stem, stride=1, debug_dir=None):
    """With both raw camera videos and data/rig.json, every pose is fitted to both views (the raw cam1 file is
    read, not the prepared copy, so both views have the same quality); otherwise the left camera alone."""
    pair = stereo_videos(dirs, stem)
    path, prepared = (pair[0], False) if pair else find_video(dirs, stem)
    detector = make_detector()
    frames = list(read_frames(path, prepared, stride=stride))
    if not frames:
        raise IOError(f"{stem}: no frames read from {path}")
    h, w = frames[0][2].shape[:2]
    cam = load_camera(dirs, w, h, quiet=True)
    detections = [detect(detector, f) for _, _, f in frames]
    t = np.array([ft for _, ft, _ in frames])
    rig, right, T_cam_table = None, None, None
    if pair:
        cams, T_lr = load_rig(dirs["base"] / "rig.json")
        cam = dict(cams["left"], placeholder=False)
        rig = (cams, T_lr)
        right = [detect(detector, f) for _, _, f in read_frames(pair[1], False, stride=stride)][:len(detections)]
        right += [{}] * (len(detections) - len(right))
        T_cam_table, n_table = stereo_table_pose(detections, right, cams, T_lr)
        table_note = f"stereo ({n_table} frames)"
    if T_cam_table is None:
        T_cam_table, n_table, table_note = table_pose(detections, cam)
    n = len(frames)
    out = dict(t=t, fps=video_fps(path, prepared) / stride, stride=stride, image_size=np.array([w, h]),
               K=cam["K"], dist=cam["dist"], placeholder_intrinsics=bool(cam["placeholder"]),
               table_found=T_cam_table is not None, table_frames=n_table, table_note=table_note,
               table_visible=np.array([C.TABLE_ID in d for d in detections]), stereo=rig is not None)
    if T_cam_table is None:
        np.savez(dirs["tracks"] / f"{stem}.npz", **out)
        return out
    T_table_cam = invert_pose(T_cam_table)
    out["T_table_cam"] = T_table_cam
    for name in ("cube", "hand"):
        out[f"{name}_pos_raw"] = np.full((n, 3), np.nan)
        out[f"{name}_R_raw"] = np.full((n, 3, 3), np.nan)
        out[f"{name}_err"] = np.full(n, np.nan)
        out[f"{name}_cams"] = np.zeros((n, 2), bool)  # which cameras (left, right) the pose used
    out["cube_ids"] = np.zeros((n, len(C.CUBE_IDS)), bool)       # markers used in the fused pose
    out["cube_ids_seen"] = np.zeros((n, len(C.CUBE_IDS)), bool)  # markers detected at all, by either camera
    for k, det in enumerate(detections):
        det_r = right[k] if right else {}
        out["cube_ids_seen"][k] = [i in det or i in det_r for i in C.CUBE_IDS]
        T, err, used, cameras = track_cube(det, det_r, cam, rig, T_table_cam)
        if T is not None:
            Tt = T_table_cam @ T
            out["cube_pos_raw"][k], out["cube_R_raw"][k], out["cube_err"][k] = Tt[:3, 3], Tt[:3, :3], err
            out["cube_ids"][k] = [i in used for i in C.CUBE_IDS]
            out["cube_cams"][k] = [c in cameras for c in ("left", "right")]
        T, err, cameras = track_hand(det, det_r, cam, rig, T_table_cam)
        if T is not None:
            Tt = T_table_cam @ T
            out["hand_pos_raw"][k], out["hand_R_raw"][k], out["hand_err"][k] = Tt[:3, 3], Tt[:3, :3], err
            out["hand_cams"][k] = [c in cameras for c in ("left", "right")]
    window = max(1, int(round(SMOOTH_S * out["fps"]))) | 1
    for name in ("cube", "hand"):
        valid = ~np.isnan(out[f"{name}_err"])
        # One camera alone gets the yaw right (0.2 deg median against both) but the position 2 cm off along its
        # line of sight: with the rig, positions only come from fits to both views. The cube keeps its
        # one-camera frames for the rotation (half its frames while held); the hand is seen by both 98% of that time.
        pos_valid = valid & out[f"{name}_cams"].all(1) if rig is not None else valid.copy()
        if name == "hand":
            valid = pos_valid.copy()
        checked = pos_valid.copy()
        out[f"{name}_implausible"] = reject_implausible(out[f"{name}_pos_raw"], pos_valid)
        valid &= ~(checked & ~pos_valid)
        out[f"{name}_flips"] = reject_flips(out[f"{name}_R_raw"], valid)
        pos_valid &= valid
        out[f"{name}_visible"], out[f"{name}_pos_visible"] = valid, pos_valid
        quat = np.full((n, 4), np.nan)
        quat[valid] = quat_from_R(out[f"{name}_R_raw"][valid])
        out[f"{name}_quat_raw"] = quat
        out[f"{name}_pos"], out[f"{name}_quat"] = smooth_track(out[f"{name}_pos_raw"], quat, valid, window, pos_valid)
    for name, yaw_fn in (("cube", yaw_of), ("hand", hand_yaw_of)):  # wrapped here; extract.py unwraps
        visible = out[f"{name}_visible"]
        out[f"{name}_yaw"] = np.full(n, np.nan)
        if visible.any():
            out[f"{name}_yaw"][visible] = yaw_fn(R_from_quat(out[f"{name}_quat"][visible]))
    out["n_detections"] = np.array([len(d) for d in detections])
    np.savez(dirs["tracks"] / f"{stem}.npz", **out)
    if debug_dir is not None:
        write_debug_video(frames, detections, out, cam, debug_dir / f"{stem}_track.mp4", is_synthetic(dirs["base"]))
    return out


def draw_pose(img, T_cam_obj, cam, length, scale):
    rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
    K = cam["K"].copy()
    K[:2] *= scale
    cv2.drawFrameAxes(img, K, cam["dist"], rvec, T_cam_obj[:3, 3], length, 2)


def write_debug_video(frames, detections, track, cam, path, synthetic, scale=0.5):
    """Detections outlined, smoothed cube / hand axes and the table axes drawn on every frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    T_cam_table = invert_pose(track["T_table_cam"])
    size = (int(track["image_size"][0] * scale), int(track["image_size"][1] * scale))
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=track["fps"], codec="libx264", macro_block_size=2,
                                         quality=None, output_params=["-crf", "23", "-preset", "veryfast"])
    writer.send(None)
    for k, ((_, t, frame), det) in enumerate(zip(frames, detections)):
        img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if det:
            cv2.aruco.drawDetectedMarkers(img, [c[None] * scale for c in det.values()], np.array(list(det))[:, None])
        draw_pose(img, T_cam_table, cam, 0.05, scale)
        lines = [f"t={t:5.2f}s"]
        for name, length in (("cube", 0.04), ("hand", 0.04)):
            if track[f"{name}_visible"][k]:
                if track[f"{name}_pos_visible"][k]:
                    T = T_cam_table @ make_pose(R_from_quat(track[f"{name}_quat"][k]), track[f"{name}_pos"][k])
                    draw_pose(img, T, cam, length, scale)
                one = "" if track[f"{name}_pos_visible"][k] else " (one camera: no position)"
                lines.append(f"{name} yaw {np.degrees(track[f'{name}_yaw'][k]):+7.1f}{one}")
            else:
                lines.append(f"{name}: not tracked")
        lines.append(f"cube markers used: {[i for i, u in zip(C.CUBE_IDS, track['cube_ids'][k]) if u]}")
        for j, text in enumerate(lines):
            cv2.putText(img, text, (8, 20 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (8, 20 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.send(cv2.cvtColor(label_frame(img, synthetic), cv2.COLOR_BGR2RGB))
    writer.close()


def _job(args):
    dirs, stem, stride, debug_dir = args
    cv2.setNumThreads(1)
    try:
        tr = track_clip(dirs, stem, stride, debug_dir)
        return stem, None, tr
    except Exception as e:  # one bad clip should not stop the batch
        return stem, f"{type(e).__name__}: {e}", None


def track_all(raw_dir, out_dir, clips=None, stride=1, debug=True, workers=8):
    dirs = data_dirs(raw_dir)
    dirs["tracks"].mkdir(parents=True, exist_ok=True)
    stems = clips or [clip_stem(p) for p in list_videos(dirs["raw"])[0]]
    debug_dir = Path(out_dir) / "debug" if debug else None
    jobs = [(dirs, s, stride, debug_dir) for s in stems]
    if stems:
        h, w = cv2.VideoCapture(str(find_video(dirs, stems[0])[0])).read()[1].shape[:2]
        load_camera(dirs, w, h)  # warn once, loudly, if the placeholder is used
    results = {}
    with multiprocessing.get_context("spawn").Pool(min(workers, max(1, len(jobs)))) as pool:  # fork can deadlock
        for stem, err, tr in pool.imap_unordered(_job, jobs):
            if err:
                print(f"FAILED {stem}: {err}")
                continue
            results[stem] = tr
            if not tr["table_found"]:
                print(f"{stem}: TABLE MARKER NOT FOUND, clip unusable")
            else:
                one = 100 * (tr["cube_visible"] & ~tr["cube_pos_visible"]).mean()
                print(f"{stem}: cube tracked {100 * tr['cube_visible'].mean():4.0f}% | hand "
                      f"{100 * tr['hand_visible'].mean():4.0f}% | flips rejected {tr['cube_flips']}+{tr['hand_flips']}, "
                      f"implausible {tr['cube_implausible']}+{tr['hand_implausible']}"
                      + (f" | cube seen by one camera only (yaw, no position) in {one:.0f}% of frames" if tr["stereo"] else "")
                      + f" | table pose: {tr['table_note']}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--out", default=None, help="where debug videos go (default outputs/real or outputs/synthetic)")
    p.add_argument("--clips", nargs="*", default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--no-debug-video", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    out = args.out or str(ROOT / "outputs" / ("synthetic" if is_synthetic(args.data) else "real"))
    track_all(args.data, out, args.clips, args.stride, not args.no_debug_video, args.workers)


if __name__ == "__main__":
    main()
