"""Synthetic phone clips with known ground truth, to test the whole pipeline before real data exists.

    python -m tests.make_synthetic [--out data/synthetic] [--clips N] [--workers 8]

A pinhole camera with mild lens distortion sees the table marker, the cube with markers on
five faces (constants.CUBE_MARKER_AXES) and a hand made of skin-coloured quads: the wrist
marker on the back of the hand, a palm-down fingertip pinch (thumb on the cube face toward
the forearm, two fingers on the opposite face, palm raised) and a forearm. Surfaces are
textured quads rendered with a z-buffer, so the hand really occludes cube markers.
Clips 1-8 use the planned camera (behind and above the shoulder, ~57 deg down, ~50 cm from
the cube); clips 9-16 a front-left 3/4 camera (~45 deg down, ~60 cm), which sees far more of the
cube during the carry.

Trajectory: hand out of view for 1 s, reach, grasp, lift 8 cm, rotate the hand by the clip's
angle (the cube follows with a gain ~0.9 plus slow noise), place, release, retreat, out of
view for the last second. Motion blur (sub-frame averaging), brightness drift, sensor noise.
Clip 15 is re-encoded with dropped frames (variable frame rate) and clip 16 as HEVC with
rotation metadata, to exercise prepare_videos.

Writes <out>/raw/clip_<nn>_rot<deg>_<spot>.mp4, <out>/raw/calib_checkerboard.mp4 and
<out>/ground_truth/<clip>.npz (poses in the table frame, per-marker visible fraction).
"""
import argparse
import subprocess
import multiprocessing
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.spatial.transform import Rotation

from src import constants as C
from src.common import ROOT
from src.geometry import hand_yaw_of, make_pose, marker_in_cube, rotz, yaw_of

W, H = 1920, 1080
K_TRUE = np.array([[1500.0, 0.0, 955.0], [0.0, 1500.0, 545.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([0.05, -0.02, 0.0, 0.0, 0.0])
CAMERAS = {"behind_shoulder": (np.array([0.16, -0.20, 0.44]), np.array([0.16, 0.08, 0.0])),
           "front_left": (np.array([-0.14, 0.44, 0.42]), np.array([0.13, 0.07, 0.08]))}
SPOTS = {"A": (0.10, 0.08), "B": (0.20, 0.06), "C": (0.15, 0.15)}
ANGLES = (-90, -60, -30, 0, 30, 60, 90, 120)
HAND_TO_CUBE = np.array([0.0, 0.085, -0.085])  # cube centre in the hand (wrist marker) frame at grasp
OFFSCREEN = np.array([0.05, -0.15, 0.55])      # grasp point relative to the cube while out of view (above both cameras)
SKIN, FOREARM = (205, 160, 135), (185, 140, 115)
GAIN_MEAN = 0.9
N_SUB = 3                                    # sub-frames averaged for motion blur
DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
TABLE_DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.TABLE_DICT))
PX_PER_MM = 8


# ---------- camera and rendering ----------

def camera_extrinsics(camera="behind_shoulder"):
    """R, t with X_cam = R @ X_table + t (OpenCV camera: x right, y down, z forward)."""
    CAM_POS, CAM_LOOK = CAMERAS[camera]
    z = (CAM_LOOK - CAM_POS) / np.linalg.norm(CAM_LOOK - CAM_POS)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ CAM_POS


def distortion_maps():
    """Maps for cv2.remap that turn a pinhole render into the distorted image the phone would record."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], axis=1)[:, None, :]
    und = cv2.undistortPoints(pts, K_TRUE, DIST_TRUE, P=K_TRUE).reshape(H, W, 2)
    return und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)


def marker_texture(marker_id, size, margin, dictionary=DICTIONARY):
    """Marker of side `size` with a white `margin` (metres), drawn at PX_PER_MM so it maps exactly onto its quad."""
    marker_px, margin_px = round(size * 1000 * PX_PER_MM), round(margin * 1000 * PX_PER_MM)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    img = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def solid_texture(rgb, rng, size=32):
    tex = np.full((size, size, 3), rgb, np.float32) + rng.normal(0, 6, (size, size, 3))
    return np.clip(tex, 0, 255).astype(np.uint8)


def table_texture(rng, px_per_mm=2, extent=((-0.40, 0.80), (-0.45, 0.75))):
    (x0, x1), (y0, y1) = extent
    w, h = int((x1 - x0) * 1000 * px_per_mm), int((y1 - y0) * 1000 * px_per_mm)
    grain = cv2.resize(rng.normal(0, 1, (h // 40, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    base = np.array([176, 140, 100], np.float32)
    tex = base + 14 * grain[..., None] + rng.normal(0, 3, (h, w, 1))
    corners = np.array([[x0, y1, 0], [x1, y1, 0], [x1, y0, 0], [x0, y0, 0]])
    return np.clip(tex, 0, 255).astype(np.uint8), corners


def quad(corners_local, T, tex, cull=False, name=None):
    """A textured planar quad; corners in ArUco order (TL, TR, BR, BL of the texture), posed by T."""
    pts = corners_local @ T[:3, :3].T + T[:3, 3]
    return dict(corners=pts, tex=tex, cull=cull, name=name)


def rect(x0, x1, y0, y1):
    return np.array([[x0, y1, 0], [x1, y1, 0], [x1, y0, 0], [x0, y0, 0]], float)


def render_quads(img, zbuf, idmap, quads, R, t, first_id=0):
    """Draw quads into img with a z-buffer; idmap records which quad owns each pixel."""
    Kinv = np.linalg.inv(K_TRUE)
    for qi, q in enumerate(quads, start=first_id):
        Xc = q["corners"] @ R.T + t
        if Xc[:, 2].min() < 0.03:
            continue
        normal = np.cross(Xc[3] - Xc[0], Xc[1] - Xc[0])  # = marker +z for corners in TL, TR, BR, BL order
        if q["cull"] and np.dot(normal, Xc[0]) >= 0:
            continue
        uv = (Xc @ K_TRUE.T)[:, :2] / Xc[:, 2:3]
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        th, tw = q["tex"].shape[:2]
        src = np.float32([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]])
        M = cv2.getPerspectiveTransform(src, (uv - [x0, y0]).astype(np.float32))
        size = (x1 - x0, y1 - y0)
        color = cv2.warpPerspective(q["tex"], M, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.ones((th, tw), np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
        n = normal / np.linalg.norm(normal)
        uu, vv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        rays = np.stack([uu, vv, np.ones_like(uu)], -1) @ Kinv.T
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.dot(n, Xc[0]) / (rays @ n)
        win = mask & (depth > 0) & (depth < zbuf[y0:y1, x0:x1])
        img[y0:y1, x0:x1][win] = color[win]
        zbuf[y0:y1, x0:x1][win] = depth[win]
        idmap[y0:y1, x0:x1][win] = qi


def render_background(R, t, rng, tiles=12):
    """Static table (split into tiles so a tile behind the camera does not hide the whole table) and table marker."""
    tex, corners = table_texture(rng)
    (x0, y1), (x1, y0) = corners[0, :2], corners[2, :2]
    th, tw = tex.shape[0] // tiles, tex.shape[1] // tiles
    quads = []
    for r in range(tiles):
        for c in range(tiles):
            xa, xb = x0 + (x1 - x0) * c / tiles, x0 + (x1 - x0) * (c + 1) / tiles
            ya, yb = y1 - (y1 - y0) * r / tiles, y1 - (y1 - y0) * (r + 1) / tiles
            quads.append(dict(corners=np.array([[xa, ya, 0], [xb, ya, 0], [xb, yb, 0], [xa, yb, 0]]),
                              tex=tex[r * th:(r + 1) * th, c * tw:(c + 1) * tw], cull=False, name="table"))
    table_marker = marker_texture(C.TABLE_ID, C.TABLE_MARKER, 0.010, TABLE_DICTIONARY)
    h = C.TABLE_MARKER / 2 + 0.010
    quads.append(quad(rect(-h, h, -h, h) + [0, 0, 0.0003], np.eye(4), table_marker, name=C.TABLE_ID))
    img = np.full((H, W, 3), 60, np.uint8)
    zbuf = np.full((H, W), np.inf)
    idmap = np.full((H, W), -1, np.int32)
    render_quads(img, zbuf, idmap, quads, R, t)
    return img, zbuf, idmap, quads


# ---------- scene and trajectory ----------

GRASP = "forearm"  # "forearm": thumb / fingers on the faces toward the forearm / fingertips; "side": on the side faces


def hand_quads(T_hand, textures):
    """Wrist marker on the back of the hand, a fingertip pinch with the palm raised above the cube
    (see GRASP for which two faces the thumb and fingers touch) and the forearm."""
    hw = C.WRIST_MARKER / 2 + 0.004
    half = C.CUBE_SIDE / 2
    cy, cz = HAND_TO_CUBE[1], HAND_TO_CUBE[2]
    quads = [
        quad(rect(-hw, hw, -hw, hw) + [0, 0, 0.001], T_hand, textures["wrist"], name=C.WRIST_ID),
        quad(rect(-0.042, 0.042, -0.06, 0.05), T_hand, textures["skin"]),
        quad(np.array([[-0.035, -0.28, 0.10], [0.035, -0.28, 0.10], [0.04, -0.06, -0.005], [-0.04, -0.06, -0.005]]),
             T_hand, textures["forearm"]),
    ]
    if GRASP == "side":
        for x_face, width in ((-half - 0.006, 0.022), (half + 0.006, 0.035)):  # thumb, two fingers
            corners = np.array([[x_face, cy - width / 2, -0.01], [x_face, cy + width / 2, -0.01],
                                [x_face, cy + width / 2, cz - 0.01], [x_face, cy - width / 2, cz - 0.01]])
            quads.append(quad(corners, T_hand, textures["skin"]))
        return quads
    for y_face, width, y_root in ((cy - half - 0.006, 0.022, 0.03), (cy + half + 0.006, 0.035, 0.05)):
        corners = np.array([[-width / 2, y_root, -0.01], [width / 2, y_root, -0.01],
                            [width / 2, y_face, cz - 0.01], [-width / 2, y_face, cz - 0.01]])
        quads.append(quad(corners, T_hand, textures["skin"]))
    return quads


def cube_quads(T_cube, textures):
    h = C.CUBE_SIDE / 2
    return [quad(rect(-h, h, -h, h), T_cube @ marker_in_cube(i), textures[i], cull=True, name=i) for i in C.CUBE_IDS]


def ease_keyframes(t, key_t, key_v):
    """Cosine ease between keyframes (zero velocity at each keyframe)."""
    key_t, key_v = np.asarray(key_t, float), np.asarray(key_v, float)
    i = np.clip(np.searchsorted(key_t, t, side="right") - 1, 0, len(key_t) - 2)
    u = np.clip((t - key_t[i]) / (key_t[i + 1] - key_t[i]), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * u)
    w = w[:, None] if key_v.ndim > 1 else w
    return key_v[i] * (1 - w) + key_v[i + 1] * w


def make_trajectory(angle_deg, spot, rng, place_spot=None):
    """Keyframed grasp point, hand yaw and the cube's response; returns a function of time and the key times.
    The cube is put back near its start spot, or carried to place_spot (a transport clip)."""
    cube0 = np.array([*SPOTS[spot], C.CUBE_SIDE / 2]) + [*rng.uniform(-0.01, 0.01, 2), 0]
    cube_yaw0 = np.radians(rng.uniform(-25, 25))
    target = cube0 if place_spot is None else np.array([*SPOTS[place_spot], C.CUBE_SIDE / 2])
    place = target + [*rng.uniform(-0.03, 0.03, 2), 0]
    angle = np.radians(angle_deg)
    yaw_grasp = np.radians(90) - angle / 2 + np.radians(rng.uniform(-10, 10))  # wind up for big turns
    t_rot = 1.0 + abs(angle_deg) / 90
    t_grasp = 3.6
    t_touch = t_grasp + 1.0 + t_rot + 1.0
    t_open = t_touch + 0.4
    keys = [
        (0.0, cube0 + OFFSCREEN, yaw_grasp + 0.35), (1.0, cube0 + OFFSCREEN, yaw_grasp + 0.35),
        (2.6, cube0 + [0, 0, 0.07], yaw_grasp), (3.2, cube0, yaw_grasp), (t_grasp, cube0, yaw_grasp),
        (t_grasp + 1.0, cube0 + [0, 0, 0.08], yaw_grasp),
        (t_grasp + 1.0 + t_rot, place + [0, 0, 0.08], yaw_grasp + angle),
        (t_touch, place, yaw_grasp + angle), (t_open, place, yaw_grasp + angle),
        (t_open + 0.3, place + [0, 0, 0.05], yaw_grasp + angle),
        (t_open + 1.8, place + OFFSCREEN, yaw_grasp + angle), (t_open + 2.8, place + OFFSCREEN, yaw_grasp + angle),
    ]
    key_t = [k[0] for k in keys]
    gain = GAIN_MEAN + rng.normal(0, 0.03)
    noise_f, noise_p = rng.uniform(0.3, 1.2, 3), rng.uniform(0, 2 * np.pi, 3)
    tilt_p = rng.uniform(0, 2 * np.pi, 2)

    def noise(t):  # slow yaw noise of the cube in the hand, ~1.5 deg
        return np.radians(1.5) * np.sum(np.sin(2 * np.pi * noise_f * t[..., None] + noise_p), -1) / np.sqrt(1.5)

    def state(t):
        t = np.atleast_1d(np.asarray(t, float))
        point = ease_keyframes(t, key_t, [k[1] for k in keys])
        yaw = ease_keyframes(t, key_t, [k[2] for k in keys])
        tilt = np.radians(4) * np.stack([np.sin(1.3 * t + tilt_p[0]), np.sin(0.9 * t + tilt_p[1])], -1)
        R_hand = np.array([rotz(y - np.pi / 2) @ Rotation.from_rotvec([a, b, 0]).as_matrix()
                           for y, (a, b) in zip(yaw, tilt)])
        hand_pos = point - R_hand @ HAND_TO_CUBE
        held = (t >= t_grasp) & (t < t_open)
        t_c = np.clip(t, t_grasp, t_open)
        yaw_at = ease_keyframes(t_c, key_t, [k[2] for k in keys])
        cube_yaw = cube_yaw0 + gain * (yaw_at - yaw_grasp) + (noise(t_c) - noise(np.full_like(t_c, t_grasp)))
        cube_pos = np.where(held[:, None], point, cube0)
        cube_pos[t >= t_open] = ease_keyframes(np.array([t_open]), key_t, [k[1] for k in keys])[0]
        return dict(hand_R=R_hand, hand_pos=hand_pos, cube_pos=cube_pos, cube_yaw=cube_yaw)

    times = dict(t_grasp=t_grasp, t_touchdown=t_touch, t_open=t_open, t_end=key_t[-1])
    return state, times, dict(gain=gain, cube0=cube0, cube_yaw0=cube_yaw0)


# ---------- clip writing ----------

def textures(rng):
    tex = {i: marker_texture(i, C.MARKER_SIZE[i], (C.CUBE_SIDE - C.MARKER_SIZE[i]) / 2) for i in C.CUBE_IDS}
    tex["wrist"] = marker_texture(C.WRIST_ID, C.WRIST_MARKER, 0.004)  # 4 mm card margin
    tex["skin"], tex["forearm"] = solid_texture(SKIN, rng), solid_texture(FOREARM, rng)
    return tex


def finish_frame(acc, t, t_end, maps, rng, phase):
    gain = 1.0 + 0.08 * np.sin(2 * np.pi * t / t_end * 1.3 + phase)  # slow brightness drift
    frame = acc * gain + rng.normal(0, 2.0, acc.shape)
    frame = cv2.remap(np.clip(frame, 0, 255).astype(np.uint8), maps[0], maps[1], cv2.INTER_LINEAR)
    return frame


def write_clip(job):
    name, angle, spot, fps, camera, seed, out, *place_spot = job
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    R, t = camera_extrinsics(camera)
    maps = distortion_maps()
    bg, zbg, idbg, bg_quads = render_background(R, t, rng)
    tex = textures(rng)
    state, times, info = make_trajectory(angle, spot, rng, place_spot[0] if place_spot else None)
    frame_t = np.arange(0, times["t_end"], 1 / fps)
    exposure = 0.5 / fps
    gt = {k: [] for k in ("hand_pos", "hand_quat", "hand_yaw", "cube_pos", "cube_quat", "cube_yaw")}
    vis = {i: [] for i in (C.TABLE_ID, *C.CUBE_IDS, C.WRIST_ID)}
    writer = imageio_ffmpeg.write_frames(str(out / "raw" / f"{name}.mp4"), (W, H), fps=fps, codec="libx264",
                                         quality=None, macro_block_size=8,
                                         output_params=["-crf", "17", "-preset", "veryfast"])
    writer.send(None)
    phase = rng.uniform(0, 2 * np.pi)
    for ft in frame_t:
        acc = np.zeros((H, W, 3), np.float32)
        for k, st in enumerate(ft + exposure * (np.arange(N_SUB) / (N_SUB - 1) - 0.5)):
            s = state(st)
            T_hand = make_pose(s["hand_R"][0], s["hand_pos"][0])
            T_cube = make_pose(rotz(s["cube_yaw"][0]), s["cube_pos"][0])
            quads = cube_quads(T_cube, tex) + hand_quads(T_hand, tex)
            img, zbuf, idmap = bg.copy(), zbg.copy(), idbg.copy()
            render_quads(img, zbuf, idmap, quads, R, t, first_id=len(bg_quads))
            acc += img
            if k == N_SUB // 2:
                all_quads = bg_quads + quads
                for i in vis:
                    qi = next(j for j, q in enumerate(all_quads) if q["name"] == i)
                    vis[i].append(np.count_nonzero(idmap == qi) / max(quad_area(all_quads[qi], R, t), 1.0))
                for key, val in (("hand_pos", s["hand_pos"][0]), ("hand_quat", Rotation.from_matrix(s["hand_R"][0]).as_quat()),
                                 ("hand_yaw", hand_yaw_of(s["hand_R"][0])), ("cube_pos", s["cube_pos"][0]),
                                 ("cube_quat", Rotation.from_matrix(T_cube[:3, :3]).as_quat()),
                                 ("cube_yaw", yaw_of(T_cube[:3, :3]))):
                    gt[key].append(val)
        writer.send(finish_frame(acc / N_SUB, ft, times["t_end"], maps, rng, phase))
    writer.close()
    gt = {k: np.array(v) for k, v in gt.items()}
    gt["hand_yaw"], gt["cube_yaw"] = np.unwrap(gt["hand_yaw"]), np.unwrap(gt["cube_yaw"])
    np.savez(out / "ground_truth" / f"{name}.npz", t=frame_t, fps=fps, **gt,
             **{f"visible_{i}": np.minimum(np.array(v), 1.0) for i, v in vis.items()},
             **times, angle=angle, spot=spot, **info, K=K_TRUE, dist=DIST_TRUE, image_size=(W, H),
             T_cam_table=make_pose(R, t), camera=camera)
    return name


def quad_area(q, R, t):
    """Projected area of a quad in pixels (unoccluded), used to turn pixel counts into visible fractions."""
    Xc = q["corners"] @ R.T + t
    uv = (Xc @ K_TRUE.T)[:, :2] / Xc[:, 2:3]
    return 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], 1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], 1)))


def write_calibration_video(out, fps=30, seconds=20, seed=0):
    """A printed checkerboard waved in front of the same camera."""
    rng = np.random.default_rng(seed)
    R, t = camera_extrinsics()
    CAM_POS, CAM_LOOK = CAMERAS["behind_shoulder"]
    maps = distortion_maps()
    bg, zbg, idbg, _ = render_background(R, t, rng)
    cols, rows = C.CHECKER_INNER[0] + 1, C.CHECKER_INNER[1] + 1
    sq_px = 50  # 2 px per mm
    board = np.full((rows * sq_px + 200, cols * sq_px + 200, 3), 255, np.uint8)  # 50 mm white border
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[100 + r * sq_px:100 + (r + 1) * sq_px, 100 + c * sq_px:100 + (c + 1) * sq_px] = 0
    bw, bh = board.shape[1] / 4000, board.shape[0] / 4000  # half-sizes in metres at 2 px/mm
    n_keys = 9
    key_t = np.linspace(0, seconds, n_keys)
    key_pos = CAM_POS + (CAM_LOOK - CAM_POS) / np.linalg.norm(CAM_LOOK - CAM_POS) * rng.uniform(0.35, 0.6, (n_keys, 1))
    key_pos = key_pos + rng.uniform(-0.12, 0.12, (n_keys, 3))
    key_rot = rng.uniform(-0.6, 0.6, (n_keys, 3))
    writer = imageio_ffmpeg.write_frames(str(out / "raw" / "calib_checkerboard.mp4"), (W, H), fps=fps,
                                         codec="libx264", quality=None, macro_block_size=8,
                                         output_params=["-crf", "17", "-preset", "veryfast"])
    writer.send(None)
    face_cam = np.linalg.inv(R) @ np.diag([1.0, -1.0, -1.0])  # board facing the camera, upright in the image
    for ft in np.arange(0, seconds, 1 / fps):
        pos = ease_keyframes(np.array([ft]), key_t, key_pos)[0]
        rot = Rotation.from_rotvec(ease_keyframes(np.array([ft]), key_t, key_rot)[0]).as_matrix()
        T = make_pose(face_cam @ rot, pos)
        img, zbuf, idmap = bg.copy(), zbg.copy(), idbg.copy()
        render_quads(img, zbuf, idmap, [quad(rect(-bw, bw, -bh, bh), T, board)], R, t)
        writer.send(finish_frame(img.astype(np.float32), ft, seconds, maps, rng, 0.0))
    writer.close()


def phone_quirks(out, vfr_clip, hevc_clip):
    """Re-encode one clip with dropped frames (VFR) and one as rotated HEVC with display rotation metadata."""
    src = out / "raw" / f"{vfr_clip}.mp4"
    tmp = src.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "select='not(eq(mod(n\\,7)\\,3))'",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "17", "-preset", "veryfast", str(tmp)], check=True)
    tmp.replace(src)
    src = out / "raw" / f"{hevc_clip}.mp4"
    rotated = src.with_suffix(".rot.mp4")
    # Store the pixels turned 90 deg clockwise and tag the stream so players turn them back.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "transpose=1", "-c:v", "libx265",
                    "-crf", "18", "-preset", "fast", "-tag:v", "hvc1", "-x265-params", "log-level=error",
                    str(rotated)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "90", "-i", str(rotated),
                    "-c", "copy", str(src.with_suffix(".mov"))], check=True)
    rotated.unlink()
    src.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    p.add_argument("--clips", type=int, default=16, help="first N of the 16 planned clips")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", type=int, nargs="*", default=None, help="regenerate only these clip numbers")
    p.add_argument("--no-calib", action="store_true", help="skip the checkerboard video")
    args = p.parse_args(argv)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    jobs = []
    for n, (rep, angle) in enumerate([(r, a) for r in (0, 1) for a in ANGLES], start=1):
        spot = "ABC"[(n - 1) % 3]
        camera = "behind_shoulder" if rep == 0 else "front_left"
        jobs.append((f"clip_{n:02d}_rot{angle:+d}_{spot}", angle, spot, 30 if rep == 0 else 60, camera, 1000 + n, out))
    jobs = jobs[:args.clips]
    quirks = (jobs[14][0], jobs[15][0]) if len(jobs) >= 16 else None
    if args.only:
        jobs = [j for j in jobs if int(j[0][5:7]) in args.only]
        for j in jobs:  # drop stale files, including the .mov variant
            for old in (out / "raw").glob(j[0] + ".*"):
                old.unlink()
    with multiprocessing.get_context("spawn").Pool(min(args.workers, len(jobs) + 1)) as pool:
        result = None if args.no_calib else pool.map_async(write_calibration_video, [out])
        for name in pool.imap_unordered(write_clip, jobs):
            print(f"wrote {name}")
        if result is not None:
            result.get()
            print("wrote calib_checkerboard")
    names = {j[0] for j in jobs}
    if quirks and quirks[0] in names and quirks[1] in names:
        phone_quirks(out, *quirks)
        print(f"re-encoded {quirks[0]} with dropped frames and {quirks[1]} as rotated HEVC .mov")


if __name__ == "__main__":
    main()
