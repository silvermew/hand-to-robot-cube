"""Pilot checks with the calibrated stereo head, before filming the full set.

    python -m src.pilot_check static [--data data/raw]   # pilot_static_cam1 / _cam2: nothing moves
    python -m src.pilot_check signs  [--data data/raw]   # pilot_signs_cam1 (+ cam2): cube turned +90 x4, then the hand

static: frame by frame, every marker that both cameras see is triangulated with the rig
  calibration (<data>/../rig.json, from src.import_rig); frames whose corners do not match between
  the views (epipolar error > 1.5 px: blur, a steeply seen face) are left out. That gives each
  marker's true printed side length (median), and its single-camera pose error against stereo.
  The cube's size comes from the distance between adjacent face centres (side / sqrt 2), which
  does not depend on the printed marker sizes: it checks the stereo scale. Something moving (the
  hand) is fine: its position spread is printed. Other ArUco / AprilTag dictionaries are scanned
  too, in case the table marker is not DICT_4X4_50 (then set TABLE_DICT in src/constants.py).
signs: tracks the left camera with the normal pipeline and prints the cube yaw and hand yaw of
  each still period relative to the first: expected 0, +90, +180, +270 (cube) and +90 (hand).
"""
import argparse
import json

import cv2
import numpy as np

from src import constants as C
from src.common import ROOT, data_dirs
from src.geometry import invert_pose, unwrap_valid
from src.import_rig import load_rig
from src.stereo import MAX_EPIPOLAR_PX, epipolar_px, marker_pose, plane_frame, triangulate
from src.track import detect, make_detector, marker_solutions, track_clip
from src.video import read_frames

OTHER_DICTS = ["DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11"]


def frame_detections(path, detector):
    """Per-frame {id: corners} for a whole video, plus its first frame."""
    dets, first = [], None
    for _, _, frame in read_frames(path, prepared=False):
        first = frame if first is None else first
        dets.append(detect(detector, frame))
    return dets, first


def single_camera_pose(corners, size, cam, normal_hint):
    """IPPE pose whose normal is closest to the triangulated one (resolves the planar ambiguity)."""
    sols = marker_solutions(corners, size, cam)
    return max(sols, key=lambda s: float(s[0][:3, 2] @ normal_hint))


def static_check(dirs):
    cams, T_lr = load_rig(dirs["base"] / "rig.json")
    detector = make_detector()
    left, f_left = frame_detections(dirs["raw"] / "pilot_static_cam1.avi", detector)
    right, _ = frame_detections(dirs["raw"] / "pilot_static_cam2.avi", detector)
    seen_l, seen_r = set().union(*left), set().union(*right)
    print(f"pipeline markers seen (table ID {C.TABLE_ID}, hand ID {C.WRIST_ID}, cube {list(C.CUBE_IDS)}): "
          f"left {sorted(seen_l)}, right {sorted(seen_r)}  ({len(left)} frames)")
    for k in (C.TABLE_ID, C.WRIST_ID):
        if k not in seen_l and k not in seen_r:
            print(f"  ID {k} NOT FOUND in {C.TABLE_DICT if k == C.TABLE_ID else C.ARUCO_DICT}: check the scan below")
    gray = cv2.cvtColor(f_left, cv2.COLOR_BGR2GRAY)
    for name in OTHER_DICTS:
        d = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)))
        _, ids, _ = d.detectMarkers(gray)
        if ids is not None and name != "DICT_4X4_100":
            print(f"  also found in {name}: IDs {sorted(ids.ravel().tolist())}  <- if this is the table marker, set "
                  f"TABLE_DICT = \"{name}\" in src/constants.py")
    T_rl = invert_pose(T_lr)
    good = {}  # id -> list of (frame, triangulated corners)
    print(f"\nID  frames used  epipolar   printed side (stereo)      expected | single-camera vs stereo (median): "
          f"left, right | moved")
    for k in sorted(seen_l & seen_r):
        both = [(i, a[k], b[k]) for i, (a, b) in enumerate(zip(left, right)) if k in a and k in b]
        epi = np.array([epipolar_px(a, b, cams, T_lr) for _, a, b in both])
        keep = [(i, a, b) for (i, a, b), e in zip(both, epi) if e < MAX_EPIPOLAR_PX]
        if not keep:
            print(f"{k:3d}  0/{len(both)}  corners never match between the views (median epipolar {np.median(epi):.1f} px)")
            continue
        P = [triangulate(a, b, cams, T_lr) for _, a, b in keep]
        good[k] = list(zip([i for i, _, _ in keep], P))
        sides = np.array([1000 * np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)) for q in P])
        centres = np.array([q.mean(0) for q in P])
        expected = C.MARKER_SIZE.get(k)
        line = (f"{k:3d}  {len(keep):4d}/{len(both):<4d}  {np.median(epi):5.2f} px  {np.median(sides):6.1f} mm "
                f"({np.percentile(sides, 10):5.1f}-{np.percentile(sides, 90):5.1f})  "
                + (f"{1000 * expected:6.1f} mm" if expected else "  not a pipeline ID"))
        if expected:
            errs = {"left": [], "right": []}
            for (i, a, b), q in zip(keep, P):
                centre, n = plane_frame(q)
                Tl, _ = single_camera_pose(a, expected, cams["left"], n)
                Tr, _ = single_camera_pose(b, expected, cams["right"], T_lr[:3, :3] @ n)
                errs["left"].append(1000 * np.linalg.norm(Tl[:3, 3] - centre))
                errs["right"].append(1000 * np.linalg.norm((T_rl @ Tr)[:3, 3] - centre))
            line += f" | {np.median(errs['left']):5.1f} mm, {np.median(errs['right']):5.1f} mm"
        spread = 1000 * np.linalg.norm(centres.std(0))
        line += f" | {spread:.0f} mm" + (" (moving)" if spread > 5 else "")
        print(line)
    for k in sorted(seen_l ^ seen_r):
        print(f"{k:3d}  seen by the {'left' if k in seen_l else 'right'} camera only")
    if C.CUBE_TOP_ID in good:
        top = dict(good[C.CUBE_TOP_ID])
        print("\ncube size from adjacent face centres (centre distance = side / sqrt 2; does not depend on marker sizes):")
        for k in C.CUBE_SIDE_IDS:
            pairs = [(top[i], q) for i, q in good.get(k, []) if i in top]
            if pairs:
                d = np.array([1000 * np.linalg.norm(plane_frame(a)[0] - plane_frame(b)[0]) for a, b in pairs])
                ang = np.array([np.degrees(np.arccos(np.clip(plane_frame(a)[1] @ plane_frame(b)[1], -1, 1))) for a, b in pairs])
                print(f"  top-{k}: {len(pairs)} frames, cube side {np.median(d) * np.sqrt(2):5.1f} mm "
                      f"(expected {1000 * C.CUBE_SIDE:.1f}), face angle {np.median(ang):5.1f} deg (expected 90)")
    if C.TABLE_ID in good:
        table = np.median(np.array([q for _, q in good[C.TABLE_ID]]), axis=0)
        c13, n13 = plane_frame(table)
        if C.CUBE_TOP_ID in good:
            tops = [plane_frame(q) for _, q in good[C.CUBE_TOP_ID]]
            ang = np.median([np.degrees(np.arccos(np.clip(abs(n @ n13), -1, 1))) for _, n in tops])
            height = np.median([1000 * abs((c - c13) @ n13) for c, _ in tops])
            print(f"\ncube top vs table marker: tilt {ang:.1f} deg, height {height:.1f} mm above the table-marker plane "
                  f"(expected ~0 deg and {1000 * C.CUBE_SIDE:.0f} mm if the cube rests flat and the sheet is flat)")
        tilt = np.degrees(np.arccos(abs(n13 @ (-c13 / np.linalg.norm(c13)))))
        print(f"table marker: {100 * np.linalg.norm(c13):.1f} cm from the left camera, viewed {90 - tilt:.0f} deg above "
              f"the table plane")
        save_head_pose(table, dirs["base"] / "head_pose.json")


def save_head_pose(P, path):
    """Left camera pose in the table frame from the triangulated table-marker corners (ArUco order), so
    tests/make_preview.py can render from where the head really is."""
    T_table_cam = invert_pose(marker_pose(P))
    pose = dict(position=T_table_cam[:3, 3].tolist(), forward=T_table_cam[:3, 2].tolist(), down=T_table_cam[:3, 1].tolist())
    path.write_text(json.dumps(pose, indent=1))
    print(f"left camera in the table frame: position {np.round(pose['position'], 3)} m, looking along "
          f"{np.round(pose['forward'], 2)}; saved {path} (use: python -m tests.make_preview --head-pose {path})")


def still_periods(t, yaw, visible, min_s=0.8, tol_deg=1.5):
    """(start, end, mean yaw) of runs where the tracked yaw stays within tol_deg for at least min_s."""
    idx = np.flatnonzero(visible)
    periods, start = [], 0
    for j in range(1, len(idx) + 1):
        end_run = j == len(idx) or abs(np.degrees(yaw[idx[j]] - yaw[idx[start]])) > tol_deg or t[idx[j]] - t[idx[j - 1]] > 0.5
        if end_run:
            if t[idx[j - 1]] - t[idx[start]] >= min_s:
                periods.append((t[idx[start]], t[idx[j - 1]], float(np.median(yaw[idx[start]:idx[j - 1] + 1][visible[idx[start]:idx[j - 1] + 1]]))))
            start = j
    return periods


def signs_check(dirs):
    dirs["tracks"].mkdir(parents=True, exist_ok=True)
    tr = track_clip(dirs, "pilot_signs", debug_dir=ROOT / "outputs" / "real" / "debug")
    if not tr["table_found"]:
        print("table marker not found: run the static check first")
        return
    t = tr["t"]
    for name, expected in (("cube", "0, +90, +180, +270"), ("hand", "0, +90")):
        yaw = unwrap_valid(tr[f"{name}_yaw"], tr[f"{name}_visible"])  # hidden frames stay NaN, not 0
        periods = still_periods(t, yaw, tr[f"{name}_visible"])
        print(f"\n{name} yaw at each still period, relative to the first (expected {expected}):")
        for a, b, y in periods:
            used = ""
            if name == "cube":
                m = (t >= a) & (t <= b)
                used = f"  markers used {sorted({i for i, u in zip(C.CUBE_IDS, tr['cube_ids'][m].any(0)) if u})}"
            print(f"  {a:5.1f}-{b:5.1f} s: {np.degrees(y - periods[0][2]):+7.1f} deg{used}")
    print(f"\ndebug video: outputs/real/debug/pilot_signs_track.mp4")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("check", choices=("static", "signs"))
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    args = p.parse_args()
    dirs = data_dirs(args.data)
    if not (dirs["base"] / "rig.json").exists():
        raise SystemExit("no rig.json: run python -m src.import_rig --rig <rig_halcon20.json> first")
    static_check(dirs) if args.check == "static" else signs_check(dirs)


if __name__ == "__main__":
    main()
