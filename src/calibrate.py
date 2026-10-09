"""Camera intrinsics from a checkerboard video; saves <data>/../camera.npz.

    python -m src.calibrate [--data data/raw] [--video <path>]

How to film it (same phone, lens, resolution, fps and zoom as the clips; stabilisation and
HDR off; focus locked at ~50 cm): tape the printed checkerboard (9x6 inner corners, 25 mm)
flat to a book. Hold the phone still on its stand and move the board slowly for 30-40 s:
through the centre, all four corners and the edges of the image, at 30-70 cm, tilted up to
~45 deg left/right/up/down. Keep the whole board in frame and avoid motion blur.

If no calibration exists, load_camera() returns PLACEHOLDER intrinsics (70 deg horizontal
field of view, no distortion) and warns loudly when used on real data.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src import constants as C
from src.common import ROOT, data_dirs, is_synthetic, list_videos, loud_warning
from src.video import find_video, read_frames

MAX_VIEWS = 40
PLACEHOLDER_HFOV_DEG = 70.0


def board_points():
    cols, rows = C.CHECKER_INNER
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * C.CHECKER_SQUARE
    return grid


def detect_boards(path, prepared, samples=150):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
    cap.release()
    stride = max(1, n // samples)
    found, size = [], None
    for i, t, frame in read_frames(path, prepared, stride=stride):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCornersSB(gray, C.CHECKER_INNER, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            found.append((i, corners.reshape(-1, 2)))
    return found, size


def calibrate(found, size):
    """Calibrate on up to MAX_VIEWS evenly spaced detections, drop views with >2x the median error, redo once."""
    views = found[:: max(1, len(found) // MAX_VIEWS)][:MAX_VIEWS]
    for _ in range(2):
        obj = [board_points()] * len(views)
        img = [v[1].astype(np.float32) for v in views]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=cv2.CALIB_FIX_K3)
        errs = []
        for o, im, r, t in zip(obj, img, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, r, t, K, dist)
            errs.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im) ** 2, axis=1))))
        errs = np.array(errs)
        keep = errs <= 2 * np.median(errs)
        if keep.all():
            break
        views = [v for v, k in zip(views, keep) if k]
    return dict(K=K, dist=dist.ravel(), rms=rms, n_views=len(views), per_view_err=errs)


def placeholder_camera(width, height):
    f = (width / 2) / np.tan(np.radians(PLACEHOLDER_HFOV_DEG) / 2)
    return dict(K=np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]), dist=np.zeros(5),
                image_size=np.array([width, height]), rms=np.nan, placeholder=True)


def load_camera(dirs, width, height, quiet=False):
    """Calibrated intrinsics if <data>/../camera.npz exists and matches the frame size, else a loud placeholder."""
    path = dirs["camera"]
    if path.exists():
        cam = dict(np.load(path))
        if tuple(cam["image_size"]) == (width, height):
            cam["placeholder"] = False
            return cam
        print(f"WARNING: {path} is for {tuple(cam['image_size'])} frames but the video is {width}x{height}")
    cam = placeholder_camera(width, height)
    if not quiet:
        message = (f"USING PLACEHOLDER INTRINSICS ({PLACEHOLDER_HFOV_DEG:.0f} deg FOV, no distortion): "
                   "poses will be biased. Run src.calibrate on a checkerboard video first.")
        if is_synthetic(dirs["base"]):
            print("note: " + message)
        else:
            loud_warning(message)
    return cam


def main(raw_dir=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(raw_dir or ROOT / "data" / "raw"))
    p.add_argument("--video", default=None, help="checkerboard video (default: calib*.* in --data)")
    args = p.parse_args([] if raw_dir else None)
    dirs = data_dirs(args.data)
    if args.video:
        path, prepared = Path(args.video), False
    else:
        _, calib = list_videos(dirs["raw"], warn=False)
        if not calib:
            if dirs["camera"].exists():
                print(f"no calib* video; using the existing {dirs['camera']} "
                      f"(source: {np.load(dirs['camera'])['source']})")
            else:
                print(f"no calib* video in {dirs['raw']} and no camera.npz: the placeholder will be used "
                      "(or import a calibrated rig with src.import_rig)")
            return
        path, prepared = find_video(dirs, calib[0].stem)
    found, size = detect_boards(path, prepared)
    print(f"{path.name}: checkerboard found in {len(found)} sampled frames")
    if len(found) < 10:
        print("too few views (need >= 10): refilm with the whole board in frame, moving slowly")
        return
    cal = calibrate(found, size)
    K, dist = cal["K"], cal["dist"]
    hfov = 2 * np.degrees(np.arctan(size[0] / 2 / K[0, 0]))
    np.savez(dirs["camera"], K=K, dist=dist, image_size=np.array(size), rms=cal["rms"], n_views=cal["n_views"],
             source=str(path))
    print(f"RMS reprojection error {cal['rms']:.3f} px over {cal['n_views']} views (good: < 0.5 px; "
          f"per-view max {cal['per_view_err'].max():.2f} px)")
    print(f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f} (horizontal FOV {hfov:.1f} deg), "
          f"dist {np.round(dist, 4)}")
    print(f"saved {dirs['camera']}")
    gt = sorted(dirs["ground_truth"].glob("*.npz")) if dirs["ground_truth"].exists() else []
    if gt:
        truth = np.load(gt[0])
        Kt = truth["K"]
        print(f"SYNTHETIC check vs true intrinsics: fx {K[0, 0] - Kt[0, 0]:+.1f} fy {K[1, 1] - Kt[1, 1]:+.1f} "
              f"cx {K[0, 2] - Kt[0, 2]:+.1f} cy {K[1, 2] - Kt[1, 2]:+.1f} px; dist true {truth['dist'][:2]}")


if __name__ == "__main__":
    main()
