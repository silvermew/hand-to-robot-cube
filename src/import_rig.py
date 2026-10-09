"""Use an already-calibrated stereo rig instead of a checkerboard video.

    python -m src.import_rig --rig <stereo rig calibration .json> [--data data/raw]

The rig file (OpenCV form of the HALCON calibration) holds K, dist (k1, k2, p1, p2, k3), image
size for "left" and "right", and T_lr with X_right = T_lr @ X_left. This writes the camera the
monocular pipeline uses (left = cam1 by default) to <data>/../camera.npz, and a copy of the
whole rig to <data>/../rig.json for the stereo checks. The rig file itself is only read.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.common import ROOT, data_dirs


def load_rig(path):
    rig = json.loads(Path(path).expanduser().read_text())
    cams = {side: dict(K=np.array(rig[side]["K"], float), dist=np.array(rig[side]["dist"], float),
                       image_size=np.array([rig[side]["width"], rig[side]["height"]])) for side in ("left", "right")}
    return cams, np.array(rig["T_lr"], float)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rig", required=True)
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--camera", choices=("left", "right"), default="left", help="left = cam1, right = cam2")
    args = p.parse_args()
    dirs = data_dirs(args.data)
    cams, T_lr = load_rig(args.rig)
    cam = cams[args.camera]
    dirs["base"].mkdir(parents=True, exist_ok=True)
    np.savez(dirs["camera"], K=cam["K"], dist=cam["dist"], image_size=cam["image_size"], rms=np.nan, n_views=0,
             source=f"{Path(args.rig).expanduser()} ({args.camera})")
    (dirs["base"] / "rig.json").write_text(Path(args.rig).expanduser().read_text())
    K = cam["K"]
    hfov = 2 * np.degrees(np.arctan(cam["image_size"][0] / 2 / K[0, 0]))
    print(f"wrote {dirs['camera']} from the {args.camera} camera: fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} "
          f"cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, {cam['image_size'][0]}x{cam['image_size'][1]}, horizontal FOV {hfov:.1f} deg")
    print(f"stereo baseline {1000 * np.linalg.norm(T_lr[:3, 3]):.1f} mm, angle between cameras "
          f"{np.degrees(np.arccos((np.trace(T_lr[:3, :3]) - 1) / 2)):.2f} deg; rig copied to {dirs['base'] / 'rig.json'}")


if __name__ == "__main__":
    main()
