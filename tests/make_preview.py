"""Synthetic preview of the real setup, to rehearse the pilot before filming.

    python -m tests.make_preview [--rig data/rig.json] [--out data/synthetic/preview]

Same renderer as tests/make_synthetic.py, changed to match the robot head: the rig's left-camera
intrinsics and distortion (1936x1096), 30 fps, the camera facing the person, a black table with
the table marker (ID 13) at the image centre and three white crosses beyond it, a 100 mm hand
marker. Writes the three pilot clips (rot+60 at A, rot-90 at B, rot+0 at C) as <clip>_cam1.mp4
with ground truth.

ASSUMED unless --head-pose is given: the head pose (left camera ~78 cm from the table marker,
~40 deg down); `pilot_check static` measures the real one into data/head_pose.json. Always assumed:
the crosses (15 cm beyond the marker, 12 cm apart). Crosses are named from the person's seat:
A = their left, B = middle, C = their right (mirrored in the camera image).
--look aims the camera elsewhere (e.g. at the crosses: --look 0 -0.15 0.08); --grasp side
renders the side pinch (thumb and fingers on the cube's left and right faces).
"""
import argparse
import json
import multiprocessing
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT
from src.import_rig import load_rig
from tests import make_synthetic as S

HEAD_POS, HEAD_LOOK = np.array([0.0, 0.60, 0.50]), np.array([0.0, 0.0, 0.0])  # person on the -y side
SPOTS = {"A": (-0.12, -0.15), "B": (0.0, -0.15), "C": (0.12, -0.15)}          # person's left = -x
CLIPS = [("clip_01_rot+60_A", 60, "A", None), ("clip_02_rot-90_B", -90, "B", None), ("clip_03_rot+0_C", 0, "C", None)]
TRANSPORT_CLIPS = [("clip_04_rot+0_A", 0, "A", "C"), ("clip_05_rot+60_C", 60, "C", "B")]  # carried to another cross
FPS = 30


def black_table(rng, px_per_mm=2, extent=((-0.40, 0.80), (-0.45, 0.75))):
    """Matte black table with three white crosses (40 mm arms, 6 mm wide) at the spots."""
    (x0, x1), (y0, y1) = extent
    w, h = int((x1 - x0) * 1000 * px_per_mm), int((y1 - y0) * 1000 * px_per_mm)
    tex = np.clip(22 + rng.normal(0, 3, (h, w, 1)), 0, 255).repeat(3, axis=2).astype(np.uint8)
    for sx, sy in SPOTS.values():
        cx, cy = int((sx - x0) * 1000 * px_per_mm), int((y1 - sy) * 1000 * px_per_mm)
        arm, half_w = 20 * px_per_mm, 3 * px_per_mm
        tex[cy - half_w:cy + half_w, cx - arm:cx + arm] = 235
        tex[cy - arm:cy + arm, cx - half_w:cx + half_w] = 235
    corners = np.array([[x0, y1, 0], [x1, y1, 0], [x1, y0, 0], [x0, y0, 0]])
    return tex, corners


def setup(rig_path, look=None, grasp="forearm", pos=None):
    """Point the shared renderer at the robot head (runs in every worker process)."""
    if look is not None:
        HEAD_LOOK[:] = look
    if pos is not None:
        HEAD_POS[:] = pos
    S.GRASP = grasp
    cams, _ = load_rig(rig_path)
    S.W, S.H = (int(v) for v in cams["left"]["image_size"])
    S.K_TRUE, S.DIST_TRUE = cams["left"]["K"], cams["left"]["dist"]
    S.CAMERAS["head"] = (HEAD_POS, HEAD_LOOK)
    S.SPOTS = SPOTS
    S.table_texture = black_table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rig", default=str(ROOT / "data" / "rig.json"))
    p.add_argument("--out", default=str(ROOT / "data" / "synthetic" / "preview"))
    p.add_argument("--look", type=float, nargs=3, default=None,
                   help="point the camera aims at, table frame (m); default: the table marker")
    p.add_argument("--grasp", choices=("forearm", "side"), default="forearm",
                   help="pinch on the faces toward forearm / fingertips, or on the two side faces")
    p.add_argument("--head-pose", default=None, help="data/head_pose.json written by `pilot_check static`")
    p.add_argument("--transport", action="store_true", help="add two clips that carry the cube to another cross")
    args = p.parse_args()
    if args.head_pose:  # the measured head pose; the renderer's look-at camera assumes no roll
        pose = json.loads(Path(args.head_pose).read_text())
        HEAD_POS[:] = pose["position"]
        HEAD_LOOK[:] = np.array(pose["position"]) + 0.8 * np.array(pose["forward"])
    if args.look:
        HEAD_LOOK[:] = args.look
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    setup(args.rig, grasp=args.grasp)
    clips = CLIPS + (TRANSPORT_CLIPS if args.transport else [])
    jobs = [(name, angle, spot, FPS, "head", 2000 + n, out, place) for n, (name, angle, spot, place) in enumerate(clips)]
    with multiprocessing.get_context("spawn").Pool(len(jobs), initializer=setup,
                                                    initargs=(args.rig, HEAD_LOOK.copy(), args.grasp, HEAD_POS.copy())) as pool:
        for name in pool.imap_unordered(S.write_clip, jobs):
            (out / "raw" / f"{name}.mp4").replace(out / "raw" / f"{name}_cam1.mp4")  # stereo naming, left camera
            print(f"wrote {name}_cam1.mp4")
    # one still per clip at mid-carry, to see what the head camera sees
    stills = []
    for name, *_ in clips:
        cap = cv2.VideoCapture(str(out / "raw" / f"{name}_cam1.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(5.0 * FPS))
        ok, frame = cap.read()
        if ok:
            stills.append(cv2.resize(frame, (S.W // 3, S.H // 3)))
    if stills:
        cv2.imwrite(str(out / "head_view.png"), np.hstack(stills))
        print(f"wrote {out / 'head_view.png'}")


if __name__ == "__main__":
    main()
