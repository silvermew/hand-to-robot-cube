"""Clip naming, data folders, and the SYNTHETIC / PLACEHOLDER labels shared by all scripts."""
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
CLIP_RE = re.compile(r"^clip_(\d+)_rot([+-]?\d+)_([ABC])(?:_cam(\d))?$")
PIPELINE_CAMERA = 1   # stereo recordings come as <clip>_cam1 / <clip>_cam2; the monocular pipeline uses cam1 (left)
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
WATERMARK = "SYNTHETIC"


def parse_clip_name(path):
    """{'number', 'angle', 'spot', 'camera'} from clip_<nn>_rot<+/-deg>_<spot>[_cam<k>], or None for other names."""
    m = CLIP_RE.match(Path(path).stem)
    if m is None:
        return None
    return dict(number=int(m.group(1)), angle=int(m.group(2)), spot=m.group(3),
                camera=int(m.group(4)) if m.group(4) else None)


def clip_stem(path):
    """Clip name without a _cam<k> suffix: what every derived file is called."""
    return re.sub(r"_cam\d$", "", Path(path).stem)


def is_calibration(path):
    return Path(path).stem.lower().startswith("calib")


def list_videos(folder, warn=True):
    """(clips, calibration videos) in a folder. Of a stereo pair only cam PIPELINE_CAMERA is a clip;
    pilot_* recordings belong to src.pilot_check; any other name gets a warning."""
    clips, calib = [], []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        info = parse_clip_name(p)
        if is_calibration(p):
            calib.append(p)
        elif info:
            if info["camera"] in (None, PIPELINE_CAMERA):
                clips.append(p)
        elif p.stem.lower().startswith("pilot"):
            continue
        elif warn:
            print(f"WARNING: skipping {p.name}: expected clip_<nn>_rot<+/-deg>_<A|B|C>[_cam1|_cam2].<ext> or calib*")
    return clips, calib


def data_dirs(raw_dir):
    """Folders derived from a raw-video folder: data/raw -> data/{prepared,tracks,...}."""
    base = Path(raw_dir).resolve().parent
    return dict(base=base, raw=Path(raw_dir).resolve(), prepared=base / "prepared", tracks=base / "tracks",
                processed=base / "processed", sim_demos=base / "sim_demos", camera=base / "camera.npz",
                ground_truth=base / "ground_truth")


def is_synthetic(path):
    return "synthetic" in Path(path).resolve().parts


def label_figure(fig, synthetic):
    """Watermark every figure made from synthetic data."""
    if synthetic:
        fig.text(0.5, 0.5, WATERMARK, fontsize=60, color="red", alpha=0.12, ha="center", va="center",
                 rotation=30, weight="bold")
        fig.text(0.01, 0.99, WATERMARK + " DATA", fontsize=10, color="red", ha="left", va="top", weight="bold")


def label_frame(img, synthetic):
    if synthetic:
        cv2.putText(img, WATERMARK, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
                    cv2.LINE_AA)
    return img


def loud_warning(text):
    bar = "!" * 78
    print(f"\n{bar}\n!! {text}\n{bar}\n", file=sys.stderr)
