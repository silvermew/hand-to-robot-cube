"""Frame reading with honest timestamps: prepared constant-rate videos, or raw videos via OpenCV."""
import json
from pathlib import Path

import cv2

from src.common import PIPELINE_CAMERA, VIDEO_EXTS


def find_video(dirs, stem):
    """(path, prepared?) for a clip: the prepared H.264 copy if it exists, else the raw file."""
    prepared = dirs["prepared"] / f"{stem}.mp4"
    if prepared.exists():
        return prepared, True
    for p in sorted(dirs["raw"].glob(stem + ".*")) + sorted(dirs["raw"].glob(f"{stem}_cam{PIPELINE_CAMERA}.*")):
        if p.suffix.lower() in VIDEO_EXTS:
            print(f"WARNING: {stem}: no prepared video, reading the raw file with OpenCV timestamps "
                  "(run src.prepare_videos; HEVC, VFR and rotation metadata may be mishandled)")
            return p, False
    raise FileNotFoundError(f"no video for {stem} in {dirs['prepared']} or {dirs['raw']}")


def video_fps(path, prepared):
    if prepared and Path(path).with_suffix(".json").exists():
        return json.loads(Path(path).with_suffix(".json").read_text())["fps"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def read_frames(path, prepared, stride=1, scale=1.0):
    """Yield (frame index, time in s, BGR frame). Prepared videos are constant-rate: t = index / fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = video_fps(path, prepared)
    i = 0
    while True:
        if i % stride:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps if prepared else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        yield i, t, frame
        i += 1
    cap.release()
