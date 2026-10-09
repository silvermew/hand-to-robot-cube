"""Re-encode every clip to constant-frame-rate H.264 that OpenCV reads reliably.

    python -m src.prepare_videos [--data data/raw]

Phones record HEVC, variable frame rate (VFR) and rotation metadata, which OpenCV handles
badly. ffmpeg applies the rotation, resamples to the stream's nominal frame rate (duplicating
frames where a VFR stream skipped some) and writes <data>/../prepared/<clip>.mp4 plus a .json
with what it found. Without ffmpeg the tracker reads the raw files with OpenCV timestamps
(see src/video.py) and warns.
"""
import argparse
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from src.common import ROOT, clip_stem, data_dirs, list_videos, loud_warning


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt:stream_side_data=rotation"
           ":stream_tags=rotate:format=duration", "-of", "json", str(path)]
    info = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    stream = info["streams"][0]
    rotation = 0
    for side in stream.get("side_data_list", []):
        rotation = int(side.get("rotation", rotation))
    rotation = int(stream.get("tags", {}).get("rotate", rotation))
    r_fps, avg_fps = Fraction(stream["r_frame_rate"]), Fraction(stream["avg_frame_rate"] or stream["r_frame_rate"])
    return dict(codec=stream["codec_name"], width=stream["width"], height=stream["height"],
                pix_fmt=stream.get("pix_fmt", ""), rotation=rotation, r_fps=str(r_fps), avg_fps=float(avg_fps),
                vfr=abs(float(r_fps) - float(avg_fps)) > 0.01 * float(r_fps),
                duration=float(info.get("format", {}).get("duration", 0.0)))


def target_fps(info):
    """The nominal rate; a VFR phone stream reports its nominal rate as r_frame_rate."""
    r = Fraction(info["r_fps"])
    return r if 1 <= r <= 240 else Fraction(round(info["avg_fps"]))


def prepare(src, dst_dir):
    dst = dst_dir / (clip_stem(src) + ".mp4")
    meta = dst.with_suffix(".json")
    if dst.exists() and meta.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return json.loads(meta.read_text()), False
    info = probe(src)
    fps = target_fps(info)
    if "10" in info["pix_fmt"] or "12" in info["pix_fmt"]:
        print(f"  WARNING {src.name}: {info['pix_fmt']} looks like HDR video; turn HDR recording off if markers look washed out")
    # ffmpeg rotates by the display matrix by default; the fps filter makes the output constant-rate
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", f"fps={fps}", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    out = probe(dst)
    info.update(fps=float(fps), fps_fraction=str(fps), out_width=out["width"], out_height=out["height"],
                source=str(src))
    meta.write_text(json.dumps(info, indent=1))
    return info, True


def main(raw_dir=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(raw_dir or ROOT / "data" / "raw"), help="folder with the phone videos")
    args = p.parse_args([] if raw_dir else None)
    dirs = data_dirs(args.data)
    if not ffmpeg_available():
        loud_warning("ffmpeg not found: tracking will read raw videos with OpenCV timestamps (CAP_PROP_POS_MSEC). "
                     "Install it with: sudo apt install ffmpeg")
        return
    dirs["prepared"].mkdir(parents=True, exist_ok=True)
    clips, calib = list_videos(dirs["raw"])
    for src in calib + clips:
        info, done = prepare(src, dirs["prepared"])
        notes = [f"{info['codec']} {info['width']}x{info['height']}", f"-> {info['out_width']}x{info['out_height']}",
                 f"{info['fps']:.3f} fps"]
        if info["vfr"]:
            notes.append(f"VFR (avg {info['avg_fps']:.2f} fps) -> constant")
        if info["rotation"]:
            notes.append(f"rotation {info['rotation']} applied")
        print(f"{'prepared' if done else 'up to date'}: {src.name}: {', '.join(notes)}")


if __name__ == "__main__":
    main()
