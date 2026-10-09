"""Name the robot-head recordings after the shot list, in recording order.

    python -m src.rename_recordings --recordings <folder> --since 20261006 [--apply]
    python -m src.rename_recordings --recordings <folder> --since 20261006_1430 --names clip_17_rot-30_B --apply --overwrite

The recorder writes recording_<YYYYMMDD>_<HHMMSS>_cam1.avi and _cam2.avi. Recordings from --since on
are sorted by time, and each one (both cameras) gets the next unused name of shot_list.csv. Every copy
is logged in data/raw/renamed_log.csv, so a later run (e.g. the main set after the pilot) skips
recordings it already copied and continues at the next unused name. --names gives explicit names
instead (for a reshoot) to the first new recordings; a second run without --names names the ones after
them from the shot list. Without --apply it only prints the mapping. It COPIES into data/raw/ (the
originals are not touched) and does not overwrite unless --overwrite. Delete failed takes (both camera
files) before running, so the order stays right.
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2

from src.common import ROOT

STAMP = re.compile(r"recording_(\d{8}_\d{6})_cam1")


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_log(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--recordings", required=True, help="folder with recording_*_cam1/_cam2 files")
    p.add_argument("--since", required=True, help="only recordings at or after this time: YYYYMMDD[_HHMMSS]")
    p.add_argument("--shot-list", default=str(ROOT / "shot_list.csv"))
    p.add_argument("--names", nargs="*", default=None, help="explicit clip names instead of the shot list")
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--apply", action="store_true", help="copy the files (default: only print the mapping)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    out = Path(args.data)
    log_path = out / "renamed_log.csv"
    log = read_log(log_path)
    done_sources = {r["source"] for r in log}
    cam1 = [f for f in sorted(Path(args.recordings).expanduser().glob("recording_*_cam1.*"))
            if STAMP.match(f.name) and STAMP.match(f.name).group(1) >= args.since and f.name not in done_sources]
    if args.names:
        names = args.names
        if len(cam1) > len(names):  # reshoots filmed before the next clips: those keep their shot-list names
            print(f"naming the first {len(names)} of {len(cam1)} new recordings; run again without --names for the rest\n")
            cam1 = cam1[:len(names)]
    else:
        used = {r["name"] for r in log}
        names = [r["name"] for r in csv.DictReader(open(args.shot_list)) if r["name"] not in used]
    if len(cam1) > len(names):
        raise SystemExit(f"{len(cam1)} new recordings but only {len(names)} names left; delete failed takes or pass --names")
    plan = []
    for src1, name in zip(cam1, names):
        src2 = src1.with_name(src1.name.replace("_cam1.", "_cam2."))
        n1, n2 = frame_count(src1), frame_count(src2) if src2.exists() else -1
        note = "cam2 MISSING" if n2 < 0 else f"{n1} frames" + (f", cam2 has {n2}: NOT IN SYNC?" if abs(n1 - n2) > 1 else "")
        plan.append((src1, src2 if src2.exists() else None, name))
        print(f"{src1.name:36s} -> {name}_cam1{src1.suffix}  ({note})")
    print(f"\n{len(plan)} new recordings" + ("" if args.apply else "   (dry run: add --apply to copy)"))
    if not args.apply or not plan:
        return
    out.mkdir(parents=True, exist_ok=True)
    for src1, src2, name in plan:
        for src, cam in ((src1, 1), (src2, 2)):
            dst = out / f"{name}_cam{cam}{src1.suffix}"
            if src is not None and dst.exists() and not args.overwrite:
                raise SystemExit(f"{dst} exists already; not overwriting (use --overwrite for a reshoot)")
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if not log:
            w.writerow(["source", "name"])
        for src1, src2, name in plan:
            for src, cam in ((src1, 1), (src2, 2)):
                dst = out / f"{name}_cam{cam}{src1.suffix}"
                if src is not None:
                    shutil.copy2(src, dst)
                elif dst.exists():  # a reshoot without cam2 must not keep the old take's cam2
                    dst.unlink()
                    print(f"removed {dst.name} from the previous take (this recording has no cam2)")
            w.writerow([src1.name, name])
    print(f"copied into {out}; logged in {log_path}")


if __name__ == "__main__":
    main()
