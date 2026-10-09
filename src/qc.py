"""One-screen quality check per clip, to decide quickly whether to reshoot.

    python -m src.qc [--data data/raw] [--clips NAME ...] [--stride 2]

Tracks any clip that has no track yet (every --stride frames, no debug video, so it is fast),
then prints one row per clip and a verdict, OK or RESHOOT with the reason. Rates:
  top%     frames where the cube's top marker was used
  cube%    frames with any cube pose
  held%    frames with a cube pose while the cube is held (grasp..release); the world model
           learns from these
  hand%    wrist marker, between its first and last detection
  err px   median reprojection error of the accepted cube poses
  jump     largest rotation between accepted cube poses in adjacent frames (deg); a planar
           flip shows up here (pairs separated by a tracking gap are skipped: the cube may
           really have turned while hidden)
  cube     measured cube yaw change (rest pose after vs before) / the angle in the file name
  hand     measured hand yaw change between grasp and release
  lift     how far the hand rose while holding the cube (cm)
"""
import argparse

import numpy as np

from src.common import ROOT, clip_stem, data_dirs, is_synthetic, list_videos, parse_clip_name
from src.extract import detect_phases, interp_valid, pos_visible
from src.geometry import R_from_quat, angle_between, unwrap_valid
from src.track import track_all

MIN_CUBE_AT_REST = 0.8
MIN_HAND = 0.7
MIN_HELD = 0.25
MAX_ERR_PX = 2.5   # two-camera cube fits at rest: 0.7-2.4 px on 6 Oct, mostly the shadowed front-face marker
MAX_JUMP_DEG = 30
MAX_TURN_MISMATCH_DEG = 30
MIN_LIFT_CM, MAX_LIFT_CM = 3, 30   # the Panda reaches a 30 cm lift easily


def clip_qc(tr, intended):
    row = dict(table="yes" if tr["table_found"] else "NO")
    reasons = []
    if not tr["table_found"]:
        return row, ["table marker not found in the first frames"]
    t, cv, hv = tr["t"], tr["cube_visible"], tr["hand_visible"]
    row["top%"] = 100 * tr["cube_ids"][:, 0].mean()
    row["cube%"] = 100 * cv.mean()
    span = (t >= t[hv][0]) & (t <= t[hv][-1]) if hv.any() else np.zeros_like(hv)
    row["hand%"] = 100 * hv[span].mean() if span.any() else 0.0
    rest = cv & ~span  # the hand covers no marker: blur, focus and calibration alone
    row["err px"] = float(np.nanmedian(tr["cube_err"][rest])) if rest.any() else np.nan
    q = tr["cube_quat_raw"]
    pairs = np.flatnonzero(cv[:-1] & cv[1:])
    jumps = np.degrees(angle_between(R_from_quat(q[pairs]), R_from_quat(q[pairs + 1]))) if len(pairs) else [0.0]
    row["jump"] = float(np.max(jumps))
    try:
        ph = detect_phases(t, tr["cube_pos"], unwrap_valid(tr["cube_yaw"], cv), cv, tr["hand_pos"], hv, pos_visible(tr))
        held = (t >= ph["t_grasp"]) & (t <= ph["t_release"])
        row["held%"] = 100 * cv[held].mean()
        # the rest poses come from the frames before the hand appears and after it leaves
        before = t < (t[hv][0] if hv.any() else ph["t_grasp"])
        after = t > (t[hv][-1] if hv.any() else ph["t_release"])
        hand_yaw = unwrap_valid(tr["hand_yaw"], hv)
        hand_turn = np.degrees(interp_valid(ph["t_release"], t, hand_yaw, hv) - interp_valid(ph["t_grasp"], t, hand_yaw, hv))
        cube_turn = np.degrees(ph["end_yaw"] - ph["rest_yaw"])
        row["cube"] = f"{cube_turn:+.0f}/{intended:+.0f}" if np.isfinite(intended) else f"{cube_turn:+.0f}"
        row["hand"] = f"{hand_turn:+.0f}"
        z = np.where(held & hv, tr["hand_pos"][:, 2], np.nan)
        row["lift"] = 100 * (np.nanmax(z) - interp_valid(ph["t_grasp"], t, tr["hand_pos"][:, 2], hv)) if (held & hv).any() else np.nan
        if (before.sum() < 5 or after.sum() < 5 or cv[before].mean() < MIN_CUBE_AT_REST
                or cv[after].mean() < MIN_CUBE_AT_REST):
            reasons.append("cube not seen at rest while the hand is out of frame (start or end): keep the hand "
                           "out of view for ~1 s at both ends")
        if row["held%"] < 100 * MIN_HELD:
            reasons.append(f"cube hidden while held ({row['held%']:.0f}% tracked; world model needs >= {100 * MIN_HELD:.0f}%)")
        if np.isfinite(intended) and abs(cube_turn - intended) > MAX_TURN_MISMATCH_DEG:
            direction = " (the other way)" if cube_turn * intended < 0 and abs(cube_turn) > 15 else ""
            reasons.append(f"cube turned {cube_turn:+.0f} deg but the name says {intended:+.0f}{direction}")
        if np.isfinite(row["lift"]) and row["lift"] < MIN_LIFT_CM:
            reasons.append(f"cube barely lifted ({row['lift']:.0f} cm): pick it up ~8 cm")
        if np.isfinite(row["lift"]) and row["lift"] > MAX_LIFT_CM:
            reasons.append(f"lifted {row['lift']:.0f} cm: too high for the robot, ~8 cm is enough")
    except ValueError as e:
        row["held%"], row["cube"], row["hand"], row["lift"] = np.nan, "?", "?", np.nan
        reasons.append(f"phases not found: {e}")
    if row["hand%"] < 100 * MIN_HAND:
        reasons.append(f"wrist marker lost too often ({row['hand%']:.0f}%)")
    if row["err px"] > MAX_ERR_PX:
        reasons.append(f"high reprojection error {row['err px']:.1f} px (blur, focus or calibration?)")
    if row["jump"] > MAX_JUMP_DEG:
        reasons.append(f"cube pose jumps {row['jump']:.0f} deg between frames (a flip got through)")
    return row, reasons


def print_table(rows):
    cols = ["clip", "table", "top%", "cube%", "held%", "hand%", "err px", "jump", "cube", "hand", "lift", "verdict"]
    widths = [18, 5, 4, 5, 5, 5, 6, 4, 9, 5, 4, 0]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        cells = []
        for c, w in zip(cols, widths):
            v = r.get(c, "")
            cells.append((f"{v:.0f}" if c in ("top%", "cube%", "held%", "hand%", "jump", "lift") else
                          f"{v:.2f}" if c == "err px" else str(v)).ljust(w) if not isinstance(v, str) else v.ljust(w))
        print("  ".join(cells))


def qc_report(raw_dir, clips=None, stride=2, workers=8, retrack=False):
    """Track clips that have no track yet (fast settings), then print the one-screen table."""
    dirs = data_dirs(raw_dir)
    stems = clips or [clip_stem(c) for c in list_videos(dirs["raw"])[0]]
    missing = [s for s in stems if retrack or not (dirs["tracks"] / f"{s}.npz").exists()]
    if missing:
        print(f"tracking {len(missing)} clip(s) (every {stride} frames)...")
        out = ROOT / "outputs" / ("synthetic" if is_synthetic(raw_dir) else "real")
        track_all(raw_dir, out, missing, stride, debug=False, workers=workers)
    rows, placeholder = [], False
    for s in stems:
        path = dirs["tracks"] / f"{s}.npz"
        if not path.exists():
            rows.append(dict(clip=s, verdict="RESHOOT: tracking failed"))
            continue
        tr = dict(np.load(path))
        placeholder |= bool(tr["placeholder_intrinsics"])
        info = parse_clip_name(s)
        row, reasons = clip_qc(tr, info["angle"] if info else np.nan)
        row["clip"] = s
        row["verdict"] = "OK" if not reasons else "RESHOOT: " + "; ".join(reasons)
        rows.append(row)
    print_table(rows)
    n_ok = sum(r["verdict"] == "OK" for r in rows)
    print(f"\n{n_ok}/{len(rows)} OK" + ("  (PLACEHOLDER INTRINSICS: run src.calibrate first)" if placeholder else ""))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--clips", nargs="*", default=None)
    p.add_argument("--stride", type=int, default=2, help="track every Nth frame for clips without a track")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retrack", action="store_true", help="track again even if a track exists")
    args = p.parse_args()
    qc_report(args.data, args.clips, args.stride, args.workers, args.retrack)


if __name__ == "__main__":
    main()
