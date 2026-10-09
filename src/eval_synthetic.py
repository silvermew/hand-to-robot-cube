"""Tracking and extraction accuracy against the synthetic ground truth (synthetic data only).

    python -m src.eval_synthetic [--data data/synthetic/raw] [--out outputs/synthetic]

Per clip: median and 95th-percentile position / rotation error of the smoothed cube and hand
poses on tracked frames, yaw error, how often a marker that was fully visible got tracked,
phase-time errors (grasp vs the true moment the cube starts moving, release vs touchdown),
and the error of the measured cube yaw change. Writes <out>/eval_vs_truth.txt.
"""
import argparse
from pathlib import Path

import numpy as np

from src import constants as C
from src.common import ROOT, data_dirs
from src.geometry import R_from_quat, angle_between, wrap


def at_times(t_query, gt_t, x):
    return np.stack([np.interp(t_query, gt_t, x[:, k]) for k in range(x.shape[1])], 1) if x.ndim > 1 else \
        np.interp(t_query, gt_t, x)


def nearest_idx(t_query, gt_t):
    return np.clip(np.round(np.interp(t_query, gt_t, np.arange(len(gt_t)))).astype(int), 0, len(gt_t) - 1)


def clip_errors(tr, proc, gt):
    row = dict(clip=Path(str(proc["clip"])).name if proc is not None else "")
    for name in ("cube", "hand"):
        v = tr[f"{name}_visible"]
        if not v.any():
            continue
        t = tr["t"][v]
        pos_err = 1000 * np.linalg.norm(tr[f"{name}_pos"][v] - at_times(t, gt["t"], gt[f"{name}_pos"]), axis=1)
        idx = nearest_idx(t, gt["t"])
        rot_err = np.degrees(angle_between(R_from_quat(gt[f"{name}_quat"][idx]), R_from_quat(tr[f"{name}_quat"][v])))
        yaw_err = np.degrees(np.abs(wrap(tr[f"{name}_yaw"][v] - wrap(at_times(t, gt["t"], gt[f"{name}_yaw"])))))
        vis_key = f"visible_{C.WRIST_ID}" if name == "hand" else None
        if vis_key:
            fully = at_times(tr["t"], gt["t"], gt[vis_key]) > 0.97
        else:
            fully = np.max([at_times(tr["t"], gt["t"], gt[f"visible_{i}"]) for i in C.CUBE_IDS], 0) > 0.97
        row.update({f"{name}_pos_mm": np.median(pos_err), f"{name}_pos_p95": np.percentile(pos_err, 95),
                    f"{name}_rot_deg": np.median(rot_err), f"{name}_rot_p95": np.percentile(rot_err, 95),
                    f"{name}_yaw_deg": np.median(yaw_err),
                    f"{name}_recall": 100 * tr[f"{name}_visible"][fully].mean() if fully.any() else np.nan})
    if proc is not None:
        true_change = np.degrees(gt["cube_yaw"][-1] - gt["cube_yaw"][0])
        row.update(grasp_s=float(proc["t_grasp"]) - float(gt["t_grasp"]),
                   release_s=float(proc["t_release"]) - float(gt["t_touchdown"]),
                   dyaw_err=float(np.degrees(proc["cube_yaw_change"])) - true_change,
                   camera=str(gt["camera"]))
    return row


def main(raw_dir=None, out_dir=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(raw_dir or ROOT / "data" / "synthetic" / "raw"))
    p.add_argument("--out", default=str(out_dir or ROOT / "outputs" / "synthetic"))
    args = p.parse_args([] if raw_dir else None)
    dirs = data_dirs(args.data)
    rows = []
    for gt_path in sorted(dirs["ground_truth"].glob("*.npz")):
        stem = gt_path.stem
        if not (dirs["tracks"] / f"{stem}.npz").exists():
            continue
        tr = dict(np.load(dirs["tracks"] / f"{stem}.npz"))
        proc_path = dirs["processed"] / f"{stem}.npz"
        proc = dict(np.load(proc_path)) if proc_path.exists() else None
        row = clip_errors(tr, proc, dict(np.load(gt_path)))
        row["clip"] = stem
        rows.append(row)
    cols = [("clip", "{:20s}"), ("camera", "{:16s}"), ("cube_pos_mm", "{:6.1f}"), ("cube_rot_deg", "{:5.2f}"),
            ("cube_rot_p95", "{:5.2f}"), ("cube_yaw_deg", "{:5.2f}"), ("cube_recall", "{:5.0f}"),
            ("hand_pos_mm", "{:6.1f}"), ("hand_yaw_deg", "{:5.2f}"), ("hand_recall", "{:5.0f}"),
            ("grasp_s", "{:+5.2f}"), ("release_s", "{:+5.2f}"), ("dyaw_err", "{:+5.1f}")]
    header = "  ".join(c for c, _ in cols)
    lines = ["SYNTHETIC DATA: tracking and extraction vs ground truth", "pos in mm and rot/yaw in deg are medians over "
             "tracked frames (p95 = 95th percentile); recall = % of frames with a fully visible marker that were tracked;",
             "grasp_s / release_s = detected minus true time; dyaw_err = measured minus true cube yaw change (deg)", "",
             header]
    for r in rows:
        lines.append("  ".join(fmt.format(r[c]) if c in r and not (isinstance(r[c], float) and np.isnan(r[c]))
                               else "-".ljust(len(fmt.format(0)) if "s}" not in fmt else 5) for c, fmt in cols))
    for cam in sorted({r.get("camera", "") for r in rows}):
        sub = [r for r in rows if r.get("camera") == cam]
        lines.append(f"\n{cam}: cube pos {np.median([r['cube_pos_mm'] for r in sub]):.1f} mm, cube rot "
                     f"{np.median([r['cube_rot_deg'] for r in sub]):.2f} deg, hand yaw {np.median([r['hand_yaw_deg'] for r in sub if 'hand_yaw_deg' in r]):.2f} deg, "
                     f"|grasp| {np.mean([abs(r['grasp_s']) for r in sub if 'grasp_s' in r]):.2f} s, |release| "
                     f"{np.mean([abs(r['release_s']) for r in sub if 'release_s' in r]):.2f} s, |yaw change err| "
                     f"{np.mean([abs(r['dyaw_err']) for r in sub if 'dyaw_err' in r]):.2f} deg (n={len(sub)})")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "eval_vs_truth.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
