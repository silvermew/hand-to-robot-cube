"""Replay every processed clip in the simulator; per-clip table, success rate, embodiment-gap plot, GIFs.

    python -m src.replay_all [--data data/raw] [--out outputs/real] [--turn hand|cube] [--gif CLIP ...]

The sim cube starts at the same spot for every clip (CUBE_HOME, yaw: whichever multiple of 90 deg
keeps the carry in reach); the human motion is applied relative to it (see sim_replay.retarget).
This is a scripted controller driven by the human data, not a learned policy. Writes
<out>/replay_table.csv, <out>/embodiment_gap.png, <out>/sim_replay/<clip>.mp4, side-by-side GIFs
for the best clip and one failure, and saves every replay to <data>/../sim_demos/.
With --turn cube the gripper turns by the cube's measured turn instead of the hand's
(sim_replay.with_cube_turn); that writes only <out>/replay_table_cube_turn.csv and
<out>/embodiment_gap_cube_turn.png.
"""
import argparse
import csv
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import constants as C
from src.common import ROOT, data_dirs, is_synthetic, label_figure, label_frame
from src.geometry import R_from_quat, invert_pose, make_pose, yaw_of
from src.sim_replay import (CUBE_HOME_XY, PHASES, TABLE_Z, demo_path, eef_pose, format_metrics, hold_mask, make_env,
                            retarget, run_replay, save_demo, save_video, with_cube_turn)
from src.video import find_video, read_frames


def sim_start():
    return np.r_[CUBE_HOME_XY, TABLE_Z + C.CUBE_SIDE / 2], 0.0


def start_yaw_within_reach(env, clip, pos0, candidates=(0.0, np.pi / 2, np.pi, -np.pi / 2)):
    """Sim cube start yaw (multiples of 90 deg) whose retargeted hold needs the least reach clamping.
    The human displacement is applied in the cube's start frame, so its direction in the sim depends on
    this yaw; a 24 cm carry pointed away from the robot base would otherwise leave the arm's reach."""
    env.reset()
    heading_home = yaw_of(eef_pose(env)[1])
    hold = hold_mask(clip)
    clamped = [retarget(clip, pos0, y, heading_home)[2][hold].mean() for y in candidates]
    return candidates[int(np.argmin(clamped))]


def replay_clips(raw_dir, out_dir, render=True, turn="hand"):
    """turn="hand": the gripper turns like the human hand. turn="cube": like the cube (with_cube_turn); those
    replays are not saved as demos or videos."""
    dirs = data_dirs(raw_dir)
    out = Path(out_dir)
    synthetic = is_synthetic(raw_dir)
    env = make_env(render=render)
    pos0, _ = sim_start()
    rows = []
    for path in sorted(dirs["processed"].glob("*.npz")):
        clip = dict(np.load(path))
        name = path.stem
        if turn == "cube":
            clip = with_cube_turn(clip)
        try:
            yaw0 = start_yaw_within_reach(env, clip, pos0)
            metrics, frames, log = run_replay(env, clip, pos0, yaw0, True, record=render, synthetic=synthetic)
        except ValueError as e:  # e.g. too little of the hand tracked to build a trajectory
            print(f"{name}: SKIPPED ({e})")
            continue
        if turn == "hand":
            save_demo(demo_path(dirs["sim_demos"], name, pos0, yaw0, True), log, metrics, name, pos0, yaw0, True)
        if render:
            save_video(frames, out / "sim_replay", name, gif=False)
        rows.append(dict(clip=name, intended_angle=float(clip["intended_angle"]), sim_start_yaw_deg=np.degrees(yaw0),
                         human_displacement_cm=100 * float(np.linalg.norm(clip["cube1"][:2] - clip["cube0"][:2])), **metrics))
        print(f"{name}: {format_metrics(metrics)}")
    env.close()
    return rows


def write_table(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})


def plot_embodiment_gap(rows, path, synthetic, turn="hand"):
    x = np.array([r["human_dyaw_deg"] for r in rows])
    y = np.array([r["sim_dyaw_deg"] for r in rows])
    ok = np.array([r["success"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = np.max(np.abs(np.r_[x, y, 30])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x (no gap)")
    ax.fill_between([-lim, lim], [-lim - 15, lim - 15], [-lim + 15, lim + 15], color="0.9", label="±15 deg (success band)")
    ax.scatter(x[ok], y[ok], c="C2", s=30, edgecolor="k", lw=0.4, label="success", zorder=3)
    ax.scatter(x[~ok], y[~ok], c="C3", s=30, marker="X", edgecolor="k", lw=0.4, label="failure", zorder=3)
    gap = np.abs(y - x)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("human cube yaw change (deg)")
    ax.set_ylabel("sim cube yaw change (deg)")
    ax.set_title(f"Gripper turned like the {turn}: mean |sim - human| = {gap.mean():.1f} deg (n={len(x)})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def project_axes(img, T_cam_obj, K, dist, length):
    rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
    cv2.drawFrameAxes(img, K, dist, rvec, T_cam_obj[:3, 3], length, 2)


def side_by_side_gif(raw_dir, clip_name, out_path, step=3, height=270):
    """Left: the human video with the tracked cube / hand axes; right: the sim agentview at the same clip time."""
    dirs = data_dirs(raw_dir)
    synthetic = is_synthetic(raw_dir)
    demo_file = next(p for p in sorted(dirs["sim_demos"].glob(f"{clip_name}__*.npz")) if "rawwrist" not in p.name)
    demo = np.load(demo_file)
    tr = np.load(dirs["tracks"] / f"{clip_name}.npz")
    steps = np.arange(0, len(demo["clip_time"]), step)
    fps = float(tr["fps"]) * int(tr["stride"])
    wanted = np.round(demo["clip_time"][steps] * fps).astype(int)
    path, prepared = find_video(dirs, clip_name)
    scale = height / float(tr["image_size"][1])
    video = {i: f for i, _, f in read_frames(path, prepared, scale=scale) if i in set(wanted)}
    K = tr["K"].copy()
    K[:2] *= scale
    T_cam_table = invert_pose(tr["T_table_cam"])
    t_track = tr["t"]
    frames = []
    for s, i in zip(steps, wanted):
        left = video.get(i, video[max(k for k in video if k <= i)] if any(k <= i for k in video) else None)
        if left is None:
            continue
        left = left.copy()
        k = int(np.argmin(np.abs(t_track - demo["clip_time"][s])))
        for name in ("cube", "hand"):
            if tr[f"{name}_visible"][k] and (f"{name}_pos_visible" not in tr or tr[f"{name}_pos_visible"][k]):
                project_axes(left, T_cam_table @ make_pose(R_from_quat(tr[f"{name}_quat"][k]), tr[f"{name}_pos"][k]),
                             K, tr["dist"], 0.04)
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right = cv2.resize(demo["agentview"][s], (height, height), interpolation=cv2.INTER_AREA)
        frame = np.hstack([left, right])
        phase = PHASES[int(demo["phase"][s])]
        for j, text in enumerate([f"t={demo['clip_time'][s]:4.1f}s {phase}", "human (tracked axes) | sim replay"]):
            cv2.putText(frame, text, (6, 16 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (6, 16 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(label_frame(frame, synthetic))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=step * 50, loop=0)  # ms per frame; sim steps are 50 ms
    return out_path


def main(raw_dir=None, out_dir=None, turn="hand"):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(raw_dir or ROOT / "data" / "raw"))
    p.add_argument("--out", default=out_dir)
    p.add_argument("--gif", nargs="*", default=None, help="only make side-by-side GIFs for these clips")
    p.add_argument("--turn", choices=("hand", "cube"), default=turn,
                   help="turn the gripper like the human hand (default) or like the cube (its measured turn)")
    args = p.parse_args([] if raw_dir else None)
    synthetic = is_synthetic(args.data)
    out = Path(args.out or ROOT / "outputs" / ("synthetic" if synthetic else "real"))
    out.mkdir(parents=True, exist_ok=True)
    if args.gif:
        for name in args.gif:
            print(f"wrote {side_by_side_gif(args.data, name, out / 'gifs' / f'{name}_side_by_side.gif')}")
        return
    cube = args.turn == "cube"
    rows = replay_clips(args.data, out, render=not cube, turn=args.turn)
    if not rows:
        print("no processed clips")
        return
    suffix = "_cube_turn" if cube else ""
    write_table(rows, out / f"replay_table{suffix}.csv")
    plot_embodiment_gap(rows, out / f"embodiment_gap{suffix}.png", synthetic, args.turn)
    ok = [r for r in rows if r["success"]]
    print(f"\nreplay success {len(ok)}/{len(rows)} = {100 * len(ok) / len(rows):.0f}% (gripper turned like the {args.turn})"
          + ("  (SYNTHETIC)" if synthetic else ""))
    if cube:
        return
    best = min(ok or rows, key=lambda r: abs(r["sim_dyaw_deg"] - r["human_dyaw_deg"]) + r["pos_err_cm"])
    picks = [("best", best)]
    failures = [r for r in rows if not r["success"]]
    if failures:
        picks.append(("failure", max(failures, key=lambda r: abs(r["sim_dyaw_deg"] - r["human_dyaw_deg"]))))
    for tag, r in picks:
        gif = side_by_side_gif(args.data, r["clip"], out / "gifs" / (tag + "_" + r["clip"] + ".gif"))
        print(f"wrote {gif}")


if __name__ == "__main__":
    main()
