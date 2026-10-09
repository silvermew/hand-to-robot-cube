"""Tracks -> processed clips: hand and cube pose in the table frame, yaws and phase times.

    python -m src.extract [--data data/raw] [--out outputs/real]

Hand yaw = heading of the wrist marker's top edge (toward the fingertips); cube yaw = heading
of the cube's +x axis (toward the ID 2 face). Both are positive counter-clockwise seen from
above, unwrapped over the frames where they were tracked, and also stored relative to the
grasp instant. Untracked frames stay NaN.

Phases, by a simple rule on the tracked cube. The cube has a start rest pose (before the
hand appears) and a final rest pose (after the hand has left):
  grasp   = the last frame at the start rest pose before the cube is clearly away from it
            (> 1.5 cm, > 8 deg, or lifted > 1.5 cm), i.e. when it starts moving with the hand;
  release = the first frame of the final rest after the cube was last clearly away from it,
            i.e. when it stops moving after being put down.
If the cube is hidden across one of these events (> 0.2 s), the hand decides. While it holds
the cube on the table it is at its holding height: within 1 cm of its lowest height at which it
is slow, above the cube's top (resting flat on the table is lower, the top of the lift higher).
The grasp is the last time it is there before the cube is seen moving (it starts lifting), the
release the first time after (the cube touches down). If the cube is never seen moving, the
hold is the longest stretch without a rest frame: the grasp comes from its first half, the
release from its second. Which rule was used is stored in phase_notes.
The world model's gripper flag comes from these times, i.e. partly from cube motion.

Writes <data>/../processed/<clip>.npz, <out>/phases/<clip>.png, <out>/grip_gain.png and
<out>/extract_summary.csv.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import constants as C
from src.common import ROOT, data_dirs, is_synthetic, label_figure, parse_clip_name
from src.geometry import unwrap_valid

AT_REST_POS, AT_REST_YAW = 0.006, np.radians(3)   # tolerant of single-marker pose bias (~2-4 mm)
AWAY_POS, AWAY_YAW = 0.015, np.radians(8)          # clearly moved
MIN_LIFT = 0.015
HAND_SLOW = 0.10                                   # m/s; the hand holding the cube on the table moves 1-7 cm/s
HAND_LEVEL = 0.01                                  # within this of its holding height, the hand holds the cube
                                                   # on the table (before lifting / after putting it down)
MAX_GAP_S = 0.2                                    # larger gaps around an event use the hand
SUSTAIN_S, SUSTAIN_FRAC = 0.5, 0.6                 # "moved" must persist: a pose glitch while the hand covers
                                                   # the top marker lasts a few frames
NEAR_HAND = 0.18                                   # wrist marker to cube centre at grasp, metres
MIN_HAND_FRAMES = 10


def interp_valid(t_query, t, x, valid):
    return np.interp(t_query, t[valid], x[valid]) if valid.any() else np.nan


def hand_at_level(t, hand_pos, hand_vis, t_a, t_b):
    """Times in [t_a, t_b] when the tracked hand is at its holding height there: within HAND_LEVEL of the lowest
    height at which it is slow, above the cube's top (a hand resting flat on the table is lower). The top of the
    lift is higher, so the grasp is the last of these times before the lift and the release the first after it.
    Fast turns leave no dwell to find, only a turning point; at a transport's far cross the height differs."""
    m = hand_vis & (t >= t_a) & (t <= t_b) & (hand_pos[:, 2] > C.CUBE_SIDE)
    if m.sum() < 2:
        return np.array([])
    tt, p = t[m], hand_pos[m]
    slow = np.linalg.norm(np.gradient(p, tt, axis=0), axis=1) < HAND_SLOW
    level = (p[slow] if slow.any() else p)[:, 2].min()
    return tt[p[:, 2] <= level + HAND_LEVEL]


def sustained(t, visible, flag, forward):
    """Keep a flagged frame only if at least 3 tracked frames in the next (or previous) SUSTAIN_S follow it and
    most are flagged too: a glimpse of the cube between the fingers is not evidence that it moved. visible:
    the frames that can show the flag."""
    out = np.zeros_like(flag)
    for i in np.flatnonzero(flag):
        w = visible & ((t > t[i]) & (t <= t[i] + SUSTAIN_S) if forward else (t < t[i]) & (t >= t[i] - SUSTAIN_S))
        out[i] = w.sum() >= 3 and flag[w].mean() >= SUSTAIN_FRAC
    return out


def rest_pose(cube_pos, cube_yaw, mask, pos_mask):
    if not (mask & pos_mask).any():
        raise ValueError("cube position at rest never seen by both cameras")
    return np.median(cube_pos[mask & pos_mask], axis=0), float(np.median(cube_yaw[mask]))


def pos_visible(tr):
    """Frames with a trusted cube position: both cameras (stereo tracks), else every tracked frame."""
    return tr["cube_pos_visible"] if "cube_pos_visible" in tr else tr["cube_visible"]


def detect_phases(t, cube_pos, cube_yaw, cube_vis, hand_pos, hand_vis, cube_pos_vis=None):
    """Grasp and release times, rest poses and notes; cube_yaw must be unwrapped. Raises ValueError when impossible.
    cube_pos_vis: frames whose cube position is trusted (default all tracked); the others count for the yaw only."""
    v = cube_vis
    pv = v if cube_pos_vis is None else v & cube_pos_vis
    idx = np.flatnonzero(v)
    if len(idx) < 10:
        raise ValueError("cube tracked in fewer than 10 frames")
    t_hand0 = t[hand_vis][0] if hand_vis.any() else np.inf
    t_hand1 = t[hand_vis][-1] if hand_vis.any() else -np.inf
    start = v & (t < t_hand0)
    end = v & (t > t_hand1)
    if start.sum() < 3:
        start = v & (t <= t[idx[0]] + 0.5)
    if end.sum() < 3:
        end = v & (t >= t[idx[-1]] - 0.5)
    pos0, yaw0 = rest_pose(cube_pos, cube_yaw, start, pv)
    pos1, yaw1 = rest_pose(cube_pos, cube_yaw, end, pv)
    lifted = pv & (cube_pos[:, 2] - pos0[2] > MIN_LIFT)

    def near(pos, yaw, tol_pos, tol_yaw):
        return pv & (np.linalg.norm(cube_pos - pos, axis=1) < tol_pos) & (np.abs(cube_yaw - yaw) < tol_yaw)

    def away(pos, yaw):
        """Clearly moved: by yaw in any tracked frame, by position or lift where the position is trusted."""
        moved = pv & (np.linalg.norm(cube_pos - pos, axis=1) > AWAY_POS)
        return v & (moved | (np.abs(cube_yaw - yaw) > AWAY_YAW) | lifted)

    at0 = near(pos0, yaw0, AT_REST_POS, AT_REST_YAW) & ~lifted
    at1 = near(pos1, yaw1, AT_REST_POS, AT_REST_YAW) & ~lifted
    a0, a1 = away(pos0, yaw0), away(pos1, yaw1)
    away0 = sustained(t, pv | a0, a0, forward=True)  # a one-camera frame at the old yaw cannot show a lift
    away1 = sustained(t, pv | a1, a1, forward=False)
    moving = away0 & away1  # clearly away from both rest poses: lifted, turning or carried
    notes = []
    if moving.any():
        i_first, i_last = np.flatnonzero(moving)[[0, -1]]
        before = np.flatnonzero(at0[:i_first])
        t_a = t[before[-1]] if len(before) else t[idx[0]]
        t_grasp = t_a
        if t[i_first] - t_a > MAX_GAP_S:
            at = hand_at_level(t, hand_pos, hand_vis, t_a, t[i_first])
            t_grasp = at[-1] if len(at) else 0.5 * (t_a + t[i_first])
            notes.append(f"cube not seen at rest in the {t[i_first] - t_a:.2f} s before it moved; used the hand")
        after = i_last + 1 + np.flatnonzero(at1[i_last + 1:])
        if not len(after):
            raise ValueError("cube never settles at its final rest pose")
        t_b = t[after[0]]
        t_release = t_b
        if t_b - t[i_last] > MAX_GAP_S:
            at = hand_at_level(t, hand_pos, hand_vis, t[i_last], t_b)
            t_release = at[0] if len(at) else 0.5 * (t[i_last] + t_b)
            notes.append(f"cube not seen at rest in the {t_b - t[i_last]:.2f} s after it last moved; used the hand")
    else:  # never seen moving (hidden while held): the hold is the longest stretch without a rest frame
        rest = np.flatnonzero(at0 | at1)
        if len(rest) < 2:
            raise ValueError("cube never seen at rest by both cameras")
        k = int(np.argmax(np.diff(t[rest])))
        t_a, t_b = t[rest[k]], t[rest[k + 1]]
        mid = 0.5 * (t_a + t_b)  # the grasp is in the first half, the release in the second
        first, second = (hand_at_level(t, hand_pos, hand_vis, a, b) for a, b in ((t_a, mid), (mid, t_b)))
        if not len(first) or not len(second):
            raise ValueError(f"cube not seen moving and the hand not tracked at it in {t_a:.1f}-{t_b:.1f} s")
        t_grasp, t_release = first[-1], second[0]
        notes.append(f"cube not seen moving ({t_b - t_a:.1f} s without a rest frame); grasp and release from the hand")
    if t_release <= t_grasp:
        raise ValueError(f"release ({t_release:.2f}s) not after grasp ({t_grasp:.2f}s)")
    seen_lift = (v & lifted & (t > t_grasp) & (t < t_release)).any()
    if not seen_lift:
        notes.append("lift not seen (cube hidden while held)")
    hand_at_grasp = np.array([interp_valid(t_grasp, t, hand_pos[:, k], hand_vis) for k in range(3)])
    if np.linalg.norm(hand_at_grasp - pos0) > NEAR_HAND:
        notes.append("hand not near the cube at the detected grasp")
    return dict(t_grasp=float(t_grasp), t_release=float(t_release), rest_pos=pos0, rest_yaw=yaw0,
                end_pos=pos1, end_yaw=yaw1, notes=notes)


def extract_clip(dirs, stem):
    tr = dict(np.load(dirs["tracks"] / f"{stem}.npz"))
    if not tr["table_found"]:
        raise ValueError("table marker not found")
    t = tr["t"]
    cv, hv = tr["cube_visible"], tr["hand_visible"]
    if hv.sum() < MIN_HAND_FRAMES:
        raise ValueError(f"hand (wrist marker) tracked in only {hv.sum()} frames: nothing to retarget")
    cube_yaw = unwrap_valid(tr["cube_yaw"], cv)
    hand_yaw = unwrap_valid(tr["hand_yaw"], hv)
    ph = detect_phases(t, tr["cube_pos"], cube_yaw, cv, tr["hand_pos"], hv, pos_visible(tr))
    tg, trel = ph["t_grasp"], ph["t_release"]
    cube1, cube_yaw1 = ph["end_pos"], ph["end_yaw"]
    hand_yaw_g = interp_valid(tg, t, hand_yaw, hv)
    info = parse_clip_name(stem) or dict(number=-1, angle=np.nan, spot="?")
    proc = dict(t=t, fps=tr["fps"], clip=stem, intended_angle=info["angle"], spot=info["spot"],
                hand_pos=tr["hand_pos"], hand_quat=tr["hand_quat"], hand_yaw=hand_yaw, hand_yaw_rel=hand_yaw - hand_yaw_g,
                hand_visible=hv, cube_pos=tr["cube_pos"], cube_quat=tr["cube_quat"], cube_yaw=cube_yaw,
                cube_yaw_rel=cube_yaw - ph["rest_yaw"], cube_visible=cv, cube_pos_visible=pos_visible(tr), cube_top_visible=tr["cube_ids"][:, 0],
                cube_n_markers=tr["cube_ids"].sum(1), t_grasp=tg, t_release=trel, cube0=ph["rest_pos"],
                cube_yaw0=ph["rest_yaw"], cube1=cube1, cube_yaw1=cube_yaw1,
                hand_yaw_change=interp_valid(trel, t, hand_yaw, hv) - hand_yaw_g, cube_yaw_change=cube_yaw1 - ph["rest_yaw"],
                phase_notes=np.array(ph["notes"], dtype=str), placeholder_intrinsics=tr["placeholder_intrinsics"])
    np.savez(dirs["processed"] / f"{stem}.npz", **proc)
    return proc


def plot_phases(proc, path, synthetic, gt=None):
    t = proc["t"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 2, 1]))
    cv, hv = proc["cube_visible"], proc["hand_visible"]
    pv = proc["cube_pos_visible"]
    ax[0].plot(t[pv], 100 * proc["cube_pos"][pv, 2], ".", ms=3, label="cube centre height")
    ax[0].plot(t[hv], 100 * proc["hand_pos"][hv, 2], ".", ms=3, label="wrist marker height")
    ax[0].set_ylabel("height above table (cm)")
    ax[1].plot(t[cv], np.degrees(proc["cube_yaw_rel"][cv]), ".", ms=3, label="cube yaw")
    ax[1].plot(t[hv], np.degrees(proc["hand_yaw_rel"][hv]), ".", ms=3, label="hand yaw (rel. to grasp)")
    if np.isfinite(proc["intended_angle"]):
        ax[1].axhline(proc["intended_angle"], color="0.6", ls=":", lw=1, label=f"intended {proc['intended_angle']:+.0f}")
    ax[1].set_ylabel("yaw change (deg)")
    for k, (name, flag) in enumerate((("cube", cv), ("top marker", proc["cube_top_visible"]), ("hand", hv))):
        ax[2].fill_between(t, k, k + 0.8, where=flag, step="mid", alpha=0.6)
        ax[2].text(t[0], k + 0.4, name, va="center", fontsize=8)
    ax[2].set_yticks([])
    ax[2].set_ylabel("tracked")
    ax[2].set_xlabel("time (s)")
    for a in ax:
        a.axvline(proc["t_grasp"], color="g", lw=1)
        a.axvline(proc["t_release"], color="r", lw=1)
        if gt is not None:
            a.axvline(gt["t_grasp"], color="g", lw=1, ls="--")
            a.axvline(gt["t_touchdown"], color="r", lw=1, ls="--")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[1].legend(fontsize=8, loc="best")
    notes = "; ".join(proc["phase_notes"])
    ax[0].set_title(f"{proc['clip']}: grasp (green) {proc['t_grasp']:.2f} s, release (red) {proc['t_release']:.2f} s"
                    + (" (dashed: ground truth)" if gt is not None else "") + (f"\n{notes}" if notes else ""), fontsize=9)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_grip_gain(rows, path, synthetic):
    x = np.array([r["hand_yaw_change_deg"] for r in rows])
    y = np.array([r["cube_yaw_change_deg"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sc = ax.scatter(x, y, c=[r["intended_angle"] for r in rows], cmap="coolwarm", s=30, edgecolor="k", lw=0.4)
    lim = np.nanmax(np.abs(np.r_[x, y, 30])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot([-lim, lim], [slope * -lim + intercept, slope * lim + intercept], "C1", lw=1.2,
                label=f"fit: y = {slope:.2f} x {intercept:+.1f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("hand yaw change at release (deg, rel. to grasp)")
    ax.set_ylabel("cube yaw change (deg)")
    ax.set_title("Human grip gain")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="intended angle (deg)", shrink=0.8)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def extract_all(raw_dir, out_dir):
    dirs = data_dirs(raw_dir)
    out = Path(out_dir)
    dirs["processed"].mkdir(parents=True, exist_ok=True)
    (out / "phases").mkdir(parents=True, exist_ok=True)
    synthetic = is_synthetic(raw_dir)
    rows = []
    for track in sorted(dirs["tracks"].glob("*.npz")):
        stem = track.stem
        if parse_clip_name(stem) is None:  # pilot_* and other checks are not clips
            continue
        try:
            proc = extract_clip(dirs, stem)
        except ValueError as e:
            print(f"{stem}: SKIPPED ({e})")
            continue
        gt_path = dirs["ground_truth"] / f"{stem}.npz"
        plot_phases(proc, out / "phases" / f"{stem}.png", synthetic, dict(np.load(gt_path)) if gt_path.exists() else None)
        rows.append(dict(clip=stem, intended_angle=proc["intended_angle"],
                         hand_yaw_change_deg=float(np.degrees(proc["hand_yaw_change"])),
                         cube_yaw_change_deg=float(np.degrees(proc["cube_yaw_change"])),
                         t_grasp=proc["t_grasp"], t_release=proc["t_release"],
                         cube_tracked_pct=100 * proc["cube_visible"].mean(), notes="; ".join(proc["phase_notes"])))
        print(f"{stem}: grasp {proc['t_grasp']:.2f}s release {proc['t_release']:.2f}s | hand {rows[-1]['hand_yaw_change_deg']:+6.1f} "
              f"cube {rows[-1]['cube_yaw_change_deg']:+6.1f} deg" + (f" | {rows[-1]['notes']}" if rows[-1]["notes"] else ""))
    if rows:
        plot_grip_gain(rows, out / "grip_gain.png", synthetic)
        with open(out / "extract_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    extract_all(args.data, args.out or str(ROOT / "outputs" / ("synthetic" if is_synthetic(args.data) else "real")))


if __name__ == "__main__":
    main()
