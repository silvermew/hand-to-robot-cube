"""Cut the push tracks into pushes and moves between crosses.

    python -m src.push_extract [--tracks data/push/tracks] [--out data/push/processed]

A push is a run of cube motion (faster than 1 cm/s or 10 deg/s, smoothed over 0.17 s); runs less than 1 s apart
are one push. A move goes from the cross the cube rests on to the next rest within 4 cm of another cross (or to
the end of the video), plus any nudges after it that keep the cube on that cross. The straight pushes are one push
per move; the turning ones are 2-6 pushes, each turning the cube by about 90 deg while my hand lifts off between
them. Each move is saved with its cube track (table-marker frame, 30 Hz, yaw unwrapped), the hand card track and
its pushes, and listed in outputs/push/moves.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from src.common import ROOT
from src.sim_replay import HUMAN_CROSSES

RATE = 30.0
SMOOTH = 5           # samples (0.17 s)
MOVING_SPEED = 0.01  # m/s
MOVING_TURN = np.radians(10)  # rad/s
MIN_PUSH = (0.02, np.radians(5))  # a push moves the cube at least this far or turns it at least this much
MERGE_GAP = 1.0      # s
AT_CROSS = 0.04      # m
MARGIN = 1.0         # s of rest kept before and after each move
STRAIGHT_DEG = 30    # a move turning less than this is straight


def resample(track):
    """Cube xy and unwrapped yaw on a uniform grid from the tracked frames, smoothed, and the hand card."""
    t, ok = track["t"], track["cube_source"] > 0
    tu = np.arange(t[ok][0], t[ok][-1], 1 / RATE)
    kernel = np.ones(SMOOTH) / SMOOTH
    smooth = lambda x: np.convolve(np.pad(x, SMOOTH // 2, mode="edge"), kernel, "valid")
    xy = np.stack([smooth(np.interp(tu, t[ok], track["cube_pos"][ok, k])) for k in range(2)], 1)
    yaw = smooth(np.interp(tu, t[ok], np.unwrap(track["cube_yaw"][ok])))
    hv = track["hand_visible"]
    # a card sample counts for a grid point within half a frame of it
    near = np.abs(tu[:, None] - t[hv][None, :]).min(1) < 0.6 / RATE if hv.any() else np.zeros(len(tu), bool)
    hand = np.stack([np.interp(tu, t[hv], track["hand_pos"][hv, k]) for k in range(3)], 1) if hv.any() else \
        np.full((len(tu), 3), np.nan)
    hand_yaw = np.interp(tu, t[hv], np.unwrap(track["hand_yaw"][hv])) if hv.any() else np.full(len(tu), np.nan)
    return dict(t=tu, xy=xy, yaw=yaw, hand_pos=hand, hand_yaw=hand_yaw, hand_visible=near)


def find_pushes(r):
    """(start, stop) sample indices of each push."""
    speed = np.r_[0, np.linalg.norm(np.diff(r["xy"], axis=0), axis=1) * RATE]
    turn = np.r_[0, np.abs(np.diff(r["yaw"])) * RATE]
    edges = np.diff(np.r_[0, ((speed > MOVING_SPEED) | (turn > MOVING_TURN)).astype(int), 0])
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
    merged = []
    for a, b in runs:
        if merged and r["t"][a] - r["t"][merged[-1][1]] < MERGE_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged
            if np.linalg.norm(r["xy"][b] - r["xy"][a]) >= MIN_PUSH[0] or abs(r["yaw"][b] - r["yaw"][a]) >= MIN_PUSH[1]]


def nearest_cross(xy):
    name = min(HUMAN_CROSSES, key=lambda c: np.linalg.norm(xy - HUMAN_CROSSES[c]))
    return name, float(np.linalg.norm(xy - HUMAN_CROSSES[name]))


def find_moves(r, pushes):
    """Lists of consecutive pushes, each taking the cube from one cross to another."""
    moves, current, start = [], [], None
    for a, b in pushes:
        if start is None:
            start = nearest_cross(r["xy"][a])[0]
        end, dist = nearest_cross(r["xy"][b])
        if not current and moves and end == moves[-1][2] and dist < AT_CROSS:
            moves[-1][0].append((a, b))  # a nudge after the move: the cube stays on its cross
            continue
        current.append((a, b))
        if end != start and dist < AT_CROSS:
            moves.append((current, start, end))
            current, start = [], end
    if current and nearest_cross(r["xy"][current[-1][1]])[0] != start:
        moves.append((current, start, nearest_cross(r["xy"][current[-1][1]])[0]))
    return moves


def move_record(r, pushes, video):
    a = max(0, pushes[0][0] - int(MARGIN * RATE))
    b = min(len(r["t"]), pushes[-1][1] + 1 + int(MARGIN * RATE))
    sl = slice(a, b)
    t = r["t"][sl] - r["t"][a]
    out = {k: v[sl] for k, v in r.items() if k != "t"}
    p = np.array([(r["t"][i] - r["t"][a], r["t"][j] - r["t"][a]) for i, j in pushes])
    turn = np.array([r["yaw"][j] - r["yaw"][i] for i, j in pushes])
    return dict(t=t, cube_xy=out["xy"], cube_yaw=out["yaw"], hand_pos=out["hand_pos"], hand_yaw=out["hand_yaw"],
                hand_visible=out["hand_visible"], pushes=p, push_turn=turn,
                push_dist=np.array([np.linalg.norm(r["xy"][j] - r["xy"][i]) for i, j in pushes]),
                cube0=out["xy"][0], cube1=out["xy"][-1], cube_yaw0=out["yaw"][0], cube_yaw1=out["yaw"][-1],
                video=video, video_t0=float(r["t"][a]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tracks", default=str(ROOT / "data" / "push" / "tracks"))
    p.add_argument("--out", default=str(ROOT / "data" / "push" / "processed"))
    p.add_argument("--table", default=str(ROOT / "outputs" / "push" / "moves.csv"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("push_*.npz"):
        old.unlink()
    rows = []
    for path in sorted(Path(args.tracks).glob("*.npz")):
        r = resample(dict(np.load(path)))
        for pushes, start, end in find_moves(r, find_pushes(r)):
            rec = move_record(r, pushes, path.stem)
            total = float(np.degrees(rec["cube_yaw1"] - rec["cube_yaw0"]))
            kind = "straight" if abs(total) < STRAIGHT_DEG else ("ccw" if total > 0 else "cw")
            name = f"push_{len(rows) + 1:02d}_{start}{end}_{kind}"
            np.savez(out / f"{name}.npz", **rec, start_cross=start, end_cross=end, kind=kind)
            rows.append(dict(name=name, video=path.stem, video_t0=round(rec["video_t0"], 2), start=start, end=end,
                             kind=kind, pushes=len(pushes), dist_cm=round(100 * float(np.linalg.norm(rec["cube1"] - rec["cube0"])), 1),
                             turn_deg=round(total, 1), start_off_cm=round(100 * nearest_cross(rec["cube0"])[1], 1),
                             end_off_cm=round(100 * nearest_cross(rec["cube1"])[1], 1), duration_s=round(float(rec["t"][-1]), 1)))
            print(", ".join(f"{k} {v}" for k, v in rows[-1].items()))
    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    with open(args.table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} moves -> {out}, {args.table}")


if __name__ == "__main__":
    main()
