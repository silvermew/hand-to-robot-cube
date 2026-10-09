"""World-model data: processed clips resampled to a fixed 10 Hz step around the interaction.

Per step k: cube yaw c_k and hand yaw h_k (radians, relative to the grasp instant), the action
a_k = h_{k+1} - h_k (the hand's yaw change over the next step) and the gripper flag g_k
(1 between grasp and release). The target is c_{k+1}. The cube yaw is observed only where
the cube was tracked (obs flag); the hand yaw is linearly bridged over short tracking gaps
(the bridged steps are counted). Synthetic clips also carry the true cube yaw.
"""
from pathlib import Path

import numpy as np

DT = 0.1
MARGIN_S = 0.5           # window = [grasp - MARGIN_S, release + MARGIN_S]
MAX_HAND_GAP_S = 0.5


def nearest_sample(t_query, t, x, valid):
    """Value of the nearest source frame and whether it was tracked (no interpolation of the target)."""
    i = np.clip(np.searchsorted(t, t_query), 1, len(t) - 1)
    i = np.where(np.abs(t[i - 1] - t_query) <= np.abs(t[i] - t_query), i - 1, i)
    return x[i], valid[i]


def load_sequence(path, gt_dir=None):
    clip = dict(np.load(path, allow_pickle=False))
    t, tg, tr = clip["t"], float(clip["t_grasp"]), float(clip["t_release"])
    hv = clip["hand_visible"]
    t_hand = t[hv] if hv.any() else np.array([tg, tr])
    ticks = np.arange(max(tg - MARGIN_S, t_hand[0]), min(tr + MARGIN_S, t_hand[-1]) + 1e-9, DT)  # hand must be in view
    c, obs = nearest_sample(ticks, t, clip["cube_yaw_rel"], clip["cube_visible"])
    h = np.interp(ticks, t[hv], clip["hand_yaw_rel"][hv])
    _, h_obs = nearest_sample(ticks, t, clip["hand_yaw_rel"], hv)
    gap_ok = np.ones(len(ticks), bool)
    for k in np.flatnonzero(~h_obs):  # bridged hand steps must be inside a short gap
        before, after = t[hv & (t <= ticks[k])], t[hv & (t >= ticks[k])]
        gap_ok[k] = len(before) and len(after) and after[0] - before[-1] <= MAX_HAND_GAP_S
    seq = dict(name=Path(path).stem, angle=float(clip["intended_angle"]), t=ticks, c=np.where(obs, c, np.nan),
               obs=obs.astype(bool), h=h, a=np.r_[np.diff(h), 0.0], g=((ticks >= tg) & (ticks < tr)).astype(float),
               hand_bridged=int((~h_obs).sum()), usable=bool(gap_ok.all()),
               cube_yaw_change=float(clip["cube_yaw_change"]))
    if gt_dir is not None and (Path(gt_dir) / f"{seq['name']}.npz").exists():
        gt = np.load(Path(gt_dir) / f"{seq['name']}.npz")
        gt_rel = gt["cube_yaw"] - np.interp(float(gt["t_grasp"]), gt["t"], gt["cube_yaw"])
        seq["c_true"] = np.interp(ticks, gt["t"], gt_rel)
    return seq


def load_sequences(processed_dir, gt_dir=None, quiet=False):
    seqs = [load_sequence(p, gt_dir) for p in sorted(Path(processed_dir).glob("*.npz"))]
    keep = []
    for s in seqs:
        held = s["g"] > 0
        reason = (f"hand untracked for > {MAX_HAND_GAP_S} s" if not s["usable"] else
                  "cube seen in < 3 steps" if s["obs"].sum() < 3 else
                  "cube never seen while held" if not s["obs"][held].any() else None)
        if reason and not quiet:
            print(f"world model: dropping {s['name']}: {reason}")
        if not reason:
            keep.append(s)
    return keep


def split_by_clip(seqs, seed=0):
    """Stratified by intended angle: per angle one test clip (if >= 2 clips), one val clip (if >= 4), rest train."""
    rng = np.random.default_rng(seed)
    split = {}
    for angle in sorted({s["angle"] for s in seqs}):
        names = sorted(s["name"] for s in seqs if s["angle"] == angle)
        rng.shuffle(names)
        for k, name in enumerate(names):
            split[name] = "test" if k == 0 and len(names) >= 2 else "val" if k == 1 and len(names) >= 4 else "train"
    return split


def heldout_split(seqs, train_max_abs=90, test_angle=120):
    return {s["name"]: "train" if abs(s["angle"]) <= train_max_abs else "test" if s["angle"] == test_angle else "unused"
            for s in seqs}


def transitions(seqs):
    """One-step training pairs where both c_k and c_{k+1} were observed: X = [c, h, a, g], y = c_{k+1} - c_k."""
    X, y = [], []
    for s in seqs:
        ok = s["obs"][:-1] & s["obs"][1:]
        X.append(np.stack([s["c"][:-1], s["h"][:-1], s["a"][:-1], s["g"][:-1]], 1)[ok])
        y.append((s["c"][1:] - s["c"][:-1])[ok])
    return np.concatenate(X), np.concatenate(y)
