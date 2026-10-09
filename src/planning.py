"""Planning with the world model as a learned simulator, executed in the robot simulator.

    python -m src.planning [--data data/raw] [--out outputs/real] [--targets 20]

For each target cube yaw change: the human clip with the closest hand rotation (and its hand
tracked through >= 90% of the hold) is the template; candidate hand trajectories rescale its hand-rotation profile; the world model
(ensemble mean, free rollout from rest) predicts each candidate's final cube yaw; the closest
one is replayed in the sim. The naive baseline rotates the hand by exactly the target.
Success = lifted >= 2 cm AND placed within 3 cm AND |sim cube yaw change - target| < 15 deg.

The naive baseline may well win: the model learned how the cube turns in a human hand,
while the robot's parallel gripper turns it almost rigidly. Either result is reported.
Writes <out>/planning.csv, <out>/planning.png and <out>/planning.txt.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import constants as C
from src.common import ROOT, data_dirs, is_synthetic, label_figure
from src.sim_replay import (CUBE_HOME_XY, POS_TOL, TABLE_Z, YAW_TOL_DEG, hold_mask, make_env, run_replay,
                            with_rotation)
from src.world_model.data import load_sequences
from src.world_model.models import GRUModel, ensemble_rollout, make_mlp, model_step_fn

MIN_TEMPLATE_DEG = 20
MIN_TEMPLATE_HAND = 0.9  # share of the hold with the hand tracked: gaps are bridged linearly, and a fast turn
                         # inside one becomes a heading jump the arm cannot follow (clip 104, 6 Oct)
SCALES = np.linspace(0.5, 1.6, 23)


def load_ensemble(path):
    models = []
    for m in torch.load(path, weights_only=False):
        net = make_mlp() if m["kind"] == "mlp" else GRUModel()
        net.load_state_dict(m["state"])
        net.eval()
        models.append(model_step_fn(dict(kind=m["kind"], net=net, stats=m["stats"])))
    return models


def rotation_profile(clip, t_query):
    """Hand rotation relative to the grasp, normalised so it reaches 1 at the release."""
    hv = clip["hand_visible"]
    h = np.interp(t_query, clip["t"][hv], clip["hand_yaw_rel"][hv])
    return h / float(clip["hand_yaw_change"])


def predicted_final(steps, seq, clip, phi):
    h = phi * rotation_profile(clip, seq["t"])
    a = np.r_[np.diff(h), 0.0]
    mean, _ = ensemble_rollout(steps, seq, [0], h=h, a=a)
    return float(mean[0, -1])


def run(raw_dir, out_dir, n_targets=20, seed=0):
    dirs = data_dirs(raw_dir)
    out = Path(out_dir)
    synthetic = is_synthetic(raw_dir)
    model_path = out / "world_model" / "model.pt"
    if not model_path.exists():
        print("planning: no world model yet (run src.world_model.run first)")
        return None
    steps = load_ensemble(model_path)
    seqs = {s["name"]: s for s in load_sequences(dirs["processed"])}
    clips = {name: dict(np.load(dirs["processed"] / f"{name}.npz")) for name in seqs}
    templates = [n for n, c in clips.items() if abs(np.degrees(float(c["hand_yaw_change"]))) >= MIN_TEMPLATE_DEG
                 and c["hand_visible"][hold_mask(c)].mean() >= MIN_TEMPLATE_HAND]
    if not templates:
        print("planning: no clip with a hand rotation >= 20 deg and the hand tracked while held to use as a template")
        return None
    rng = np.random.default_rng(seed)
    targets = np.radians(rng.uniform(-90, 90, n_targets))
    env = make_env(render=False)
    pos0 = np.r_[CUBE_HOME_XY, TABLE_Z + C.CUBE_SIDE / 2]
    rows = []
    for target in targets:
        name = min(templates, key=lambda n: abs(abs(float(clips[n]["hand_yaw_change"])) - abs(target)))
        seq, clip = seqs[name], clips[name]
        candidates = np.r_[SCALES * target, target]
        preds = np.array([predicted_final(steps, seq, clip, phi) for phi in candidates])
        phi_plan = candidates[int(np.argmin(np.abs(preds - target)))]
        row = dict(target_deg=np.degrees(target), template=name)
        for method, phi in (("planned", phi_plan), ("naive", target)):
            m, _, _ = run_replay(env, with_rotation(clip, phi), pos0, 0.0, True, record=False)
            ok = m["lifted"] and m["pos_err_cm"] < 100 * POS_TOL and abs(m["sim_dyaw_deg"] - np.degrees(target)) < YAW_TOL_DEG
            row.update({f"{method}_hand_deg": np.degrees(phi), f"{method}_sim_deg": m["sim_dyaw_deg"],
                        f"{method}_pos_err_cm": m["pos_err_cm"], f"{method}_success": bool(ok)})
        row["model_pred_for_plan_deg"] = np.degrees(preds[int(np.argmin(np.abs(preds - target)))])
        rows.append(row)
        print(f"target {row['target_deg']:+6.1f}: planned hand {row['planned_hand_deg']:+6.1f} -> sim {row['planned_sim_deg']:+6.1f} "
              f"({'ok' if row['planned_success'] else 'FAIL'}) | naive hand {row['naive_hand_deg']:+6.1f} -> sim "
              f"{row['naive_sim_deg']:+6.1f} ({'ok' if row['naive_success'] else 'FAIL'})")
    env.close()
    with open(out / "planning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for method in ("planned", "naive"):
        ok = np.mean([r[f"{method}_success"] for r in rows])
        err = np.mean([abs(r[f"{method}_sim_deg"] - r["target_deg"]) for r in rows])
        summary.append(f"{method:8s}: success {100 * ok:5.1f}% over {len(rows)} targets, mean |sim yaw - target| {err:5.1f} deg")
    text = (("SYNTHETIC DATA\n" if synthetic else "") + "Planning with the world model vs naive (rotate by the target):\n"
            + "\n".join(summary) + "\n")
    (out / "planning.txt").write_text(text)
    print(text)
    plot(rows, out / "planning.png", synthetic)
    return rows


def plot(rows, path, synthetic):
    t = np.array([r["target_deg"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = 130
    ax.fill_between([-lim, lim], [-lim - YAW_TOL_DEG, lim - YAW_TOL_DEG], [-lim + YAW_TOL_DEG, lim + YAW_TOL_DEG],
                    color="0.9", label=f"±{YAW_TOL_DEG:.0f} deg")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.scatter(t, [r["naive_sim_deg"] for r in rows], marker="s", s=28, label="naive: hand turns by the target")
    ax.scatter(t, [r["planned_sim_deg"] for r in rows], marker="o", s=28, label="planned with the world model")
    ax.set_xlabel("target cube yaw change (deg)")
    ax.set_ylabel("sim cube yaw change (deg)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Planning in the sim with a model learned from human data", fontsize=10)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(raw_dir=None, out_dir=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(raw_dir or ROOT / "data" / "raw"))
    p.add_argument("--out", default=out_dir)
    p.add_argument("--targets", type=int, default=20)
    args = p.parse_args([] if raw_dir else None)
    run(args.data, args.out or str(ROOT / "outputs" / ("synthetic" if is_synthetic(args.data) else "real")), args.targets)


if __name__ == "__main__":
    main()
