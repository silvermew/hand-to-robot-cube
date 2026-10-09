"""Train and evaluate the small world model on processed clips (CPU, a minute or two).

    python -m src.world_model.run [--data data/raw] [--out outputs/real]

A small low-dimensional state-space model, not a video model: it predicts the next cube yaw
(10 Hz) from the cube yaw, the hand yaw, the hand's yaw change and the gripper flag. Learned
models (MLP, GRU; ensembles of 5) are compared against persistence, "cube follows the hand"
and a best-fit linear gain, all gated by the gripper flag. Writes <out>/world_model/:
report.txt, metrics.json, horizon.png, trust.png, by_angle.png, occlusion.png,
embodiment_model.png and model.pt (the chosen ensemble, used by src.planning).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from src.common import ROOT, data_dirs, is_synthetic, label_figure
from src.world_model.data import DT, heldout_split, load_sequences, split_by_clip
from src.world_model.models import (follow_step, fit_gain, gain_step_fn, model_step_fn, persistence_step,
                                    ensemble_rollout, train_ensemble)

HORIZONS = np.arange(1, 31)
GAP_BINS = [(0.05, 0.35), (0.35, 0.75), (0.75, 1.5), (1.5, 99)]
SIMPLER_UNLESS = 0.10    # keep the MLP unless the GRU's final error is >10% lower


def build_models(train):
    gain = fit_gain(train)
    mlp, gru = train_ensemble("mlp", train), train_ensemble("gru", train)
    models = {"persistence": [persistence_step], "follow hand": [follow_step],
              f"linear gain {gain:.2f}": [gain_step_fn(gain)],
              "MLP x5": [model_step_fn(m) for m in mlp], "GRU x5": [model_step_fn(m) for m in gru]}
    return models, dict(mlp=mlp, gru=gru, gain=gain)


def horizon_errors(steps, seqs):
    """Mean |error| (deg) per horizon over all observed start points, plus (ensemble std, error) pairs."""
    errs = {H: [] for H in HORIZONS}
    trust = []
    for s in seqs:
        starts = np.flatnonzero(s["obs"])
        mean, std = ensemble_rollout(steps, s, starts)
        for b, st in enumerate(starts):
            for H in HORIZONS:
                k = st + H
                if k < len(s["t"]) and s["obs"][k]:
                    e = abs(mean[b, k] - s["c"][k])
                    errs[H].append(e)
                    trust.append((std[b, k], e))
    return np.array([np.degrees(np.mean(errs[H])) if errs[H] else np.nan for H in HORIZONS]), np.degrees(np.array(trust))


def final_errors(steps, seqs):
    """Free rollout from the window start (cube at rest) to release + 0.5 s vs the measured cube yaw change."""
    rows = []
    for s in seqs:
        mean, std = ensemble_rollout(steps, s, [0])
        rows.append(dict(name=s["name"], angle=s["angle"], pred=np.degrees(mean[0, -1]),
                         meas=np.degrees(s["cube_yaw_change"]), std=np.degrees(std[0, -1])))
        rows[-1]["err"] = abs(rows[-1]["pred"] - rows[-1]["meas"])
    return rows


def occlusion_errors(steps, seqs):
    """Filtered rollout (observations used whenever available). Re-acquisition error after each gap by gap
    length; and, where the true cube yaw is known (synthetic), error on tracked vs untracked steps."""
    reacq = {b: [] for b in GAP_BINS}
    vis, occ = [], []
    for s in seqs:
        T = len(s["t"])
        mean, _ = ensemble_rollout(steps, s, [T])
        pred = mean[0]
        obs_idx = np.flatnonzero(s["obs"])
        for a, b in zip(obs_idx[:-1], obs_idx[1:]):
            gap = (b - a - 1) * DT
            for lo, hi in GAP_BINS:
                if lo <= gap < hi:
                    reacq[(lo, hi)].append(np.degrees(abs(pred[b] - s["c"][b])))
        if "c_true" in s:
            # one-step-ahead prediction (made before the step's observation) vs the truth
            e = np.degrees(np.abs(pred - s["c_true"]))
            vis += list(e[s["obs"]])
            occ += list(e[~s["obs"]])
    out = dict(reacquisition={f"{lo}-{hi}s": (float(np.mean(v)) if v else np.nan, len(v)) for (lo, hi), v in reacq.items()})
    if vis or occ:
        out["vs_truth_tracked"] = (float(np.mean(vis)) if vis else np.nan, len(vis))
        out["vs_truth_untracked"] = (float(np.mean(occ)) if occ else np.nan, len(occ))
    return out


def plot_horizon(curves, path, title, synthetic):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, c in curves.items():
        ax.plot(HORIZONS * DT, c, label=name, lw=2 if "x5" in name else 1.2, ls="-" if "x5" in name else "--")
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("mean |cube yaw error| (deg)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trust(pairs, path, name, synthetic):
    std, err = pairs[:, 0], pairs[:, 1]
    rho = spearmanr(std, err).correlation if len(std) > 2 else np.nan
    edges = np.unique(np.quantile(std, np.linspace(0, 1, 6)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(std, err, s=4, alpha=0.15, color="0.4")
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (std >= lo) & (std <= hi)
        if m.any():
            centers.append(np.median(std[m]))
            means.append(err[m].mean())
    ax.plot(centers, means, "o-", color="C3", label="mean error per disagreement quintile")
    ax.set_xlabel("ensemble disagreement (std of 5 models, deg)")
    ax.set_ylabel("|error| (deg)")
    ax.set_title(f"Where the model can be trusted ({name}): Spearman rho = {rho:.2f}", fontsize=10)
    ax.legend(fontsize=8)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return float(rho), list(zip(centers, means))


def plot_by_angle(results, path, synthetic, title):
    angles = sorted({r["angle"] for rows in results.values() for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(results)
    for j, (name, rows) in enumerate(results.items()):
        vals = [np.mean([r["err"] for r in rows if r["angle"] == a]) if any(r["angle"] == a for r in rows) else np.nan
                for a in angles]
        ax.bar(np.arange(len(angles)) + j * width, vals, width, label=name)
    ax.set_xticks(np.arange(len(angles)) + 0.4 - width / 2)
    ax.set_xticklabels([f"{a:+.0f}" for a in angles])
    ax.set_xlabel("intended rotation (deg)")
    ax.set_ylabel("|final cube yaw error| (deg)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_embodiment(rows, path, synthetic):
    x = np.array([r["model_pred"] for r in rows])
    y = np.array([r["sim"] for r in rows])
    test = np.array([r["split"] == "test" for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = np.max(np.abs(np.r_[x, y, 30])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
    ax.scatter(x[test], y[test], c="C0", s=36, edgecolor="k", lw=0.4, label="test clips")
    ax.scatter(x[~test], y[~test], c="C1", s=20, alpha=0.6, label="train/val clips (in-sample)")
    gap = np.abs(y[test] - x[test]).mean() if test.any() else np.nan
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("world model: predicted human cube yaw change (deg)")
    ax.set_ylabel("simulator: robot cube yaw change (deg)")
    ax.set_title(f"Embodiment gap through the model: mean |sim - model| = {gap:.1f} deg (test)", fontsize=9)
    ax.legend(fontsize=8)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return float(gap)


def save_ensemble(path, ensemble):
    torch.save([dict(kind=m["kind"], state=m["net"].state_dict(), stats=m["stats"]) for m in ensemble], path)


def run(raw_dir, out_dir):
    dirs = data_dirs(raw_dir)
    synthetic = is_synthetic(raw_dir)
    out = Path(out_dir) / "world_model"
    out.mkdir(parents=True, exist_ok=True)
    seqs = load_sequences(dirs["processed"], dirs["ground_truth"] if synthetic else None)
    split = split_by_clip(seqs)
    train = [s for s in seqs if split[s["name"]] == "train"]
    val = [s for s in seqs if split[s["name"]] == "val"]
    test = [s for s in seqs if split[s["name"]] == "test"]
    if not train or not test:
        print(f"world model: skipped, {len(seqs)} usable clip(s) give no train/test split "
              "(needs >= 2 clips of the same intended angle)")
        return None
    lines = [f"{'SYNTHETIC DATA: ' if synthetic else ''}world model on {len(seqs)} clips: "
             f"{len(train)} train, {len(val)} val, {len(test)} test (split by clip, stratified by angle, seed 0)"]
    models, trained = build_models(train)
    metrics = dict(split=split, gain=trained["gain"])

    curves = {name: horizon_errors(steps, test)[0] for name, steps in models.items()}
    plot_horizon(curves, out / "horizon.png", "Autoregressive rollout error vs horizon (test clips)", synthetic)
    finals = {name: final_errors(steps, test) for name, steps in models.items()}
    plot_by_angle(finals, out / "by_angle.png", synthetic, "Final cube yaw error by rotation amount (test clips)")
    lines.append("\nTest clips: mean |error| in deg (rollout at 0.5 s / 1 s / 2 s / 3 s; final = free rollout from rest)")
    for name in models:
        c = curves[name]
        lines.append(f"  {name:22s} {c[4]:6.1f} {c[9]:6.1f} {c[19]:6.1f} {c[29]:6.1f} | final {np.mean([r['err'] for r in finals[name]]):6.1f}")
    metrics["horizon_deg"] = {k: v.tolist() for k, v in curves.items()}
    metrics["final_err_deg"] = {k: float(np.mean([r["err"] for r in v])) for k, v in finals.items()}

    pick_set = val or test
    final_on = {k: np.mean([r["err"] for r in final_errors(models[k], pick_set)]) for k in ("MLP x5", "GRU x5")}
    chosen = "GRU x5" if final_on["GRU x5"] < (1 - SIMPLER_UNLESS) * final_on["MLP x5"] else "MLP x5"
    lines.append(f"\nModel choice on {'val' if val else 'TEST (no val clips: choice is not independent of the test score)'}: "
                 f"MLP {final_on['MLP x5']:.1f} vs GRU {final_on['GRU x5']:.1f} deg final error -> keep {chosen}"
                 f" (MLP unless the GRU is >{100 * SIMPLER_UNLESS:.0f}% better)")
    best_baseline = min((k for k in models if "x5" not in k), key=lambda k: metrics["final_err_deg"][k])
    verdict = "beats" if metrics["final_err_deg"][chosen] < metrics["final_err_deg"][best_baseline] else "DOES NOT beat"
    lines.append(f"{chosen} {verdict} the best baseline ({best_baseline}) on final error: "
                 f"{metrics['final_err_deg'][chosen]:.1f} vs {metrics['final_err_deg'][best_baseline]:.1f} deg")
    metrics["chosen"] = chosen
    save_ensemble(out / "model.pt", trained["mlp" if chosen == "MLP x5" else "gru"])

    for name in ("MLP x5", "GRU x5"):
        _, pairs = horizon_errors(models[name], test)
        rho, bins = plot_trust(pairs, out / f"trust_{name.split()[0].lower()}.png", name, synthetic)
        metrics[f"trust_spearman_{name}"] = rho
        lines.append(f"Trust ({name}): Spearman rho(disagreement, error) = {rho:.2f}; mean error per disagreement "
                     f"quintile: " + ", ".join(f"{e:.1f}" for _, e in bins) + " deg")

    occ = {name: occlusion_errors(steps, test) for name, steps in models.items()}
    metrics["occlusion"] = occ
    lines.append("\nRe-acquisition error after a tracking gap (deg, n), by gap length:")
    for name, o in occ.items():
        lines.append(f"  {name:22s} " + "  ".join(f"{k}: {v[0]:5.1f} ({v[1]})" for k, v in o["reacquisition"].items()))
        if "vs_truth_untracked" in o:
            lines.append(f"  {'':22s} vs truth: tracked steps {o['vs_truth_tracked'][0]:.1f} deg, "
                         f"untracked steps {o['vs_truth_untracked'][0]:.1f} deg (n={o['vs_truth_untracked'][1]})")
    plot_occlusion(occ, out / "occlusion.png", synthetic)

    held = heldout_split(seqs)
    h_train = [s for s in seqs if held[s["name"]] == "train"]
    h_test = [s for s in seqs if held[s["name"]] == "test"]
    if h_train and h_test:
        h_models, _ = build_models(h_train)
        h_curves = {name: horizon_errors(steps, h_test)[0] for name, steps in h_models.items()}
        h_finals = {name: final_errors(steps, h_test) for name, steps in h_models.items()}
        plot_horizon(h_curves, out / "heldout_horizon.png",
                     f"Held-out range: train |angle| <= 90, test +120 ({len(h_test)} clips)", synthetic)
        lines.append(f"\nHeld-out rotation range (train |angle| <= 90 on {len(h_train)} clips, test +120 on {len(h_test)}):")
        for name in h_models:
            lines.append(f"  {name:22s} final {np.mean([r['err'] for r in h_finals[name]]):6.1f} deg | rollout 1 s "
                         f"{h_curves[name][9]:6.1f} deg")
        metrics["heldout_final_err_deg"] = {k: float(np.mean([r["err"] for r in v])) for k, v in h_finals.items()}
    else:
        lines.append("\nHeld-out rotation range: skipped (no +120 clips or no |angle| <= 90 clips)")

    table = Path(out_dir) / "replay_table.csv"
    if table.exists():
        sim = {r["clip"]: float(r["sim_dyaw_deg"]) for r in csv.DictReader(open(table))}
        rows = []
        for s in seqs:
            if s["name"] in sim:
                mean, _ = ensemble_rollout(models[chosen], s, [0])
                rows.append(dict(name=s["name"], model_pred=np.degrees(mean[0, -1]), sim=sim[s["name"]], split=split[s["name"]]))
        if rows:
            gap = plot_embodiment(rows, out / "embodiment_model.png", synthetic)
            metrics["embodiment_gap_through_model_deg"] = gap
            lines.append(f"\nEmbodiment gap through the model ({chosen}): mean |sim cube yaw - model-predicted human "
                         f"cube yaw| = {gap:.1f} deg on test clips")
    else:
        lines.append("\nEmbodiment gap through the model: skipped (run src.replay_all first)")

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print("\n".join(lines))
    return metrics


def plot_occlusion(occ, path, synthetic):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = list(next(iter(occ.values()))["reacquisition"])
    width = 0.8 / len(occ)
    for j, (name, o) in enumerate(occ.items()):
        vals = [o["reacquisition"][k][0] for k in labels]
        ax.bar(np.arange(len(labels)) + j * width, vals, width, label=name)
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    ax.set_xticklabels([f"gap {k}" for k in labels])
    ax.set_ylabel("|error| when the cube is seen again (deg)")
    ax.set_title("Predicting through occlusion (test clips)", fontsize=10)
    ax.legend(fontsize=7)
    label_figure(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(raw_dir=None, out_dir=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(raw_dir or ROOT / "data" / "raw"))
    p.add_argument("--out", default=out_dir)
    args = p.parse_args([] if raw_dir else None)
    run(args.data, args.out or str(ROOT / "outputs" / ("synthetic" if is_synthetic(args.data) else "real")))


if __name__ == "__main__":
    main()
