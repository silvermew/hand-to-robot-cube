"""Rebuild every result from the raw videos with one command.

    python -m src.run_all --synthetic                        # synthetic clips (generated if missing)
    python -m src.run_all --data data/raw --out outputs/real
    python -m src.run_all --data data/raw --out outputs/real --from replay   # resume at a step

Steps: generate (synthetic only, if missing or --regen) -> prepare -> calibrate -> track ->
qc -> extract -> eval (synthetic only: vs ground truth) -> replay -> world_model -> planning.
Derived files of a step (tracks, processed clips, sim demos) are cleared before it runs, so
a clip that fails a step cannot leave stale results behind.
"""
import argparse
import shutil
import time

from src import calibrate, eval_synthetic, extract, prepare_videos, qc, track
from src.common import ROOT, data_dirs, is_synthetic, list_videos, loud_warning

STEPS = ["generate", "prepare", "calibrate", "track", "qc", "extract", "eval", "replay", "world_model", "planning"]


def clear(folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--synthetic", action="store_true", help="use data/synthetic/raw and outputs/synthetic")
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--from", dest="start", choices=STEPS, default="generate")
    p.add_argument("--regen", action="store_true", help="re-render the synthetic clips")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    raw = args.data or str(ROOT / "data" / ("synthetic" if args.synthetic else "") / "raw")
    synthetic = is_synthetic(raw)
    out = args.out or str(ROOT / "outputs" / ("synthetic" if synthetic else "real"))
    if synthetic != args.synthetic and args.synthetic:
        raise SystemExit("--synthetic needs a data folder under data/synthetic")
    dirs = data_dirs(raw)
    steps = STEPS[STEPS.index(args.start):]
    t0 = time.time()

    def banner(name):
        print(f"\n=== {name} ({time.time() - t0:.0f} s) " + "=" * 40)

    if "generate" in steps and synthetic and (args.regen or not dirs["raw"].exists()):
        banner("generate synthetic clips")
        from tests import make_synthetic
        make_synthetic.main(["--out", str(dirs["base"]), "--workers", str(args.workers)])
    if not dirs["raw"].exists() or not list_videos(dirs["raw"], warn=False)[0]:
        raise SystemExit(f"no clips in {dirs['raw']} (expected clip_<nn>_rot<+/-deg>_<A|B|C>.mp4)")
    if "prepare" in steps:
        banner("prepare videos")
        prepare_videos.main(raw)
    if "calibrate" in steps:
        banner("calibrate")
        calibrate.main(raw)
    if "track" in steps:
        banner("track")
        clear(dirs["tracks"])
        track.track_all(raw, out, workers=args.workers)
    if "qc" in steps:
        banner("quality check")
        qc.qc_report(raw, workers=args.workers)
    if "extract" in steps:
        banner("extract")
        clear(dirs["processed"])
        extract.extract_all(raw, out)
    if "eval" in steps and synthetic:
        banner("tracking vs ground truth")
        eval_synthetic.main(raw, out)
    # robosuite and torch are imported only now: worker pools above must not fork a process with their threads
    from src import planning, replay_all
    from src.world_model import run as world_model
    if "replay" in steps:
        banner("sim replay")
        clear(dirs["sim_demos"])
        replay_all.main(raw, out)
        replay_all.main(raw, out, turn="cube")
    if "world_model" in steps:
        banner("world model")
        world_model.main(raw, out)
    if "planning" in steps:
        banner("planning")
        planning.main(raw, out)
    print(f"\ndone in {time.time() - t0:.0f} s; results in {out}")
    if not synthetic and not dirs["camera"].exists():
        loud_warning("these results used PLACEHOLDER intrinsics (no camera.npz): calibrate and rerun")


if __name__ == "__main__":
    main()
