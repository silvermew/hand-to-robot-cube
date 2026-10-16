"""Merge LeRobot datasets with the same features into one (run with the lerobot venv's python).

    ~/.venvs/lerobot/bin/python -m src.merge_datasets OUT_ROOT IN_ROOT [IN_ROOT ...]

Converting demos is slow (the frames are encoded to H.264 one by one), so each batch of demos is converted once,
in parallel if need be, and merged with lerobot's aggregate_datasets (checked against lerobot 0.6.1).
"""
import argparse
import shutil
from pathlib import Path


def main():
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out")
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()
    out, inputs = Path(args.out), [Path(q) for q in args.inputs]
    if out.exists():
        shutil.rmtree(out)  # fully derived from the inputs
    aggregate_datasets(repo_ids=[f"local/{q.name}" for q in inputs], aggr_repo_id=f"local/{out.name}", roots=inputs,
                       aggr_root=out)
    meta = LeRobotDatasetMetadata(f"local/{out.name}", root=out)
    print(f"wrote {meta.total_episodes} episodes, {meta.total_frames} frames, {len(meta.tasks)} tasks to {out}")


if __name__ == "__main__":
    main()
