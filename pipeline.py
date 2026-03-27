from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rsu import run_rsu_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RSU pipeline -> parquet snapshots")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="snapshots/csv")
    parser.add_argument(
        "--snapshot-profile",
        choices=["dashboard", "full"],
        default="dashboard",
        help="dashboard: faster/lighter snapshots for app runtime, full: persist all intermediate frames",
    )
    parser.add_argument(
        "--timeseries-freq",
        choices=["weekly", "daily"],
        default="weekly",
        help="Cadence for score/near-threshold timeseries snapshots.",
    )
    parser.add_argument(
        "--timeseries-batch-size",
        type=int,
        default=0,
        help="Optional number of periods to process per batch (0 = all periods in one pass).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    profile = "dashboard" if args.snapshot_profile == "dashboard" else "full"
    frames = run_rsu_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        snapshot_profile=profile,
        timeseries_freq=("D" if args.timeseries_freq == "daily" else "W-MON"),
        timeseries_batch_size=max(0, int(args.timeseries_batch_size)),
    )
    print(f"Built {len(frames)} frames")
    for name, df in frames.items():
        print(f"- {name}: {len(df)} rows x {len(df.columns)} cols")


if __name__ == "__main__":
    main()
