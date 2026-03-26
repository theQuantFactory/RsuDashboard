from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qfpytoolbox import run_rsu_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RSU pipeline -> parquet snapshots")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="snapshots/csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frames = run_rsu_pipeline(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(f"Built {len(frames)} frames")
    for name, df in frames.items():
        print(f"- {name}: {len(df)} rows x {len(df.columns)} cols")


if __name__ == "__main__":
    main()
