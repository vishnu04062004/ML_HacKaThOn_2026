#!/usr/bin/env python3
"""Run the complete challenge solution."""

from __future__ import annotations

import argparse
from pathlib import Path

from er_pipeline.pipeline import Config, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--training-targets-per-source", type=int, default=250_000)
    parser.add_argument(
        "--retrieval-top-n", type=int, default=8,
        help="Word candidates per target; extra typo-tolerant candidates are added automatically",
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume", action="store_true",
        help="Preserve cached chunks and continue an interrupted run",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Train, calibrate macro F0.5, save a checkpoint, and stop before inference",
    )
    parser.add_argument(
        "--resume-threshold", type=float, default=None,
        help="Optional threshold printed by the interrupted run",
    )
    parser.add_argument(
        "--resume-margin", type=float, default=None,
        help="Optional probability margin printed by the interrupted run",
    )
    args = parser.parse_args()
    metrics = run(Config(**vars(args)))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import json
    main()
