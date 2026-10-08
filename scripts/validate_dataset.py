#!/usr/bin/env python
"""CLI: validate one or more TemperatureVLA HDF5 files.

    python scripts/validate_dataset.py data/pick_hot_cylinder_aluminum.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempvla.tools.validate import validate_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="HDF5 dataset file(s) to validate")
    parser.add_argument(
        "--no-strict-channels",
        action="store_true",
        help="Do not require proprio/tactile/temperature channels (structure-only check).",
    )
    args = parser.parse_args()

    failed = False
    for path in args.files:
        report = validate_file(path, strict_channels=not args.no_strict_channels)
        print(report.summary())
        print()
        failed |= not report.ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
