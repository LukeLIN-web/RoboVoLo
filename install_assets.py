#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install the RoboVoLo asset directories into a RoboLab checkout.

By default installation is strict: none of the destination asset directories
may already exist. Use --merge or --overwrite to select a destructive/conflict
handling mode; both modes display a warning and require confirmation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR / "assets" / "objects"
ASSET_DIRS = (
    "art_shapes",
    "chemistry_cubes",
    "math_wood_cubes",
)
SIMREADY_DOWNLOADER = SCRIPT_DIR / "download_simready_assets.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install bundled RoboVoLo custom assets into RoboLab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""modes:
  default      Fail if any destination asset directory already exists.
  --merge      Merge into existing directories and replace conflicting files.
  --overwrite  Delete existing destination directories before copying.

examples:
  %(prog)s /path/to/RoboLab
  %(prog)s --merge /path/to/RoboLab
  %(prog)s --overwrite /path/to/RoboLab
""",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--merge",
        action="store_true",
        help="merge assets into existing directories (confirmation required)",
    )
    modes.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing asset directories (confirmation required)",
    )
    parser.add_argument("robolab", type=Path, help="path to the RoboLab repository")
    return parser.parse_args()


def find_lfs_pointers() -> list[Path]:
    """Return bundled files that are still Git LFS pointer stubs."""
    pointer_header = b"version https://git-lfs.github.com/spec/v1"
    pointers = []
    for path in SOURCE_ROOT.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(len(pointer_header)) == pointer_header:
                    pointers.append(path)
    return pointers


def validate_robolab(path: Path) -> Path:
    root = path.expanduser().resolve()
    expected = root / "assets" / "objects"
    if not (root / "assets").is_dir() or not (root / "robolab").is_dir():
        raise SystemExit(
            f"Error: {root} does not look like a RoboLab repository "
            "(expected assets/ and robolab/)."
        )
    if not expected.is_dir():
        raise SystemExit(f"Error: RoboLab object directory does not exist: {expected}")
    return expected


def confirm(mode: str, existing: list[Path]) -> None:
    if mode == "merge":
        action = "merge into existing directories and replace conflicting files"
    else:
        action = "DELETE existing destination directories and replace them"

    print(f"WARNING: --{mode} will {action}.", file=sys.stderr)
    if existing:
        print("Existing destinations:", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
    else:
        print(
            "No destination asset directories currently exist, but explicit "
            f"--{mode} mode was requested.",
            file=sys.stderr,
        )

    if not sys.stdin.isatty():
        raise SystemExit(
            "Error: confirmation requires an interactive terminal. "
            "Run the command interactively."
        )
    answer = input("Continue? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Installation cancelled.")


def copy_directory(source: Path, destination: Path, mode: str) -> None:
    if mode == "overwrite" and destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    if mode == "merge":
        if destination.exists() and not destination.is_dir():
            destination.unlink()
        shutil.copytree(
            source, destination, dirs_exist_ok=True, copy_function=shutil.copy2
        )
    else:
        shutil.copytree(source, destination, copy_function=shutil.copy2)


def main() -> None:
    args = parse_args()

    missing_sources = [SOURCE_ROOT / name for name in ASSET_DIRS if not (SOURCE_ROOT / name).is_dir()]
    if missing_sources:
        lines = "\n".join(f"  {path}" for path in missing_sources)
        raise SystemExit("Error: bundled asset directories are missing:\n" + lines)

    lfs_pointers = find_lfs_pointers()
    if lfs_pointers:
        examples = "\n".join(f"  {path}" for path in lfs_pointers[:5])
        remainder = len(lfs_pointers) - 5
        if remainder > 0:
            examples += f"\n  ... and {remainder} more"
        raise SystemExit(
            "Error: bundled assets contain unresolved Git LFS pointers. "
            "Run 'git lfs pull' in the RoboVoLo repository before installing:\n"
            + examples
        )

    destination_root = validate_robolab(args.robolab)
    destinations = [(SOURCE_ROOT / name, destination_root / name) for name in ASSET_DIRS]
    existing = [destination for _, destination in destinations if destination.exists()]

    mode = "merge" if args.merge else "overwrite" if args.overwrite else "strict"
    if mode == "strict" and existing:
        lines = "\n".join(f"  {path}" for path in existing)
        raise SystemExit(
            "Error: strict installation requires that none of the destination "
            "asset directories already exist:\n"
            + lines
            + "\nRe-run with --merge or --overwrite to handle existing assets."
        )
    if mode != "strict":
        confirm(mode, existing)

    for source, destination in destinations:
        print(f"Installing {source.name} -> {destination}")
        copy_directory(source, destination, mode)

    print(f"Installed {len(destinations)} custom asset directories into {destination_root}")
    if not (destination_root / "simready").exists():
        print("\nSimReady assets are not distributed in this repository.")
        print("Download them separately with:")
        print(
            f"  {sys.executable} {SIMREADY_DOWNLOADER} "
            f"--agree-external-terms {args.robolab.expanduser().resolve()}"
        )


if __name__ == "__main__":
    main()
