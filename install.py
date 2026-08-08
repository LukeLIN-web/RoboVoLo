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
"""Install the RoboVoLo benchmark pack into a local RoboLab checkout."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ASSETS = SCRIPT_DIR / "assets" / "objects"
ASSET_DIRS = ("art_shapes", "chemistry_cubes", "math_wood_cubes")
PACK_NAME = "robovolo"
REQUIRED_ROBOLAB_VERSION = "0.3.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install RoboVoLo assets, tasks, and scenes into RoboLab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  %(prog)s /path/to/RoboLab
""",
    )
    parser.add_argument(
        "robolab",
        type=Path,
        help=f"path to a RoboLab v{REQUIRED_ROBOLAB_VERSION} checkout",
    )
    return parser.parse_args()


def find_lfs_pointers() -> list[Path]:
    """Return bundled files that are still Git LFS pointer stubs."""
    pointer_header = b"version https://git-lfs.github.com/spec/v1"
    pointers: list[Path] = []
    for path in SOURCE_ASSETS.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(len(pointer_header)) == pointer_header:
                    pointers.append(path)
    return pointers


def validate_sources() -> None:
    required = [SCRIPT_DIR / "tasks", SCRIPT_DIR / "scenes", SCRIPT_DIR / "pack.json"]
    required.extend(SOURCE_ASSETS / name for name in ASSET_DIRS)
    missing = [path for path in required if not path.exists()]
    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise SystemExit("Error: RoboVoLo source content is missing:\n" + lines)

    metadata = json.loads((SCRIPT_DIR / "pack.json").read_text(encoding="utf-8"))
    if metadata.get("name") != PACK_NAME or metadata.get("source_split") != PACK_NAME:
        raise SystemExit(
            "Error: pack.json name and source_split must both be "
            f"{PACK_NAME!r}."
        )
    task_count = len(list((SCRIPT_DIR / "tasks").glob("*.py")))
    scene_count = len(list((SCRIPT_DIR / "scenes").glob("*.usda")))
    if task_count != metadata["num_tasks"] or scene_count != metadata["num_scenes"]:
        raise SystemExit(
            "Error: pack metadata does not match source content: "
            f"tasks={task_count}/{metadata['num_tasks']}, "
            f"scenes={scene_count}/{metadata['num_scenes']}."
        )

    pointers = find_lfs_pointers()
    if pointers:
        examples = "\n".join(f"  {path}" for path in pointers[:5])
        remainder = len(pointers) - 5
        if remainder:
            examples += f"\n  ... and {remainder} more"
        raise SystemExit(
            "Error: bundled assets contain unresolved Git LFS pointers. "
            "Run 'git lfs pull' in the RoboVoLo repository first:\n" + examples
        )


def validate_robolab(path: Path) -> Path:
    root = path.expanduser().resolve()
    expected = (root / "assets", root / "assets" / "objects", root / "robolab" / "tasks")
    if not all(item.is_dir() for item in expected):
        raise SystemExit(
            f"Error: {root} does not look like a RoboLab checkout "
            "(expected assets/, assets/objects/, and robolab/tasks/)."
        )
    return root


def validate_robolab_version(robolab: Path) -> None:
    """Require the public RoboLab release validated for this content pack."""
    pyproject = robolab / "pyproject.toml"
    try:
        contents = pyproject.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SystemExit(f"Error: cannot read {pyproject}.") from error

    project = re.search(
        r"(?ms)^\[project\][ \t]*$\n(?P<body>.*?)(?=^\[|\Z)",
        contents,
    )
    version_match = (
        re.search(
            r'''(?m)^\s*version\s*=\s*["'](?P<version>[^"']+)["']\s*(?:#.*)?$''',
            project.group("body"),
        )
        if project
        else None
    )
    if version_match is None:
        raise SystemExit(
            f"Error: cannot determine the RoboLab version from {pyproject}."
        )
    version = version_match.group("version")
    if version != REQUIRED_ROBOLAB_VERSION:
        raise SystemExit(
            "Error: RoboVoLo requires RoboLab "
            f"v{REQUIRED_ROBOLAB_VERSION}; found version {version!r}."
        )


def destinations(robolab: Path) -> list[tuple[Path, Path]]:
    pairs = [
        (SOURCE_ASSETS / name, robolab / "assets" / "objects" / name)
        for name in ASSET_DIRS
    ]
    pairs.extend(
        [
            (SCRIPT_DIR / "scenes", robolab / "assets" / "scenes" / PACK_NAME),
            (SCRIPT_DIR / "tasks", robolab / "robolab" / "tasks" / PACK_NAME),
        ]
    )
    return pairs


def copy_managed_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def update_manifest(robolab: Path) -> None:
    manifest_path = robolab / "content_packs.json"
    pack_metadata = json.loads((SCRIPT_DIR / "pack.json").read_text(encoding="utf-8"))
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Error: cannot read {manifest_path}: {error}") from error
    else:
        manifest = {"packs": {}}
    packs = manifest.setdefault("packs", {})
    packs[PACK_NAME] = {
        "name": PACK_NAME,
        "description": "RoboVoLo benchmark task pack",
        "version": pack_metadata["version"],
        "scenes_dir": f"assets/scenes/{PACK_NAME}",
        "tasks_dir": f"robolab/tasks/{PACK_NAME}",
        "num_tasks": pack_metadata["num_tasks"],
        "num_scenes": pack_metadata["num_scenes"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ensure_gitignored(robolab: Path) -> None:
    gitignore = robolab / ".gitignore"
    start = f"# content-pack:{PACK_NAME}"
    end = f"# /content-pack:{PACK_NAME}"
    managed_paths = [
        "content_packs.json",
        *(f"assets/objects/{name}/" for name in ASSET_DIRS),
        f"assets/scenes/{PACK_NAME}/",
        f"robolab/tasks/{PACK_NAME}/",
    ]
    block = "\n".join([start, *managed_paths, end])
    text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    lines = text.splitlines()
    result: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == start:
            inside = True
            if result and not result[-1].strip():
                result.pop()
            continue
        if line.strip() == end:
            inside = False
            continue
        if not inside:
            result.append(line)
    while result and not result[-1].strip():
        result.pop()
    result.extend(["", block])
    gitignore.write_text("\n".join(result) + "\n", encoding="utf-8")


def _parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return None


def _find_function(tree: ast.AST | None, name: str) -> ast.FunctionDef | None:
    if tree is None:
        return None
    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def _string_constants(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _names(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _assignment(tree: ast.AST | None, name: str) -> ast.AST | None:
    if tree is None:
        return None
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return node.value
    return None


def supports_content_pack_discovery(robolab: Path) -> bool:
    """Check RoboLab's native VoLo registration for robovolo discovery."""
    registration = _parse_python(robolab / "policies" / "volo" / "registration.py")
    volo_task_dirs = _assignment(registration, "VOLO_TASK_SUBFOLDERS")
    register_volo_envs = _find_function(registration, "register_volo_envs")
    if PACK_NAME not in _string_constants(volo_task_dirs) or register_volo_envs is None:
        return False

    task_dirs_defaulted = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "task_dirs"
            for target in node.targets
        )
        and "VOLO_TASK_SUBFOLDERS" in _names(node.value)
        for node in ast.walk(register_volo_envs)
    )
    registration_calls = [
        node
        for node in ast.walk(register_volo_envs)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "auto_register_droid_envs"
    ]
    forwards_task_dirs = any(
        any(
            keyword.arg == "task_dirs" and "task_dirs" in _names(keyword.value)
            for keyword in call.keywords
        )
        for call in registration_calls
    )
    return task_dirs_defaulted and forwards_task_dirs


def supports_grouped_predicates(robolab: Path) -> bool:
    """Check the grouped task APIs required by RoboVoLo definitions."""
    conditionals = _parse_python(
        robolab / "robolab" / "core" / "task" / "conditionals.py"
    )
    grouped = _find_function(conditionals, "pick_and_place_grouped")
    termination = _find_function(conditionals, "object_groups_in_containers")
    grouped_args = {
        arg.arg for arg in (*grouped.args.args, *grouped.args.kwonlyargs)
    } if grouped else set()
    return (
        {"groups", "logical", "K", "score"}.issubset(grouped_args)
        and {"object_grabbed", "object_above_bottom", "object_dropped", "object_in_container"}
        .issubset(_names(grouped))
        and termination is not None
    )


def main() -> None:
    args = parse_args()
    validate_sources()
    robolab = validate_robolab(args.robolab)
    validate_robolab_version(robolab)

    if not supports_content_pack_discovery(robolab):
        raise SystemExit(
            "Error: this RoboLab checkout does not natively discover the installed "
            "robovolo task directory. Install RoboLab v0.3.0 first."
        )
    if not supports_grouped_predicates(robolab):
        raise SystemExit(
            "Error: this RoboLab checkout lacks the grouped task predicates required by "
            "RoboVoLo. Install RoboLab v0.3.0 first."
        )

    pairs = destinations(robolab)
    existing = [destination for _, destination in pairs if destination.exists()]

    if existing:
        lines = "\n".join(f"  {path}" for path in existing)
        raise SystemExit(
            "Error: RoboVoLo requires a fresh installation, but managed "
            "destinations already exist:\n" + lines
        )

    for source, destination in pairs:
        print(f"Installing {source.relative_to(SCRIPT_DIR)} -> {destination}")
        copy_managed_directory(source, destination)

    init_file = robolab / "robolab" / "tasks" / PACK_NAME / "__init__.py"
    init_file.touch(exist_ok=True)
    update_manifest(robolab)
    ensure_gitignored(robolab)

    metadata = json.loads((SCRIPT_DIR / "pack.json").read_text(encoding="utf-8"))
    print(
        f"\nInstalled RoboVoLo: {metadata['num_tasks']} tasks, "
        f"{metadata['num_scenes']} scenes, and 3 custom asset families."
    )
    simready = robolab / "assets" / "objects" / "simready"
    if not simready.exists():
        print("SimReady assets are not distributed with RoboVoLo. Download them with:")
        print(
            f"  {sys.executable} {SCRIPT_DIR / 'download_simready_assets.py'} "
            f"--agree-external-terms {robolab}"
        )


if __name__ == "__main__":
    main()
