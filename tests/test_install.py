# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility checks for the RoboVoLo installer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "install.py"
_SPEC = importlib.util.spec_from_file_location("robovolo_install", _SCRIPT)
assert _SPEC and _SPEC.loader
install = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(install)


def _write_tree(root: Path, *, constants: str, registration: str | None = None) -> None:
    constants_path = root / "robolab" / "constants.py"
    constants_path.parent.mkdir(parents=True, exist_ok=True)
    constants_path.write_text(constants, encoding="utf-8")
    if registration is not None:
        registration_path = root / "policies" / "volo" / "registration.py"
        registration_path.parent.mkdir(parents=True, exist_ok=True)
        registration_path.write_text(registration, encoding="utf-8")


def test_requires_robolab_v030(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "nvidia-robolab"\nversion = "0.3.0"\n',
        encoding="utf-8",
    )
    install.validate_robolab_version(tmp_path)

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "nvidia-robolab"\nversion = "0.2.1"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match=r"requires RoboLab v0\.3\.0"):
        install.validate_robolab_version(tmp_path)


def test_accepts_volo_scoped_robovolo_discovery(tmp_path):
    _write_tree(
        tmp_path,
        constants="DEFAULT_TASK_SUBFOLDERS = ['benchmark']\n",
        registration="""
VOLO_TASK_SUBFOLDERS = [*DEFAULT_TASK_SUBFOLDERS, 'robovolo']
def register_volo_envs(task_dirs=None):
    if task_dirs is None:
        task_dirs = VOLO_TASK_SUBFOLDERS
    auto_register_droid_envs(task_dirs=task_dirs)
""",
    )

    assert install.supports_content_pack_discovery(tmp_path)


def test_rejects_unrelated_or_unwired_discovery(tmp_path):
    incomplete = (
        "VOLO_TASK_SUBFOLDERS = ['benchmark', 'other_pack']\n",
        "VOLO_TASK_SUBFOLDERS = ['benchmark', 'robovolo']\n",
        "DOCUMENTATION = 'robovolo is unsupported'\n",
        """
VOLO_TASK_SUBFOLDERS = ['benchmark', 'robovolo']
def register_volo_envs():
    pass
""",
        """
VOLO_TASK_SUBFOLDERS = ['benchmark', 'robovolo']
def register_volo_envs():
    auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS)
""",
    )
    for registration in incomplete:
        _write_tree(
            tmp_path,
            constants="DEFAULT_TASK_SUBFOLDERS = ['benchmark', 'robovolo']\n",
            registration=registration,
        )
        assert not install.supports_content_pack_discovery(tmp_path)


def test_install_paths_manifest_and_gitignore_use_robovolo(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setattr(install, "SCRIPT_DIR", source_root)
    monkeypatch.setattr(install, "SOURCE_ASSETS", source_root / "assets" / "objects")

    robolab = tmp_path / "RoboLab"
    robolab.mkdir()
    pairs = install.destinations(robolab)
    assert [destination for _, destination in pairs[-2:]] == [
        robolab / "assets" / "scenes" / "robovolo",
        robolab / "robolab" / "tasks" / "robovolo",
    ]

    (source_root / "pack.json").write_text(
        '{"version":"1.0","num_tasks":126,"num_scenes":123}\n',
        encoding="utf-8",
    )
    (robolab / ".gitignore").write_text("keep-this\n", encoding="utf-8")
    install.update_manifest(robolab)
    install.ensure_gitignored(robolab)

    manifest = (robolab / "content_packs.json").read_text(encoding="utf-8")
    assert '"robovolo"' in manifest
    assert "assets/scenes/robovolo" in manifest
    assert "robolab/tasks/robovolo" in manifest
    gitignore = (robolab / ".gitignore").read_text(encoding="utf-8")
    assert "keep-this" in gitignore
    assert "# content-pack:robovolo" in gitignore
    assert "content_packs.json" in gitignore
    for family in install.ASSET_DIRS:
        assert f"assets/objects/{family}/" in gitignore
    assert "assets/scenes/robovolo/" in gitignore
    assert "robolab/tasks/robovolo/" in gitignore


def test_fresh_install_writes_only_native_pack_layout(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    (source_root / "assets" / "objects" / "family").mkdir(parents=True)
    (source_root / "assets" / "objects" / "family" / "asset.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    (source_root / "scenes").mkdir()
    (source_root / "scenes" / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (source_root / "tasks").mkdir()
    (source_root / "tasks" / "task.py").write_text("class Task: pass\n", encoding="utf-8")
    (source_root / "pack.json").write_text(
        '{"name":"robovolo","source_split":"robovolo","version":"1.0",'
        '"num_tasks":1,"num_scenes":1}\n',
        encoding="utf-8",
    )

    robolab = tmp_path / "RoboLab"
    (robolab / "assets" / "objects").mkdir(parents=True)
    (robolab / "robolab" / "tasks").mkdir(parents=True)
    (robolab / ".gitignore").write_text("keep-this\n", encoding="utf-8")
    (robolab / "pyproject.toml").write_text(
        '[project]\nname = "nvidia-robolab"\nversion = "0.3.0"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(install, "SCRIPT_DIR", source_root)
    monkeypatch.setattr(install, "SOURCE_ASSETS", source_root / "assets" / "objects")
    monkeypatch.setattr(install, "ASSET_DIRS", ("family",))
    monkeypatch.setattr(install, "parse_args", lambda: type("Args", (), {"robolab": robolab})())
    monkeypatch.setattr(install, "supports_content_pack_discovery", lambda root: True)
    monkeypatch.setattr(install, "supports_grouped_predicates", lambda root: True)

    install.main()

    assert (robolab / "assets" / "objects" / "family" / "asset.usda").is_file()
    assert (robolab / "assets" / "scenes" / "robovolo" / "scene.usda").is_file()
    assert (robolab / "robolab" / "tasks" / "robovolo" / "task.py").is_file()
    assert (robolab / "robolab" / "tasks" / "robovolo" / "__init__.py").is_file()
    manifest = (robolab / "content_packs.json").read_text(encoding="utf-8")
    assert '"robovolo"' in manifest
    assert "assets/scenes/robovolo" in manifest
    assert "robolab/tasks/robovolo" in manifest
    gitignore = (robolab / ".gitignore").read_text(encoding="utf-8")
    assert "assets/objects/family/" in gitignore


def test_incompatible_checkout_fails_before_copying(tmp_path, monkeypatch):
    robolab = tmp_path / "RoboLab"
    for path in (
        robolab / "assets" / "objects",
        robolab / "robolab" / "tasks",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (robolab / "robolab" / "constants.py").write_text(
        "DEFAULT_TASK_SUBFOLDERS = ['benchmark']\n",
        encoding="utf-8",
    )
    (robolab / "pyproject.toml").write_text(
        '[project]\nname = "nvidia-robolab"\nversion = "0.3.0"\n',
        encoding="utf-8",
    )

    copied = False

    def fail_if_called(*args, **kwargs):
        nonlocal copied
        copied = True

    monkeypatch.setattr(install, "validate_sources", lambda: None)
    monkeypatch.setattr(install, "parse_args", lambda: type("Args", (), {"robolab": robolab})())
    monkeypatch.setattr(install, "copy_managed_directory", fail_if_called)

    with pytest.raises(SystemExit, match="does not natively discover"):
        install.main()
    assert not copied
    assert not (robolab / "robolab" / "tasks" / "robovolo").exists()


def test_grouped_predicate_validation(tmp_path):
    path = tmp_path / "robolab" / "core" / "task" / "conditionals.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
def pick_and_place_grouped(groups, logical='all', K=None, score=1.0):
    partial(object_grabbed); partial(object_above_bottom)
    partial(object_dropped); partial(object_in_container)
def object_groups_in_containers(env, groups): pass
""",
        encoding="utf-8",
    )
    assert install.supports_grouped_predicates(tmp_path)

    path.write_text("def pick_and_place_grouped(groups): pass\n", encoding="utf-8")
    assert not install.supports_grouped_predicates(tmp_path)
