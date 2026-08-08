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

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    pick_and_place_on_surface,
    stacked,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


def both_towers_split(env, env_id=None, object=None, objects=None):
    """Compound termination: both the bottom-half and top-half towers must be correctly stacked.

    Accepts and ignores ``object``/``objects`` kwargs so callers can
    pass them via ``partial(both_towers_split, object=[…])`` for the
    GTStateExporter's ``object_completed`` lookup (audit Gap 1).
    """
    bottom_ok = stacked(env, objects=["red_block", "blue_block"], order="bottom_to_top", env_id=env_id)
    top_ok = stacked(env, objects=["green_block", "yellow_block"], order="bottom_to_top", env_id=env_id)
    if env_id is not None:
        return bottom_ok and top_ok
    return bottom_ok & top_ok


@configclass
class SplitTowerHalvesTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=both_towers_split,
        params={},
    )


@dataclass
class SplitTowerHalvesTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'yellow_block', 'table']
    scene = import_scene("lh_four_tower_bowl_bin.usda", contact_object_list)
    terminations = SplitTowerHalvesTerminations
    instruction = {
        "default": "The four blocks are stacked in a tower. Take them all apart. Then build two separate two-block towers \u2014 one from the top half of the original tower and one from the bottom half, keeping each pair in their original order",
        "vague": "Split the tower into two pairs keeping original order",
        "specific": "Unstack all four blocks, then build one stack from the bottom two and another from the top two, each in their original stacking order",
    }
    episode_length_s: int = 150
    attributes = ['memory', 'ordering', 'stacking']
    # Three ordered unstacks (top-down) then a single split-success
    # check.  NOTE: the underlying ``stacked`` predicate is pairwise
    # contact-based and is True at episode start for both pairs (since
    # the original 4-tower contains red→blue and green→yellow as
    # adjacent pairs in contact); the actual termination
    # ``both_towers_split`` therefore needs strengthening separately.
    subtasks = [
        pick_and_place_on_surface(
            object=["yellow_block"], surface="table",
            logical="all", score=0.25,
        ),
        pick_and_place_on_surface(
            object=["green_block"], surface="table",
            logical="all", score=0.25,
        ),
        pick_and_place_on_surface(
            object=["blue_block"], surface="table",
            logical="all", score=0.25,
        ),
        # Final phase: two independent pair-stack checks, parallel under
        # ``logical="all"`` so they can complete in any order.  Each
        # group's partial keyword ``objects=[…]`` is read by
        # GTStateExporter._export_object_completed so per-block
        # completion fires when its pair is stacked.
        Subtask(
            name="rebuild_pairs",
            conditions={
                "bottom_pair": partial(
                    stacked,
                    objects=["red_block", "blue_block"],
                    order="bottom_to_top",
                ),
                "top_pair": partial(
                    stacked,
                    objects=["green_block", "yellow_block"],
                    order="bottom_to_top",
                ),
            },
            logical="all",
            score=0.25,
        ),
    ]
