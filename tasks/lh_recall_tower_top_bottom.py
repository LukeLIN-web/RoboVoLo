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


@configclass
class RecallTowerTopBottomTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={"objects": ["red_block", "yellow_block"]},
    )


@dataclass
class RecallTowerTopBottomTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'yellow_block', 'table']
    scene = import_scene("lh_four_tower_bowl_bin.usda", contact_object_list)
    terminations = RecallTowerTopBottomTerminations
    instruction = {
        "default": "Unstack all four blocks. Then build a two-block tower using only the block that was on the very top and the one that was on the very bottom \u2014 leave the middle two on the table.",
        "vague": "Unstack all and stack just the extremes",
        "specific": "Remove all blocks, then stack the originally-bottom block and the originally-top block together in either order",
    }
    episode_length_s: int = 120
    attributes = ['memory', 'recall', 'stacking']
    # Four ordered phases (top-down unstack is physically required —
    # only the topmost block of the original tower is graspable):
    #   1. yellow (top) → table
    #   2. green (next) → table
    #   3. blue (next) → table — red remains in place at the bottom
    #   4. yellow on red (= termination)
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
        Subtask(
            name="build_pair",
            conditions=partial(stacked, objects=["red_block", "yellow_block"]),
            score=0.25,
        ),
    ]
