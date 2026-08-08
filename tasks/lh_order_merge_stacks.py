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
class MergeStacksTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={"objects": ["red_block", "blue_block", "green_block", "yellow_block"], "order": "bottom_to_top"},
    )


@dataclass
class MergeStacksTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'yellow_block', 'table']
    scene = import_scene("lh_two_paired_stacks.usda", contact_object_list)
    terminations = MergeStacksTerminations
    instruction = {
        "default": "Merge the two two-block stacks into a single four-block tower. The left stack's blocks form the bottom half and the right stack's blocks go on top \u2014 keep the original stacking order within each pair",
        "vague": "Combine both stacks into one tower",
        "specific": "Put the right stack's blocks on top of the left stack, preserving internal order",
    }
    episode_length_s: int = 90
    attributes = ['memory', 'ordering', 'stacking']
    # Three ordered steps to merge two paired stacks:
    #   1. Unstack yellow (top of right stack) onto the table.
    #   2. Place green on top of the left stack (red+blue+green).
    #   3. Place yellow on top of green (= termination, full 4-tower).
    subtasks = [
        pick_and_place_on_surface(
            object=["yellow_block"], surface="table",
            logical="all", score=1 / 3,
        ),
        Subtask(
            name="stack_g_on_b",
            conditions=partial(
                stacked,
                objects=["red_block", "blue_block", "green_block"],
                order="bottom_to_top",
            ),
            score=1 / 3,
        ),
        Subtask(
            name="stack_y_on_g",
            conditions=partial(
                stacked,
                objects=["red_block", "blue_block", "green_block", "yellow_block"],
                order="bottom_to_top",
            ),
            score=1 / 3,
        ),
    ]
