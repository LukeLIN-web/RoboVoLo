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
    object_in_container,
    object_on_top,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@atomic
def _compound_success(env, env_id=None):
    """Top half (green, yellow) in bin AND bottom half (red, blue) unstacked on the table."""
    return (
        object_in_container(env, object=["green_block", "yellow_block"], container="bin_a02", logical="all", require_gripper_detached=True)
        & object_on_top(env, object=["red_block", "blue_block"], reference_object="table", logical="all", require_gripper_detached=True)
    )


@configclass
class RecallTopHalfToBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_compound_success, params={})


@dataclass
class RecallTopHalfToBinTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'yellow_block', 'bin_a02', 'table']
    scene = import_scene("lh_four_tower_bowl_bin.usda", contact_object_list)
    terminations = RecallTopHalfToBinTerminations
    instruction = {
        "default": "Unstack all four blocks onto the table. Then put the two blocks that were in the top half of the tower into the bin.",
        "vague": "Unstack and bin the top two",
        "specific": "Remove all blocks to the table, then identify the two that were at the top of the stack and place them in the bin",
    }
    episode_length_s: int = 120
    attributes = ['memory', 'recall', 'stacking']
    # Single subtask with 3 parallel object-keyed groups, each tracking
    # only the final destination.  Any-order, any-path completion: the
    # policy is free to first park objects on the table during unstack
    # and later move them to the bin — only the ultimate placement is
    # scored.  ``logical="all"`` requires all 3 destinations satisfied.
    subtasks = [
        Subtask(
            name="place_all",
            conditions={
                "blue_block": partial(
                    object_on_top, object="blue_block",
                    reference_object="table",
                    require_gripper_detached=True,
                ),
                "green_block": partial(
                    object_in_container, object="green_block",
                    container="bin_a02", tolerance=0.01,
                    require_gripper_detached=True,
                ),
                "yellow_block": partial(
                    object_in_container, object="yellow_block",
                    container="bin_a02", tolerance=0.01,
                    require_gripper_detached=True,
                ),
            },
            logical="all",
            score=1.0,
        ),
    ]
