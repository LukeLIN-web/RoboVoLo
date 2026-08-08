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
    object_left_of,
    object_right_of,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class SwapBlockPositionsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_left_of,
        params={"object": "red_block", "reference_object": "blue_block", "require_gripper_detached": True},
    )


@dataclass
class SwapBlockPositionsTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'table']
    scene = import_scene("lh_two_blocks_table.usda", contact_object_list)
    terminations = SwapBlockPositionsTerminations
    instruction = {
        "default": "Swap the positions of the red block and the blue block on the table",
        "vague": "Switch the two blocks around",
        "specific": "Move one block aside, move the other to its spot, then place the first where the second was",
    }
    episode_length_s: int = 90
    attributes = ['memory', 'swapping', 'spatial']
    subtasks = [
        Subtask(
            name="complete_swap",
            conditions={
                "red_block": partial(
                    object_left_of, object="red_block",
                    reference_object="blue_block",
                    require_gripper_detached=True,
                ),
                "blue_block": partial(
                    object_right_of, object="blue_block",
                    reference_object="red_block",
                    require_gripper_detached=True,
                ),
            },
            logical="all",
            score=1.0,
        ),
    ]
