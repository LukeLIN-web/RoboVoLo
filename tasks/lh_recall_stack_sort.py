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

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_groups_in_containers,
    pick_and_place_grouped,
)
from robolab.core.task.task import Task


@configclass
class RecallStackSortTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["blue_block"], "container": "bowl", "logical": "all", "require_gripper_detached": True},
                {"object": ["red_block"], "container": "bin_a01", "logical": "all", "require_gripper_detached": True},
            ]
        },
    )


@dataclass
class RecallStackSortTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'bowl', 'bin_a01', 'table']
    scene = import_scene("lh_partial_stack_3blocks_bowl_bin.usda", contact_object_list)
    terminations = RecallStackSortTerminations
    instruction = {
        "default": "Three blocks are on the table \u2014 two are stacked and one is loose. Take the top block off the stack and put it in the bowl. Then put the block that was on the bottom of the stack into the bin.",
        "vague": "Route the stacked blocks to the right containers",
        "specific": "Put the top of the stack into the bowl, then identify and place the bottom-of-stack block into the bin",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'recall', 'stacking']
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["blue_block"], "container": "bowl"},
                {"object": ["red_block"], "container": "bin_a01"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
