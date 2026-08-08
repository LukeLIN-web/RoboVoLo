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
    object_in_container,
    object_outside_of,
    object_outside_of_and_on_surface,
    object_groups_in_containers,
    pick_and_place,
    stacked,
)
from robolab.core.task.task import Task


def _success(env, env_id=None):
    # Original condition: all objects in their respective containers
    original = object_groups_in_containers(env, groups=[
        {"object": ["spam_can", "tuna_can", "corn_can"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True},
        {"object": ["lemon_02", "orange_01"], "container": "open_cardboard_box_004", "logical": "all", "require_gripper_detached": True},
    ])

    # Alternative condition: at least one object from each group is in the container
    # AND all objects in that group are stacked (handles top of stack poking out)
    group1_any_in = object_in_container(env, object=["spam_can", "tuna_can", "corn_can"], container="serving_bowl", logical="any", require_gripper_detached=True)
    group1_stacked = stacked(env, objects=["spam_can", "tuna_can", "corn_can"])

    group2_any_in = object_in_container(env, object=["lemon_02", "orange_01"], container="open_cardboard_box_004", logical="any", require_gripper_detached=True)
    group2_stacked = stacked(env, objects=["lemon_02", "orange_01"])

    alternative = group1_any_in & group1_stacked & group2_any_in & group2_stacked

    return original | alternative


@configclass
class XrOddEvenTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrOddEvenTask(Task):
    contact_object_list = ['spam_can', 'lemon_02', 'tuna_can', 'orange_01', 'corn_can', 'serving_bowl', 'open_cardboard_box_004', 'table']
    scene = import_scene("xr_count_odd_even.usda", contact_object_list)
    terminations = XrOddEvenTaskTerminations
    instruction = {'default': 'Five items are in a row. Put the odd-numbered ones (1st, 3rd, 5th) into the bowl, and the even-numbered ones (2nd, 4th) into the bin', 'vague': 'Sort by odd and even positions', 'specific': 'From left to right: spam can (1), lemon (2), tuna can (3), orange (4), corn can (5). Put the spam, tuna, and corn cans in the bowl; put the lemon and orange in the bin.'}
    episode_length_s: int = 150
    attributes = ['complex_reference', 'semantics', 'counting']
    subtasks = [
        pick_and_place(object=["spam_can", "tuna_can", "corn_can"], container="serving_bowl", logical="all", score=0.5),
        pick_and_place(object=["lemon_02", "orange_01"], container="open_cardboard_box_004", logical="all", score=0.5),
    ]
