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
)
from robolab.core.task.task import Task


def _success(env, env_id=None):
    return object_groups_in_containers(env, groups=[
        {"object": ["lychee01"], "container": "bowl", "logical": "all", "require_gripper_detached": True},
        {"object": ["lemon_02"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True},
        {"object": ["orange_01"], "container": "blackandbrassbowl_small", "logical": "all", "require_gripper_detached": True},
    ])


@configclass
class XrSortThreeBowlsTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrSortThreeBowlsTask(Task):
    contact_object_list = ['lychee01', 'lemon_02', 'orange_01', 'bowl', 'serving_bowl', 'blackandbrassbowl_small', 'table']
    scene = import_scene("xr_size_sort_three_bowls.usda", contact_object_list)
    terminations = XrSortThreeBowlsTaskTerminations
    instruction = {'default': 'Sort the three fruits by size into the three bowls — put the smallest in the left bowl, the medium one in the middle bowl, and the biggest in the right bowl', 'vague': 'Size-sort the fruits into the bowls', 'specific': 'Put the lychee (smallest) in the red bowl on the left, the lemon (medium) in the silver bowl in the middle, and the orange (biggest) in the brass bowl on the right.'}
    episode_length_s: int = 270
    attributes = ['complex_reference', 'semantics', 'size_reasoning', 'sorting']
    subtasks = [
        pick_and_place(object="lychee01", container="bowl", score=1/3),
        pick_and_place(object="lemon_02", container="serving_bowl", score=1/3),
        pick_and_place(object="orange_01", container="blackandbrassbowl_small", score=1/3),
    ]
