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
        {"object": ["banana_2", "bitter_melon"], "container": "bin_a03", "logical": "all", "require_gripper_detached": True},
        {"object": ["lemon_02", "lychee01"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True},
    ])


@configclass
class XrSortBigSmallTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrSortBigSmallTask(Task):
    contact_object_list = ['banana_2', 'bitter_melon', 'lemon_02', 'lychee01', 'bin_a03', 'serving_bowl', 'table']
    scene = import_scene("xr_size_sort_big_small.usda", contact_object_list)
    terminations = XrSortBigSmallTaskTerminations
    instruction = {'default': 'Sort the items by size: put the two biggest into the bin and the two smallest into the bowl', 'vague': 'Big ones in the bin, small ones in the bowl', 'specific': 'The banana and bitter melon are the two biggest. The lemon and lychee are the two smallest. Put the banana and bitter melon in the bin, and the lemon and lychee in the serving bowl.'}
    episode_length_s: int = 120
    attributes = ['complex_reference', 'semantics', 'size_reasoning', 'sorting']
    subtasks = [
        pick_and_place(object=["banana_2", "bitter_melon"], container="bin_a03", logical="all", score=0.5),
        pick_and_place(object=["lemon_02", "lychee01"], container="serving_bowl", logical="all", score=0.5),
    ]
