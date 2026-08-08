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
        {"object": ["lime01", "lemon_02"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True},
        {"object": ["orange_01", "lychee01"], "container": "mini_plastic_basket", "logical": "all", "require_gripper_detached": True},
    ])


@configclass
class XrSplitByBlockTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrSplitByBlockTask(Task):
    contact_object_list = ['orange_01', 'lime01', 'lemon_02', 'lychee01', 'red_block', 'serving_bowl', 'mini_plastic_basket', 'table']
    scene = import_scene("xr_spatial_split_by_block.usda", contact_object_list)
    terminations = XrSplitByBlockTaskTerminations
    instruction = {'default': 'From the perspective of the robot, put the fruits that are in front of the red block into the bowl, and the fruits behind the red block into the basket', 'vague': 'Split the fruits by position relative to the block from the robot perspective', 'specific': 'From the perspective of the robot, the lime and lemon are in front of the block. The orange and lychee are behind. Put the lime and lemon in the serving bowl; put the orange and lychee in the basket.'}
    episode_length_s: int = 120
    attributes = ['complex_reference', 'semantics', 'spatial']
    subtasks = [
        pick_and_place(object=["lime01", "lemon_02"], container="serving_bowl", logical="all", score=0.5),
        pick_and_place(object=["orange_01", "lychee01"], container="mini_plastic_basket", logical="all", score=0.5),
    ]
