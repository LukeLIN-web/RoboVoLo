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
        {"object": ["lemon_02"], "container": "bowl", "logical": "all", "require_gripper_detached": True},
        {"object": ["orange_01"], "container": "bin_a02", "logical": "all", "require_gripper_detached": True},
    ])


@configclass
class XrLeftRightMugTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrLeftRightMugTask(Task):
    contact_object_list = ['lemon_02', 'orange_01', 'ceramic_mug', 'bowl', 'bin_a02', 'table']
    scene = import_scene("xr_spatial_left_right_mug.usda", contact_object_list)
    terminations = XrLeftRightMugTaskTerminations
    instruction = {'default': 'Put the fruit to the right of the mug into the bowl, and the fruit to the left of the mug into the bin', 'vague': 'Sort the fruits by their position relative to the mug', 'specific': 'The lemon is to the right of the mug and the orange is to the left. Put the lemon in the bowl and the orange in the bin.'}
    episode_length_s: int = 180
    attributes = ['complex_reference', 'semantics', 'spatial']
    subtasks = [
        pick_and_place(object="lemon_02", container="bowl", score=0.5),
        pick_and_place(object="orange_01", container="bin_a02", score=0.5),
    ]
