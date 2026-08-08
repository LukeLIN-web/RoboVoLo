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
        {"object": ["lime01", "lemon_02", "lychee01"], "container": "bowl", "logical": "all", "require_gripper_detached": True},
        {"object": ["orange_01"], "container": "bin_a01", "logical": "all", "require_gripper_detached": True},
    ])


@configclass
class XrBiggestSeparateTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrBiggestSeparateTask(Task):
    contact_object_list = ['orange_01', 'lime01', 'lemon_02', 'lychee01', 'bowl', 'bin_a01', 'table']
    scene = import_scene("xr_size_biggest_separate.usda", contact_object_list)
    terminations = XrBiggestSeparateTaskTerminations
    instruction = {'default': 'Put all fruits except the biggest one into the bowl, and put the biggest fruit into the bin', 'vague': 'Separate the biggest fruit from the rest', 'specific': 'The orange is the biggest. Put the lime, lemon, and lychee into the bowl, and put the orange into the bin.'}
    episode_length_s: int = 120
    attributes = ['complex_reference', 'semantics', 'size_reasoning', 'negation']
    subtasks = [
        pick_and_place(object=["lime01", "lemon_02", "lychee01"], container="bowl", logical="all", score=0.5),
        pick_and_place(object="orange_01", container="bin_a01", score=0.5),
    ]
