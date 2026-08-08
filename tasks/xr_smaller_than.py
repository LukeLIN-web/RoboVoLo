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
    return (
        object_in_container(env, object=["lemon_02", "lime01", "lychee01"], container="wooden_bowl", logical="all", require_gripper_detached=True)
        & object_outside_of(env, object="orange_01", container="wooden_bowl")
    )


@configclass
class XrSmallerThanTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrSmallerThanTask(Task):
    contact_object_list = ['orange_01', 'lemon_02', 'lime01', 'lychee01', 'wooden_bowl', 'table']
    scene = import_scene("xr_size_smaller_than.usda", contact_object_list)
    terminations = XrSmallerThanTaskTerminations
    instruction = {'default': 'Move all fruits that are smaller than the orange into the bowl', 'vague': 'Put the smaller fruits in the bowl', 'specific': 'The orange is the largest fruit. Pick up the lemon, lime, and lychee and place them into the wooden bowl. Leave the orange on the table.'}
    episode_length_s: int = 90
    attributes = ['complex_reference', 'semantics', 'size_reasoning']
    subtasks = [
        pick_and_place(object=["lemon_02", "lime01", "lychee01"], container="wooden_bowl", logical="all", score=1.0),
    ]
