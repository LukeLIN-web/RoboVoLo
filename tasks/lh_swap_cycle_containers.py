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
class CycleContainersTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["lychee01"], "container": "bowl", "logical": "all", "require_gripper_detached": True},
                {"object": ["lemon_02"], "container": "bin_a01", "logical": "all", "require_gripper_detached": True},
                {"object": ["chocolate_pudding_mix"], "container": "wooden_bowl", "logical": "all", "require_gripper_detached": True},
            ]
        },
    )


@dataclass
class CycleContainersTask(Task):
    contact_object_list = ['wooden_bowl', 'bowl', 'bin_a01', 'lychee01', 'lemon_02', 'chocolate_pudding_mix', 'table']
    scene = import_scene("lh_three_containers_cycle.usda", contact_object_list)
    terminations = CycleContainersTerminations
    instruction = {
        "default": "Cycle the items through three containers: move the wooden bowl's item to the bowl, the bowl's item to the bin, and the bin's item to the wooden bowl",
        "vague": "Rotate items between the three containers",
        "specific": "Buffer the wooden bowl item, move the bowl item to the wooden bowl wait no: move bin item to wooden bowl, bowl item to bin, buffer to bowl",
    }
    episode_length_s: int = 120
    attributes = ['memory', 'swapping', 'cyclic', 'multi_container']
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["lychee01"], "container": "bowl"},
                {"object": ["lemon_02"], "container": "bin_a01"},
                {"object": ["chocolate_pudding_mix"], "container": "wooden_bowl"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
