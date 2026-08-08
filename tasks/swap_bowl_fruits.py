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
class SwapBowlFruitsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {
                    "object": ["lemon_02"],
                    "container": "serving_bowl",
                    "logical": "all",
                    "require_gripper_detached": True,
                },
                {
                    "object": ["lychee01"],
                    "container": "bowl",
                    "logical": "all",
                    "require_gripper_detached": True,
                },
            ]
        },
    )


@dataclass
class SwapBowlFruitsTask(Task):
    contact_object_list = ["bowl", "serving_bowl", "lemon_02", "lychee01", "table"]
    scene = import_scene("lh_swap_bowl_fruits.usda", contact_object_list)
    terminations = SwapBowlFruitsTerminations
    instruction = {
        "default": "Swap the fruits between the two bowls",
        "vague": "Switch what's in each bowl",
        "specific": "Move the lemon from the red bowl to the white serving bowl, and the lychee from the serving bowl to the red bowl, using the table as a temporary buffer",
    }
    episode_length_s: int = 120
    attributes = ["memory", "swapping"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["lemon_02"], "container": "serving_bowl"},
                {"object": ["lychee01"], "container": "bowl"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
