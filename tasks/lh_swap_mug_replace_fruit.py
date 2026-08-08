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
    pick_and_place,
    pick_and_place_on_surface,
)
from robolab.core.task.task import Task


@configclass
class SwapMugReplaceFruitTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["lychee01"], "container": "ceramic_mug", "logical": "all", "require_gripper_detached": True},
            ]
        },
    )


@dataclass
class SwapMugReplaceFruitTask(Task):
    contact_object_list = ['ceramic_mug', 'orange_01', 'lychee01', 'table']
    scene = import_scene("lh_mug_fruits_replace.usda", contact_object_list)
    terminations = SwapMugReplaceFruitTerminations
    instruction = {
        "default": "Take the item out of the mug and place it on the table. Then put the item that was on the table into the mug.",
        "vague": "Swap what's in the mug with what's outside",
        "specific": "Remove the fruit from the mug to the table, then pick up the other fruit and place it in the mug",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'swapping', 'minimal_memory']
    subtasks = [
        pick_and_place_on_surface(
            object=["orange_01"], surface="table",
            logical="all", score=0.5,
        ),
        pick_and_place(
            object=["lychee01"], container="ceramic_mug",
            logical="all", score=0.5,
        ),
    ]
