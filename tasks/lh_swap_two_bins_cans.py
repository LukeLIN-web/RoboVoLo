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
class SwapTwoBinsCansTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["tuna_can"], "container": "container_b04", "logical": "all", "require_gripper_detached": True},
                {"object": ["corn_can"], "container": "container_b03", "logical": "all", "require_gripper_detached": True},
            ]
        },
    )


@dataclass
class SwapTwoBinsCansTask(Task):
    contact_object_list = ['container_b03', 'container_b04', 'tuna_can', 'corn_can', 'table']
    scene = import_scene("lh_two_bins_cans.usda", contact_object_list)
    terminations = SwapTwoBinsCansTaskTerminations
    instruction = {
        "default": "Swap the tuna can and the corn can between the two containers",
        "vague": "Exchange the cans between containers",
        "specific": "Remove one can to the table, move the other can over, then place the first can in the now-empty container",
    }
    episode_length_s: int = 90
    attributes = ['memory', 'swapping']
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["tuna_can"], "container": "container_b04"},
                {"object": ["corn_can"], "container": "container_b03"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
