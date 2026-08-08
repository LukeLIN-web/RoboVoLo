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
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    pick_and_place_on_surface,
    stacked,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class ReverseStackGYTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={"objects": ["yellow_block", "green_block"], "order": "bottom_to_top"},
    )


@dataclass
class ReverseStackGYTask(Task):
    contact_object_list = ['green_block', 'yellow_block', 'table']
    scene = import_scene("lh_stack_gy.usda", contact_object_list)
    terminations = ReverseStackGYTaskTerminations
    instruction = {
        "default": "Reverse the two-block stack",
        "vague": "Flip the stack upside down",
        "specific": "Unstack and rebuild in opposite order",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'ordering', 'stacking']
    subtasks = [
        pick_and_place_on_surface(
            object=["yellow_block"], surface="table",
            logical="all", score=0.5,
        ),
        Subtask(
            name="reverse",
            conditions=partial(stacked, objects=["yellow_block", "green_block"], order="bottom_to_top"),
            score=0.5,
        ),
    ]
