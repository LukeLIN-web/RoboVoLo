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
    object_on_top,
    pick_and_place_on_surface,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class RestackTopOnLooseTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={"object": "corn_can", "reference_object": "tuna_can", "require_gripper_detached": True},
    )


@dataclass
class RestackTopOnLooseTask(Task):
    contact_object_list = ['spam_can', 'corn_can', 'tuna_can', 'table']
    scene = import_scene("lh_partial_stack_3cans.usda", contact_object_list)
    terminations = RestackTopOnLooseTerminations
    instruction = {
        "default": "Two of the three cans are stacked. Take the stack apart. Then stack the one that was on top of the old stack onto the one that was NOT stacked.",
        "vague": "Rearrange the cans based on their starting roles",
        "specific": "Unstack the top can and place it on top of the can that was sitting alone on the table",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'ordering', 'stacking']
    subtasks = [
        # 1. Unstack corn (top of old stack) onto the table.
        pick_and_place_on_surface(
            object=["corn_can"], surface="table",
            logical="all", score=0.5,
        ),
        # 2. Stack corn on tuna (= termination).
        Subtask(
            name="restack",
            conditions=partial(
                object_on_top, object="corn_can",
                reference_object="tuna_can",
                require_gripper_detached=True,
            ),
            score=0.5,
        ),
    ]
