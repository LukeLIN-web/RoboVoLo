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
    object_left_of,
    object_right_of,
    pick_and_place_on_surface,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class SpatialByStackOrderTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_left_of,
        params={"object": "spam_can", "reference_object": "jello", "require_gripper_detached": True},
    )


@dataclass
class SpatialByStackOrderTask(Task):
    contact_object_list = ['spam_can', 'jello', 'table']
    scene = import_scene("lh_stack_spam_orange.usda", contact_object_list)
    terminations = SpatialByStackOrderTerminations
    instruction = {
        "default": "The two items are stacked. Take them apart. Place the one that was on the bottom to the left of the one that was on top.",
        "vague": "Unstack and arrange by original position",
        "specific": "Remove the top item, then place the bottom item to the left of where you put the top one",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'ordering', 'spatial']
    subtasks = [
        # 1. Unstack jello (top) onto the table — physically required.
        pick_and_place_on_surface(
            object=["jello"], surface="table",
            logical="all", score=0.5,
        ),
        # 2. Final spatial arrangement (both objects exposed as targets
        #    so picking either one for the rearrangement is valid).
        Subtask(
            name="arrange",
            conditions={
                "spam_can": partial(
                    object_left_of, object="spam_can",
                    reference_object="jello",
                    require_gripper_detached=True,
                ),
                "jello": partial(
                    object_right_of, object="jello",
                    reference_object="spam_can",
                    require_gripper_detached=True,
                ),
            },
            logical="all",
            score=0.5,
        ),
    ]
