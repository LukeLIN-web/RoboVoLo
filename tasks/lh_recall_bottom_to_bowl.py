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
    pick_and_place,
)
from robolab.core.task.task import Task


@configclass
class RecallBottomToBowlTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "red_block", "container": "bowl", "require_gripper_detached": True},
    )


@dataclass
class RecallBottomToBowlTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'bowl', 'table']
    scene = import_scene("lh_stack_rb_bowl_bin.usda", contact_object_list)
    terminations = RecallBottomToBowlTerminations
    instruction = {
        "default": "The blocks are stacked. Take them apart. Then put only the block that was on the bottom into the bowl \u2014 leave the other on the table.",
        "vague": "Unstack and bowl the bottom one",
        "specific": "Remove the top block to the table, then pick up the bottom block and place it in the bowl",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'recall', 'stacking']
    subtasks = [
        pick_and_place(
            object=["red_block"], container="bowl",
            logical="all", score=1.0,
        ),
    ]
