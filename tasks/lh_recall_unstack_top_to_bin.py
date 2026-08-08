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
    pick_and_place_on_surface,
)
from robolab.core.task.task import Task


@configclass
class RecallUnstackTopToBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "blue_block", "container": "bin_a01", "require_gripper_detached": True},
    )


@dataclass
class RecallUnstackTopToBinTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'bin_a01', 'table']
    scene = import_scene("lh_tower_rbg_bin.usda", contact_object_list)
    terminations = RecallUnstackTopToBinTerminations
    instruction = {
        "default": "Unstack all three blocks onto the table. Then put the block that was originally in the middle into the bin.",
        "vague": "Take apart the tower and bin the middle one",
        "specific": "Remove all three blocks to the table, then pick up the one that was in the middle of the stack and place it in the bin",
    }
    episode_length_s: int = 90
    attributes = ['memory', 'recall', 'stacking']
    # Two ordered phases: unstack top (green) before the middle block
    # (blue) is graspable, then place blue into the bin.
    subtasks = [
        pick_and_place_on_surface(
            object=["green_block"], surface="table",
            logical="all", score=0.5,
        ),
        pick_and_place(
            object=["blue_block"], container="bin_a01",
            logical="all", score=0.5,
        ),
    ]
