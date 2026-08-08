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
class SwapBinReplaceBlockCanTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "corn_can", "container": "bin_a04", "require_gripper_detached": True},
    )


@dataclass
class SwapBinReplaceBlockCanTask(Task):
    contact_object_list = ['bin_a04', 'green_block', 'corn_can', 'table']
    scene = import_scene("lh_bin_block_can.usda", contact_object_list)
    terminations = SwapBinReplaceBlockCanTaskTerminations
    instruction = {
        "default": "Take the item out of the bin and place it on the table. Then put the item that was on the table into the bin.",
        "vague": "Swap the bin contents with the table item",
        "specific": "Remove the block from the bin, then put the can into the bin",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'swapping', 'minimal_memory']
    subtasks = [
        pick_and_place_on_surface(
            object=["green_block"], surface="table",
            logical="all", score=0.5,
        ),
        pick_and_place(
            object=["corn_can"], container="bin_a04",
            logical="all", score=0.5,
        ),
    ]
