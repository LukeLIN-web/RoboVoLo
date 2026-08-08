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
    object_groups_in_containers,
    object_in_container,
    object_on_top,
    object_outside_of,
    pick_and_place,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class KitBalanceBinsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["orange_01", "lime01"], "container": "bin_a04", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["spam_can", "corn_can", "tuna_can"], "container": "bin_a03", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["lemon_02"], "container": "bin_a04", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class KitBalanceBinsTask(Task):
    contact_object_list = ["bin_a03", "bin_a04", "spam_can", "corn_can", "tuna_can", "lemon_02", "orange_01", "lime01", "table"]
    scene = import_scene("lh_kit_balance_bins.usda", contact_object_list)
    terminations = KitBalanceBinsTerminations
    instruction = {
        "default": "Distribute the table items so both bins end up with the same number of items.",
        "vague": "Balance them out.",
        "specific": "One bin has three items and the other has one. Put both table items in the bin that only has one, making it three each.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "kitting", "common_sense", "counting"]
    subtasks = [
        pick_and_place(object=["orange_01", "lime01"], container="bin_a04", logical="all", score=1.0),
    ]
