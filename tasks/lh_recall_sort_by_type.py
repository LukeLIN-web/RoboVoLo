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
    object_grabbed,
    object_groups_in_containers,
    object_in_container,
    object_in_front_of,
    object_left_of,
    object_on_top,
    object_outside_of,
    pick_and_place,
    stacked,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class RecallRestoreContainersTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["tuna_can", "corn_can", "lemon_02"], "container": "bin_a03", "logical": "all", "require_gripper_detached": True},
            ]
        },
    )


@dataclass
class RecallRestoreContainersTask(Task):
    contact_object_list = ['serving_bowl', 'bin_a03', 'tuna_can', 'lemon_02', 'corn_can', 'table']
    scene = import_scene("lh_bowls_bin_sort.usda", contact_object_list)
    terminations = RecallRestoreContainersTerminations
    instruction = {
        "default": "Move all items from the container that has less items to the container that has more items.",
        "vague": "Transfer everything from the emptier container into the fuller one.",
        "specific": "The bowl has one item and the bin has two. Move the tuna can from the bowl into the bin.",
    }
    episode_length_s: int = 240
    attributes = ['memory', 'recall', 'HL', 'sorting', 'categorization']
    subtasks = [
        pick_and_place(object=["tuna_can"], container="bin_a03", logical="all", score=1.0),
    ]
