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
class KitFruitUtensilTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["fork_small", "lime01"], "container": "bowl", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["lemon_02", "fork_big"], "container": "serving_bowl", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class KitFruitUtensilTask(Task):
    contact_object_list = ["serving_bowl", "bowl", "lemon_02", "fork_big", "lime01", "fork_small", "table"]
    scene = import_scene("lh_kit_fruit_utensil.usda", contact_object_list)
    terminations = KitFruitUtensilTerminations
    instruction = {
        "default": "Both bowls should have a similar setup. Complete the incomplete one.",
        "vague": "Make them match.",
        "specific": "One bowl has a fruit and a fork. The other only has a fruit. Add the small fork to complete it.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "kitting", "common_sense", "pattern_completion"]
    subtasks = [
        pick_and_place(object="fork_small", container="bowl", score=1.0),
    ]
