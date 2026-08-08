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
class KitUtensilSetsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["spoon_small", "spoon_big"], "container": "serving_bowl", "require_gripper_detached": True, "require_contact_with": False},
                {"object": ["fork_big", "fork_small"], "container": "bowl", "require_gripper_detached": True, "require_contact_with": False},
            ]
        },
    )


@dataclass
class KitUtensilSetsTask(Task):
    contact_object_list = ["bowl", "serving_bowl", "fork_big", "fork_small", "spoon_big", "spoon_small", "table"]
    scene = import_scene("lh_kit_utensil_sets.usda", contact_object_list)
    terminations = KitUtensilSetsTerminations
    instruction = {
        "default": "Each bowl groups one type of utensil. Add the missing one to complete the set.",
        "vague": "Finish the collection.",
        "specific": "One bowl has two forks, the other has one spoon. Add the small spoon to the spoon bowl.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "kitting", "common_sense", "pattern_completion"]
    subtasks = [
        pick_and_place(object="spoon_small", container="serving_bowl", score=1.0),
    ]
