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
    object_groups_in_containers,
    object_in_container,
    object_on_top,
    object_outside_of,
    pick_and_place_grouped,
)
from robolab.core.task.task import Task


@configclass
class SortForksAndSpoonsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["fork_small", "fork_big"], "container": "bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["spoon_small", "spoon_big"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class SortForksAndSpoonsTask(Task):
    contact_object_list = ["bowl", "serving_bowl", "fork_big", "spoon_big", "fork_small", "spoon_small", "table"]
    scene = import_scene("lh_sort_forks_spoons.usda", contact_object_list)
    terminations = SortForksAndSpoonsTerminations
    instruction = {
        "default": "Put each utensil in the bowl with matching ones.",
        "vague": "Match them up.",
        "specific": "The left bowl has forks and the right has spoons. Put the small fork with the forks and the small spoon with the spoons.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "sorting", "common_sense"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["fork_small"], "container": "bowl"},
                {"object": ["spoon_small"], "container": "serving_bowl"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_forks_in_bowl",
            "predicate": "object_in_container",
            "object": ["fork_small", "fork_big"],
            "container": "bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_spoons_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["spoon_small", "spoon_big"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "fork_exemplar_remains_in_bowl",
            "predicate": "object_in_container",
            "object": "fork_big",
            "container": "bowl",
            "require_gripper_detached": True,
        },
        {
            "name": "spoon_exemplar_remains_in_serving_bowl",
            "predicate": "object_in_container",
            "object": "spoon_big",
            "container": "serving_bowl",
            "require_gripper_detached": True,
        },
    ]
