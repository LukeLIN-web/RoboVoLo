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
class SortFoodAndUtensilsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["lychee01", "lemon_02", "lime01"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["fork_small", "fork_big", "spoon_big"], "container": "bin_a04", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class SortFoodAndUtensilsTask(Task):
    contact_object_list = ["serving_bowl", "bin_a04", "lemon_02", "lime01", "fork_big", "spoon_big", "lychee01", "fork_small", "table"]
    scene = import_scene("lh_sort_food_utensils.usda", contact_object_list)
    terminations = SortFoodAndUtensilsTerminations
    instruction = {
        "default": "Put each table item where similar ones already are.",
        "vague": "Put things where they belong.",
        "specific": "The bowl has fruits and the bin has utensils. Put the lychee in the bowl and the fork in the bin.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "sorting", "common_sense"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["lychee01"], "container": "serving_bowl"},
                {"object": ["fork_small"], "container": "bin_a04"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_food_items_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["lychee01", "lemon_02", "lime01"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_utensils_in_bin",
            "predicate": "object_in_container",
            "object": ["fork_small", "fork_big", "spoon_big"],
            "container": "bin_a04",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "food_exemplars_remain_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["lemon_02", "lime01"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "utensil_exemplars_remain_in_bin",
            "predicate": "object_in_container",
            "object": ["fork_big", "spoon_big"],
            "container": "bin_a04",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
