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
class SortFoodVsNonFoodTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["lime01", "lemon_02", "lychee01"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["blue_block", "rubiks_cube", "crabbypenholder"], "container": "bin_a02", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class SortFoodVsNonFoodTask(Task):
    contact_object_list = ["serving_bowl", "bin_a02", "lemon_02", "lychee01", "rubiks_cube", "crabbypenholder", "lime01", "blue_block", "table"]
    scene = import_scene("lh_sort_food_nonfood.usda", contact_object_list)
    terminations = SortFoodVsNonFoodTerminations
    instruction = {
        "default": "Sort the items on the table — each container shows you what type goes there.",
        "vague": "Put things where they belong.",
        "specific": "The bowl has food items and the bin has non-food items. Put the lime in the bowl and the blue block in the bin.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "sorting", "common_sense"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["lime01"], "container": "serving_bowl"},
                {"object": ["blue_block"], "container": "bin_a02"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_food_items_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["lime01", "lemon_02", "lychee01"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_nonfood_items_in_bin",
            "predicate": "object_in_container",
            "object": ["blue_block", "rubiks_cube", "crabbypenholder"],
            "container": "bin_a02",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "food_exemplars_remain_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["lemon_02", "lychee01"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "nonfood_exemplars_remain_in_bin",
            "predicate": "object_in_container",
            "object": ["rubiks_cube", "crabbypenholder"],
            "container": "bin_a02",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
