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
class SortCannedVeggiesAndFruitsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["corn_can", "green_beans_can", "peas_and_carrots"], "container": "bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["canned_peaches", "pineapple_slices_can", "pitted_cherries"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class SortCannedVeggiesAndFruitsTask(Task):
    contact_object_list = ["bowl", "serving_bowl", "corn_can", "green_beans_can", "canned_peaches", "pineapple_slices_can", "peas_and_carrots", "pitted_cherries", "table"]
    scene = import_scene("lh_sort_canned_veggies_fruits.usda", contact_object_list)
    terminations = SortCannedVeggiesAndFruitsTerminations
    instruction = {
        "default": "Sort the cans on the table into the correct bowls.",
        "vague": "Put each can where it goes.",
        "specific": "One bowl has canned vegetables, the other has canned fruits. Put the peas and carrots with the vegetables and the cherries with the fruits.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "sorting", "common_sense", "semantics"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["peas_and_carrots"], "container": "bowl"},
                {"object": ["pitted_cherries"], "container": "serving_bowl"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_vegetable_cans_in_bowl",
            "predicate": "object_in_container",
            "object": ["corn_can", "green_beans_can", "peas_and_carrots"],
            "container": "bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_fruit_cans_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["canned_peaches", "pineapple_slices_can", "pitted_cherries"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "vegetable_can_exemplars_remain_in_bowl",
            "predicate": "object_in_container",
            "object": ["corn_can", "green_beans_can"],
            "container": "bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "fruit_can_exemplars_remain_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["canned_peaches", "pineapple_slices_can"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
