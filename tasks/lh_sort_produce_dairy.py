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
class SortProduceAndDairyTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["lime01", "orange_01", "lemon_02"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["yogurt_cup", "butter", "cream_cheese"], "container": "bin_a03", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class SortProduceAndDairyTask(Task):
    contact_object_list = ["serving_bowl", "bin_a03", "orange_01", "lemon_02", "butter", "cream_cheese", "lime01", "yogurt_cup", "table"]
    scene = import_scene("lh_sort_produce_dairy.usda", contact_object_list)
    terminations = SortProduceAndDairyTerminations
    instruction = {
        "default": "Each container holds a category. Put the table items where they belong.",
        "vague": "Sort them by type.",
        "specific": "The bowl has fresh produce and the bin has dairy items. Put the lime with the produce and the yogurt with the dairy.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "sorting", "common_sense"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["lime01"], "container": "serving_bowl"},
                {"object": ["yogurt_cup"], "container": "bin_a03"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_produce_items_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["lime01", "orange_01", "lemon_02"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_dairy_items_in_bin",
            "predicate": "object_in_container",
            "object": ["yogurt_cup", "butter", "cream_cheese"],
            "container": "bin_a03",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "produce_exemplars_remain_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["orange_01", "lemon_02"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "dairy_exemplars_remain_in_bin",
            "predicate": "object_in_container",
            "object": ["butter", "cream_cheese"],
            "container": "bin_a03",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
