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
    pick_and_place_grouped,
)
from robolab.core.task.task import Task


@configclass
class SortFreshVsCannedTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["lime01"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["spam_can"], "container": "bin_a04", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["orange_01", "lemon_02"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["tuna_can", "corn_can"], "container": "bin_a04", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class SortFreshVsCannedTask(Task):
    contact_object_list = ["serving_bowl", "bin_a04", "orange_01", "lemon_02", "tuna_can", "corn_can", "lime01", "spam_can", "table"]
    scene = import_scene("lh_sort_fresh_vs_canned.usda", contact_object_list)
    terminations = SortFreshVsCannedTerminations
    instruction = {
        "default": "Sort the remaining items into the correct containers.",
        "vague": "Put things where they belong.",
        "specific": "The bowl has fresh produce and the bin has canned food. Put the lime in the bowl and the spam can in the bin.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "sorting", "common_sense"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["lime01"], "container": "serving_bowl"},
                {"object": ["spam_can"], "container": "bin_a04"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_fresh_items_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["lime01", "orange_01", "lemon_02"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_canned_items_in_bin",
            "predicate": "object_in_container",
            "object": ["spam_can", "tuna_can", "corn_can"],
            "container": "bin_a04",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "fresh_exemplars_remain_in_serving_bowl",
            "predicate": "object_in_container",
            "object": ["orange_01", "lemon_02"],
            "container": "serving_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "canned_exemplars_remain_in_bin",
            "predicate": "object_in_container",
            "object": ["tuna_can", "corn_can"],
            "container": "bin_a04",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
