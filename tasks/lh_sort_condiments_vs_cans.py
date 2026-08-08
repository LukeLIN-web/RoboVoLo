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
class SortCondimentsVsCansTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["bbq_sauce_bottle", "ranch_dressing", "ketchup_bottle", "mustard_bottle"], "container": "bin_a04", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["spam_can", "corn_can", "tuna_can"], "container": "bin_a03", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class SortCondimentsVsCansTask(Task):
    contact_object_list = ["bin_a04", "bin_a03", "ketchup_bottle", "mustard_bottle", "ranch_dressing", "corn_can", "tuna_can", "bbq_sauce_bottle", "spam_can", "table"]
    scene = import_scene("lh_sort_condiments_vs_cans.usda", contact_object_list)
    terminations = SortCondimentsVsCansTerminations
    instruction = {
        "default": "Put each item in the bin with similar ones.",
        "vague": "Sort them into the right bins.",
        "specific": "One bin has sauce bottles, the other has canned food. Put the BBQ sauce and ranch dressing with the bottles, and the spam can with the cans.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "sorting", "common_sense"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["bbq_sauce_bottle", "ranch_dressing"], "container": "bin_a04"},
                {"object": ["spam_can"], "container": "bin_a03"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_condiments_in_condiment_bin",
            "predicate": "object_in_container",
            "object": ["bbq_sauce_bottle", "ranch_dressing", "ketchup_bottle", "mustard_bottle"],
            "container": "bin_a04",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_cans_in_can_bin",
            "predicate": "object_in_container",
            "object": ["spam_can", "corn_can", "tuna_can"],
            "container": "bin_a03",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "condiment_exemplars_remain_in_condiment_bin",
            "predicate": "object_in_container",
            "object": ["ketchup_bottle", "mustard_bottle"],
            "container": "bin_a04",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "can_exemplars_remain_in_can_bin",
            "predicate": "object_in_container",
            "object": ["corn_can", "tuna_can"],
            "container": "bin_a03",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
