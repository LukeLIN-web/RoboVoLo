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
class RecoverSpillTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["banana", "lemon_02", "orange_01", "lime01"], "container": "wooden_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["tuna_can", "corn_can", "spam_can"], "container": "bin_a04", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class RecoverSpillTask(Task):
    contact_object_list = ["wooden_bowl", "banana", "lemon_02", "orange_01", "lime01", "bin_a04", "tuna_can", "corn_can", "spam_can", "table"]
    scene = import_scene("lh_recover_spill.usda", contact_object_list)
    terminations = RecoverSpillTerminations
    instruction = {
        "default": "Some items got disorganized. Clean it up.",
        "vague": "Tidy up.",
        "specific": "Put all the fruits in the bowl and all the cans in the bin.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "recovery", "common_sense", "spatial_inference"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["banana", "lemon_02", "orange_01", "lime01"], "container": "wooden_bowl"},
                {"object": ["tuna_can", "corn_can", "spam_can"], "container": "bin_a04"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
    gt_success_checks = [
        {
            "name": "all_fruits_in_bowl",
            "predicate": "object_in_container",
            "object": ["banana", "lemon_02", "orange_01", "lime01"],
            "container": "wooden_bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
        {
            "name": "all_cans_in_bin",
            "predicate": "object_in_container",
            "object": ["tuna_can", "corn_can", "spam_can"],
            "container": "bin_a04",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = []
