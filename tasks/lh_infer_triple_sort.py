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
    pick_and_place_grouped,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class InferTripleSortTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["lime01"], "container": "bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["crabbypenholder"], "container": "bin_a01", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["corn_can"], "container": "bin_a03", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["lemon_02"], "container": "bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["rubiks_cube"], "container": "bin_a01", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["tuna_can"], "container": "bin_a03", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class InferTripleSortTask(Task):
    contact_object_list = ["bowl", "bin_a01", "bin_a03", "lemon_02", "rubiks_cube", "tuna_can", "lime01", "crabbypenholder", "corn_can", "table"]
    scene = import_scene("lh_infer_triple_sort.usda", contact_object_list)
    terminations = InferTripleSortTerminations
    instruction = {
        "default": "Three containers, three items. Each item has a home — figure out where.",
        "vague": "Sort them all.",
        "specific": "The bowl has a fruit, the small bin has a toy, and the larger bin has a can. Put the lime in the bowl, the crab toy in the small bin, and the corn can in the larger bin.",
    }
    episode_length_s: int = 150
    attributes = ["reasoning", "inference", "common_sense", "multi_container"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["lime01", "lemon_02"], "container": "bowl"},
                {"object": ["crabbypenholder", "rubiks_cube"], "container": "bin_a01"},
                {"object": ["corn_can", "tuna_can"], "container": "bin_a03"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
