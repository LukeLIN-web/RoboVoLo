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
class InferClearTableTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["orange_01"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["rubiks_cube"], "container": "bin_a01", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["lemon_02"], "container": "serving_bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
            {"object": ["crabbypenholder"], "container": "bin_a01", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class InferClearTableTask(Task):
    contact_object_list = ["serving_bowl", "bin_a01", "lemon_02", "crabbypenholder", "orange_01", "rubiks_cube", "table"]
    scene = import_scene("lh_infer_clear_table.usda", contact_object_list)
    terminations = InferClearTableTerminations
    instruction = {
        "default": "Clear the table — put each item in the container with similar things.",
        "vague": "Put things away.",
        "specific": "The bowl has a lemon (food) and the bin has a toy. Put the orange in the bowl and the Rubik's cube in the bin.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "inference", "common_sense", "exemplar_matching"]
    subtasks = [
        pick_and_place_grouped(
            groups=[
                {"object": ["orange_01"], "container": "serving_bowl"},
                {"object": ["rubiks_cube"], "container": "bin_a01"},
            ],
            logical="all",
            score=1.0,
        ),
    ]
