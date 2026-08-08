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
    pick_and_place,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class RecoverMisplacedTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["tuna_can", "corn_can", "spam_can", "green_beans_can"], "container": "container_b03", "require_contact_with": False},
                {"object": ["orange_01", "lemon_02"], "container": "serving_bowl", "require_contact_with": False},
            ],
        },
    )


@dataclass
class RecoverMisplacedTask(Task):
    contact_object_list = ["serving_bowl", "container_b03", "orange_01", "lemon_02", "tuna_can", "corn_can", "spam_can", "green_beans_can", "table"]
    scene = import_scene("lh_recover_misplaced.usda", contact_object_list)
    terminations = RecoverMisplacedTerminations
    instruction = {
        "default": "One item is in the wrong container. Move it to where it belongs.",
        "vague": "Fix the mistake.",
        "specific": "The tuna can is in the fruit bowl but belongs with the other cans in the bin. Move it.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "recovery", "common_sense", "categorization"]
    subtasks = [
        pick_and_place(object="tuna_can", container="container_b03", score=1.0),
    ]
    gt_success_checks = [
        {
            "name": "tuna_can_moved_to_can_bin",
            "predicate": "object_in_container",
            "object": "tuna_can",
            "container": "container_b03",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "other_cans_remain_in_can_bin",
            "predicate": "object_in_container",
            "object": ["corn_can", "spam_can", "green_beans_can"],
            "container": "container_b03",
            "logical": "all",
        },
        {
            "name": "fruit_remains_in_bowl",
            "predicate": "object_in_container",
            "object": ["orange_01", "lemon_02"],
            "container": "serving_bowl",
            "logical": "all",
        },
    ]
