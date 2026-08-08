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

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import (
    object_groups_in_containers,
    pick_and_place,
)
from robolab.core.scenes.utils import import_scene


@configclass
class RecycleSort4Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {
                    "object": ["banana", "eggplant"],
                    "container": "compost_bin",
                    "logical": "all",
                    "require_gripper_detached": True,
                },
                {
                    "object": ["soda_can"],
                    "container": "recycle_bin",
                    "logical": "all",
                    "require_gripper_detached": True,
                },
                {
                    "object": ["coffee_cup"],
                    "container": "trash_bin",
                    "logical": "all",
                    "require_gripper_detached": True,
                },
            ],
        },
    )


@dataclass
class RecycleSort4Task(Task):
    contact_object_list = [
        "table",
        "compost_bin",
        "trash_bin",
        "recycle_bin",
        "banana",
        "eggplant",
        "soda_can",
        "coffee_cup",
    ]
    scene = import_scene("wk_recycle_sort_4.usda", contact_object_list)
    terminations = RecycleSort4Terminations
    instruction = {
        "default": "Sort all four items into the correct bins",
        "vague": "Clean up everything",
        "specific": "The banana and eggplant are food waste — compost them. The soda can is aluminum — recycle it. The coffee cup is non-recyclable — trash it.",
    }
    episode_length_s: int = 120
    attributes = ["world_knowledge", "recycling", "sorting", "multi_step"]
    subtasks = [
        pick_and_place(
            object=["banana", "eggplant"],
            container="compost_bin",
            logical="all",
            score=0.5,
        ),
        pick_and_place(
            object=["coffee_cup"],
            container="trash_bin",
            logical="all",
            score=0.25,
        ),
        pick_and_place(
            object=["soda_can"],
            container="recycle_bin",
            logical="all",
            score=0.25,
        ),
    ]
