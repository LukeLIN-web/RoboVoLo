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
class RecycleSort2Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {
                    "object": ["banana"],
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
            ],
        },
    )


@dataclass
class RecycleSort2Task(Task):
    contact_object_list = [
        "table",
        "compost_bin",
        "trash_bin",
        "recycle_bin",
        "banana",
        "soda_can",
    ]
    scene = import_scene("wk_recycle_sort_2.usda", contact_object_list)
    terminations = RecycleSort2Terminations
    instruction = {
        "default": "Sort both items into the correct bins",
        "vague": "Clean up the table",
        "specific": "The banana is food waste — compost it. The soda can is aluminum — recycle it. Put each in the correct bin.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "recycling", "sorting", "multi_step"]
    subtasks = [
        pick_and_place(
            object=["banana"],
            container="compost_bin",
            logical="all",
            score=0.5,
        ),
        pick_and_place(
            object=["soda_can"],
            container="recycle_bin",
            logical="all",
            score=0.5,
        ),
    ]
