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
class ChemSortNobleMetalTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {
                    "object": ["element_he", "element_ar"],
                    "container": "bin_left",
                    "logical": "all",
                    "require_gripper_detached": True,
                },
                {
                    "object": ["element_fe", "element_cu"],
                    "container": "bin_right",
                    "logical": "all",
                    "require_gripper_detached": True,
                },
            ],
        },
    )


@dataclass
class ChemSortNobleMetalTask(Task):
    contact_object_list = [
        "table",
        "bin_left",
        "bin_right",
        "element_he",
        "element_ar",
        "element_fe",
        "element_cu",
    ]
    scene = import_scene("wk_chem_sort_noble_metal.usda", contact_object_list)
    terminations = ChemSortNobleMetalTerminations
    instruction = {
        "default": "Sort the elements: put noble gases in the left bin and transition metals in the right bin",
        "vague": "Classify the elements into the bins",
        "specific": "Helium (He) and Argon (Ar) are noble gases — put them in the left bin. Iron (Fe) and Copper (Cu) are transition metals — put them in the right bin.",
    }
    episode_length_s: int = 120
    attributes = ["world_knowledge", "chemistry", "sorting", "multi_step"]
    subtasks = [
        pick_and_place(
            object=["element_he", "element_ar"],
            container="bin_left",
            logical="all",
            score=0.5,
        ),
        pick_and_place(
            object=["element_fe", "element_cu"],
            container="bin_right",
            logical="all",
            score=0.5,
        ),
    ]
