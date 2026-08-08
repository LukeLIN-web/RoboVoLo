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
    object_in_container,
    object_outside_of_and_on_surface,
    pick_and_place,
)
from robolab.core.task.decorators import atomic
from robolab.core.scenes.utils import import_scene


@atomic
def _lightest_success(env, env_id=None):
    """Hydrogen (lightest element) must be in the bin. Others must stay out."""
    correct = object_in_container(
        env,
        object=["element_h"],
        container="bowl",
        logical="all",
        require_gripper_detached=True,
    )
    wrong_out = object_outside_of_and_on_surface(
        env,
        object=["element_he", "element_li", "element_be"],
        container="bowl",
        surface="table",
        logical="all",
        require_gripper_detached=True,
    )
    return correct & wrong_out


@configclass
class ChemLightestTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_lightest_success, params={})


@dataclass
class ChemLightestTask(Task):
    contact_object_list = [
        "table",
        "bowl",
        "element_h",
        "element_he",
        "element_li",
        "element_be",
    ]
    scene = import_scene("wk_chem_lightest.usda", contact_object_list)
    terminations = ChemLightestTerminations
    instruction = {
        "default": "Put the lightest element into the bowl",
        "vague": "Find the lightest one",
        "specific": "Among Hydrogen (H), Helium (He), Lithium (Li), and Beryllium (Be), Hydrogen is the lightest element on the periodic table. Place it in the bowl.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "chemistry", "semantics"]
    subtasks = [
        pick_and_place(
            object=["element_h"],
            container="bowl",
            logical="all",
            score=1.0,
        ),
    ]
