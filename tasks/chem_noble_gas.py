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
def _noble_gas_success(env, env_id=None):
    """Helium (the noble gas) must be in the bin. Metals must stay out."""
    correct = object_in_container(
        env,
        object=["element_he"],
        container="bin",
        logical="all",
        require_gripper_detached=True,
    )
    wrong_out = object_outside_of_and_on_surface(
        env,
        object=["element_fe", "element_cu", "element_na"],
        container="bin",
        surface="table",
        logical="all",
        require_gripper_detached=True,
    )
    return correct & wrong_out


@configclass
class ChemNobleGasTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_noble_gas_success, params={})


@dataclass
class ChemNobleGasTask(Task):
    contact_object_list = [
        "table",
        "bin",
        "element_he",
        "element_fe",
        "element_cu",
        "element_na",
    ]
    scene = import_scene("wk_chem_noble_gas.usda", contact_object_list)
    terminations = ChemNobleGasTerminations
    instruction = {
        "default": "Put the non-reactive noble gas into the bin",
        "vague": "Find the inert element",
        "specific": "Among the element cubes on the table, identify Helium (He) — the noble gas that does not react — and place it into the bin. Leave Iron, Copper, and Sodium on the table.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "chemistry", "semantics"]
    subtasks = [
        pick_and_place(
            object=["element_he"],
            container="bin",
            logical="all",
            score=1.0,
        ),
    ]
