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
    object_in_contact,
    object_in_container,
    object_dropped,
    object_outside_of_and_on_surface,
    pick_and_place,
)
from robolab.core.task.predicate_logic import _and, _not
from robolab.core.task.decorators import atomic
from robolab.core.scenes.utils import import_scene


@atomic
def _salt_success(env, env_id=None):
    """Chlorine must be in (or stacked atop Na in) the bowl to complete NaCl.
    Na must remain in bowl. Gold and Sulfur must stay out.
    Handles the case where Cl is stacked on Na so its centroid is above the bowl rim."""
    na_in_bowl = object_in_container(
        env,
        object=["element_na"],
        container="bowl_002",
        logical="all",
        require_gripper_detached=True,
    )
    cl_in_bowl = object_in_container(
        env,
        object=["element_cl"],
        container="bowl_002",
        logical="all",
        require_gripper_detached=True,
    )
    cl_touching_na = object_in_contact(
        env,
        object1="element_cl",
        object2="element_na",
    )
    cl_detached = object_dropped(env, object="element_cl")
    # Cl is valid if directly in bowl, OR (touching Na which is in bowl and not held)
    cl_stacked = _and(_and(cl_touching_na, na_in_bowl), cl_detached)
    cl_valid = _not(_and(_not(cl_in_bowl), _not(cl_stacked)))
    wrong_out = object_outside_of_and_on_surface(
        env,
        object=["element_au", "element_s"],
        container="bowl_002",
        surface="table",
        logical="all",
        require_gripper_detached=True,
    )
    return _and(_and(cl_valid, na_in_bowl), wrong_out)


@configclass
class ChemSaltTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_salt_success, params={})


@dataclass
class ChemSaltTask(Task):
    contact_object_list = [
        "table",
        "bowl_002",
        "element_na",
        "element_cl",
        "element_au",
        "element_s",
    ]
    scene = import_scene("wk_chem_salt.usda", contact_object_list)
    terminations = ChemSaltTerminations
    instruction = {
        "default": "Complete table salt by adding the missing element to the bowl",
        "vague": "Finish making salt",
        "specific": "The bowl already contains Sodium. Table salt is made of Sodium and Chlorine, so pick the Chlorine cube and place it in the bowl. Leave Gold and Sulfur on the table.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "chemistry", "semantics"]
    subtasks = [
        pick_and_place(
            object=["element_cl"],
            container="bowl_002",
            logical="all",
            score=1.0,
        ),
    ]
