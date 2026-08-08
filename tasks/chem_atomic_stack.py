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
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import (
    stacked,
)
from robolab.core.task.decorators import atomic
from robolab.core.scenes.utils import import_scene


@atomic
def _atomic_stack_success(env, env_id=None):
    """H (bottom), He (middle), Li (top) stacked in ascending atomic number."""
    return stacked(
        env,
        objects=["element_h", "element_he", "element_li"],
        order="bottom_to_top",
    )


@configclass
class ChemAtomicStackTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_atomic_stack_success, params={})


@dataclass
class ChemAtomicStackTask(Task):
    contact_object_list = [
        "table",
        "element_h",
        "element_he",
        "element_li",
        "element_be",
        "element_b",
    ]
    scene = import_scene("wk_chem_atomic_stack.usda", contact_object_list)
    terminations = ChemAtomicStackTerminations
    instruction = {
        "default": "Stack the first three elements of the periodic table in order of atomic number, lowest on the bottom",
        "vague": "Build a tower with the lightest elements in order",
        "specific": "Hydrogen (H, atomic number 1) goes on the bottom, then Helium (He, 2) in the middle, then Lithium (Li, 3) on top. Ignore Beryllium and Boron.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "chemistry", "stacking", "ordering", "multi_step"]
    subtasks = [
        Subtask(
            name="he_on_h",
            conditions=partial(
                stacked,
                objects=["element_h", "element_he"],
                order="bottom_to_top",
            ),
            score=0.5,
        ),
        Subtask(
            name="li_on_he",
            conditions=partial(
                stacked,
                objects=["element_h", "element_he", "element_li"],
                order="bottom_to_top",
            ),
            score=0.5,
        ),
    ]
