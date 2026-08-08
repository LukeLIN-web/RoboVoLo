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
from robolab.core.task.conditionals import stacked
from robolab.core.scenes.utils import import_scene


@configclass
class MathPrimeStackTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={
            "objects": ["wood_math_cube_2", "wood_math_cube_3", "wood_math_cube_5"],
            "order": None,
        },
    )


@dataclass
class MathPrimeStackTask(Task):
    contact_object_list = [
        "table",
        "wood_math_cube_2",
        "wood_math_cube_3",
        "wood_math_cube_4",
        "wood_math_cube_5",
        "wood_math_cube_7",
        "wood_math_cube_8",
    ]
    scene = import_scene("wk_math_prime_stack.usda", contact_object_list)
    terminations = MathPrimeStackTerminations
    instruction = {
        "default": "Build a stack of cubes out of the prime factors of 30 only",
        "vague": "Build a tower with the right numbers",
        "specific": "The prime factors of 30 are 2, 3, and 5. Stack the cubes showing 2, 3, and 5 on top of each other in any order",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "number_theory", "stacking"]
    subtasks = [
        Subtask(
            name="stack_prime_factors",
            conditions=partial(
                stacked,
                objects=["wood_math_cube_2", "wood_math_cube_3", "wood_math_cube_5"],
                order=None,
            ),
            score=1.0,
        ),
    ]
