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
from robolab.core.task.conditionals import object_at
from robolab.core.scenes.utils import import_scene


def _equation_solved(env, env_id=None):
    """Success: all cubes involved in the equation are at their correct positions.

    Equation: 9 - 6 = 3
    Only checks cubes that are part of the equation; ignores distractor cubes.
    """
    tol = 0.07
    # Check each equation cube is at its designated position
    result = object_at(
        env, object="wood_math_cube_9",
        position=(0.55, 0.16, 0.033), tolerance=tol, env_id=env_id,
    )
    result = result & object_at(
        env, object="wood_math_cube_minus",
        position=(0.55, 0.08, 0.033), tolerance=tol, env_id=env_id,
    )
    result = result & object_at(
        env, object="wood_math_cube_6",
        position=(0.55, 0.00, 0.033), tolerance=tol, env_id=env_id,
    )
    result = result & object_at(
        env, object="wood_math_cube_equals",
        position=(0.55, -0.08, 0.033), tolerance=tol, env_id=env_id,
    )
    result = result & object_at(
        env, object="wood_math_cube_3",
        position=(0.55, -0.16, 0.033), tolerance=tol,
        require_gripper_detached=True, env_id=env_id,
    )
    return result


@configclass
class MathSubtractionTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_equation_solved,
    )


@dataclass
class MathSubtractionTask(Task):
    contact_object_list = [
        "table",
        "wood_math_cube_9",
        "wood_math_cube_minus",
        "wood_math_cube_6",
        "wood_math_cube_equals",
        "wood_math_cube_2",
        "wood_math_cube_3",
        "wood_math_cube_5",
    ]
    scene = import_scene("wk_math_subtraction.usda", contact_object_list)
    terminations = MathSubtractionTerminations
    instruction = {
        "default": "Move a cube to complete the equation",
        "vague": "Solve the math problem",
        "specific": "Complete the equation: place the correct cube after the equals sign to solve 9 − 6 = ?",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "arithmetic"]
    subtasks = [
        Subtask(
            name="place_answer",
            conditions=partial(
                _equation_solved,
            ),
            score=1.0,
        ),
    ]
