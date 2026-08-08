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


def equation_solved(env, env_id=None):
    """Success: cube_5 placed in the blank AND equation blocks haven't moved."""
    answer_ok = object_at(
        env,
        object="wood_math_cube_5",
        position=(0.55, -0.16, 0.032),
        tolerance=0.07,
        require_gripper_detached=True,
        env_id=env_id,
    )
    plus_ok = object_at(
        env, object="wood_math_cube_plus", position=(0.55, -0.08, 0.032), tolerance=0.07, env_id=env_id
    )
    three_ok = object_at(
        env, object="wood_math_cube_3", position=(0.55, 0.0, 0.032), tolerance=0.07, env_id=env_id
    )
    equals_ok = object_at(
        env, object="wood_math_cube_equals", position=(0.55, 0.08, 0.032), tolerance=0.07, env_id=env_id
    )
    eight_ok = object_at(
        env, object="wood_math_cube_8", position=(0.55, 0.16, 0.032), tolerance=0.07, env_id=env_id
    )

    if env_id is not None:
        return answer_ok and plus_ok and three_ok and equals_ok and eight_ok
    else:
        return answer_ok & plus_ok & three_ok & equals_ok & eight_ok


@configclass
class MathMissingOperandTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=equation_solved,
        params={},
    )


@dataclass
class MathMissingOperandTask(Task):
    contact_object_list = [
        "table",
        "wood_math_cube_plus",
        "wood_math_cube_3",
        "wood_math_cube_equals",
        "wood_math_cube_8",
        "wood_math_cube_2",
        "wood_math_cube_4",
        "wood_math_cube_5",
        "wood_math_cube_7",
    ]
    scene = import_scene("wk_math_missing_operand.usda", contact_object_list)
    terminations = MathMissingOperandTerminations
    instruction = {
        "default": "Move a cube to complete the equation",
        "vague": "Find the missing number",
        "specific": "Complete the equation: place the correct cube in the blank to solve ? + 3 = 8",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "algebra"]
    subtasks = [
        Subtask(
            name="place_answer",
            conditions=partial(
                equation_solved,
            ),
            score=1.0,
        ),
    ]
