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
from robolab.core.task.decorators import atomic
from robolab.core.scenes.utils import import_scene


@atomic
def _math_addition_success(env, env_id=None):
    """Success: cube 7 at answer position AND equation blocks (3, +, 4, =) not moved."""
    answer_placed = object_at(
        env,
        object="wood_math_cube_7",
        position=(0.566, -0.182, 0.033),
        tolerance=0.07,
        require_gripper_detached=True,
        env_id=env_id,
    )
    cube_3_unmoved = object_at(
        env,
        object="wood_math_cube_3",
        position=(0.569, 0.137, 0.033),
        tolerance=0.07,
        env_id=env_id,
    )
    cube_plus_unmoved = object_at(
        env,
        object="wood_math_cube_plus",
        position=(0.568, 0.057, 0.033),
        tolerance=0.07,
        env_id=env_id,
    )
    cube_4_unmoved = object_at(
        env,
        object="wood_math_cube_4",
        position=(0.568, -0.022, 0.033),
        tolerance=0.07,
        env_id=env_id,
    )
    cube_equals_unmoved = object_at(
        env,
        object="wood_math_cube_equals",
        position=(0.567, -0.102, 0.033),
        tolerance=0.07,
        env_id=env_id,
    )
    return answer_placed & cube_3_unmoved & cube_plus_unmoved & cube_4_unmoved & cube_equals_unmoved


@configclass
class MathAdditionTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_math_addition_success,
        params={},
    )


@dataclass
class MathAdditionTask(Task):
    contact_object_list = [
        "table",
        "wood_math_cube_3",
        "wood_math_cube_plus",
        "wood_math_cube_4",
        "wood_math_cube_equals",
        "wood_math_cube_5",
        "wood_math_cube_7",
        "wood_math_cube_9",
    ]
    scene = import_scene("wk_math_addition.usda", contact_object_list)
    terminations = MathAdditionTerminations
    instruction = {
        "default": "Place the correct cube in the blank to complete the equation",
        "vague": "Solve the math problem",
        "specific": "Complete the equation: place the correct cube after the equals sign to solve 3 + 4 = ?",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "arithmetic"]
    subtasks = [
        Subtask(
            name="place_answer",
            conditions=_math_addition_success,
            score=1.0,
        ),
    ]
