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
import torch
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_at
from robolab.core.scenes.utils import import_scene


def math_multiply_success(
    env,
    answer_object: str,
    answer_position: tuple,
    equation_objects: list,
    equation_positions: list,
    tolerance: float = 0.07,
    require_gripper_detached: bool = True,
    env_id: int = 0,
):
    """Check that the answer cube is placed correctly AND equation blocks are unmoved."""
    # Check answer placement
    answer_ok = object_at(
        env,
        object=answer_object,
        position=answer_position,
        tolerance=tolerance,
        require_gripper_detached=require_gripper_detached,
        env_id=env_id,
    )
    # Check each equation block is still in place
    for obj, pos in zip(equation_objects, equation_positions):
        obj_ok = object_at(
            env,
            object=obj,
            position=pos,
            tolerance=tolerance,
            env_id=env_id,
        )
        answer_ok = answer_ok & obj_ok
    return answer_ok


@configclass
class MathMultiplyTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=math_multiply_success,
        params={
            "answer_object": "wood_math_cube_8",
            "answer_position": (0.55, -0.16, 0.032),
            "equation_objects": [
                "wood_math_cube_4",
                "wood_math_cube_multiply",
                "wood_math_cube_2",
                "wood_math_cube_equals",
            ],
            "equation_positions": [
                (0.55, 0.16, 0.032),
                (0.55, 0.08, 0.032),
                (0.55, 0.0, 0.032),
                (0.55, -0.08, 0.032),
            ],
            "tolerance": 0.07,
            "require_gripper_detached": True,
        },
    )


@dataclass
class MathMultiplyTask(Task):
    contact_object_list = [
        "table",
        "wood_math_cube_4",
        "wood_math_cube_multiply",
        "wood_math_cube_2",
        "wood_math_cube_equals",
        "wood_math_cube_6",
        "wood_math_cube_8",
        "wood_math_cube_9",
    ]
    scene = import_scene("wk_math_multiply.usda", contact_object_list)
    terminations = MathMultiplyTerminations
    instruction = {
        "default": "Move a cube to complete the equation",
        "vague": "Solve the math problem",
        "specific": "Complete the equation: place the correct cube after the equals sign to solve 4 × 2 = ?",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "arithmetic"]
    subtasks = [
        Subtask(
            name="place_answer",
            conditions=partial(
                math_multiply_success,
                answer_object="wood_math_cube_8",
                answer_position=(0.55, -0.16, 0.032),
                equation_objects=[
                    "wood_math_cube_4",
                    "wood_math_cube_multiply",
                    "wood_math_cube_2",
                    "wood_math_cube_equals",
                ],
                equation_positions=[
                    (0.55, 0.16, 0.032),
                    (0.55, 0.08, 0.032),
                    (0.55, 0.0, 0.032),
                    (0.55, -0.08, 0.032),
                ],
                tolerance=0.07,
                require_gripper_detached=True,
            ),
            score=1.0,
        ),
    ]
