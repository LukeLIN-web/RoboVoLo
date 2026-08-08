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
from robolab.core.task.predicate_logic import _and
from robolab.core.task.decorators import atomic
from robolab.core.scenes.utils import import_scene

# Gap positions in equation (left-to-right): 4 × [gap_left] = 5 + [gap_right]
# gap_left at Y=0.08, gap_right at Y=-0.24
LEFT_GAP = (0.55, 0.08, 0.032)
RIGHT_GAP = (0.55, -0.24, 0.032)

# Positions of equation blocks that should NOT be moved
EQUATION_BLOCK_POSITIONS = {
    "wood_math_cube_4": (0.55, 0.24, 0.032),
    "wood_math_cube_multiply": (0.55, 0.16, 0.032),
    "wood_math_cube_equals": (0.55, 0.0, 0.032),
    "wood_math_cube_5": (0.55, -0.08, 0.032),
    "wood_math_cube_plus": (0.55, -0.16, 0.032),
}


@atomic
def _balance_equation_success(env, env_id=None):
    """Check that cube_3 is in the left gap and cube_7 is in the right gap (4×3=12=5+7),
    and that the equation blocks (4, ×, =, 5, +) have not been moved."""
    # Check answer cubes are placed correctly. Predicates return a (num_envs,)
    # tensor on the vectorized eval path (env_id=None) or a bool on the scalar
    # path, so combine with the tensor-safe _and helper instead of Python `and`.
    result = object_at(env, object="wood_math_cube_3", position=LEFT_GAP, tolerance=0.07, require_gripper_detached=True, env_id=env_id)
    result = _and(result, object_at(env, object="wood_math_cube_7", position=RIGHT_GAP, tolerance=0.07, require_gripper_detached=True, env_id=env_id))
    # Check equation blocks have not been moved
    for obj_name, pos in EQUATION_BLOCK_POSITIONS.items():
        result = _and(result, object_at(env, object=obj_name, position=pos, tolerance=0.07, env_id=env_id))
    return result


@configclass
class MathBalanceEquationTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_balance_equation_success, params={})


@dataclass
class MathBalanceEquationTask(Task):
    contact_object_list = [
        "table",
        "wood_math_cube_4",
        "wood_math_cube_multiply",
        "wood_math_cube_equals",
        "wood_math_cube_5",
        "wood_math_cube_plus",
        "wood_math_cube_3",
        "wood_math_cube_7",
        "wood_math_cube_9",
    ]
    scene = import_scene("wk_math_balance_equation.usda", contact_object_list)
    terminations = MathBalanceEquationTerminations
    instruction = {
        "default": "Place the appropriate blocks in both blanks to complete the equation",
        "vague": "Make both sides equal",
        "specific": "Balance the equation: place cubes in both blanks so that 4 × ? = 5 + ?",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "algebra", "multi_step"]
    subtasks = [
        Subtask(
            name="balance_both_sides",
            conditions=_balance_equation_success,
            score=1.0,
        ),
    ]
