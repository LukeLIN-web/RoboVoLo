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

# Equation: 4 × ? = 5 + ?  (reading left to right, +Y to -Y)
# Layout (Y positions): 4(0.24) ×(0.16) gap1(0.08) =(0.00) 5(-0.08) +(-0.16) gap2(-0.24)
# Solution: 4 × 2 = 5 + 3 → 8 = 8
GAP1_POS = (0.55, 0.08, 0.032)   # first blank (multiplication operand)
GAP2_POS = (0.55, -0.24, 0.032)  # second blank (addition operand)

# Fixed equation cube positions (should not be moved)
CUBE_4_POS = (0.55, 0.24, 0.032)
MULTIPLY_POS = (0.55, 0.16, 0.032)
EQUALS_POS = (0.55, 0.0, 0.032)
CUBE_5_POS = (0.55, -0.08, 0.032)
PLUS_POS = (0.55, -0.16, 0.032)


@atomic
def _factor_pair_success(env, env_id=None):
    """Check that cubes 2 and 3 fill the two gaps correctly (4×2=5+3),
    and that the fixed equation cubes have not been moved."""
    # Check answer cubes are in the correct positions
    answer_correct = (
        object_at(env, object="wood_math_cube_2", position=GAP1_POS, tolerance=0.07, require_gripper_detached=True)
        & object_at(env, object="wood_math_cube_3", position=GAP2_POS, tolerance=0.07, require_gripper_detached=True)
    )

    # Check that fixed equation cubes have not been moved
    fixed_cubes_intact = (
        object_at(env, object="wood_math_cube_4", position=CUBE_4_POS, tolerance=0.07, require_gripper_detached=True)
        & object_at(env, object="wood_math_cube_multiply", position=MULTIPLY_POS, tolerance=0.07, require_gripper_detached=True)
        & object_at(env, object="wood_math_cube_equals", position=EQUALS_POS, tolerance=0.07, require_gripper_detached=True)
        & object_at(env, object="wood_math_cube_5", position=CUBE_5_POS, tolerance=0.07, require_gripper_detached=True)
        & object_at(env, object="wood_math_cube_plus", position=PLUS_POS, tolerance=0.07, require_gripper_detached=True)
    )

    return answer_correct & fixed_cubes_intact


@configclass
class MathFactorPairTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_factor_pair_success, params={})


@dataclass
class MathFactorPairTask(Task):
    contact_object_list = [
        "table",
        "wood_math_cube_4",
        "wood_math_cube_multiply",
        "wood_math_cube_equals",
        "wood_math_cube_5",
        "wood_math_cube_plus",
        "wood_math_cube_1",
        "wood_math_cube_2",
        "wood_math_cube_3",
        "wood_math_cube_6",
        "wood_math_cube_9",
    ]
    scene = import_scene("wk_math_factor_pair.usda", contact_object_list)
    terminations = MathFactorPairTerminations
    instruction = {
        "default": "Place the appropriate blocks in the blanks to complete the equation",
        "vague": "Make both sides of the equation equal",
        "specific": "Balance the equation: place cubes in both blanks so that 4 × ? = 5 + ?",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "algebra", "multi_step"]
    subtasks = [
        Subtask(
            name="place_both_factors",
            conditions=_factor_pair_success,
            score=1.0,
        ),
    ]
