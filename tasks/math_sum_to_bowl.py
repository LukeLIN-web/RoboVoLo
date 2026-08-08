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
    object_in_container,
    object_outside_of_and_on_surface,
)
from robolab.core.task.decorators import atomic
from robolab.core.scenes.utils import import_scene


@atomic
def _sum_to_bowl_success(env, env_id=None):
    """Cubes 1, 3, 5 must be in the bowl (1+3+5=9). Cubes 7, 8 must NOT be in the bowl and must remain on the table."""
    correct_in = object_in_container(
        env,
        object=["wood_math_cube_1", "wood_math_cube_3", "wood_math_cube_5"],
        container="bowl_015",
        logical="all",
        require_gripper_detached=True,
    )
    wrong_out = object_outside_of_and_on_surface(
        env,
        object=["wood_math_cube_7", "wood_math_cube_8"],
        container="bowl_015",
        surface="table",
        logical="all",
        require_gripper_detached=True,
    )
    return correct_in & wrong_out


@configclass
class MathSumToBowlTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_sum_to_bowl_success, params={})


@dataclass
class MathSumToBowlTask(Task):
    contact_object_list = [
        "table",
        "bowl_015",
        "wood_math_cube_1",
        "wood_math_cube_3",
        "wood_math_cube_5",
        "wood_math_cube_7",
        "wood_math_cube_8",
    ]
    scene = import_scene("wk_math_sum_to_bowl.usda", contact_object_list)
    terminations = MathSumToBowlTerminations
    instruction = {
        "default": "Put exactly three cubes that add up to 9 into the bowl",
        "vague": "Pick cubes that sum to the target",
        "specific": "Find three number cubes from the table whose values add up to 9 and place them into the bowl. Leave the others on the table.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "math", "combinatorics"]
    subtasks = [
        Subtask(
            name="correct_cubes_in_bowl",
            conditions=_sum_to_bowl_success,
            score=1.0,
        ),
    ]
