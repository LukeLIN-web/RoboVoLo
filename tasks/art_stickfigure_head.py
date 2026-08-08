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


def _stickfigure_head_success(env, env_id=None):
    """Success: head placed at target AND other stick figure parts remain unmoved (4cm tolerance)."""
    # Check head is at its target position
    result = object_at(
        env, object="head",
        position=(0.6, 0.0, 0.0095), tolerance=0.04,
        require_gripper_detached=True, env_id=env_id,
    )
    # Check other stick figure parts remain near their starting positions (same 4cm tolerance)
    result = result & object_at(
        env, object="body",
        position=(0.520, 0.0, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="left_arm",
        position=(0.540, 0.050, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="right_arm",
        position=(0.540, -0.050, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="left_leg",
        position=(0.437, 0.020, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="right_leg",
        position=(0.436, -0.020, 0.011), tolerance=0.04, env_id=env_id,
    )
    return result


@configclass
class ArtStickfigureHeadTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_stickfigure_head_success,
    )


@dataclass
class ArtStickfigureHeadTask(Task):
    contact_object_list = ['table', 'body', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'head', 'distractor_triangle', 'distractor_square']
    scene = import_scene("wk_art_stickfigure_head.usda", contact_object_list)
    terminations = ArtStickfigureHeadTaskTerminations
    instruction = {'default': 'Complete the stick figure by placing the missing head', 'vague': 'Finish the drawing', 'specific': 'Pick up the circle piece and place it at the top of the stick figure where the head should be'}
    episode_length_s: int = 90
    attributes = ["world_knowledge", "art", "spatial_reasoning"]
    subtasks = [
        Subtask(
            name="place_piece",
            conditions=partial(
                _stickfigure_head_success,
            ),
            score=1.0,
        ),
    ]
