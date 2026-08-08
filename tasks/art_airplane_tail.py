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


def _airplane_tail_success(env, env_id=None):
    """Success: tail_right placed at target AND other airplane parts remain unmoved (4cm tolerance)."""
    # Check tail_right is at its target position
    result = object_at(
        env, object="tail_right",
        position=(0.43, -0.03, 0.0095), tolerance=0.04,
        require_gripper_detached=True, env_id=env_id,
    )
    # Check other airplane parts remain near their starting positions (same 4cm tolerance)
    result = result & object_at(
        env, object="fuselage",
        position=(0.520, 0.0, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="left_wing",
        position=(0.534, 0.064, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="right_wing",
        position=(0.530, -0.062, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="tail_left",
        position=(0.428, 0.031, 0.011), tolerance=0.04, env_id=env_id,
    )
    return result


@configclass
class ArtAirplaneTailTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_airplane_tail_success,
    )


@dataclass
class ArtAirplaneTailTask(Task):
    contact_object_list = ['table', 'fuselage', 'left_wing', 'right_wing', 'tail_left', 'tail_right', 'distractor_red', 'distractor_blue']
    scene = import_scene("wk_art_airplane_tail.usda", contact_object_list)
    terminations = ArtAirplaneTailTaskTerminations
    instruction = {'default': 'Complete the airplane by attaching the missing tail fin', 'vague': 'Finish the airplane', 'specific': 'Pick up the small charcoal rectangle and place it on the tail end of the airplane, symmetric with the existing tail fin'}
    episode_length_s: int = 90
    attributes = ["world_knowledge", "art", "spatial_reasoning"]
    subtasks = [
        Subtask(
            name="place_piece",
            conditions=partial(
                _airplane_tail_success,
            ),
            score=1.0,
        ),
    ]
