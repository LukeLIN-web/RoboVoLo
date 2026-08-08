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


def _car_wheel_success(env, env_id=None):
    """Success: rear_wheel placed at target AND other painting parts remain unmoved (4cm tolerance)."""
    # Check rear_wheel is at its target position
    result = object_at(
        env, object="rear_wheel",
        position=(0.47, -0.05, 0.0095), tolerance=0.04,
        require_gripper_detached=True, env_id=env_id,
    )
    # Check other painting parts remain near their starting positions (same tolerance as manipulated object)
    result = result & object_at(
        env, object="road",
        position=(0.443, 0.0, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="car_body",
        position=(0.507, 0.0, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="car_roof",
        position=(0.541, -0.001, 0.011), tolerance=0.04, env_id=env_id,
    )
    result = result & object_at(
        env, object="front_wheel",
        position=(0.476, 0.051, 0.012), tolerance=0.04, env_id=env_id,
    )
    return result


@configclass
class ArtCarWheelTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_car_wheel_success,
    )


@dataclass
class ArtCarWheelTask(Task):
    contact_object_list = ['table', 'road', 'car_body', 'car_roof', 'front_wheel', 'rear_wheel', 'distractor_square', 'distractor_triangle', 'distractor_rect']
    scene = import_scene("wk_art_car_wheel.usda", contact_object_list)
    terminations = ArtCarWheelTaskTerminations
    instruction = {'default': 'Install the missing wheel on the car', 'vague': 'Fix the car', 'specific': 'Pick up the charcoal circle and place it under the rear of the car as the missing wheel, matching the front wheel position'}
    episode_length_s: int = 90
    attributes = ["world_knowledge", "art", "spatial_reasoning", "distractors"]
    subtasks = [
        Subtask(
            name="place_piece",
            conditions=partial(
                _car_wheel_success,
            ),
            score=1.0,
        ),
    ]
