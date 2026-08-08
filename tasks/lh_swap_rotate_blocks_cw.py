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

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_in_front_of,
    object_left_of,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@atomic
def _rotated_square_success(env, env_id=None, object=None, objects=None):
    """After clockwise rotation the square must be:
        front-left: yellow   front-right: red
        back-left:  green    back-right:  blue
    Check all four pairwise relationships to fully specify the arrangement.

    Accepts and ignores ``object``/``objects`` kwargs so callers can
    pass them via ``partial(_rotated_square_success, object=[…])`` for
    the GTStateExporter's ``object_completed`` lookup (audit Gap 1).
    """
    return (
        object_left_of(env, object="yellow_block", reference_object="red_block",
                        require_gripper_detached=True, env_id=env_id)
        & object_left_of(env, object="green_block", reference_object="blue_block",
                           require_gripper_detached=True, env_id=env_id)
        & object_in_front_of(env, object="yellow_block", reference_object="green_block",
                               require_gripper_detached=True, env_id=env_id)
        & object_in_front_of(env, object="red_block", reference_object="blue_block",
                               require_gripper_detached=True, env_id=env_id)
    )


@configclass
class RotateBlocksClockwiseTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_rotated_square_success,
        params={},
    )


@dataclass
class RotateBlocksClockwiseTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'yellow_block', 'table']
    scene = import_scene("lh_four_blocks_square.usda", contact_object_list)
    terminations = RotateBlocksClockwiseTerminations
    instruction = {
        "default": "Rotate each block one position clockwise \u2014 each block moves to the next position around the square",
        "vague": "Shift all blocks one spot clockwise",
        "specific": "Move each block to the next corner position going clockwise, using a buffer for the first block",
    }
    episode_length_s: int = 120
    attributes = ['memory', 'swapping', 'spatial', 'cyclic']
    subtasks = [
        Subtask(
            name="complete",
            conditions=partial(
                _rotated_square_success,
                object=["red_block", "blue_block", "green_block", "yellow_block"],
            ),
            score=1.0,
        ),
    ]
