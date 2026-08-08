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
    object_left_of,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@atomic
def _compound_success(env, env_id=0, object=None, objects=None):
    """corn_can left of tomato_soup_can AND tomato_soup_can left of tuna_can (full order check).

    Accepts and ignores ``object``/``objects`` kwargs so callers can
    pass them via ``partial(_compound_success, object=[…])`` for the
    GTStateExporter's ``object_completed`` lookup (audit Gap 1).
    """
    return (
        object_left_of(env, object="corn_can", reference_object="tomato_soup_can", require_gripper_detached=True)
        & object_left_of(env, object="tomato_soup_can", reference_object="tuna_can", require_gripper_detached=True)
    )


@configclass
class RotateCansLeftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_compound_success, params={})


@dataclass
class RotateCansLeftTask(Task):
    contact_object_list = ['tuna_can', 'corn_can', 'tomato_soup_can', 'table']
    scene = import_scene("lh_three_cans_row.usda", contact_object_list)
    terminations = RotateCansLeftTerminations
    instruction = {
        "default": "Rotate the three cans one position to the left \u2014 the leftmost can wraps around to the right position",
        "vague": "Shift all cans one spot to the left",
        "specific": "Move the leftmost can to a buffer spot, shift the middle can left, shift the right can to the middle, then place the buffer can on the right",
    }
    episode_length_s: int = 120
    attributes = ['memory', 'swapping', 'spatial', 'cyclic']
    subtasks = [
        Subtask(
            name="complete",
            conditions=partial(
                _compound_success,
                object=["corn_can", "tomato_soup_can", "tuna_can"],
            ),
            score=1.0,
        ),
    ]
